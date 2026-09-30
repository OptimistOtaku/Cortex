"""Comms windows: short, scarce, metered contact with ground control.

The scarcity is imposed on purpose (like a relay orbiter's pass), but the bytes and the network are real: each
downlink request body is built to fit the window's byte budget and measured before it is sent.

Per window:
  1. uplink   - GET signed commands (mission intent, questions, relabels, supersedes, discards, full-res requests)
                and fleet knowledge. Commands whose HMAC doesn't verify are refused.
  2. downlink - fill the budget with the sync gate's most valuable memories and POST them.
Anything not acknowledged stays pending and competes again in the next window.

Windows open on a timer (SIETCH_GAP="25,45" seconds), or only when asked (SIETCH_GAP=manual): the Mars simulator
opens one whenever the relay orbiter is overhead.
"""

import json
import os
import random
import threading
import time

import requests

from sietch.common.signing import verify

MANUAL = float("inf")


def _range(env, default):
    lo, hi = (float(x) for x in os.environ.get(env, default).split(","))
    return lo, hi


class Comms:
    def __init__(self, cfg, mem, store, brain=None):
        self.cfg, self.mem, self.store, self.brain = cfg, mem, store, brain
        self.manual = os.environ.get("SIETCH_GAP", "") == "manual"
        self.gap_s = (0.0, 0.0) if self.manual else _range("SIETCH_GAP", "25,45")
        self.window_s = float(os.environ.get("SIETCH_WINDOW", "15"))
        self.budget_b = _range("SIETCH_BUDGET", "20000,60000")
        self.link_up = True
        self.window_no = int(store.meta("window_no", "0"))
        self.cmd_since = int(store.meta("cmd_since", "0"))
        self.fleet_since = float(store.meta("fleet_since", "0"))
        self.phase = "waiting"
        self.next_open = MANUAL if self.manual else time.time() + random.uniform(*self.gap_s) / 3
        self.next_budget = None
        self.last = {}
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def open_now(self, budget=None):
        self.next_budget = int(budget) if budget else None
        self.next_open = time.time()
        self._wake.set()

    def status(self):
        return {
            "phase": self.phase, "link_up": self.link_up, "window": self.window_no, "manual": self.manual,
            "next_open_in_s": None if self.next_open == MANUAL else max(0.0, round(self.next_open - time.time(), 1)),
            "window_s": self.window_s, "last": self.last,
        }

    def _loop(self):
        while True:
            self._wake.wait(timeout=None if self.next_open == MANUAL else max(0.0, self.next_open - time.time()))
            self._wake.clear()
            if time.time() < self.next_open:
                continue
            if not self.link_up:
                self.store.event("comms", "window missed: link down (blackout); memory keeps working offline")
            else:
                try:
                    self.run_window()
                except requests.RequestException as e:
                    self.store.event("comms", f"window {self.window_no} lost contact: {type(e).__name__}; unsent items stay queued")
                except Exception as e:  # keep the loop alive through bugs during the demo
                    self.store.event("error", f"window {self.window_no}: {type(e).__name__}: {e}")
            self.phase = "waiting"
            self.next_open = MANUAL if self.manual else time.time() + random.uniform(*self.gap_s)

    def run_window(self):
        self.window_no += 1
        self.store.set_meta("window_no", str(self.window_no))
        budget = self.next_budget or int(random.uniform(*self.budget_b))
        self.next_budget = None
        closes = time.time() + self.window_s
        self.phase = "open"
        url = self.cfg.ground_url

        # 1. uplink: commands + fleet knowledge
        status = json.dumps({"window": self.window_no, "memories": self.store.count(), "intent": self.mem.intent_text,
                             "sent": self.store.bytes_sent()})
        r = requests.get(f"{url}/api/uplink", params={"node": self.cfg.node_id, "since": self.cmd_since,
                                                      "fleet_since": self.fleet_since, "status": status}, timeout=5)
        r.raise_for_status()
        up = r.json()
        commands = self._apply(up)

        # 2. downlink: the gate's most valuable memories that fit the budget
        selected, skipped = self.mem.plan_window(budget)
        body, sent = self._fill(selected, budget)
        self.last = {"window": self.window_no, "ts": time.time(), "budget": budget, "uplink_bytes": len(r.content),
                     "commands": commands, "bytes": 0, "items": [], "waiting": len(skipped),
                     "why_skipped": [f"{it.get('ref', it['id'][:8])}: {why}" for it, why in skipped if "budget" not in why][:5]}
        if not sent:
            return
        resp = requests.post(f"{url}/api/downlink", data=body, headers={"Content-Type": "application/json"},
                             timeout=max(1.0, closes - time.time()))
        resp.raise_for_status()
        for it, nbytes in sent:
            self.mem.mark_sent(it)
            self.store.record_sent(self.window_no, it["id"], it["rep"], nbytes)
        self.last.update(bytes=len(body), items=[{"id": it["id"], "ref": it.get("ref"), "kind": it.get("kind", "observation"),
                                                   "rep": it["rep"], "bytes": nbytes, "tags": it["tags"]} for it, nbytes in sent])
        self.store.event("downlink", f"window {self.window_no} sent {len(sent)} items, {len(body)}/{budget} B")

    def _fill(self, selected, budget):
        """Add packaged items while the serialized request body stays within budget."""
        env = {"node": self.cfg.node_id, "window": self.window_no, "items": []}
        chosen, body = [], json.dumps(env).encode()
        for it in selected:
            pkg = self.mem.package(it)
            env["items"].append(pkg)
            trial = json.dumps(env).encode()
            if len(trial) > budget:
                env["items"].pop()
                continue
            chosen.append((it, len(json.dumps(pkg))))
            body = trial
        return body, chosen

    def _apply(self, up):
        applied = []
        for c in up["commands"]:
            kind, b = c["kind"], c["body"]
            self.cmd_since = max(self.cmd_since, c["id"])
            if not verify(c):
                self.store.event("security", f"REFUSED unsigned/forged '{kind}' command #{c['id']} {json.dumps(b)[:80]}")
                applied.append({"kind": kind, "refused": True})
                continue
            applied.append({"kind": kind, "refused": False})
            if kind == "intent":
                self.mem.set_intent(b["text"], c["hlc"])
            elif kind == "relabel":
                self.mem.relabel(b["item"], b["label"], b.get("by", "ground"), c["hlc"])
            elif kind == "discard":
                self.mem.discard(b["item"], c["hlc"])
            elif kind == "request_full":
                self.mem.request_full(b["item"])
            elif kind == "supersede":
                self.mem.supersede(b["item"], b.get("by", "mission control"), b.get("winner", ""), c["hlc"])
            elif kind == "ask":
                self.store.event("uplink", f"question from mission control: {b['question'][:100]!r}")
                if not (self.brain and self.brain.submit("ask", question=b["question"], qid=b["qid"], source="ground")):
                    if self.brain:
                        self.brain.answer_without_brain(b["question"], qid=b["qid"])
        if up["fleet"]:
            self.mem.apply_fleet(up["fleet"])
            self.fleet_since = max(self.fleet_since, max(f["received_at"] for f in up["fleet"]))
            self.store.event("uplink", f"fleet knowledge: {len(up['fleet'])} items other robots already delivered")
        self.store.set_meta("cmd_since", str(self.cmd_since))
        self.store.set_meta("fleet_since", str(self.fleet_since))
        return applied
