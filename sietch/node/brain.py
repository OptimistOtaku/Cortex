"""The robot's brain: a local LLM that thinks with SIETCH memory.

Reflex vs deliberation: every observation gets the millisecond reflex (embedding + Formula gate). The LLM
(qwen3-vl:2b-instruct on Ollama: tools + vision, ~10 s per deliberation on a GTX 1650, spike_12) only deliberates
when something deserves it:
  examine - the reflex flagged a frame (relevant to the mission and novel) or an operator asked it to look.
            It captions the frame, recalls related memories, remembers an insight citing evidence, and acts.
  ask     - a question from the operator or from mission control; answered offline from memory, with citations.
One worker, one task at a time (one GPU). If Ollama is missing or fails, the robot keeps its reflexes (reflex-only).
"""

import base64
import json
import os
import queue
import re
import threading
import time
import uuid

import requests

from sietch.common.geo import where
from sietch.node.memory import TEXT_KINDS, ref

TOOLS = [
    {"type": "function", "function": {
        "name": "recall",
        "description": "Search your memory (your own observations and conclusions, plus what the fleet shared).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what to look for, a few words"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "remember",
        "description": "Store ONE insight: what the observation means for the mission. It reaches mission control first.",
        "parameters": {"type": "object", "properties": {
            "insight": {"type": "string", "description": "a conclusion, not a description; cite the memory ids you were given, in square brackets"},
            "importance": {"type": "number", "description": "0..1, how much mission control needs this"}},
            "required": ["insight", "importance"]}}},
    {"type": "function", "function": {
        "name": "act",
        "description": "Choose what to do next.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["investigate", "flag_for_ground", "move_on"]},
            "reason": {"type": "string"}}, "required": ["action", "reason"]}}},
]

EXAMINE_SYSTEM = """You are the onboard brain of {node}, an autonomous explorer with no link to mission control right now.
Mission: {intent}
You have long-term memory. Work in this order:
1. Call recall once with a short query about what you see.
2. Call remember with ONE insight: what this observation MEANS for the mission, combined with what you recalled.
   Write a conclusion, not a description, and cite the memory ids you rely on in square brackets.
   Shape: "<what you see> [{me}] together with <what you recalled> [<its id>] suggests <what it means for the mission>."
   Use your own words about THIS frame. A conclusion, not a description ("The image shows X" is useless).
   Cite only ids that appear in this conversation; never invent one.
   Be a sceptical scientist: judge only what THIS frame shows. If it shows nothing the mission is looking for
   (for example only wind-blown sand, plain soil or hardware), say so plainly, e.g. "No sign of <mission target> here:
   only ...". A false alarm wastes the link.
3. Call act: investigate (worth a closer look), flag_for_ground (mission control must know soon), or move_on (not useful).
Keep each text under 40 words."""

ANSWER_PROMPT = """You are the onboard brain of {node}. Mission: {intent}
Answer the question using ONLY the memories below. Cite the ids you use in square brackets exactly as shown, e.g. [{example}].
Some memories are unrelated search hits: ignore any that do not bear on the question. Start with a direct yes/no when
the question asks one, and do not contradict yourself. If the memories do not answer it, say so. At most 3 sentences.

Memories:
{memories}

Question: {question}"""

# what the brain reasons from: knowledge, not its own audit trail (decisions) or earlier answers
KNOWLEDGE = ("insight", "note")

ACTION_IMPORTANCE = {"flag_for_ground": 1.0, "investigate": 0.8, "move_on": 0.1}


class Brain:
    def __init__(self, cfg, mem, store):
        self.cfg, self.mem, self.store = cfg, mem, store
        self.model = cfg.brain_model
        self.enabled = self.model.lower() not in ("off", "none", "")
        self.q = queue.Queue(maxsize=6)
        self.state = {"status": "idle" if self.enabled else "off (reflex-only)", "model": self.model if self.enabled else None,
                      "task": None, "trace": [], "partial": "", "answers": [], "prompt": None, "last_error": None}
        self._lock = threading.Lock()
        if self.enabled:
            threading.Thread(target=self._worker, daemon=True, name="brain").start()

    # ------------------------------------------------------------ public

    def submit(self, kind, **args):
        if not self.enabled:
            return False
        try:
            self.q.put_nowait((kind, args))
            return True
        except queue.Full:
            self.store.event("brain", f"busy: skipped {kind} ({args.get('why') or args.get('question', '')[:40]})")
            return False

    def consider(self, obs):
        """Reflex hook after a capture: wake the brain only for relevant, novel frames."""
        if not self.enabled or obs.get("kind") != "observation":
            return False
        if obs.get("camera") == "navcam":  # navigation frames feed memory and the gate; science cameras wake the brain
            return False
        row = next((r for r in self.mem.rank() if r["id"] == obs["id"]), None)
        if row and row["relevance"] >= self.cfg.brain_relevance and row["novelty"] >= self.cfg.brain_novelty:
            why = f"reflex: relevance {row['relevance']:.2f}, novelty {row['novelty']:.2f}"
            return self.submit("examine", pid=obs["id"], why=why)
        return False

    def snapshot(self):
        with self._lock:
            return json.loads(json.dumps(self.state, default=str)) | {"queued": self.q.qsize()}

    # ------------------------------------------------------------ worker

    def _set(self, **kw):
        with self._lock:
            self.state.update(kw)

    def _worker(self):
        while True:
            kind, args = self.q.get()
            self._set(status="thinking", task={"kind": kind, **{k: v for k, v in args.items() if k != "pid"},
                                               "ref": ref("observation", args["pid"]) if "pid" in args else None,
                                               "started": time.time()}, trace=[], partial="")
            try:
                if kind == "examine":
                    self._examine(**args)
                elif kind == "ask":
                    self._ask(**args)
                self._set(last_error=None)
            except Exception as e:  # the robot must keep working without its brain
                self._set(last_error=f"{type(e).__name__}: {e}")
                self.store.event("brain", f"{kind} failed ({type(e).__name__}: {str(e)[:120]}); reflexes continue")
                if kind == "ask":
                    self._answer_fallback(args, reason=f"brain error: {type(e).__name__}")
            finally:
                self._set(status="idle", task=None)

    # ------------------------------------------------------------ LLM calls

    def _chat(self, messages, tools=None, stream=False, on_token=None):
        body = {"model": self.model, "messages": messages, "stream": stream, "keep_alive": "30m",
                "options": {"temperature": 0, "num_ctx": 4096}}
        if tools:
            body["tools"] = tools
        # Ollama's runner can die under host-RAM pressure (GGML_ASSERT mem_buffer) and reload a few seconds later;
        # retry transient failures instead of losing the deliberation
        for attempt, backoff in enumerate((5, 15, 30, None)):
            try:
                r = requests.post(f"{self.cfg.ollama_url}/api/chat", json=body, stream=stream, timeout=180)
                if r.status_code < 500:
                    break
                err = f"HTTP {r.status_code}"
            except requests.ConnectionError as e:
                err = type(e).__name__
            if backoff is None:
                break
            self._set(status=f"brain unavailable ({err}), retrying in {backoff}s")
            self._trace("retry", f"{err}; retry {attempt + 1} in {backoff}s")
            time.sleep(backoff)
        self._set(status="thinking")
        r.raise_for_status()
        if not stream:
            return r.json()["message"]
        text = ""
        for line in r.iter_lines():
            if line:
                ev = json.loads(line)
                piece = ev.get("message", {}).get("content", "")
                if piece:
                    text += piece
                    if on_token:
                        on_token(text)
        return {"role": "assistant", "content": text}

    def _trace(self, step, detail):
        with self._lock:
            task = self.state.get("task")
            if task:
                self.state["trace"].append({"t": round(time.time() - task["started"], 1), "step": step, "detail": detail})

    # ------------------------------------------------------------ examine

    def _image_b64(self, pid):
        for kind in ("orig", "full", "thumb"):
            path = self.mem._path(pid, kind)
            if os.path.exists(path):
                from PIL import Image
                import io

                im = Image.open(path).convert("RGB")
                im.thumbnail((448, 448))  # enough for a caption, much faster than full size
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=85)
                return base64.b64encode(buf.getvalue()).decode()
        return None

    def _examine(self, pid, why="asked"):
        rec = self.store.get([pid])
        if not rec:
            return
        obs = rec[0].payload
        me = ref("observation", pid)
        self._trace("wake", why)
        caption = obs.get("caption")
        img = self._image_b64(pid)
        if img:
            msg = self._chat([{"role": "user", "images": [img], "content":
                               # no mission in this prompt: told what to look for, a small vision model sees it everywhere
                               "Describe this camera frame in one sentence: terrain, rocks, textures, colours and any "
                               "objects. Only what is visible; do not interpret."}])
            caption = msg.get("content", "").strip() or caption
            self.mem.set_caption(pid, caption)
            self._trace("see", caption)
        messages = [
            {"role": "system", "content": EXAMINE_SYSTEM.format(node=self.cfg.node_id, intent=self.mem.intent_text, me=me)},
            {"role": "user", "content": f"New observation [{me}] ({why}). Camera: {caption}. Tags: {', '.join(obs['tags'])}. "
                                        f"Novelty vs everything known: {obs.get('novelty', 0):.2f}."},
        ]
        remembered, acted = None, None
        for _ in range(5):
            msg = self._chat(messages, tools=TOOLS)
            calls = msg.get("tool_calls") or []
            if not calls:
                if msg.get("content"):
                    self._trace("think", msg["content"][:200])
                break
            messages.append(msg)
            for c in calls:
                name, args = c["function"]["name"], c["function"].get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                result = self._tool(name, args, pid, me)
                if name == "remember" and result.get("id"):
                    remembered = result
                if name == "act" and result.get("id"):
                    acted = result
                messages.append({"role": "tool", "tool_name": name, "content": result.get("for_llm", "ok")})
            if acted:
                break
        if not remembered:  # every look ends in a conclusion; "no sign of X here" is worth syncing too
            messages.append({"role": "user", "content": "Now call remember with ONE conclusion about this frame for the "
                                                        "mission. It may be negative, e.g. 'No sign of ... here: only ...'."})
            msg = self._chat(messages, tools=TOOLS)
            for c in msg.get("tool_calls") or []:
                if c["function"]["name"] == "remember":
                    args = c["function"].get("arguments") or {}
                    result = self._tool("remember", json.loads(args) if isinstance(args, str) else args, pid, me)
                    remembered = result if result.get("id") else None
                    break
        if not acted:  # the model stopped without choosing; record that honestly
            acted = self._tool("act", {"action": "move_on", "reason": "no decision from the model"}, pid, me)
        self.store.event("brain", f"examined {me}: {'insight ' + remembered['ref'] if remembered else 'no insight'}, "
                                  f"action {acted.get('action')}")

    def _tool(self, name, args, pid, me):
        if name == "recall":
            q = str(args.get("query") or "")[:120] or self.mem.intent_text
            # frames only: shown its own earlier conclusions, a 2B model copies them instead of judging this frame
            res = self.mem.recall(q, k=5, k_text=0)
            lines = [f"[{r['ref']}]{self._where(r)} {r.get('caption') or r.get('text')}"
                     for r in res["insights"] + res["results"] if r["id"] != pid]
            self._trace("recall", {"query": q, "found": [l[:90] for l in lines]})
            return {"for_llm": "\n".join(lines) or "nothing relevant in memory"}
        if name == "remember":
            text = self.mem.strip_unknown_refs(str(args.get("insight") or args.get("note") or ""))
            if not text:
                return {"for_llm": "empty insight ignored"}
            try:
                importance = float(args.get("importance", 0.5))
            except (TypeError, ValueError):
                importance = 0.5
            evidence = [pid] + [e for e in self.mem.resolve_refs(text) if e != pid]
            ins = self.mem.add_text("insight", text, importance=importance, evidence=evidence, source="brain")
            self._trace("remember", {"ref": ins["ref"], "text": text, "importance": ins["importance"],
                                     "evidence": [ref("observation", e) for e in evidence]})
            return {**ins, "for_llm": f"stored as [{ins['ref']}]"}
        if name == "act":
            action = args.get("action") if args.get("action") in ACTION_IMPORTANCE else "move_on"
            reason = self.mem.strip_unknown_refs(str(args.get("reason") or ""))[:300]
            self.mem.set_importance(pid, max(ACTION_IMPORTANCE[action], 0.0))
            cited = [pid] + [e for e in self.mem.resolve_refs(reason) if e != pid]
            dec = self.mem.add_text("decision", f"{action}: {reason}", importance=0.4, evidence=cited, source="brain",
                                    action=action, reason=reason)
            if action == "investigate":
                self._set(prompt={"text": f"Brain asks for a closer look at {me}: {reason}", "ref": me, "id": pid, "ts": time.time()})
            self._trace("act", {"action": action, "reason": reason})
            return {**dec, "action": action, "for_llm": "done"}
        return {"for_llm": f"unknown tool {name}"}

    # ------------------------------------------------------------ ask

    def _where(self, r, sep=" "):
        """Where a memory is from here ("38 m NE"), so conclusions and answers can point to places."""
        w = where(r.get("pos"), self.mem.pose)
        return f"{sep}{w}" if w else ""

    def _context(self, question):
        res = self.mem.recall(question, k=5, k_text=4, text_kinds=KNOWLEDGE)
        items = res["insights"] + res["results"]
        lines = [f"[{r['ref']}] ({r['kind']}, {r['node']}{self._where(r, ', ')}) {r.get('text') or r.get('caption')}"
                 for r in items]
        return items, "\n".join(lines)

    def _ask(self, question, qid=None, source="operator"):
        items, memories = self._context(question)
        self._trace("recall", {"query": question, "found": [m[:90] for m in memories.splitlines()]})
        prompt = ANSWER_PROMPT.format(node=self.cfg.node_id, intent=self.mem.intent_text, memories=memories or "(none)",
                                      example=items[0]["ref"] if items else "obs-00000000",
                                      question=question)
        msg = self._chat([{"role": "user", "content": prompt}], stream=True, on_token=lambda t: self._set(partial=t))
        answer = self.mem.strip_unknown_refs(msg["content"])
        self._finish_answer(question, answer, qid, source, items)

    def _finish_answer(self, question, answer, qid, source, items):
        cited_ids = self.mem.resolve_refs(answer)
        cited = [{"id": r["id"], "ref": r["ref"], "kind": r["kind"], "caption": r.get("caption") or r.get("text")}
                 for r in items if r["id"] in cited_ids]
        rec = {"qid": qid or str(uuid.uuid4()), "question": question, "answer": answer, "cited": cited,
               "source": source, "ts": time.time()}
        with self._lock:
            self.state["answers"] = ([rec] + self.state["answers"])[:10]
            self.state["partial"] = ""
        if source == "ground":  # the answer goes down in the next window: a few hundred bytes
            self.mem.add_text("answer", f"Q: {question} A: {answer}", importance=1.0, evidence=[c["id"] for c in cited],
                              source="brain", qid=qid, question=question, answer=answer)
        self._trace("answer", answer[:200])
        return rec

    def _answer_fallback(self, args, reason):
        """No brain available: answer with what recall found, so mission control still gets something."""
        items, memories = self._context(args["question"])
        top = "; ".join(f"[{r['ref']}] {(r.get('text') or r.get('caption'))[:60]}" for r in items[:3])
        answer = f"({reason}; closest memories) {top}" if top else f"({reason}) nothing relevant in memory"
        self._finish_answer(args["question"], answer, args.get("qid"), args.get("source", "operator"), items)

    def answer_without_brain(self, question, qid=None, source="ground"):
        self._answer_fallback({"question": question, "qid": qid, "source": source}, reason="reflex-only rover")
