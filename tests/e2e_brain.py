"""End-to-end: SIETCH as the memory of a robot's LLM brain, against a running fleet.

Expects ground on :8100, rover-1 on :8101 WITH a brain (Ollama) and rover-2 on :8102 reflex-only, e.g.
    python run_fleet.py --fast --fresh --model clip --brains rover-1 --storage-mb 3
    python tests/e2e_brain.py

Checks: reflex wakes the brain; the brain sees, recalls, writes an insight citing evidence and acts; the insight
reaches ground before (or without) its picture; offline Q&A with citations; ask-over-the-link round trip; a reflex-only
robot still answers; a contradiction between robots opens a dispute (Jev) that mission control resolves and the
loser's memory is superseded on the robot; a forged command is refused; storage tiers raw evidence down.
"""

import os
import sys
import time
from datetime import datetime

import requests

G = "http://127.0.0.1:8100"
R1, R2 = "http://127.0.0.1:8101", "http://127.0.0.1:8102"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GALLERY = os.path.join(ROOT, "spikes", "photos", "commons")
results, notes = [], []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def get(url, **kw):
    return requests.get(url, timeout=60, **kw).json()


def post(url, body=None):
    return requests.post(url, json=body or {}, timeout=600).json()


def wait(pred, timeout=300, every=2.0, what=""):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            v = pred()
            if v:
                return v
        except (requests.RequestException, KeyError, IndexError, StopIteration):
            pass
        time.sleep(every)
    print(f"  (timed out after {timeout}s waiting for {what})", flush=True)
    return None


def memories(url):
    m = get(f"{url}/api/memory")
    return m["queue"] + m["held"] + m["sent"]


def main():
    wait(lambda: get(f"{G}/api/metrics") and get(f"{R1}/api/status") and get(f"{R2}/api/status"), what="stack")
    b1 = get(f"{R1}/api/brain")
    check("0 rover-1 has a local LLM brain, rover-2 is reflex-only",
          b1["model"] and get(f"{R2}/api/brain")["model"] is None, f"rover-1 brain={b1['model']}")

    post(f"{G}/api/intent", {"text": "evidence of past water"})
    wait(lambda: all("past water" in get(f"{u}/api/status")["intent"] for u in (R1, R2)), what="intent")

    woken = 0
    for cat in ("water", "layered_rock", "laptop", "bottle"):
        woken += post(f"{R1}/api/traverse", {"dir": os.path.join(GALLERY, cat)})["brain_woken"]
    for cat in ("water", "plant", "red_object"):
        post(f"{R2}/api/traverse", {"dir": os.path.join(GALLERY, cat)})
    check("1 reflex (Qdrant Formula) woke the brain only for relevant, novel frames", 0 < woken < 20,
          f"{woken} of 20 frames woke the brain")

    # make sure the brain looks at a specific water frame both rovers saw, so the dispute test is deterministic
    r1_water = next(x for x in memories(R1) if x.get("kind") == "observation" and x.get("source") == "water")
    t0 = time.time()
    wait(lambda: post(f"{R1}/api/examine/{r1_water['id']}")["queued"], timeout=400, every=5, what="brain queue space")
    ins = wait(lambda: next(i for i in get(f"{R1}/api/brain")["insights"] if r1_water["id"] in i.get("evidence", [])),
               timeout=420, what="brain insight on the water frame")
    check("2 brain saw the frame, recalled, wrote an insight citing evidence", bool(ins),
          f"{time.time() - t0:.0f}s: {ins['text'][:140] if ins else ''!r}")
    # act follows remember in the same deliberation, a few seconds later
    dec = wait(lambda: next(d for d in get(f"{R1}/api/brain")["decisions"] if r1_water["id"] in d.get("evidence", [])),
               timeout=120, what="brain decision on the water frame") if ins else None
    obs = next(x for x in memories(R1) if x["id"] == r1_water["id"])
    check("2 brain acted and its decision is logged; the frame got a caption", bool(dec) and obs.get("examined") == 1,
          f"action={dec and dec['action']!r} caption={obs.get('caption', '')[:80]!r}")

    # insight-first: when the conclusion reaches ground, is its picture there yet?
    def at_ground():
        feed = get(f"{G}/api/feed", params={"limit": 500})
        g_ins = next((p for p in feed if p["item"] == ins["id"]), None)
        return (g_ins, {p["item"]: p for p in feed if p.get("kind") == "observation"}) if g_ins else None
    got = wait(at_ground, timeout=200, what="insight at ground") if ins else None
    if got:
        g_ins, obs_at_ground = got
        ev = obs_at_ground.get(r1_water["id"])
        concluded = datetime.fromisoformat(ins["captured_at"]).timestamp()
        if ev is not None and ev["received_at"] < concluded:  # picture went down before the brain had concluded
            check("3 insight-first: n/a, the picture was already down before the brain concluded", True,
                  f"insight {ins['bytes_thumb']} B written {concluded - ev['received_at']:.0f}s after the picture arrived")
        else:
            first = ev is None or ev["received_at"] >= g_ins["received_at"]
            check("3 the conclusion reached ground no later than its evidence picture", first,
                  f"insight {ins['bytes_thumb']} B; picture {'not yet down' if ev is None else f'arrived {ev['received_at'] - g_ins['received_at']:+.1f}s after'}")
    else:
        check("3 the conclusion reached ground", False, "" if ins else "(no insight from check 2)")

    # offline Q&A on rover-1 (airplane mode)
    post(f"{R1}/api/link", {"up": False})
    q = "What have you seen that suggests past water?"
    t0 = time.time()
    post(f"{R1}/api/ask", {"question": q})
    a = wait(lambda: next(x for x in get(f"{R1}/api/brain")["answers"] if x["question"] == q), timeout=240, what="offline answer")
    check("4 offline answer from local memory, with citations (airplane mode)", a and a["cited"],
          f"{time.time() - t0:.0f}s, cites {[c['ref'] for c in a['cited']] if a else []}: {a['answer'][:140] if a else ''!r}")
    post(f"{R1}/api/link", {"up": True})

    # ask over the link: question up in one window, answer down in a later one
    for node, url in (("rover-1", R1), ("rover-2", R2)):
        qid = post(f"{G}/api/ask", {"node": node, "question": "Did you see any standing water?"})["qid"]
        ans = wait(lambda: next(x for x in get(f"{G}/api/questions") if x["qid"] == qid and x["answer"]), timeout=300,
                   what=f"answer from {node}")
        label = "reflex-only robot answers from recall" if node == "rover-2" else "brain answers over the scarce link"
        check(f"5 ask {node} from ground: {label}", bool(ans),
              f"round trip {ans['latency_s']}s: {ans['answer'][:110]!r}" if ans else "")

    # dispute: rover-2's operator disagrees about the same frame
    r2_same = next(x for x in memories(R2) if x.get("file") == r1_water.get("file"))
    post(f"{R2}/api/note", {"text": "Dry dust only. There is no sign of water at this site, past or present.",
                            "evidence": [r2_same["id"]], "importance": 0.9})
    for u in (R1, R2):
        post(f"{u}/api/window_now")
    d = wait(lambda: next(x for x in get(f"{G}/api/disputes") if x["status"] != "resolved"), timeout=300, what="dispute")
    check("6 contradiction between robots about one site opens a dispute (Jev)", bool(d),
          f"p={d['p']:.2f}: {d['a']['node']} {d['a'].get('text', '')[:50]!r} vs {d['b']['node']} {d['b'].get('text', '')[:50]!r}" if d else "")
    if d:
        winner = "a" if d["a"]["node"] == "rover-1" else "b"
        post(f"{G}/api/disputes/{d['id']}/resolve", {"winner": winner})
        post(f"{R2}/api/window_now")
        sup = wait(lambda: next(n for n in get(f"{R2}/api/brain")["notes"] if n.get("superseded") == 1), timeout=120, what="supersede on rover-2")
        check("6 mission control resolved it; the loser is superseded on the robot and no longer recalled", bool(sup) and not any(
            i["id"] == sup["id"] for i in get(f"{R2}/api/search", params={"q": "no sign of water dry dust"})["insights"]))

    # forged command
    forged = post(f"{G}/api/redteam/forge", {"node": "rover-2"})
    post(f"{R2}/api/window_now")
    refused = wait(lambda: any("REFUSED" in e["text"] for e in get(f"{R2}/api/status")["events"]), timeout=120, what="refusal")
    still = any(x["id"] == forged["item"] for x in memories(R2))
    check("7 forged discard refused; the memory survives", refused and still, f"victim {forged['item'][:8]} still in memory: {still}")

    # bounded memory
    tiers = {u: get(f"{u}/api/status")["storage"] for u in (R1, R2)}
    check("8 storage budget tiers raw evidence down; insights keep the meaning",
          any(set(t["tiers"]) - {"full"} for t in tiers.values()),
          "; ".join(f"{u[-4:]}: {t['storage_mb']}/{t['budget_mb']} MB {t['tiers']}" for u, t in tiers.items()))

    print(f"\n{sum(results)}/{len(results)} checks passed", flush=True)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
