"""End-to-end rehearsal against a running fleet: ground on :8100, rovers on :8101 and :8102.

Drives the demo storyline and asserts each PS03 goal. Mission relevance of what arrives is judged with the
hand-checked categories of the spike gallery (water/rock images are useful for the water intent).

    python tests/e2e_scenario.py
"""

import os
import sys
import time

import requests

G = "http://127.0.0.1:8100"
R = {"rover-1": "http://127.0.0.1:8101", "rover-2": "http://127.0.0.1:8102"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GALLERY = os.path.join(ROOT, "spikes", "photos", "commons")
TRAVERSE = {
    "rover-1": ["bottle", "glasses_person", "laptop", "layered_rock", "water"],
    "rover-2": ["logo_sign", "person_no_glasses", "plant", "red_object", "water"],  # water overlaps on purpose
}
WATER_USEFUL = {"water", "layered_rock"}
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def wait(pred, timeout=240, every=2.0, what=""):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            v = pred()
            if v:
                return v
        except requests.RequestException:
            pass
        time.sleep(every)
    raise TimeoutError(what)


def get(url, **kw):
    return requests.get(url, timeout=30, **kw).json()


def post(url, body=None):
    return requests.post(url, json=body or {}, timeout=300).json()


def sources():
    """item id -> gallery folder, from the rovers' own memory."""
    out = {}
    for url in R.values():
        m = get(f"{url}/api/memory")
        for x in m["queue"] + m["held"] + m["sent"]:
            out[x["id"]] = x.get("source")
    return out


def main():
    wait(lambda: get(f"{G}/api/metrics") and all(get(f"{u}/api/status") for u in R.values()), what="stack up")
    for node, cats in TRAVERSE.items():
        for c in cats:
            post(f"{R[node]}/api/traverse", {"dir": os.path.join(GALLERY, c)})
    n = sum(get(f"{u}/api/status")["memories"] for u in R.values())
    check("1 semantic memory on each device", n == 50, f"{n} memories across 2 rovers")

    # 4. offline: airplane mode on rover-1, local search still answers
    post(f"{R['rover-1']}/api/link", {"up": False})
    runs = [get(f"{R['rover-1']}/api/search", params={"q": q}) for q in ("a bottle", "a bottle", "rock layers", "a bottle")]
    med = sorted(r["ms"] for r in runs)[len(runs) // 2]
    check("2 offline hybrid search (airplane mode)", runs[-1]["results"][0]["source"] == "bottle" and med < 300,
          f"top={runs[-1]['results'][0]['source']}, median {med} ms (first {runs[0]['ms']} ms; "
          f"Qdrant Edge part {runs[-1]['qdrant_ms']} ms)")
    post(f"{R['rover-1']}/api/window_now")
    wait(lambda: "window missed" in str(get(f"{R['rover-1']}/api/status")["events"]), what="missed window")
    check("4 comms window missed while offline, memory keeps working", True)
    post(f"{R['rover-1']}/api/link", {"up": True})

    # 3/8. retask with one sentence, then measure what the gate delivers in the next 2 windows per rover
    t_retask = time.time()
    post(f"{G}/api/intent", {"text": "now we care about evidence of past water"})
    wait(lambda: all("past water" in get(f"{u}/api/status")["intent"] for u in R.values()), what="intent reached rovers")
    tops = {n: [x["source"] for x in get(f"{u}/api/memory")["queue"][:3]] for n, u in R.items()}
    check("8 one-sentence retask re-ranked every rover's queue locally",
          all(sum(s in WATER_USEFUL for s in t) >= 1 for t in tops.values()), str(tops))

    def downlinks_since(t):
        return [e for e in get(f"{G}/api/events", params={"limit": 400}) if e["kind"] == "downlink" and e["ts"] > t]
    wait(lambda: all(sum(1 for e in downlinks_since(t_retask) if e["text"].startswith(n)) >= 2 for n in R),
         timeout=300, what="2 windows per rover after retask")
    # what came down after the retask, against the mission-relevant share of what the rovers hold
    src = sources()
    got = [p for p in get(f"{G}/api/feed", params={"limit": 500})
           if p["received_at"] > t_retask and p.get("kind", "observation") == "observation"]
    useful = sum(src.get(p["item"]) in WATER_USEFUL for p in got)
    base = sum(s in WATER_USEFUL for s in src.values()) / max(len(src), 1)
    check("8 under scarcity the gate spends the budget on the mission first",
          got and useful / len(got) >= 2 * base,
          f"after retask: {useful}/{len(got)} pictures mission-relevant vs {base:.0%} of everything held")

    # 5. let windows drain the queues
    wait(lambda: get(f"{G}/api/metrics")["items"] >= 40, timeout=400, what="downlinks")
    ev = get(f"{G}/api/events", params={"limit": 400})
    over = [e for e in ev if e["kind"] == "downlink" and int(e["text"].split(", ")[-1].split(" ")[0]) > 30000]
    check("5 every downlink fits its byte budget (real request bodies)", not over, f"{len(over)} over budget")

    # 3. redundancy: the gate never pays twice for the same site
    sites = {}
    for p in get(f"{G}/api/feed", params={"kind": "observation", "limit": 500}):
        sites.setdefault(p["site"], set()).add(p["node"])
    dups = sum(len(v) > 1 for v in sites.values())
    check("3 gate skips what another rover already delivered", dups == 0, f"sites delivered by both rovers: {dups}")

    # 5. demand-driven full resolution streamed in chunks across windows
    gate = get(f"{G}/api/feed", params={"kind": "observation", "limit": 500})
    target = gate[0]
    post(f"{G}/api/request_full", {"point": target["point"]})
    done = wait(lambda: next((p for p in get(f"{G}/api/feed", params={"limit": 500})
                              if p["point"] == target["point"] and p.get("full")), None), timeout=600, what="full-res")
    chunks = [e for e in get(f"{G}/api/events", params={"limit": 400}) if e["kind"] == "full" and target["item"][:8] in e["text"]]
    check("5 requested full-res streamed across windows and reassembled", bool(done), f"{len(chunks)} chunk events")

    # 6. conflicts: relabel supersedes on the rover; discard tombstones free rover memory
    rel = next(p for p in gate if p["node"] == "rover-1" and p["point"] != target["point"])
    post(f"{G}/api/relabel", {"point": rel["point"], "label": "sandstone strata"})
    dis = next(p for p in gate if p["node"] == "rover-2" and p["point"] not in (target["point"], rel["point"]))
    post(f"{G}/api/discard", {"point": dis["point"]})
    for u in R.values():
        post(f"{u}/api/window_now")

    def relabelled():
        m = get(f"{R['rover-1']}/api/memory")
        x = next((x for x in m["queue"] + m["held"] + m["sent"] if x["id"] == rel["item"]), None)
        return x if x and x["label"] == "sandstone strata" else None
    x = wait(relabelled, what="relabel")
    check("6 ground relabel supersedes the rover's label (history kept)", len(x["label_history"]) == 1, str(x["label_history"][0]))
    wait(lambda: all(y["id"] != dis["item"] for y in (lambda m: m["queue"] + m["held"] + m["sent"])(get(f"{R['rover-2']}/api/memory"))), what="discard")
    left = [f for f in os.listdir(os.path.expanduser("~/.sietch/rover-2/img")) if f.startswith(dis["item"])]
    check("6 discard tombstone removed the memory and its files on the rover", not left, f"{len(left)} files left")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
