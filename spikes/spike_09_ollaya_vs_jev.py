"""Spike 09: local Ollaya (laya) vs cloud Jev on identical cases. Same wire API (/v1/systemone).

    TYPESAFE_API_KEY=... python spikes/spike_09_ollaya_vs_jev.py
"""

import os
import statistics
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TYPESAFE_API_KEY", "unset")
import spike_07_jev as j  # noqa: E402
import spike_08_jev_conflict as c  # noqa: E402,F401  (runs its own cases on import? no: guarded below)

BACKENDS = {
    "ollaya:laya": ("http://127.0.0.1:11435/v1/systemone", "laya", None),
    "ollaya:laya:multilingual": ("http://127.0.0.1:11435/v1/systemone", "laya:multilingual", None),
    "jev (cloud)": ("https://api.typesafe.ai/v1/systemone", "jev-latest", os.environ["TYPESAFE_API_KEY"]),
}


def make_ask(url, model, key):
    s = requests.Session()
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"

    def ask(state, questions):
        t0 = time.perf_counter()
        r = s.post(url, headers=h, json={"state": state, "model": model, "questions": questions}, timeout=120)
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
        return r.json(), ms
    return ask


def contra_as_choice():
    return {"type": "choice", "instructions": c.CONTRA["instructions"],
            "criteria": {"contradiction": c.CONTRA["criteria"]["true"], "compatible": c.CONTRA["criteria"]["false"]}}


def run(name, ask):
    res, lat = {}, []

    def tally(key, good, ms):
        lat.append(ms)
        a, b = res.get(key, (0, 0))
        res[key] = (a + good, b + 1)

    ask({"warm": "up"}, {"w": j.SENS})  # warm-up, excluded
    for note, exp in j.SENSES:
        r, ms = ask({"note": note}, {"sensitivity": j.SENS})
        tally("sensitivity", r["answers"]["sensitivity"]["choice"] == exp, ms)
    for q, exp in j.ROUTES:
        r, ms = ask({"question": q}, {"route": j.ROUTE})
        tally("routing", r["answers"]["route"]["choice"] == exp, ms)
    for st, exp in j.QUALS:
        r, ms = ask(st, {"quality": j.QUALITY})
        tally("quality", round(r["answers"]["quality"]["score"]) == exp, ms)
    for a, b, exp in c.PAIRS:
        try:
            r, ms = ask({"A": a, "B": b}, {"contradiction": c.CONTRA})
            got = r["answers"]["contradiction"]["noul"] >= 0.5
            tally("contradiction(noul)", got == exp, ms)
        except RuntimeError as e:
            res["contradiction(noul)"] = (0, 0)
            res["noul_error"] = str(e)[:120]
            break
    for a, b, exp in c.PAIRS:
        r, ms = ask({"A": a, "B": b}, {"contradiction": contra_as_choice()})
        tally("contradiction(choice)", (r["answers"]["contradiction"]["choice"] == "contradiction") == exp, ms)
    batch = []
    for _ in range(5):
        _, ms = ask({"note": j.SENSES[0][0]}, {"sensitivity": j.SENS, "route": j.ROUTE, "quality": j.QUALITY})
        batch.append(ms)
    print(f"\n== {name}")
    for k, v in res.items():
        print(f"  {k:24} {v if isinstance(v, str) else f'{v[0]}/{v[1]}'}")
    print(f"  latency single median={statistics.median(lat):.0f}ms p90={sorted(lat)[int(0.9*len(lat))-1]:.0f}ms | 3-question batch median={statistics.median(batch):.0f}ms")


if __name__ == "__main__":
    for name, (url, model, key) in BACKENDS.items():
        if key == "unset":
            continue
        try:
            run(name, make_ask(url, model, key))
        except Exception as e:  # noqa: BLE001
            print(f"\n== {name}: ERROR {type(e).__name__}: {str(e)[:200]}")
