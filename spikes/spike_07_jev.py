"""Spike 07: TypeSafe Jev (cloud API) as Cortex Cloud's decision layer.

Key from env TYPESAFE_API_KEY only. Test data is synthetic. Measures latency (cold/warm/batched)
and accuracy on cases with known answers: conflict adjudication, escalation routing, answer
quality scoring, sensitivity classification.
"""

import os
import statistics
import time

import requests

URL = "https://api.typesafe.ai/v1/systemone"
HDR = {"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}", "Content-Type": "application/json"}
S = requests.Session()


def ask(state, questions):
    t0 = time.perf_counter()
    r = S.post(URL, headers=HDR, json={"state": state, "model": "jev-latest", "questions": questions}, timeout=60)
    ms = (time.perf_counter() - t0) * 1000
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    return r.json(), ms


SUPERSEDE = {
    "type": "noul",
    "instructions": "Report B was made after report A about the same asset. Does report B make report A no longer true "
                    "(the situation changed), rather than the two reports disagreeing about the same moment?",
    "criteria": {"true": "B describes a later change that replaces A", "false": "A and B contradict each other about the same state, or B does not affect A"},
}
CONFLICTS = [  # (state, expected supersede?)
    ({"asset": "Inverter INV-17", "A": {"time": "09:10", "device": "tech-1", "text": "E42 fault active, inverter offline"},
      "B": {"time": "11:40", "device": "tech-1", "text": "Replaced DC fuse, E42 cleared, inverter back online"}}, True),
    ({"asset": "Tower T-3 battery", "A": {"time": "14:02", "device": "tech-1", "text": "Battery bank at 48V, healthy"},
      "B": {"time": "14:03", "device": "tech-2", "text": "Battery bank at 41V, low-voltage alarm active"}}, False),
    ({"asset": "Gate at site 9", "A": {"time": "Mon", "device": "tech-3", "text": "Gate code is 4471"},
      "B": {"time": "Thu", "device": "office", "text": "Customer changed the gate code this week, new code 8820"}}, True),
    ({"asset": "Pump P-2", "A": {"time": "10:00", "device": "tech-2", "text": "Pump running normally"},
      "B": {"time": "10:01", "device": "tech-4", "text": "Pump P-2 is not running, no power at the panel"}}, False),
]

ROUTE = {
    "type": "choice",
    "instructions": "A field device could not answer this question confidently offline. Where should the cloud route it?",
    "criteria": {
        "fleet_knowledge": "Answerable from past fixes and notes logged by other technicians",
        "manuals": "Answerable from equipment manuals and official procedures",
        "human_expert": "Safety-critical or novel; needs a senior engineer",
    },
}
ROUTES = [
    ("Has anyone seen error E42 come back after a firmware update, and what fixed it?", "fleet_knowledge"),
    ("What is the torque spec for the DC busbar bolts on the SX-500 inverter?", "manuals"),
    ("Smoke is coming from the battery cabinet and it smells of sulphur, what do I do?", "human_expert"),
]

QUALITY = {
    "type": "score",
    "instructions": "Rate how well the answer resolves the technician's question using the cited notes, before it is cached and shared with every device.",
    "criteria": ["wrong or unsupported", "partially useful", "correct and actionable"],
}
QUALS = [
    ({"question": "How was E42 fixed on INV-17?", "cited_notes": ["Replaced DC fuse, E42 cleared"], "answer": "Replace the DC fuse; E42 cleared after that on INV-17."}, 2),
    ({"question": "How was E42 fixed on INV-17?", "cited_notes": ["Replaced DC fuse, E42 cleared"], "answer": "Reboot the router and call the ISP."}, 0),
]

SENS = {
    "type": "choice",
    "instructions": "Classify this field note for data residency.",
    "criteria": {"public": "Generic technical info, safe to share fleet-wide", "internal": "Operational detail for the company only",
                 "pii": "Personal or security-sensitive: people, phone numbers, addresses, access codes, credentials"},
}
SENSES = [
    ("Customer gate code 4471, dog on premises", "pii"),
    ("E42 on SX-500 usually means a blown DC fuse", "public"),
    ("Site 17 contract renewal is due next month", "internal"),
    ("Call Mr. Sharma on 98100 12345 before visiting", "pii"),
]


def main():
    lat = []
    # cold + accuracy runs, one question per call
    ok = 0
    for i, (st, exp) in enumerate(CONFLICTS):
        r, ms = ask(st, {"supersede": SUPERSEDE})
        lat.append(ms)
        p = r["answers"]["supersede"]["noul"]
        got = p >= 0.5
        ok += got == exp
        print(f"conflict{i}: p(supersede)={p:.2f} expected={exp} {'OK' if got == exp else 'WRONG'} {ms:.0f}ms")
    print(f"  conflict accuracy {ok}/{len(CONFLICTS)}")
    ok = 0
    for q, exp in ROUTES:
        r, ms = ask({"question": q}, {"route": ROUTE})
        lat.append(ms)
        a = r["answers"]["route"]
        ok += a["choice"] == exp
        print(f"route: {a['choice']:15} conf={a['confidence']:.2f} expected={exp} {'OK' if a['choice']==exp else 'WRONG'} {ms:.0f}ms")
    print(f"  routing accuracy {ok}/{len(ROUTES)}")
    ok = 0
    for st, exp in QUALS:
        r, ms = ask(st, {"quality": QUALITY})
        lat.append(ms)
        sc = r["answers"]["quality"]["score"]
        ok += round(sc) == exp
        print(f"quality: score={sc:.2f} expected={exp} {'OK' if round(sc)==exp else 'WRONG'} {ms:.0f}ms")
    ok = 0
    for note, exp in SENSES:
        r, ms = ask({"note": note}, {"sensitivity": SENS})
        lat.append(ms)
        a = r["answers"]["sensitivity"]
        ok += a["choice"] == exp
        print(f"sens: {a['choice']:8} conf={a['confidence']:.2f} expected={exp:8} {'OK' if a['choice']==exp else 'WRONG'} {ms:.0f}ms")
    print(f"  sensitivity accuracy {ok}/{len(SENSES)}")
    print(f"single-question latency: first={lat[0]:.0f}ms median={statistics.median(lat):.0f}ms max={max(lat):.0f}ms (n={len(lat)})")

    # batched: several questions on one state in one call
    st = CONFLICTS[0][0]
    bl = []
    for _ in range(5):
        _, ms = ask(st, {"supersede": SUPERSEDE, "route": ROUTE, "quality": QUALITY, "sensitivity": SENS})
        bl.append(ms)
    print(f"batched 4 questions/call: median={statistics.median(bl):.0f}ms min={min(bl):.0f}ms")


if __name__ == "__main__":
    main()
