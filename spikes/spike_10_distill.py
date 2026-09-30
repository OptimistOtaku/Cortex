"""Spike 10: "cloud teaches edge". Distil Jev's decisions into a <1 ms head on our on-device embeddings.

Stages (each cached under spikes/data/ so reruns don't re-call Jev):
  gen    - template-generate synthetic field notes (sensitivity) and technician questions (routing)
  label  - label them with Jev (cloud)
  train  - logistic-regression head on multilingual MiniLM embeddings
  eval   - compare on a HAND-WRITTEN held-out set (different phrasing): head, head+PII patterns,
           laya:multilingual (local Ollaya), Jev

    TYPESAFE_API_KEY=... python spikes/spike_10_distill.py
"""

import json
import os
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
import spike_07_jev as j  # noqa: E402

JEV = ("https://api.typesafe.ai/v1/systemone", "jev-latest", os.environ.get("TYPESAFE_API_KEY"))
LAYA = ("http://127.0.0.1:11435/v1/systemone", "laya:multilingual", None)
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ------------------------------------------------------------------ generation

ASSETS = ["INV-17", "inverter SX-500", "tower T-3", "BTS cabinet B-12", "pump P-2", "chiller CH-4", "UPS U-9",
          "DG set at site 21", "solar string S-4", "fibre ODF at node 7", "transformer TR-2", "rectifier R-5"]
CODES = ["E42", "E17", "F03", "ALM-221", "LVD", "OVP", "0x1F", "E09"]
FIXES = ["replaced the DC fuse", "cleaned the fan filter", "re-seated the SFP", "reflashed firmware to v3.2",
         "tightened the busbar lugs", "swapped the contactor", "reset the breaker", "replaced the surge arrester",
         "topped up coolant", "recalibrated the sensor"]
SYMPTOMS = ["overheating in the afternoon", "tripping every few hours", "low voltage alarm", "no output",
            "intermittent link drops", "loud humming", "reading zero on the display", "battery draining fast"]
NAMES = ["Mr. Sharma", "Priya Verma", "Rakesh ji", "Mrs. Iyer", "Anil Kumar", "Farhan", "Neha Gupta", "the owner Mr. Rao"]


def phone(r):
    return r.choice(["98100 12345", "+91 99887 66554", "9123456780", "080-2345 6789", "98-765-43210"])


def gen_notes(r):
    pii = [
        lambda: f"Gate code for {r.choice(ASSETS)} site is {r.randint(1000, 9999)}",
        lambda: f"Call {r.choice(NAMES)} on {phone(r)} before visiting",
        lambda: f"{r.choice(NAMES)} lives at {r.randint(1, 99)}, {r.choice(['MG Road', 'Sector 62', 'Lajpat Nagar', 'Baner'])}, key with neighbour",
        lambda: f"Wifi password at site is {r.choice(['Sun@2024', 'tower#77', 'admin123'])}",
        lambda: f"Controller login admin / {r.choice(['Pass@123', 'solar99', 'qwerty7'])}",
        lambda: f"Alarm panel disarm code {r.randint(100, 999)}#, customer asked not to share",
        lambda: f"{r.choice(NAMES)} is elderly and lives alone, knock loudly",
        lambda: f"Customer Aadhaar {r.randint(1000,9999)} {r.randint(1000,9999)} {r.randint(1000,9999)} noted for warranty",
        lambda: f"{r.choice(NAMES)} ka number {phone(r)} hai, pehle call karna",
        lambda: f"Locker code {r.randint(1000,9999)} for spare keys at {r.choice(ASSETS)}",
        lambda: f"Email the report to {r.choice(['rakesh.k', 'priya.v', 'ops.north'])}@gmail.com",
    ]
    internal = [
        lambda: f"Contract for {r.choice(ASSETS)} renews next month, customer wants a discount",
        lambda: f"SLA breached at {r.choice(ASSETS)}, penalty likely",
        lambda: f"Vendor quoted {r.randint(8, 90)}k for a new {r.choice(['inverter', 'battery bank', 'rectifier'])}",
        lambda: f"Only {r.randint(0, 4)} spare {r.choice(['fuses', 'SFPs', 'contactors'])} left in the van stock",
        lambda: f"Customer complained about the delay at {r.choice(ASSETS)}, escalate to area manager",
        lambda: f"Planned shutdown of {r.choice(ASSETS)} on Saturday for maintenance",
        lambda: f"Billing dispute on last visit to {r.choice(ASSETS)}, hold the invoice",
        lambda: f"Need two technicians for {r.choice(ASSETS)} next week, short staffed",
        lambda: f"{r.choice(ASSETS)} warranty claim rejected by OEM, cost goes to us",
        lambda: f"Stock kam hai, {r.choice(['fuse', 'SFP', 'contactor'])} order karna padega",
    ]
    public = [
        lambda: f"{r.choice(CODES)} on {r.choice(ASSETS)} usually means you should check if someone {r.choice(FIXES)}",
        lambda: f"{r.choice(ASSETS)} {r.choice(SYMPTOMS)}; fixed after I {r.choice(FIXES)}",
        lambda: f"Tip: when {r.choice(CODES)} shows, first {r.choice(FIXES).replace('replaced', 'check')}",
        lambda: f"Firmware v3.2 fixes the {r.choice(CODES)} false alarm on {r.choice(ASSETS)}",
        lambda: f"Always isolate DC before touching {r.choice(ASSETS)} terminals",
        lambda: f"{r.choice(SYMPTOMS).capitalize()} is common in summer on {r.choice(ASSETS)}",
        lambda: f"{r.choice(CODES)} aata hai toh pehle {r.choice(['fuse', 'fan', 'breaker'])} check karo",
        lambda: f"Use a torque wrench on {r.choice(ASSETS)} lugs, hand tightening causes {r.choice(CODES)}",
    ]
    out = []
    for label, gens, n in (("pii", pii, 110), ("internal", internal, 90), ("public", public, 100)):
        seen = set()
        while len(seen) < n:
            seen.add(r.choice(gens)())
        out += [{"text": t, "template_label": V2[label]} for t in seen]
    r.shuffle(out)
    return out


def gen_questions(r):
    fleet = [
        lambda: f"Has anyone seen {r.choice(CODES)} on {r.choice(ASSETS)} before, what fixed it?",
        lambda: f"What did the last technician do at {r.choice(ASSETS)}?",
        lambda: f"Did {r.choice(FIXES)} work for others with {r.choice(SYMPTOMS)}?",
        lambda: f"Kisi ne {r.choice(CODES)} pehle theek kiya hai?",
        lambda: f"Is {r.choice(SYMPTOMS)} a known issue on our {r.choice(ASSETS)} units?",
    ]
    manuals = [
        lambda: f"What is the torque spec for the lugs on {r.choice(ASSETS)}?",
        lambda: f"What is the part number of the DC fuse in {r.choice(ASSETS)}?",
        lambda: f"What does alarm {r.choice(CODES)} mean according to the manual?",
        lambda: f"What is the rated input voltage range of {r.choice(ASSETS)}?",
        lambda: f"Steps to reflash firmware on {r.choice(ASSETS)}?",
    ]
    expert = [
        lambda: f"Smoke coming out of {r.choice(ASSETS)}, what do I do?",
        lambda: f"I got a shock from the chassis of {r.choice(ASSETS)}, is it safe to continue?",
        lambda: f"Battery bank at {r.choice(ASSETS)} is swollen and hot, should I touch it?",
        lambda: f"Smell of gas near {r.choice(ASSETS)}, should I proceed?",
        lambda: f"Sparks from {r.choice(ASSETS)} when I closed the breaker, what now?",
    ]
    out = []
    for label, gens, n in (("fleet_knowledge", fleet, 50), ("manuals", manuals, 50), ("human_expert", expert, 50)):
        seen = set()
        while len(seen) < n:
            seen.add(r.choice(gens)())
        out += [{"text": t, "template_label": label} for t in seen]
    r.shuffle(out)
    return out


# hand-written held-out sets: different phrasing from the templates, my gold labels
TEST_NOTES = [
    ("Mrs. Kapoor's flat is 4B, spare key under the flowerpot", "pii"),
    ("reach site manager Deepak at 9812233445", "pii"),
    ("router admin pwd is Tower@55, change it after the visit", "pii"),
    ("Customer's son handles payments, his number is +91 90000 11122", "pii"),
    ("Door PIN changed to 2580 by the society", "pii"),
    ("the owner is recovering from surgery, avoid noisy work before noon", "pii"),
    ("Security guard Ramesh, phone 99990 88877, opens the terrace", "pii"),
    ("site ka lock code 7391 hai", "pii"),
    ("Customer is threatening to cancel the AMC over repeated visits", "internal"),
    ("We are losing money on this site, travel cost too high", "internal"),
    ("Area manager wants weekly reports for the northern cluster", "internal"),
    ("Quote from the new battery supplier is 12% cheaper", "internal"),
    ("Van 3 needs servicing, keep it off the schedule Friday", "internal"),
    ("Invoice for the March visit is still unpaid", "internal"),
    ("Parts request pending approval from finance", "internal"),
    ("Hamare paas sirf ek spare inverter bacha hai", "internal"),
    ("If the display freezes, hold reset for 10 seconds", "public"),
    ("Corroded terminals cause voltage drops; clean with a wire brush", "public"),
    ("Fan bearing noise means the fan should be replaced soon", "public"),
    ("After a firmware update, re-check the grid code setting", "public"),
    ("Moisture in the cabinet leads to earth faults during monsoon", "public"),
    ("Label every cable before disconnecting, saves time later", "public"),
    ("Low insulation resistance often comes from a damaged DC cable", "public"),
    ("Baarish ke baad earth fault aaye toh cable joint check karo", "public"),
]
TEST_QUESTIONS = [
    ("Anyone fixed the humming noise on the rectifiers in our region?", "fleet_knowledge"),
    ("How did we resolve the repeated tripping at site 21 last month?", "fleet_knowledge"),
    ("Which fix worked best for link drops on the fibre nodes?", "fleet_knowledge"),
    ("Pichli baar pump P-2 ka kya issue tha?", "fleet_knowledge"),
    ("What's the maximum DC input current for this model?", "manuals"),
    ("Where is the earthing point shown in the wiring diagram?", "manuals"),
    ("What is the recommended replacement interval for the air filter?", "manuals"),
    ("Which fuse rating does the manual specify for the battery breaker?", "manuals"),
    ("The cable insulation is melting and I can smell burning, help", "human_expert"),
    ("Water is leaking onto the live busbar, should I switch it off myself?", "human_expert"),
    ("My colleague fell from the tower ladder and is not responding", "human_expert"),
    ("Battery is hissing and venting gas, what should I do right now?", "human_expert"),
]

PII_PATTERNS = [
    re.compile(r"(\+91[\s-]?)?[6-9]\d{2}[\s-]?\d{2}[\s-]?\d{5}\b|\b[6-9]\d{4}\s?\d{5}\b"),  # IN mobile
    re.compile(r"\b0\d{2,4}[-\s]\d{3,4}\s?\d{4}\b"),  # landline
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),  # email
    re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),  # aadhaar-like
    re.compile(r"\b(code|pin|password|pwd|passcode|login)\b[^.\n]{0,20}?\d{3,}", re.I),
    re.compile(r"\b(password|pwd|login)\b\s*(is|:|/)?\s*\S+", re.I),
    re.compile(r"\bcode\s+\d{3,}\s+(hai|he)\b", re.I),
]


def pii_hit(text):
    return any(p.search(text) for p in PII_PATTERNS)


# ------------------------------------------------------------------ helpers


def make_ask(backend):
    url, model, key = backend
    s = requests.Session()
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}

    def ask(state, questions):
        for attempt in range(6):
            t0 = time.perf_counter()
            r = s.post(url, headers=h, json={"state": state, "model": model, "questions": questions}, timeout=120)
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code in (429, 529) or r.status_code >= 500:
                time.sleep(0.5 * 2 ** attempt)
                continue
            r.raise_for_status()
            return r.json(), ms
        raise RuntimeError(f"gave up after retries: {r.status_code} {r.text[:200]}")
    return ask


def jsonl_load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def jsonl_save(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# v2 labels: named by where the note may go (v1 "public/internal/pii" made Jev call 96% of technical notes internal)
SENS2 = {
    "type": "choice",
    "instructions": "A field technician wrote this note on a company device. Decide how far the note may travel.",
    "criteria": {
        "personal_or_secret": "Identifies or describes a private person (name with phone, address, health, family) or contains "
                              "an access secret (gate/door/alarm code, PIN, password, login). Must stay on the device.",
        "business_confidential": "Commercial or internal business matters: contracts, pricing, quotes, penalties, billing, "
                                 "staffing, stock levels, schedules, complaints, warranty costs. Company cloud only.",
        "technical_knowledge": "Technical observations, fault codes, symptoms, fixes, procedures and safety tips about "
                               "equipment, even when a specific asset or site is named. Share with all technicians.",
    },
}
V2 = {"pii": "personal_or_secret", "internal": "business_confidential", "public": "technical_knowledge"}

TASKS = {
    "sensitivity": {"question": SENS2, "state_key": "note", "gen": gen_notes, "test": [(t, V2[g]) for t, g in TEST_NOTES]},
    "route": {"question": j.ROUTE, "state_key": "question", "gen": gen_questions, "test": TEST_QUESTIONS},
}


def stage_gen_and_label():
    os.makedirs(DATA, exist_ok=True)
    ask = make_ask(JEV)
    for task, cfg in TASKS.items():
        path = os.path.join(DATA, f"{task}_train_v2.jsonl")
        if os.path.exists(path):
            print(f"[label] {task}: cached {path}")
            continue
        rows = cfg["gen"](random.Random(42))

        def label(row):
            r, ms = ask({cfg["state_key"]: row["text"]}, {task: cfg["question"]})
            a = r["answers"][task]
            return {**row, "jev_label": a["choice"], "jev_conf": a["confidence"], "jev_probs": a["probabilities"], "ms": ms}

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as ex:
            rows = list(ex.map(label, rows))
        jsonl_save(path, rows)
        agree = sum(r["jev_label"] == r["template_label"] for r in rows) / len(rows)
        print(f"[label] {task}: {len(rows)} rows in {time.perf_counter()-t0:.0f}s; Jev agrees with template intent {agree:.0%}")


def stage_train_eval():
    from fastembed import TextEmbedding
    from sklearn.linear_model import LogisticRegression

    emb = TextEmbedding(model_name=EMB_MODEL)

    def embed(texts):
        v = np.array(list(emb.embed(texts)))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    laya = make_ask(LAYA)
    jev = make_ask(JEV)
    for task, cfg in TASKS.items():
        rows = jsonl_load(os.path.join(DATA, f"{task}_train_v2.jsonl"))
        X = embed([r["text"] for r in rows])
        y = [r["jev_label"] for r in rows]
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(X, y)
        test_x = [t for t, _ in cfg["test"]]
        gold = [g for _, g in cfg["test"]]
        t0 = time.perf_counter()
        Xt = embed(test_x)
        emb_ms = (time.perf_counter() - t0) * 1000 / len(test_x)
        t0 = time.perf_counter()
        pred = list(clf.predict(Xt))
        head_ms = (time.perf_counter() - t0) * 1000 / len(test_x)
        conf = clf.predict_proba(Xt).max(axis=1)

        results = {"edge head (distilled)": pred}
        if task == "sensitivity":
            results["edge head + PII patterns"] = ["personal_or_secret" if pii_hit(t) else p for t, p in zip(test_x, pred)]
        lat = {"edge head (distilled)": emb_ms + head_ms}
        for name, ask in (("laya:multilingual (local)", laya), ("jev (cloud)", jev)):
            preds, ms_all = [], []
            for t in test_x:
                r, ms = ask({cfg["state_key"]: t}, {task: cfg["question"]})
                preds.append(r["answers"][task]["choice"])
                ms_all.append(ms)
            results[name] = preds
            lat[name] = statistics.median(ms_all)

        print(f"\n== {task}: train n={len(rows)} (Jev-labelled), held-out hand-written n={len(gold)}")
        for name, p in results.items():
            acc = sum(a == b for a, b in zip(p, gold)) / len(gold)
            l = f"{lat[name]:.1f} ms" if name in lat else f"{lat['edge head (distilled)']:.1f} ms + regex"
            print(f"  {name:28} acc={acc:6.1%}  latency/item={l}")
        wrong = [(t, g, p, c) for t, g, p, c in zip(test_x, gold, results.get('edge head + PII patterns', pred), conf) if g != p]
        for t, g, p, c in wrong:
            print(f"    miss: gold={g:15} pred={p:15} conf={c:.2f} | {t}")
        low = sum(c < 0.6 for c in conf)
        print(f"  head confidence <0.6 on {low}/{len(conf)} items (would escalate to Jev when online)")


if __name__ == "__main__":
    if not JEV[2]:
        sys.exit("set TYPESAFE_API_KEY")
    stage_gen_and_label()
    stage_train_eval()
