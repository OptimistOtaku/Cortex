"""Spike 06: on-device AI latency. fastembed multilingual dense embed; Ollama local LLM answer + JSON classification."""

import json
import time

import requests
from fastembed import TextEmbedding

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXTS = [
    "Inverter fault E42 fixed by replacing the DC fuse on site 17",
    "इन्वर्टर में E42 त्रुटि, DC फ्यूज बदला",
    "Customer gate code 4471, dog on premises",
    "tower battery low voltage alarm after storm",
]

t0 = time.perf_counter()
emb = TextEmbedding(model_name=MODEL)
t1 = time.perf_counter()
vecs = list(emb.embed(TEXTS))
t2 = time.perf_counter()
for _ in range(3):
    list(emb.embed([TEXTS[0]]))
t3 = time.perf_counter()
import numpy as np  # noqa: E402

cos = float(np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])))
cos_un = float(np.dot(vecs[0], vecs[3]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[3])))
print(f"fastembed {MODEL}: dim={len(vecs[0])} load={t1-t0:.1f}s batch4={1000*(t2-t1):.0f}ms single={1000*(t3-t2)/3:.1f}ms")
print(f"  cross-lingual EN~HI same fact cos={cos:.3f} vs unrelated EN cos={cos_un:.3f}")

OLL = "http://localhost:11434/api/generate"
for model in ("llama3.2:1b", "qwen3-vl:2b-instruct"):
    prompt = ("Classify this field note for data residency. Reply JSON only with keys "
              "sensitivity (public|internal|pii) and reason.\nNote: " + TEXTS[2])
    t0 = time.perf_counter()
    r = requests.post(OLL, json={"model": model, "prompt": prompt, "stream": False, "format": "json",
                                 "options": {"temperature": 0}}, timeout=300).json()
    t1 = time.perf_counter()
    r2 = requests.post(OLL, json={"model": model, "prompt": prompt, "stream": False, "format": "json",
                                  "options": {"temperature": 0}}, timeout=300).json()
    t2 = time.perf_counter()
    print(f"{model}: cold={t1-t0:.1f}s warm={t2-t1:.1f}s eval_tokens/s="
          f"{r2.get('eval_count',0)/max(r2.get('eval_duration',1)/1e9,1e-9):.1f} -> {r2.get('response','')[:160]!r}")
