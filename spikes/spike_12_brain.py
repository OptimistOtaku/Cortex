"""Spike 12: can a local LLM on this laptop (GTX 1650 4 GB) be a robot's brain on top of SIETCH memory?

For each installed Ollama model, measure:
  latency  - time to first token and tokens/s, streaming, plain chat (no JSON mode)
  tools    - does it call memory tools correctly (recall / remember / act) from a scene + mission?
  caption  - can it describe a camera frame (vision models only), and how fast?
  answer   - grounded answer from recalled memories, citing memory ids

    python spikes/spike_12_brain.py [model ...]
"""

import base64
import json
import os
import sys
import time

import requests

OLLAMA = "http://localhost:11434"
HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "photos", "commons", "water", "00.jpg")

TOOLS = [
    {"type": "function", "function": {
        "name": "recall", "description": "Search this robot's memory (and what the fleet shared) for relevant past observations.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what to look for"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "remember", "description": "Store a short insight in long-term memory. Insights are tiny, so they reach mission control first.",
        "parameters": {"type": "object", "properties": {
            "note": {"type": "string"}, "importance": {"type": "number", "description": "0..1, relevance to the mission"}},
            "required": ["note", "importance"]}}},
    {"type": "function", "function": {
        "name": "act", "description": "Choose what the robot does next.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["investigate", "flag_for_ground", "move_on"]},
            "reason": {"type": "string"}}, "required": ["action", "reason"]}}},
]
SYSTEM = ("You are the onboard brain of an exploration rover with no link to Earth right now. "
          "Mission: find evidence of past water. Use your tools: recall memory before deciding, remember insights, then act.")
SCENE = ("New observation #obs-117: camera shows a dried mud surface with polygonal cracks next to a shallow puddle. "
         "Novelty vs everything the fleet has seen: 0.71. Relevance to mission: 0.93.")
MEMORIES = [
    ("m-12", "layered sedimentary rock outcrop, fine horizontal bands, 40 m north of landing site"),
    ("m-31", "puddle on dusty ground after overnight frost, reflective surface"),
    ("m-44", "laptop on a desk (calibration image)"),
    ("m-58", "rounded pebbles in a dry channel bed"),
    ("m-73", "red plastic bottle near the hab (debris)"),
]


def stream_chat(model, messages, **extra):
    t0 = time.perf_counter()
    ttft, text, n, last = None, "", 0, {}
    with requests.post(f"{OLLAMA}/api/chat", json={"model": model, "messages": messages, "stream": True,
                                                    "options": {"temperature": 0, "num_ctx": 4096}, **extra}, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            ev = json.loads(line)
            piece = ev.get("message", {}).get("content", "")
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            text += piece
            if ev.get("done"):
                last = ev
    total = time.perf_counter() - t0
    tps = last.get("eval_count", 0) / max(last.get("eval_duration", 1) / 1e9, 1e-9)
    return text, ttft, total, tps


def chat(model, messages, **extra):
    t0 = time.perf_counter()
    r = requests.post(f"{OLLAMA}/api/chat", json={"model": model, "messages": messages, "stream": False,
                                                  "options": {"temperature": 0, "num_ctx": 4096}, **extra}, timeout=600)
    r.raise_for_status()
    return r.json(), time.perf_counter() - t0


def run(model):
    print(f"\n==================== {model}")
    info = requests.post(f"{OLLAMA}/api/show", json={"model": model}, timeout=60).json()
    caps = info.get("capabilities", [])
    print(f"capabilities: {caps}")

    # warm-up (load weights), then latency
    stream_chat(model, [{"role": "user", "content": "Say ok."}])
    text, ttft, total, tps = stream_chat(model, [{"role": "system", "content": SYSTEM},
                                                 {"role": "user", "content": "In two sentences, what should a rover look for?"}])
    print(f"[latency] ttft={ttft:.2f}s total={total:.2f}s {tps:.1f} tok/s -> {text.strip()[:160]!r}")

    # tool use
    if "tools" in caps:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": SCENE}]
        calls = []
        t_all = 0.0
        for step in range(4):  # let it recall, then remember/act
            r, dt = chat(model, msgs, tools=TOOLS)
            t_all += dt
            msg = r["message"]
            tc = msg.get("tool_calls") or []
            if not tc:
                calls.append(("text", (msg.get("content") or "")[:120]))
                break
            msgs.append(msg)
            for c in tc:
                fn, args = c["function"]["name"], c["function"]["arguments"]
                calls.append((fn, args))
                if fn == "recall":
                    result = "\n".join(f"[{i}] {t}" for i, t in MEMORIES[:3] + MEMORIES[3:4])
                else:
                    result = "ok"
                msgs.append({"role": "tool", "content": result, "tool_name": fn})
            if any(fn == "act" for fn, _ in calls):
                break
        names = [c[0] for c in calls]
        print(f"[tools] {t_all:.1f}s over {len(calls)} steps: {names}")
        for fn, args in calls:
            print(f"        {fn}: {json.dumps(args)[:180] if not isinstance(args, str) else args}")
    else:
        print("[tools] not supported by this model in Ollama")

    # grounded answer with citations
    ctx = "\n".join(f"[{i}] {t}" for i, t in MEMORIES)
    q = ("Using ONLY these memories, what evidence of past water have we found? Cite memory ids in brackets. "
         "Say if a memory is irrelevant.\n" + ctx)
    text, ttft, total, tps = stream_chat(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}])
    cited = sorted({i for i, _ in MEMORIES if i in text})
    print(f"[answer] ttft={ttft:.2f}s total={total:.2f}s cited={cited}\n         {text.strip()[:300]!r}")

    # vision caption
    if "vision" in caps:
        b64 = base64.b64encode(open(IMG, "rb").read()).decode()
        text, ttft, total, tps = stream_chat(model, [{"role": "user", "content": "Describe this camera frame for a rover's memory in one sentence: terrain, objects, colors, anything suggesting water.",
                                                      "images": [b64]}])
        print(f"[caption] ttft={ttft:.2f}s total={total:.2f}s -> {text.strip()[:200]!r}")
    ps = requests.get(f"{OLLAMA}/api/ps", timeout=10).json()
    for m in ps.get("models", []):
        if m["name"].startswith(model.split(":")[0]):
            vram = m.get("size_vram", 0) / 1e9
            print(f"[placement] size={m.get('size', 0)/1e9:.2f} GB, in VRAM {vram:.2f} GB")


if __name__ == "__main__":
    for m in (sys.argv[1:] or ["qwen3-vl:2b-instruct", "llama3.2:1b"]):
        try:
            run(m)
        except Exception as e:  # noqa: BLE001
            print(f"[{m}] ERROR {type(e).__name__}: {e}")
