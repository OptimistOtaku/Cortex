"""Run the SIETCH Mars simulation on this laptop: Qdrant Server + one process hosting ground control (:8100),
the rover (:8101) and the scout drone (:8102). Open http://127.0.0.1:8100/sim/ (add ?demo for the scripted demo).

    python run_sim.py --fresh              # clean memories, SigLIP2, both robots think with the local LLM
    python run_sim.py --fresh --model clip # lighter embedding model
    python run_sim.py --no-brain           # reflexes only (no Ollama needed)

Comms windows open only when the simulated relay orbiter is overhead (the browser drives them).
The brain needs Ollama with the model pulled; start it with OLLAMA_CONTEXT_LENGTH=4096 on a 4 GB GPU.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

from run_fleet import LOGS, spawn, start_qdrant, stop, up, wipe

OLLAMA = os.environ.get("SIETCH_OLLAMA", "http://127.0.0.1:11434")


def warm_brain(model):
    """Load the LLM before the demo starts, so the first deliberation isn't a 20 s cold start."""
    if not up(f"{OLLAMA}/api/version", timeout=2):
        print(f"  Ollama is not running at {OLLAMA}: start it (OLLAMA_CONTEXT_LENGTH=4096 ollama serve) "
              f"or use --no-brain. The robots keep their reflexes meanwhile.")
        return
    body = json.dumps({"model": model, "prompt": "ok", "stream": False, "keep_alive": "60m",
                       "options": {"num_ctx": 4096, "num_predict": 1}}).encode()
    t0 = time.time()
    try:
        urllib.request.urlopen(urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                                      headers={"Content-Type": "application/json"}), timeout=180)
        print(f"  brain {model} loaded in {time.time() - t0:.0f}s")
    except Exception as e:
        print(f"  brain warm-up failed ({e}); the robots will retry when they need it")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="wipe every memory and the ground collection first")
    ap.add_argument("--model", default="siglip2", choices=["siglip2", "clip"])
    ap.add_argument("--brain", default="qwen3-vl:2b-instruct")
    ap.add_argument("--no-brain", action="store_true")
    ap.add_argument("--storage-mb", default="4", help="per-robot image storage budget before tiering")
    ap.add_argument("--budget", default="30000", help="bytes per orbiter pass")
    a = ap.parse_args()

    procs = []
    if a.fresh:
        wipe()
    try:
        if not a.no_brain:
            warm_brain(a.brain)
        start_qdrant(procs, a.fresh)
        procs.append(spawn("sim", [sys.executable, "-m", "sietch.sim.host"], {
            "SIETCH_MODEL": a.model, "SIETCH_WORLD": "mars", "SIETCH_GAP": "manual", "SIETCH_WINDOW": "20",
            "SIETCH_BUDGET": f"{a.budget},{a.budget}", "SIETCH_STORAGE_MB": a.storage_mb,
            "SIETCH_BRAIN": "off" if a.no_brain else a.brain,
        }))
        for name, url in (("ground", "http://127.0.0.1:8100/api/metrics"), ("rover-1", "http://127.0.0.1:8101/api/status"),
                          ("scout-1", "http://127.0.0.1:8102/api/status")):
            if not up(url, timeout=300):
                sys.exit(f"{name} did not start; see {LOGS}")
        print("Mars simulation:  http://127.0.0.1:8100/sim/        scripted demo: http://127.0.0.1:8100/sim/?demo")
        print("ground control:   http://127.0.0.1:8100    rover console: :8101    scout console: :8102")
        print("Ctrl+C to stop.")
        while True:
            time.sleep(1)
            for p in procs:
                if p.poll() is not None:
                    print(f"a process exited with code {p.returncode}; see {LOGS}")
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        stop(procs)


if __name__ == "__main__":
    main()
