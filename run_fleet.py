"""Start a local SIETCH fleet: Qdrant Server, ground control (:8100) and N rovers (:8101...). Ctrl+C stops all.

    python run_fleet.py                      # ground + 2 rovers, demo pacing
    python run_fleet.py --rovers 3 --fast    # short gaps for rehearsals and tests
    python run_fleet.py --fresh              # wipe ~/.sietch and the ground collection first
    python run_fleet.py --ground-only        # this laptop is mission control; rovers run elsewhere
    python run_fleet.py --rover-only rover-2 --ground http://<ground-ip>:8100
    python run_fleet.py --brains rover-1     # only rover-1 runs the local LLM brain; others are reflex-only
    python run_fleet.py --brains none        # no LLM anywhere (low-RAM machines)

The brain is a local Ollama model (default qwen3-vl:2b-instruct, ~3.3 GB VRAM). Start Ollama with
OLLAMA_CONTEXT_LENGTH=4096 on 4 GB GPUs. Every child exits when this launcher dies (SIETCH_PARENT_PID watchdog).

Needs the Qdrant binary: set SIETCH_QDRANT_BIN, or it looks in ~/.cortex/qdrant/qdrant(.exe).
Windows: storage lives under ~/.sietch (short path; Qdrant fails beyond MAX_PATH under long dirs).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.join(os.path.expanduser("~"), ".sietch")
LOGS = os.path.join(HOME, "logs")


def up(url, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def spawn(name, args, env_extra, cwd=ROOT):
    os.makedirs(LOGS, exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1", "HF_HUB_DISABLE_XET": "1", "SIETCH_PARENT_PID": str(os.getpid()), **env_extra}
    log = open(os.path.join(LOGS, f"{name}.log"), "w")
    print(f"  {name:10} -> log {log.name}")
    return subprocess.Popen(args, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)


def wipe():
    """Fresh start: every robot's memory and ground's data (logs stay)."""
    if os.path.isdir(HOME):
        for d in os.listdir(HOME):
            if d != "logs":
                shutil.rmtree(os.path.join(HOME, d), ignore_errors=True)


def start_qdrant(procs, fresh):
    """Qdrant Server for ground (reused if one is already running); --fresh drops the fleet collection."""
    qbin = os.environ.get("SIETCH_QDRANT_BIN") or next(
        (p for p in (os.path.join(os.path.expanduser("~"), ".cortex", "qdrant", n) for n in ("qdrant.exe", "qdrant"))
         if os.path.exists(p)), None)
    if not up("http://127.0.0.1:6333/readyz", timeout=1):
        if not qbin:
            sys.exit("Qdrant binary not found: set SIETCH_QDRANT_BIN")
        qdir = os.path.join(HOME, "qdrant")
        os.makedirs(qdir, exist_ok=True)
        procs.append(spawn("qdrant", [qbin], {"QDRANT__STORAGE__STORAGE_PATH": "./storage",
                                              "QDRANT__STORAGE__SNAPSHOTS_PATH": "./snapshots",
                                              "QDRANT__TELEMETRY_DISABLED": "true"}, cwd=qdir))
        if not up("http://127.0.0.1:6333/readyz"):
            sys.exit("Qdrant did not start; see logs")
    if fresh:
        req = urllib.request.Request("http://127.0.0.1:6333/collections/fleet", method="DELETE")
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass


def stop(procs):
    for p in reversed(procs):
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    sys.stdout.reconfigure(line_buffering=True)  # progress must show even when output is redirected to a file
    ap = argparse.ArgumentParser()
    ap.add_argument("--rovers", type=int, default=2)
    ap.add_argument("--fast", action="store_true", help="8-12 s gaps, fixed 30 KB budget")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--ground-only", action="store_true")
    ap.add_argument("--rover-only", metavar="NAME")
    ap.add_argument("--ground", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="siglip2", choices=["siglip2", "clip"])
    ap.add_argument("--brain", default="qwen3-vl:2b-instruct", help="Ollama model for the robots' brains")
    ap.add_argument("--brains", default="all", help="'all', 'none', or comma-separated robot names that get a brain")
    ap.add_argument("--storage-mb", default="200", help="per-robot image storage budget before tiering")
    a = ap.parse_args()

    procs = []
    py = sys.executable
    pacing = {"SIETCH_GAP": "8,12", "SIETCH_WINDOW": "10", "SIETCH_BUDGET": "30000,30000"} if a.fast else {}

    if a.fresh:
        wipe()

    try:
        if not a.rover_only:
            start_qdrant(procs, a.fresh)
            procs.append(spawn("ground", [py, "-m", "uvicorn", "sietch.ground.app:app", "--host", "0.0.0.0",
                                          "--port", "8100", "--log-level", "warning"], {"SIETCH_MODEL": a.model}))
            if not up("http://127.0.0.1:8100/api/metrics", timeout=300):
                sys.exit("ground did not start; see logs")
            print("ground control: http://127.0.0.1:8100")

        names = [a.rover_only] if a.rover_only else ([] if a.ground_only else [f"rover-{i + 1}" for i in range(a.rovers)])
        for i, name in enumerate(names):
            port = 8101 + (int(name.split("-")[-1]) - 1 if name.split("-")[-1].isdigit() else i)
            procs.append(spawn(name, [py, "-m", "uvicorn", "sietch.node.app:app", "--host", "0.0.0.0", "--port", str(port),
                                      "--log-level", "warning"],
                               {"SIETCH_NODE": name, "SIETCH_PORT": str(port), "SIETCH_GROUND": a.ground,
                                "SIETCH_MODEL": a.model, "SIETCH_STORAGE_MB": a.storage_mb,
                                "SIETCH_BRAIN": a.brain if (a.brains == "all" or name in a.brains.split(",")) else "off",
                                **pacing}))
        for i, name in enumerate(names):
            port = 8101 + (int(name.split("-")[-1]) - 1 if name.split("-")[-1].isdigit() else i)
            ok = up(f"http://127.0.0.1:{port}/api/status", timeout=300)
            print(f"{name}: http://127.0.0.1:{port}  {'ready' if ok else 'NOT READY, see log'}")
        print("Ctrl+C to stop the fleet.")
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
