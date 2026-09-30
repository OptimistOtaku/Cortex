# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SIETCH is an entry for Code Cubicle 6.0, problem statement 03 (Qdrant Edge). It is a Qdrant-powered edge memory for the local LLM brains of robots and drones that work where the link home is scarce or gone.

Each rover:
- keeps its memory in a **Qdrant Edge** shard;
- thinks with a local LLM brain (Ollama);
- in each short comms window, syncs the most valuable memories to ground control, which runs on **Qdrant Server**.

`findings_results.md` is the research and measurement log: every spike result, gotcha and design decision, with sources. Read it before you change the engine behaviour. Record new measurements there.

## Commands

The fleet needs a Qdrant Server binary. Set `SIETCH_QDRANT_BIN`, or put the binary at `~/.cortex/qdrant/qdrant(.exe)`.

```bash
pip install -r requirements.txt
pytest tests                                    # unit/integration tests (no fleet or Ollama needed)
pytest tests/test_signing.py::test_tampered_body_is_refused   # a single test
python run_sim.py --fresh                       # THE DEMO: Qdrant + one process hosting ground :8100, rover-1 :8101,
                                                # scout-1 :8102 (shared model); sim at http://127.0.0.1:8100/sim/?demo
python run_fleet.py --fast --fresh              # multi-process: Qdrant + ground :8100 + rover-1 :8101 + rover-2 :8102
python tests/e2e_scenario.py                    # sync-gate e2e against a running fleet (not collected by pytest)
python run_fleet.py --fast --fresh --model clip --brains rover-1 --storage-mb 3
python tests/e2e_brain.py                       # brain e2e: needs the fleet above + Ollama with the brain model
```

`run_fleet.py` flags:

| Flag | Effect |
|---|---|
| `--fast` | 8–12 s comms gaps and a 30 KB budget; use it for tests |
| `--fresh` | wipes `~/.sietch` and the ground collection |
| `--model clip` | lighter than the default SigLIP2 (512 vs 768 dims) |
| `--brains all\|none\|rover-1,...` | which rovers get an LLM brain |
| `--ground-only` / `--rover-only NAME --ground URL` | multi-laptop setups |

Logs go to `~/.sietch/logs/<name>.log`. Children exit when the launcher dies (the `SIETCH_PARENT_PID` watchdog in `sietch/common/runtime.py`).

Environment details:
- **Windows:** processes need `PYTHONUTF8=1`. Otherwise fastembed reads SigLIP2's tokenizer as cp1252. `run_fleet.py` sets it, so set it yourself only when you run uvicorn or spikes directly.
- **Model downloads:** set `HF_HUB_DISABLE_XET=1`.
- **Brain:** start Ollama with `OLLAMA_CONTEXT_LENGTH=4096` on a 4 GB GPU. The brain model is `qwen3-vl:2b-instruct` (env `SIETCH_BRAIN`, or `off` for a reflex-only robot). `llama3.2:1b` tool calls are broken.
- **Ground's contradiction check:** calls the TypeSafe Jev API with `TYPESAFE_API_KEY` from the git-ignored `.env`, which `load_env()` reads.
- **Dev machine limits:** 7.3 GB RAM and a GTX 1650 4 GB. Ground, two rovers and a brain together are tight, so prefer `--model clip` and a single brain.

## Architecture

**Rover (`sietch/node/`), one FastAPI process per robot:**
- `perception.py`: image and text embeddings in one shared space (SigLIP2 or CLIP via fastembed), zero-shot tags (a per-world vocabulary; Mars tags are calibrated against hub labels).
- `store.py`: two Qdrant Edge shards plus SQLite.
  - `memory` shard: dense `img` vectors and sparse BM25 `tags`.
  - `fleet` shard: a mirror of what ground already has.
  - SQLite tables: ops, meta, events, ledger.
  - **Every write goes to the SQLite log first, then to Edge.** Qdrant Edge 0.8.0 loses unflushed writes on a crash, so the log is replayed on start. A flush thread checkpoints every 0.25 s.
- `memory.py` (`NodeMemory`): one memory holds five kinds, told apart by the payload field `kind`: `observation` (images) and the text kinds `insight`, `decision`, `answer`, `note`.
  - `recall()` runs hybrid RRF search and excludes superseded items.
  - `rank()` is the **sync gate**: one Qdrant `Formula` per group scores value per KB (relevance², novelty, influence, importance; full-resolution requests first). There is no FIFO baseline any more (removed at the user's request).
  - `plan_window`/`package` fill a comms window under its byte budget. Text is tiny, so insights go down before pictures.
  - `enforce_budget` tiers stored images: original, then full, then thumbnail, then embedding only.
- `brain.py`: two speeds of thinking.
  - **Reflex:** `consider()` runs on every ingest and takes milliseconds. It wakes the brain only when the frame is relevant and novel.
  - **Deliberation:** a single worker queue, because there is one GPU. It calls Ollama `/api/chat` with the tools `recall`, `remember` and `act`. It produces insights and decisions that cite evidence refs such as `obs-xxxxxxxx`.
  - Q&A with citations; `answer_without_brain` is the fallback for reflex-only robots.
- `comms.py`: the comms-window loop.
  - Uplink: HMAC-signed ground commands (intent, relabel, discard, request_full, supersede, ask). Unsigned or forged commands are refused and logged.
  - Downlink: one policy per window.
- `app.py` + `static/index.html`: the rover console.

**Ground (`sietch/ground/app.py`):** mission control on the Qdrant Server collection `fleet`.
- Merges duplicate sightings into sites (similarity ≥ 0.92).
- Opens disputes when rovers' conclusions about one site contradict (checked by Jev in a thread) and resolves them with supersede commands.
- Also handles retasking (`/api/intent`), ask-a-rover, metrics, and a red-team forged-command endpoint.
- Sites merge by position (12 m `GeoRadius`) when memories carry `geo`; otherwise by embedding similarity ≥0.92.

**Mars simulation (`sietch/sim/`):**
- `host.py` runs ground and both robots in one process, sharing one `Perception.shared()` model, because this laptop has 7.3 GB of RAM.
- `static/js/` is three.js with no build step (vendored in `static/vendor/`):
  - `world.js`: terrain and targets;
  - `robots.js`: models and cameras. Navcam frames are rendered; science targets use NASA photos;
  - `memviz.js`: pins drawn from `/api/map`;
  - `main.js`: HUD, polling and actions;
  - `director.js`: the scripted demo.
- Comms windows open only when the simulated orbiter calls `/api/window_now` (`SIETCH_GAP=manual`).
- Captures carry `X-Pos: x,z`, which becomes payload `pos` plus `geo` (`sietch/common/geo.py`: Earth metres-per-degree, so Qdrant's radius equals sim metres).
- Commands are signed on insert.

**Common (`sietch/common/`):**
- `hlc.py`: hybrid logical clocks that order conflicting updates.
- `signing.py`: HMAC over {id, node, kind, body, hlc}, with the key in `SIETCH_KEY`.
- `db.py`: `LockedDB` serialises all SQLite access behind one RLock; concurrent access caused "API misuse" errors.
- `config.py`: `NodeConfig`, i.e. gate weights, thresholds and brain settings, overridable through env vars.

## Qdrant Edge gotchas (qdrant-edge-py 0.8.0)

- A boolean `MatchValue` filter never matches. Store flags as integers 0/1 (`superseded`, `requested_full`, ...).
- `Expression.Div` needs `by_zero_default` as a positional argument.
- `retrieve()` requires `with_vector`.
- Qdrant Server storage must live on a short path (`~/.sietch/qdrant`) because of Windows MAX_PATH.

## Other directories

- `spikes/`: numbered standalone validation scripts (`spike_01`…`spike_12`) for the engine, sync, models and the brain. `spikes/photos/commons` is the 45-image, 9-folder test gallery that the e2e scripts traverse.
- `tests/conftest.py`: a session-scoped `perception` fixture. Model loading is slow, so reuse it.
- `test_brain_memory.py`: fakes the brain's `_chat`, so the tests need no Ollama.
