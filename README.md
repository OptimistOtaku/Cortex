# SIETCH: edge memory for robot brains that lose signal

SIETCH is a memory layer built on **Qdrant Edge** for the local LLM brains of robots and drones. It is for places where the link home is scarce or gone: planetary rovers, exploration, disaster drones, and operations with denied comms.

Each robot:
- remembers what it sees;
- recalls it in milliseconds with no connection;
- lets its onboard LLM reason over that memory and decide;
- sends the most valuable knowledge home in each short comms window, with conclusions first and pictures on demand.

Mission control on **Qdrant Server** can retask the whole fleet with one sentence, ask any robot a question, and settle disputes when two robots disagree.

**The demo is a 3D Mars simulation that runs entirely on one laptop.** A rover and an Ingenuity-style scout drone explore a Jezero-like crater. Each has its own Qdrant Edge memory and a local LLM brain. At science targets their cameras see real NASA imagery; between targets they see the rendered terrain. Every memory, conclusion, sync, dispute and refusal comes live from the real system.

Code Cubicle 6.0 · Problem Statement 03 (Qdrant Edge).

## How a robot thinks

```
camera frame ─► perception (SigLIP2/CLIP embedding, zero-shot tags)
             ─► Qdrant Edge memory ─► REFLEX (ms): one Formula query scores mission relevance, novelty, value per KB
                                        │ relevant and novel?
                                        ▼
                                  DELIBERATION (~10 s, local LLM qwen3-vl:2b on Ollama)
                                  sees the frame → recall(memory) → remember(insight citing evidence) → act
                                        │
          comms window opens ─► SYNC GATE: insights (~1.7 KB) before pictures, value per KB under the byte budget
```

- **One memory, five kinds.** Observations (images), plus the text kinds insights, decisions, answers and operator notes. All five live in one Edge shard and share SigLIP's text–image space, so a sentence the brain wrote sits next to the pictures it is about.
- **Reflex vs deliberation.** Every frame gets the millisecond reflex. The LLM wakes only when the reflex flags a frame, when someone asks a question, or when an operator asks it to look. It runs on a single worker queue because there is one GPU.
- **Reflex-only robots still work.** A robot with no GPU, or whose LLM crashes, keeps the reflex, the sync gate and recall. It answers questions from recall alone. The brain retries transient Ollama failures instead of dropping the task.

## How it maps to PS03

| PS03 goal | SIETCH |
|---|---|
| Searchable semantic memory on the device | Observations and the brain's insights and decisions sit in the robot's Qdrant Edge shard, in one shared text–image embedding space |
| Low-latency vector and hybrid search offline | Dense and BM25 search fused with weighted RRF. Recall in airplane mode takes a median of ~56 ms, of which Qdrant Edge takes ~0.5 ms. The brain answers questions offline and cites the memories it used |
| Decide what stays local vs synced | The **sync gate**: one Qdrant `Formula` ranks memories by value per KB, from mission relevance², novelty against the fleet, influence and the brain's importance. See the details below the table |
| Intermittent connectivity | Comms windows with byte budgets. Our own SQLite write-ahead log makes every write crash-safe; Qdrant Edge 0.8.0 loses unflushed writes on a hard exit |
| Sync edge ↔ Qdrant Server | Point-level downlink. Ground's knowledge flows back up, so a robot never pays twice for a site another robot already delivered. Full resolution arrives on request, streamed in chunks across windows |
| Evolving memory, updates, conflicts | Covered by the steps below the table |
| UI for memory, search, sync status, activity | Covered by the UIs below the table |
| Meaningful edge-to-cloud AI workflow | One-sentence retasking; ask-a-robot; an insight-first downlink; spatial recall ("what's near here?", with a Qdrant geo filter) |

**Sync gate details.**
- Insights go down before pictures, because a conclusion is ~1.7 KB against ~10–30 KB for a thumbnail.
- Near-duplicates are held back, as is anything another robot already delivered.

**Evolving memory and conflicts.**
- Hybrid logical clocks order updates.
- Sightings of the same place from different robots merge into one site: within 12 m when positions are known (Qdrant `GeoRadius`), otherwise near-identical pictures.
- When two robots' conclusions about one site contradict, ground opens a **dispute**. The contradiction check is the TypeSafe Jev API.
- Mission control resolves the dispute, and a signed `supersede` command removes the loser from the robot's recall.
- Storage budgets tier raw evidence down from full resolution to thumbnail to embedding only, while insights keep the meaning.

**The UIs.**
- **Mars simulation** (`/sim/`): the 3D world, where memories appear as pins where they were made. Conclusions are diamonds linked to their evidence, packets fly up on orbiter passes, and recall lights up ranked pins.
- **Robot console:** camera, brain panel with a live thinking trace, conclusions, the downlink queue with value per KB, the held-back list, offline ask and recall, windows, memory tiers and activity.
- **Ground:** conclusions and disputes, pictures received, and ask-a-robot.

**Security.** Ground's commands are HMAC-signed. A robot refuses a forged command and logs it; the ground UI has a red-team button to demonstrate this.

## Results

`tests/e2e_scenario.py` runs the sync gate against a local fleet: 2 rovers, a 45-image gallery and 30 KB windows. **11/11 checks pass.** After retasking to "evidence of past water", what comes down is at least twice as mission-relevant as what the rovers hold.

`tests/e2e_brain.py` runs the brain on rover-1 (qwen3-vl:2b on a GTX 1650) while rover-2 is reflex-only. **12/12 checks pass.** Its checks:
- the reflex wakes the brain;
- the brain writes an insight citing evidence and logs a decision;
- the insight reaches ground no later than its picture;
- offline Q&A with citations;
- ask over the link, for both rovers;
- a dispute is opened, resolved and superseded;
- a forged command is refused;
- storage tiers.

Measured in that run:
- the reflex woke the brain for 4 of 20 frames;
- a 1.7 KB conclusion reached ground while its picture was still held back;
- an offline answer took 6 s;
- ask-a-robot round trips took 15 s with the brain and 3 s reflex-only;
- Jev scored the contradiction p = 0.92.

Measurements and gotchas are in [findings_results.md](findings_results.md).

## Run it

Requirements:
- Python 3.12 or 3.13, on Windows, Linux or macOS.
- The Qdrant Server binary from https://github.com/qdrant/qdrant/releases. Set `SIETCH_QDRANT_BIN`, or put it at `~/.cortex/qdrant/qdrant(.exe)`.
- For the brain: [Ollama](https://ollama.com), then `ollama pull qwen3-vl:2b-instruct`. It needs ~3.3 GB of VRAM; on 4 GB GPUs run `OLLAMA_CONTEXT_LENGTH=4096 ollama serve`.
- For ground's contradiction check: `TYPESAFE_API_KEY` in a git-ignored `.env`.

```bash
pip install -r requirements.txt
python -m sietch.sim.fetch_assets                  # once: the NASA imagery for the science targets
python run_sim.py --fresh                          # THE DEMO: Qdrant + one process (ground, rover-1, scout-1); open
                                                   # http://127.0.0.1:8100/sim/  (scripted demo: /sim/?demo)
python run_fleet.py --fresh                        # multi-process fleet: Qdrant + ground (:8100) + 2 rovers (:8101, :8102)
python run_fleet.py --fast --fresh --model clip --brains rover-1   # 8-12 s windows; only rover-1 runs the LLM
python run_fleet.py --brains none                  # reflex-only fleet (no Ollama needed)
python tests/e2e_scenario.py                       # sync-gate checks against the running fleet
python tests/e2e_brain.py                          # brain checks (fleet started with --brains rover-1)
pytest tests                                       # unit/integration tests
```

The demo video script and recording checklist are in [docs/demo_video.md](docs/demo_video.md). In the sim, click the ground to drive the rover, shift-click to fly the scout, and use keys 1–7 for camera shots, C to capture and P for an orbiter pass.

**Several laptops (optional):**
- One laptop runs `python run_fleet.py --ground-only`.
- Each teammate runs `python run_fleet.py --rover-only rover-2 --ground http://<ground-ip>:8100` and opens `http://127.0.0.1:810N`.
- Give the brain to the robots with a 4 GB+ GPU; the rest run with `--brains none`.

**Notes:**
- The first start downloads the embedding model: ~1.6 GB for SigLIP2, or use `--model clip`, which is lighter.
- On a machine with 8 GB of RAM, close browsers before running ground, two robots and the brain together.

## Layout

- `sietch/node/`: the robot.
  - `perception.py`: embeddings and tags.
  - `store.py`: Edge shards and the SQLite write-ahead log.
  - `memory.py`: memory kinds, recall, the sync gate and storage tiers.
  - `brain.py`: the LLM brain, with its tools, questions and fallback.
  - `comms.py`: windows and signed commands.
  - `app.py` + `static/`: the robot console.
- `sietch/ground/`: mission control on Qdrant Server (sites, disputes, ask), in `app.py` + `static/`.
- `sietch/sim/`: the Mars simulation.
  - `host.py`: ground and both robots in one process.
  - `fetch_assets.py`: the NASA imagery.
  - `static/`: three.js world, robots, memory pins, HUD and the scripted director.
- `sietch/common/`: hybrid logical clocks, command signing, the locked SQLite wrapper and config.
- `spikes/`: numbered runtime validations of Qdrant Edge, the models, the gate and the brain.
- `findings_results.md`: every measurement and gotcha, with sources.
