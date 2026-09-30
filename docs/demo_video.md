# SIETCH Mars: demo video (3:00)

Everything runs on one laptop:
- Qdrant Server;
- one process hosting ground control, the rover and the scout drone;
- Ollama (qwen3-vl 2B) on the GTX 1650;
- Edge rendering the sim on the AMD iGPU.

The director (`/sim/?demo`) drives the world and the cameras, and shows the captions. Every memory, conclusion, sync, answer and refusal comes live from the real backend.

A take runs about 8–12 minutes in real time, because each LLM deliberation takes 10–40 s. Speed up the parts marked **⏩ time-lapse** to reach 3:00.

## Recording checklist
1. Close VS Code, Claude, Firefox and other apps. The laptop has 7.3 GB of RAM, and the demo needs about 5.7 GB.
2. Start Ollama: `set OLLAMA_CONTEXT_LENGTH=4096 && ollama serve`, in its own terminal.
3. Run `python run_sim.py --fresh`, and wait for the "Mars simulation" line. The script warms the brain.
4. Open Edge at `http://127.0.0.1:8100/sim/?demo` and press **F11**.
   - If the frame rate is poor, go to Windows Settings → Display → Graphics → Edge → **Power saving** (AMD).
5. Press **Win+Alt+R** (Xbox Game Bar) to start recording, then click **Play demo**. Stop with Win+Alt+R when the end card shows.
6. Record **2 takes**. The LLM's wording differs between takes, so keep the take with the clearest conclusions.
7. Edit in **Clipchamp** (built into Windows):
   - speed up the ⏩ segments ×3–6;
   - trim dead air;
   - add the voiceover below;
   - export at 1080p.
8. `python run_sim.py --fresh` again before every take, so memories start empty.

## Script

| Time | Chapter (on-screen caption) | What the viewer sees (live) | Voiceover |
|---|---|---|---|
| 0:00–0:14 | JEZERO CRATER · Earth is 12 light-minutes away | Flyover of the delta scarp, the dry channel and dunes; the rover and the scout at the landing site | "On Mars, the link home is a relay orbiter that passes a few times a day. Minutes of contact. Kilobytes of budget. Everything else, the robot has to decide alone." |
| 0:14–0:24 | SIETCH · Today's mission arrives on the pass | Sky shot as the orbiter crosses and the mission uplinks: "evidence of past water" | "SIETCH is the memory for those robots' brains: Qdrant Edge on every robot, a local LLM that thinks with it." |
| 0:24–0:45 | 1 · REMEMBER | The rover drives (⏩); each navcam frame becomes a grey pin; the scout flies ahead and photographs the slab from 14 m | "Every frame goes into the robot's own Qdrant Edge memory, where images and text share one vector space, searchable offline in under a millisecond. The scout has its own memory and its own brain." |
| 0:45–1:15 | 2 · THINK | At the cracked slab the reflex wakes the brain; the bubble shows *looking → recall → remember → act*; a cyan diamond appears, linked to its evidence pins; the decision *investigate* drives the rover closer for a close-up | "A single Qdrant Formula query is the reflex: relevance, novelty and importance per kilobyte. Only when it fires does the LLM wake. It looks, recalls what it has seen nearby, writes a conclusion that cites its evidence, and acts. Here it decides to take a closer look." |
| 1:15–1:35 | 3 · SYNC | The orbiter rises; packets fly up, cyan conclusions first, then amber pictures; the map shows green pins (sent) against grey ones (still on board); Mission Control lists conclusions "before the pictures" | "The pass carries 30 kilobytes. The gate spends them by value per kilobyte. A conclusion is one-tenth the size of a picture, so meaning reaches Earth first. The rest waits for the next pass." |
| 1:35–2:05 | 4 · RETASK | "possible signs of ancient life" goes up signed; the map recolours as the rover re-ranks its whole memory on board; the rover drives (⏩) to the leopard-spot rock, and the brain flags it for Earth | "One sentence retasks the fleet. It arrives with the next pass, and each robot re-ranks everything it remembers, on board, without retraining." |
| 2:05–2:30 | 5 · BLACKOUT | "Solar conjunction: no link". Recall "mud cracks" highlights ranked pins (ms timing shown); asked "Where did you see signs of water?", the rover answers on board with citations; cited pins light up | "No link for two weeks. The robot still remembers and reasons, and it shows its evidence." |
| 2:30–2:48 | 6 · TRUST | A forged "discard" is refused at the next pass (red shield); the storage bar tiers pictures down | "Commands are signed; a spoofed one bounces. When storage fills, pixels go first. The conclusions stay." |
| 2:48–3:00 | End card | Blue Martian sunset → end card with NASA credits | "SIETCH: Qdrant Edge memory for robot brains that lose signal, on Mars, underwater, or after a disaster. Everything you saw ran live on this laptop." |

## If a take goes wrong
- **The brain says `move_on` at the slab:** the investigate beat is skipped automatically. The chapter still works.
- **Stutter or out-of-memory:** close more apps, or run with `--model clip` (lighter embedding model).
