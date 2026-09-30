// The scripted demo (open /sim/?demo, press Play). It drives the world and the cameras; everything the robots
// remember, conclude, send and refuse comes live from the real backend, so waits are real (the LLM needs ~10-40 s
// per deliberation). Waits are marked "time-lapse" so the recording can be sped up there in editing.
import * as THREE from "three";
import { get, post, robot, sleep, until } from "./api.js";
import { heightAt } from "./world.js";

const V = (x, y, z) => new THREE.Vector3(x, y, z);

export async function runDemo(sim) {
  const T = Object.fromEntries(sim.TARGETS.map((t) => [t.key, t]));
  const { rover, scout } = sim;
  sim.state.autoPasses = false;
  const cap = (ch, t, s, ms = 0) => sim.caption(ch, t, s, ms);
  const lapse = sim.timelapse;

  async function examined(name, r) {
    if (!r.brain_woken) await post(robot(name, `/api/examine/${r.id}?why=${encodeURIComponent("mission plan: examine this target")}`)).catch(() => {});
    return r.id;
  }
  async function decisionOn(name, obsId, timeout = 420000) {
    return until(async () => (await get(robot(name, "/api/brain"))).decisions.find((d) => (d.evidence || []).includes(obsId)),
      { timeout, every: 1500 });
  }
  const approach = (t, d = 9) => [t.x, t.z + d];

  // ---------------------------------------------------------------- 0 · cold open
  sim.setShot("path", { path: { from: V(250, 150, 330), to: V(25, 30, 95), lookFrom: V(-120, 0, -240), look: V(0, 0, 0), seconds: 16, t0: performance.now() } });
  cap("JEZERO CRATER · MARS", "Earth is 12 light-minutes away", "The link home is a relay orbiter that passes a few times a sol: minutes of contact, kilobytes of budget.");
  await sleep(10000);
  cap("SIETCH", "Qdrant Edge memory for robot brains that lose signal", "a rover and a scout drone, each with its own memory and its own local LLM");
  await sim.retask("evidence of past water");
  await sleep(6000);
  sim.setShot("sky", { of: rover });
  cap("SOL 142 · ORBITER PASS", "Today's mission arrives on the pass", "signed at Earth, uplinked in the window: \"evidence of past water\"");
  await sim.orbiterPass();

  // ---------------------------------------------------------------- 1 · remember
  sim.setShot("chase", { of: rover });
  cap("1 · REMEMBER", "Every frame goes into the robot's own Qdrant Edge memory", "image and text in one vector space · searchable offline in under a millisecond");
  const C = T.mud_cracks, S = T.streambed;
  const B = T.backshell;
  const scoutJob = (async () => {
    // the scout surveys the slab from the air, then its own brain judges the landing debris
    await sim.drive("scout-1", C.x - 3, C.z + 2, { every: 40, alt: 14 });
    await sim.captureTarget("scout-1", C);
    await sim.drive("scout-1", B.x + 2, B.z + 2, { every: 60, alt: 14 });
    const rb = await sim.captureTarget("scout-1", B);
    await examined("scout-1", rb);
    return rb;
  })();
  lapse(true);
  await sim.drive("rover-1", ...approach(S, 7), { every: 15 });
  lapse(false);
  const rs = await sim.captureTarget("rover-1", S);
  await examined("rover-1", rs);
  sim.setShot("chase", { of: scout });
  cap("1 · REMEMBER", "The scout flies ahead and looks from 14 m up", "its own memory and its own brain: it photographs the slab, then judges the landing debris");
  await sleep(9000);  // the scout keeps flying and thinking in the background; both brains share one GPU
  scoutJob.catch(() => {});

  // ---------------------------------------------------------------- 2 · think
  sim.setShot("chase", { of: rover });
  cap("2 · THINK", "The rover drives to the slab", "navigation frames pile up in memory; they will mostly stay on board");
  lapse(true);
  await sim.drive("rover-1", ...approach(C), { every: 15 });
  lapse(false);
  const rc = await sim.captureTarget("rover-1", C);
  await examined("rover-1", rc);
  sim.setShot("orbit", { point: V(C.x, heightAt(C.x, C.z), C.z), radius: 22, height: 11, speed: 0.1 });
  cap("2 · THINK", "A one-query reflex decides what deserves thought", "Qdrant Formula: relevance² + novelty + importance per KB wakes the local LLM (qwen3-vl 2B on a 4 GB laptop GPU)");
  lapse(true);
  const dec = await decisionOn("rover-1", rc.id);
  lapse(false);
  if (dec) {
    cap("2 · THINK", `It looked, recalled, concluded with evidence, and decided: ${dec.action.replace(/_/g, " ")}`, (dec.reason || "").slice(0, 150), 9000);
    if (dec.action === "investigate") {
      sim.setShot("chase", { of: rover });
      await sleep(3000);
      await sim.investigate(C);
      cap("2 · THINK", "The brain's decision moved the robot", "closer look: a new close-up frame, remembered and judged like any other", 7000);
    }
  }
  lapse(true);
  await sim.brainIdle("rover-1", 180000);
  lapse(false);

  // ---------------------------------------------------------------- 3 · sync
  sim.setShot("sky", { of: rover });
  cap("3 · SYNC", "The orbiter rises: 30 KB for this pass", "the gate spends it by value per KB · conclusions first, then the best evidence");
  await sim.orbiterPass();
  const w = (await get(robot("rover-1", "/api/status"))).comms.last;
  const kinds = {};
  (w.items || []).forEach((i) => (kinds[i.kind] = (kinds[i.kind] || 0) + 1));
  sim.setShot("map", { point: V(C.x - 30, 0, C.z + 50), height: 190 });
  cap("3 · SYNC", `Sent ${Object.entries(kinds).map(([k, n]) => `${n} ${k}${n > 1 ? "s" : ""}`).join(" + ") || "nothing"} in ${(w.bytes / 1000).toFixed(1)} KB`,
    `${w.waiting} memories wait on board · a conclusion is ~1.8 KB, a picture ~10-20 KB · green pins reached Earth`);
  await sleep(9000);

  // ---------------------------------------------------------------- 4 · retask
  await sim.retask("possible signs of ancient life");
  cap("4 · RETASK", "One sentence retasks the fleet", "\"possible signs of ancient life\" · queued at Earth, it rises with the next pass");
  sim.setShot("sky", { of: rover });
  await sim.orbiterPass();
  sim.setShot("map", { point: rover.pos.clone(), height: 150 });
  cap("4 · RETASK", "Re-ranked on board, instantly, no retraining",
    "the whole memory re-scored against the new mission");
  await sleep(8000);
  const L = T.leopard;
  sim.setShot("chase", { of: rover });
  lapse(true);
  await sim.drive("rover-1", ...approach(L), { every: 15 });
  lapse(false);
  const rl = await sim.captureTarget("rover-1", L);
  await examined("rover-1", rl);
  sim.setShot("orbit", { point: V(L.x, heightAt(L.x, L.z), L.z), radius: 20, height: 10, speed: 0.1 });
  cap("4 · RETASK", "A rock with 'leopard spots'", "what Perseverance found at Cheyava Falls in 2024 · the brain judges it against the new mission");
  lapse(true);
  const dl = await decisionOn("rover-1", rl.id);
  lapse(false);
  if (dl) cap("4 · RETASK", `Decision: ${dl.action.replace(/_/g, " ")}`, (dl.reason || "").slice(0, 160), 9000);
  await sleep(6000);

  // ---------------------------------------------------------------- 5 · blackout
  await sim.setBlackout(true);
  cap("5 · BLACKOUT", "Solar conjunction: no link for 14 sols", "the robot still remembers and reasons, and shows its evidence");
  await sleep(4000);
  const r = await sim.recall("mud cracks");
  cap("5 · BLACKOUT", `Recall "mud cracks": ${r.ms} ms, of which Qdrant Edge ${r.qdrant_ms} ms`, "hybrid search (dense + BM25) on the rover itself, with a geo filter when asked what is nearby", 8000);
  await sleep(8000);
  cap("5 · BLACKOUT", "Ask the rover: where did you see signs of water?", "answered on board by the local LLM from its own memories, with citations and distances");
  lapse(true);
  const a = await sim.ask("Where did you see signs of water?");
  lapse(false);
  if (a) cap("5 · BLACKOUT", "Answered from memory, offline", a.answer.slice(0, 190), 11000);
  await sleep(11000);
  sim.mem.clearHighlight();
  await sim.setBlackout(false);

  // ---------------------------------------------------------------- 6 · trust
  const f = await sim.forge("rover-1");
  cap("6 · TRUST", "Someone spoofs mission control", `a forged 'discard ${f.item.slice(0, 8)}' is queued for the rover with a bad signature`);
  sim.setShot("chase", { of: rover });
  await sleep(4000);
  await sim.orbiterPass();
  cap("6 · TRUST", "Refused: every command is signed", "the memory it targeted is still on board · when storage fills, pixels tier down first; conclusions stay", 9000);
  sim.setShot("map", { point: rover.pos.clone(), height: 150 });
  await sleep(9000);

  // ---------------------------------------------------------------- close
  sim.caption("", "", "");
  sim.setShot("path", { path: { from: V(rover.pos.x + 40, rover.pos.y + 12, rover.pos.z + 60), to: V(rover.pos.x + 12, rover.pos.y + 4, rover.pos.z + 20),
    lookFrom: rover.pos.clone(), look: V(rover.pos.x - 200, rover.pos.y + 40, rover.pos.z - 400), seconds: 14, t0: performance.now() } });
  const t0 = performance.now();
  while (performance.now() - t0 < 12000) { sim.world.setSunset(Math.min(1, (performance.now() - t0) / 10000)); await sleep(50); }
  cap("SIETCH", "Everything you saw ran live on this laptop", "Qdrant Edge on each robot · Qdrant Server at ground · a local LLM brain · real NASA imagery at the targets");
  await sleep(7000);
  sim.caption("", "", "");
  await sim.endcard(true);
}
