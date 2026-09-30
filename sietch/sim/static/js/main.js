// SIETCH Mars: the 3D simulation. Everything the robots remember, decide and send is real (their own Qdrant Edge
// memories, the local LLM brain, the sync gate, ground on Qdrant Server); this page supplies the world, the motion,
// the cameras and the relay orbiter whose passes open the comms windows.
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/CSS2DRenderer.js";
import { GROUND, ROBOTS, capture, get, post, robot, sleep, until } from "./api.js";
import { buildWorld, heightAt, TARGETS } from "./world.js";
import { Orbiter, Rover, Scout, photoFrame, renderFrame } from "./robots.js";
import { MemoryViz } from "./memviz.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const DEMO = new URLSearchParams(location.search).has("demo");

// ---------------------------------------------------------------- scene
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
$("view").appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.setSize(innerWidth, innerHeight);
Object.assign(labels.domElement.style, { position: "fixed", inset: "0", pointerEvents: "none" });
$("view").after(labels.domElement);  // under the HUD panels

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.3, 6000);
const world = buildWorld(scene);
const rover = new Rover(scene), scout = new Scout(scene), orbiter = new Orbiter(scene);
const mem = new MemoryViz(scene);
rover.place(0, 22, 0);
scout.place(6, 28);
const bots = { "rover-1": rover, "scout-1": scout };
const controls = new OrbitControls(camera, renderer.domElement);
controls.enabled = false;
controls.maxPolarAngle = 1.45;

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight); labels.setSize(innerWidth, innerHeight);
});

// target name tags + a thought bubble over each robot
for (const t of TARGETS) {
  const el = Object.assign(document.createElement("div"), { className: "lbl", textContent: t.name });
  const o = new CSS2DObject(el);
  o.position.set(t.x, heightAt(t.x, t.z) + 4.2, t.z);
  scene.add(o);
}
const bubbles = {};
for (const [name, bot] of Object.entries(bots)) {
  const el = Object.assign(document.createElement("div"), { className: `bubble ${name === "scout-1" ? "scout" : ""}` });
  bubbles[name] = { el, obj: new CSS2DObject(el), until: 0 };
  bubbles[name].obj.visible = false;
  scene.add(bubbles[name].obj);
}

// ---------------------------------------------------------------- camera shots
const shot = { mode: "chase", of: rover, point: new THREE.Vector3(), angle: 0, radius: 32, height: 16, path: null };
const lookAt = new THREE.Vector3();
function setShot(mode, opts = {}) {
  Object.assign(shot, { mode, ...opts });
  controls.enabled = mode === "free";
  if (mode === "free") controls.target.copy(shot.of.pos);
}
function updateCamera(dt) {
  const k = 1 - Math.exp(-dt * 2.2);
  let want = new THREE.Vector3(), look = new THREE.Vector3();
  const of = shot.of;
  if (shot.mode === "free") { controls.update(); return; }
  if (shot.mode === "chase") {
    const back = new THREE.Vector3(Math.sin(of.heading), 0, Math.cos(of.heading));
    want.copy(of.pos).addScaledVector(back, of === scout ? 22 : 15).add(new THREE.Vector3(0, of === scout ? 10 : 6.5, 0));
    look.copy(of.pos).add(new THREE.Vector3(0, 2, 0));
  } else if (shot.mode === "orbit") {
    shot.angle += dt * (shot.speed ?? 0.12);
    const c = shot.point;
    want.set(c.x + Math.cos(shot.angle) * shot.radius, heightAt(c.x, c.z) + shot.height, c.z + Math.sin(shot.angle) * shot.radius);
    look.copy(c);
  } else if (shot.mode === "map") {
    const c = shot.point;
    want.set(c.x + 1, heightAt(c.x, c.z) + (shot.height || 170), c.z + 60);
    look.copy(c);
  } else if (shot.mode === "pov") {
    camera.position.copy(of.cam.position); camera.quaternion.copy(of.cam.quaternion); return;
  } else if (shot.mode === "sky") {
    // from north of the robot, looking south into the sky the orbiter crosses (the sun stays behind the camera)
    want.copy(of.pos).add(new THREE.Vector3(8, 4, -26));
    const sky = orbiter.active ? orbiter.model.position.clone() : new THREE.Vector3(of.pos.x, 420, 900);
    look.copy(of.pos).lerp(sky, 0.06);
  } else if (shot.mode === "path") {
    const p = shot.path, t = Math.min(1, (performance.now() - p.t0) / (p.seconds * 1000)), e = t * t * (3 - 2 * t);
    camera.position.lerpVectors(p.from, p.to, e);
    lookAt.lerpVectors(p.lookFrom || p.look, p.look, e);
    camera.lookAt(lookAt);
    return;
  }
  const floor = heightAt(want.x, want.z) + 2;
  if (want.y < floor) want.y = floor;
  camera.position.lerp(want, k);
  lookAt.lerp(look, k);
  camera.lookAt(lookAt);
}
camera.position.set(40, 60, 120);

// ---------------------------------------------------------------- live state from the backend
const state = { status: {}, brain: {}, maps: {}, metrics: {}, feed: [], disputes: [], lastWindow: {}, blackout: false, budget: 30000,
  nextPass: performance.now() + 45000, autoPasses: !DEMO, busy: {}, lastDecision: null };

async function poll() {
  for (const name of Object.keys(ROBOTS)) {
    try {
      const [s, m] = await Promise.all([get(robot(name, "/api/status")), get(robot(name, "/api/map"))]);
      state.status[name] = s;
      state.maps[name] = m.memories;
      mem.update(name, m.memories);
      const w = s.comms.last || {};
      if (w.window && w.window !== state.lastWindow[name]) {
        if (state.lastWindow[name] !== undefined && (w.items || []).length) mem.launch(bots[name].headPos(), orbiter.model, w.items);
        if ((w.commands || []).some((c) => c.refused)) shield(name);
        state.lastWindow[name] = w.window;
      }
    } catch (e) { /* robot starting */ }
  }
  try {
    state.brain["rover-1"] = await get(robot("rover-1", "/api/brain"));
    state.brain["scout-1"] = await get(robot("scout-1", "/api/brain"));
  } catch (e) { /* ignore */ }
  hud();
  autonomy();
}
async function pollGround() {
  try {
    [state.metrics, state.feed, state.disputes] = await Promise.all([get(`${GROUND}/api/metrics`),
      get(`${GROUND}/api/feed?kind=insight,note,answer&limit=40`), get(`${GROUND}/api/disputes`)]);
    mcPanel();
  } catch (e) { /* ignore */ }
}

// ---------------------------------------------------------------- HUD
function fmtTime(s) { return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`; }
function hud() {
  const s = state.status["rover-1"];
  if (!s) return;
  $("intent").textContent = s.intent;
  const link = $("link");
  if (state.blackout) {
    link.className = "panel down"; $("linkState").textContent = "BLACKOUT"; $("linkSub").textContent = "solar conjunction: no link for 14 sols";
  } else if (orbiter.active && orbiter.overhead) {
    link.className = "panel up"; $("linkState").textContent = "ORBITER PASS · LINK UP"; $("linkSub").textContent = `${(state.budget / 1000).toFixed(0)} KB this pass per robot`;
  } else {
    link.className = "panel"; $("linkState").textContent = "NO LINK";
    $("linkSub").textContent = state.autoPasses ? `next orbiter pass in ${fmtTime(Math.max(0, (state.nextPass - performance.now()) / 1000))}` : "waiting for the next orbiter pass";
  }
  // brain
  const b = state.brain["rover-1"] || {};
  $("brainModel").textContent = b.model || "no LLM";
  $("brainQ").textContent = b.queued ? `${b.queued} queued` : "";
  const task = b.task;
  $("brainState").innerHTML = task ? `<b class="b">${esc(task.kind)}</b> ${esc(task.ref || "")} <span class="muted">${esc((task.why || task.question || "").slice(0, 70))}</span>`
    : `<span class="muted">${esc(b.status || "idle")} · the reflex (a Qdrant Formula) wakes it for relevant, novel frames</span>`;
  const tr = (b.trace || []).slice(-6).map((t) => {
    const d = typeof t.detail === "string" ? t.detail : t.detail.text || t.detail.query || t.detail.action || JSON.stringify(t.detail);
    return `<div><span class="st">${t.t.toFixed(0)}s ${esc(t.step)}</span> ${esc(d)}</div>`;
  }).join("");
  $("trace").innerHTML = tr + (b.partial ? `<div><span class="st">answer</span> ${esc(b.partial.slice(-120))}▌</div>` : "");
  // queue
  const rows = (state.maps["rover-1"] || []).filter((m) => m.rank).sort((a, c) => a.rank - c.rank).slice(0, 5);
  const top = Math.max(...rows.map((r) => r.value_per_kb || 0), 0.001);
  $("queueRows").innerHTML = rows.map((r) => `<div class="q ${r.kind !== "observation" ? "text" : ""}"><b>${r.rank}</b>
    <span>${esc(r.ref)} <span class="muted">${esc(r.kind === "observation" ? r.label : r.kind)}</span></span>
    <div class="bar" style="width:${Math.max(6, (r.value_per_kb || 0) / top * 70)}px"></div></div>`).join("") || '<div class="muted">nothing waiting</div>';
  // memory + storage tiers
  for (const [name, el] of [["rover-1", "memRover"], ["scout-1", "memScout"]]) {
    const st = state.status[name];
    if (!st) continue;
    const t = st.storage.tiers, n = Object.values(t).reduce((a, c) => a + c, 0) || 1;
    const seg = (k, c) => `<i style="width:${((t[k] || 0) / n) * 100}%;background:${c}"></i>`;
    $(el).innerHTML = `<b class="${name === "scout-1" ? "s" : "g"}">${name}</b> ${st.memories} memories · ${st.storage.storage_mb}/${st.storage.budget_mb} MB · sent ${(st.bytes_sent.bytes / 1000).toFixed(1)} KB
      <div class="tier">${seg("full", "#3fbf7f")}${seg("thumb+full", "#8fcf6f")}${seg("thumb", "#f2a33a")}${seg("embedding", "#6f9cf0")}</div>`;
  }
  $("memBytes").innerHTML = '<span style="color:#3fbf7f">full</span> · <span style="color:#f2a33a">thumb</span> · <span style="color:#6f9cf0">embedding only</span>';
  // bubbles
  for (const [name, bot] of Object.entries(bots)) {
    const br = state.brain[name] || {}, bub = bubbles[name];
    bub.obj.position.copy(bot.headPos()).add(new THREE.Vector3(0, name === "scout-1" ? 2.5 : 2.2, 0));
    const last = (br.trace || []).slice(-1)[0];
    if (br.task || br.partial) {
      const step = last ? `<b>${esc(last.step)}</b> ${esc(typeof last.detail === "string" ? last.detail : last.detail.text || last.detail.query || last.detail.action || "")}` : "<b>waking up…</b>";
      bub.el.innerHTML = `${name === "scout-1" ? "SCOUT" : "ROVER"} BRAIN · ${esc(br.task?.ref || br.task?.kind || "")}<br>${step.slice(0, 260)}`;
      bub.until = performance.now() + 6000;
    }
    bub.obj.visible = performance.now() < bub.until && shot.mode !== "map";
  }
}

function mcPanel() {
  const m = state.metrics;
  $("mcStats").textContent = `${m.conclusions ?? 0} conclusions · ${m.pictures ?? 0} pictures · ${((m.downlink_bytes || 0) / 1000).toFixed(0)} KB`;
  const dHtml = state.disputes.slice(0, 2).map((d) => `<div class="dispute"><b>${d.status === "resolved" ? "RESOLVED" : "DISPUTE"}</b> · same place, contradiction p=${d.p.toFixed(2)} (Jev)
    <div class="vs">${["a", "b"].map((k) => `<div><b class="${d[k].node === "scout-1" ? "s" : "b"}">${esc(d[k].node)}</b> ${esc((d[k].text || "").slice(0, 110))}
      ${d.status !== "resolved" ? `<br><button onclick="window.sim.resolve(${d.id}, '${k}')">${esc(d[k].node)} is right</button>` : d.winner === d[k].point ? '<br><b style="color:#3fbf7f">upheld</b>' : '<br><b style="color:#e0565b">superseded</b>'}</div>`).join("")}</div></div>`).join("");
  const ans = state.feed.filter((p) => p.kind === "answer").slice(0, 1).map((p) => `<div class="answer"><div class="k">ANSWER FROM ${esc(p.node)}</div>${esc((p.text || "").slice(0, 220))}</div>`).join("");
  const concl = state.feed.filter((p) => p.kind !== "answer").slice(0, 6).map((p) => `<div class="concl ${p.node === "scout-1" ? "scout" : ""} ${p.superseded ? "sup" : ""}">
    <div class="k">${esc(p.kind.toUpperCase())} · ${esc(p.node)} · ${esc(p.ref || "")} · ${p.evidence?.length ? "picture " + (state.metrics.pictures ? "may follow" : "not yet down") : ""}</div>${esc((p.text || "").slice(0, 170))}</div>`).join("");
  $("mcList").innerHTML = dHtml + ans + (concl || '<div class="muted">No conclusions yet. They arrive before the pictures.</div>');
}

function feed(name, blob, tags, camLabel) {
  const el = $(name === "rover-1" ? "feedRover" : "feedScout");
  const url = URL.createObjectURL(blob);
  el.style.backgroundImage = `url(${url})`;
  $(name === "rover-1" ? "camRover" : "camScout").textContent = camLabel;
  $(name === "rover-1" ? "tagsRover" : "tagsScout").innerHTML = tags ? `<span class="muted">tags</span> ${tags.map(esc).join(" · ")}` : "";
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 400);
}

function shield(name) {
  const el = Object.assign(document.createElement("div"), { className: "shield", textContent: "REFUSED: forged command (bad signature)" });
  const o = new CSS2DObject(el);
  o.position.copy(bots[name].headPos()).add(new THREE.Vector3(0, 5, 0));
  scene.add(o);
  setTimeout(() => { scene.remove(o); el.remove(); }, 7000);
}

// ---------------------------------------------------------------- actions (used by free play and the director)
const nearTarget = (bot, r = 12) => TARGETS.find((t) => Math.hypot(t.x - bot.pos.x, t.z - bot.pos.z) < r);

async function captureNav(name) {
  const bot = bots[name];
  const blob = await renderFrame(renderer, scene, bot.cam, [mem.group, bot.model, orbiter.model]);
  const r = await capture(name, blob, { source: name === "scout-1" ? "aerial-nav" : "navcam", camera: "navcam", pos: bot.xz });
  feed(name, blob, r.tags, name === "scout-1" ? "SCOUT · NAV CAMERA" : "ROVER · NAVCAM");
  return r;
}

async function captureTarget(name, t, { zoom = 1, photo } = {}) {
  const bot = bots[name];
  const key = photo || (name === "scout-1" ? t.aerial : t.photo);
  if (!key) return captureNav(name);
  const blob = await photoFrame(key, zoom, name === "scout-1" ? 0.4 : 0);
  const cam = name === "scout-1" ? "aerial" : "mastcam";
  const r = await capture(name, blob, { source: t.key, camera: cam, pos: bot.xz });
  feed(name, blob, r.tags, `${name === "scout-1" ? "SCOUT · AERIAL" : "ROVER · MASTCAM"} · ${t.name.toUpperCase()}`);
  return r;
}

async function drive(name, x, z, { every = 16, alt = 14 } = {}) {
  const bot = bots[name];
  let last = bot.travel;
  const go = name === "scout-1" ? bot.flyTo(x, z, alt) : bot.goTo(x, z);
  let done = false;
  go.then(() => (done = true));
  while (!done) {
    await sleep(250);
    if (bot.travel - last >= every) { last = bot.travel; captureNav(name).catch(() => {}); }
    post(robot(name, "/api/pose"), { ...bot.xz, heading: bot.heading }).catch(() => {});
  }
  await post(robot(name, "/api/pose"), { ...bot.xz, heading: bot.heading }).catch(() => {});
}

async function orbiterPass(budget = 30000) {
  state.budget = budget;
  orbiter.start(26);
  await until(() => orbiter.overhead, { timeout: 20000, every: 100 });
  const before = { ...state.lastWindow };
  if (!state.blackout) {
    for (const name of Object.keys(ROBOTS)) post(robot(name, "/api/window_now"), { budget }).catch(() => {});
    await until(() => Object.keys(ROBOTS).every((n) => state.lastWindow[n] !== before[n]), { timeout: 20000, every: 300 });
  } else {
    for (const name of Object.keys(ROBOTS)) post(robot(name, "/api/window_now"), { budget }).catch(() => {});
  }
  await until(() => !orbiter.active, { timeout: 30000, every: 200 });
  pollGround();
}

async function brainIdle(name, timeout = 240000) {
  await sleep(1500);
  return until(async () => {
    const b = await get(robot(name, "/api/brain"));
    return !b.task && !b.queued && b.status === "idle" ? b : null;
  }, { timeout, every: 1500 });
}

async function recall(q) {
  const r = await get(robot("rover-1", `/api/search?q=${encodeURIComponent(q)}&k=5`));
  $("recallMs").textContent = `${r.ms} ms · Qdrant Edge ${r.qdrant_ms} ms`;
  const ids = [...r.insights.slice(0, 2), ...r.results.slice(0, 4)].map((x) => x.id);
  const pts = mem.highlight(ids);
  if (pts.length) setShot("orbit", { point: pts[0].clone(), radius: 30, height: 18, speed: 0.15 });
  return r;
}

async function ask(q) {
  const before = ((await get(robot("rover-1", "/api/brain"))).answers || [])[0]?.ts || 0;
  await post(robot("rover-1", "/api/ask"), { question: q });
  const a = await until(async () => {
    const b = await get(robot("rover-1", "/api/brain"));
    return b.answers?.[0] && b.answers[0].ts > before ? b.answers[0] : null;
  }, { timeout: 240000, every: 1000 });
  if (a) {
    $("mcList").insertAdjacentHTML("afterbegin", `<div class="answer"><div class="k">ONBOARD ANSWER · ROVER-1 (NO LINK NEEDED)</div><b>${esc(a.question)}</b><br>${esc(a.answer)}</div>`);
    const pts = mem.highlight(a.cited.map((c) => c.id));
    if (pts.length) setShot("orbit", { point: pts[0].clone(), radius: 38, height: 22, speed: 0.12 });
  }
  return a;
}

async function setBlackout(on) {
  state.blackout = on;
  await Promise.all(Object.keys(ROBOTS).map((n) => post(robot(n, "/api/link"), { up: !on })));
}

// the brain's decisions drive the rover in free play: "investigate" = go closer and take a close-up
function autonomy() {
  if (DEMO || state.busy.rover) return;
  const d = (state.brain["rover-1"]?.decisions || [])[0];
  if (!d || d.id === state.lastDecision) return;
  const first = state.lastDecision === null;
  state.lastDecision = d.id;
  if (first || d.action !== "investigate") return;
  const t = nearTarget(rover, 25);
  if (t) investigate(t);
}
async function investigate(t) {
  state.busy.rover = true;
  const dx = t.x - rover.pos.x, dz = t.z - rover.pos.z, d = Math.hypot(dx, dz);
  await drive("rover-1", t.x - (dx / d) * 4.5, t.z - (dz / d) * 4.5, { every: 99 });
  const r = await captureTarget("rover-1", t, t.close ? { photo: t.close } : { zoom: 1.8 });
  state.busy.rover = false;
  return r;
}

function caption(ch, title, sub = "", ms = 0) {
  $("capCh").textContent = ch; $("capT").textContent = title; $("capS").textContent = sub;
  $("caption").style.opacity = title ? 1 : 0;
  if (ms) setTimeout(() => { if ($("capT").textContent === title) $("caption").style.opacity = 0; }, ms);
}

window.sim = {
  rover, scout, orbiter, world, mem, state, setShot, caption, captureNav, captureTarget, drive, orbiterPass, brainIdle,
  recall, ask, setBlackout, investigate, nearTarget, TARGETS, pollGround,
  retask: (text) => post(`${GROUND}/api/intent`, { text }),
  resolve: async (id, winner) => { await post(`${GROUND}/api/disputes/${id}/resolve`, { winner }); pollGround(); },
  forge: (node = "rover-1") => post(`${GROUND}/api/redteam/forge`, { node }),
  timelapse: (on) => ($("ff").style.display = on ? "block" : "none"),
  endcard: async (on) => {
    const cat = await get("./assets/mars/catalog.json");
    $("endCredits").textContent = "Mars imagery: " + [...new Set(Object.values(cat).map((c) => c.credit))].join(" · ") +
      ` (${Object.values(cat).map((c) => c.nasa_id).join(", ")}). Everything ran live on one laptop: Qdrant Edge on each robot, Qdrant Server at ground, a local LLM brain.`;
    $("endcard").style.display = on ? "flex" : "none";
  },
};

// ---------------------------------------------------------------- free play
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
renderer.domElement.addEventListener("click", (e) => {
  if (DEMO && !state.freeplay) return;
  mouse.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObject(world.ground)[0];
  if (!hit) return;
  if (e.shiftKey) drive("scout-1", hit.point.x, hit.point.z).then(async () => { const t = nearTarget(scout, 15); if (t) captureTarget("scout-1", t); });
  else drive("rover-1", hit.point.x, hit.point.z).then(async () => { const t = nearTarget(rover); if (t) captureTarget("rover-1", t); });
});
addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  const k = e.key.toLowerCase();
  if (k === "c") { const t = nearTarget(rover); (t ? captureTarget("rover-1", t) : captureNav("rover-1")); }
  if (k === "p") orbiterPass();
  if (k === "1") setShot("chase", { of: rover });
  if (k === "2") setShot("chase", { of: scout });
  if (k === "3") setShot("map", { point: rover.pos.clone(), height: 220 });
  if (k === "4") setShot("pov", { of: rover });
  if (k === "5") setShot("orbit", { point: rover.pos.clone(), radius: 30, height: 14 });
  if (k === "6") setShot("free", { of: rover });
  if (k === "7") setShot("sky", { of: rover });
});
$("bPass").onclick = () => orbiterPass();
$("bBlackout").onclick = () => setBlackout(!state.blackout);
$("bCapture").onclick = () => { const t = nearTarget(rover); (t ? captureTarget("rover-1", t) : captureNav("rover-1")); };
$("bForge").onclick = async () => { const r = await window.sim.forge(); caption("RED TEAM", `Forged 'discard ${r.item.slice(0, 8)}' queued for rover-1`, "watch the rover refuse it at the next orbiter pass", 6000); };
$("recallForm").onsubmit = (e) => { e.preventDefault(); const q = $("recallQ").value.trim(); if (q) recall(q); else mem.clearHighlight(); };
$("askForm").onsubmit = (e) => { e.preventDefault(); const q = $("askQ").value.trim(); if (q) { $("askQ").value = ""; ask(q); } };
$("retaskForm").onsubmit = async (e) => { e.preventDefault(); const q = $("retaskQ").value.trim(); if (!q) return; $("retaskQ").value = ""; await window.sim.retask(q); caption("MISSION CONTROL", `Retask: "${q}"`, "queued at Earth · goes up signed at the next orbiter pass", 6000); };
$("toolbar").classList.remove("hidden");
if (DEMO) {
  for (const id of ["bPass", "bBlackout", "bCapture", "bForge"]) $(id).style.display = "none";
  $("bDemo").onclick = async () => {
    $("toolbar").classList.add("hidden");
    const { runDemo } = await import("./director.js");
    await runDemo(window.sim);
  };
  if (new URLSearchParams(location.search).has("autoplay")) setTimeout(() => $("bDemo").click(), 3000);
} else {
  $("bDemo").style.display = "none";
}

// ---------------------------------------------------------------- loop
let lastT = performance.now();
function loop(now) {
  const dt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;
  rover.update(dt); scout.update(dt); orbiter.update(dt); mem.tick(dt);
  world.tick(dt, shot.of?.pos || rover.pos);
  if (state.autoPasses && !orbiter.active && now > state.nextPass) { state.nextPass = now + 90000; orbiterPass(); }
  updateCamera(dt);
  renderer.render(scene, camera);
  labels.render(scene, camera);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
setInterval(poll, 1200);
setInterval(pollGround, 2500);
poll(); pollGround();
setShot("chase", { of: rover });
