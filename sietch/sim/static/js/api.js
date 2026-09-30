// Talking to the real SIETCH backend: ground control serves this page; each robot is its own HTTP server.
const host = location.hostname;
export const GROUND = location.origin;
export const ROBOTS = { "rover-1": `http://${host}:8101`, "scout-1": `http://${host}:8102` };

export async function get(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

export async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

// a camera frame goes into the robot's memory exactly like a real capture, tagged with where it was taken
export async function capture(robot, blob, { source, camera, pos }) {
  const r = await fetch(`${ROBOTS[robot]}/api/capture`, {
    method: "POST", body: blob,
    headers: { "Content-Type": "image/jpeg", "X-Source": source, "X-Camera": camera, "X-Pos": `${pos.x.toFixed(2)},${pos.z.toFixed(2)}` },
  });
  if (!r.ok) throw new Error(`capture ${r.status}`);
  return r.json();
}

export const robot = (name, path) => `${ROBOTS[name]}${path}`;
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function until(pred, { timeout = 120000, every = 1000 } = {}) {
  const t0 = performance.now();
  while (performance.now() - t0 < timeout) {
    try {
      const v = await pred();
      if (v) return v;
    } catch (e) { /* backend busy: try again */ }
    await sleep(every);
  }
  return null;
}
