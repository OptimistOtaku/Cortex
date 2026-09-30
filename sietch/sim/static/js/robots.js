// The robots and the relay orbiter: models, motion over the terrain, and their cameras.
// A camera frame is either a render of this 3D world (navigation, in transit) or, at a science target, the real
// NASA image of that target cropped to the robot's viewpoint. Either way it goes into the robot's real memory.
import * as THREE from "three";
import { heightAt } from "./world.js";

const M = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.2, ...extra });
const box = (w, h, d, mat) => new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
const cyl = (r1, r2, h, mat, seg = 16) => new THREE.Mesh(new THREE.CylinderGeometry(r1, r2, h, seg), mat);

function roverModel() {
  const g = new THREE.Group(), white = M(0xe8e2d6), dark = M(0x2a2a2e), gold = M(0xb58a2a, { metalness: 0.6 }), grey = M(0x77777c);
  const body = box(2.1, 0.55, 2.9, white); body.position.y = 1.15; g.add(body);
  const deck = box(2.0, 0.08, 2.7, gold); deck.position.y = 1.46; g.add(deck);
  const rtg = cyl(0.32, 0.36, 1.0, dark, 10); rtg.rotation.x = 1.1; rtg.position.set(0, 1.55, 1.75); g.add(rtg);
  for (let i = 0; i < 6; i++) { const fin = box(0.02, 0.5, 0.9, dark); fin.rotation.set(1.1, (i / 6) * Math.PI * 2, 0); fin.position.copy(rtg.position); g.add(fin); }
  const mast = cyl(0.07, 0.08, 1.7, white); mast.position.set(0.7, 2.3, -1.05); g.add(mast);
  const head = box(0.62, 0.32, 0.34, white); head.position.set(0.7, 3.2, -1.1); g.add(head);
  for (const dx of [-0.16, 0.16]) { const eye = cyl(0.07, 0.07, 0.1, dark); eye.rotation.x = Math.PI / 2; eye.position.set(0.7 + dx, 3.2, -1.3); g.add(eye); }
  const dish = cyl(0.42, 0.42, 0.05, grey, 20); dish.rotation.set(0.6, 0, 0.3); dish.position.set(-0.7, 1.9, 0.6); g.add(dish);
  const uhf = cyl(0.03, 0.03, 0.6, grey); uhf.position.set(-0.8, 1.8, -0.6); g.add(uhf);
  const arm = box(0.14, 0.14, 1.4, white); arm.position.set(-0.4, 1.0, -1.9); g.add(arm);
  const wheels = [];
  for (const x of [-1.25, 1.25]) {
    const rocker = box(0.1, 0.12, 2.6, grey); rocker.position.set(x, 0.75, 0); rocker.rotation.x = 0.08; g.add(rocker);
    for (const z of [-1.15, 0, 1.15]) {
      const w = cyl(0.36, 0.36, 0.34, dark, 14); w.rotation.z = Math.PI / 2; w.position.set(x, 0.36, z);
      g.add(w); wheels.push(w);
    }
  }
  g.userData.wheels = wheels;
  return g;
}

function scoutModel() {
  const g = new THREE.Group(), body = M(0xcfc9bf), dark = M(0x333337), blade = M(0x2d2a2a);
  const b = box(0.55, 0.45, 0.55, body); b.position.y = 0.9; g.add(b);
  for (const [x, z] of [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]) {
    const leg = cyl(0.03, 0.03, 1.1, dark); leg.position.set(x * 0.8, 0.45, z * 0.8); leg.rotation.set(z * 0.5, 0, -x * 0.5); g.add(leg);
  }
  const mast = cyl(0.05, 0.05, 1.2, dark); mast.position.y = 1.7; g.add(mast);
  const rotors = [];
  for (const y of [1.55, 1.9]) { const r = box(3.0, 0.03, 0.18, blade); r.position.y = y; g.add(r); rotors.push(r); }
  const panel = box(0.9, 0.03, 0.55, M(0x1c2b55, { metalness: 0.5 })); panel.position.y = 2.35; g.add(panel);
  g.userData.rotors = rotors;
  g.scale.setScalar(1.6);
  return g;
}

function orbiterModel() {
  const g = new THREE.Group(), gold = M(0xc9a13b, { metalness: 0.8, emissive: 0x332200 }), panel = M(0x1c2b55, { emissive: 0x0a1030 });
  g.add(box(4, 3, 3, gold));
  for (const x of [-9, 9]) { const p = box(12, 0.2, 4, panel); p.position.x = x; g.add(p); }
  const dish = cyl(2.2, 2.2, 0.3, M(0xdddddd), 20); dish.position.y = -2.2; g.add(dish);
  g.scale.setScalar(11);
  return g;
}

// ---------------------------------------------------------------- cameras
const rt = new THREE.WebGLRenderTarget(384, 288);
const pixels = new Uint8Array(384 * 288 * 4);
const canvas = Object.assign(document.createElement("canvas"), { width: 384, height: 288 });

export function renderFrame(renderer, scene, cam, hide = []) {
  hide.forEach((o) => (o.visible = false));
  renderer.setRenderTarget(rt);
  renderer.render(scene, cam);
  renderer.readRenderTargetPixels(rt, 0, 0, 384, 288, pixels);
  renderer.setRenderTarget(null);
  hide.forEach((o) => (o.visible = true));
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(384, 288);
  for (let y = 0; y < 288; y++) img.data.set(pixels.subarray((287 - y) * 384 * 4, (288 - y) * 384 * 4), y * 384 * 4);
  ctx.putImageData(img, 0, 0);
  return new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.9));
}

const photos = {};
function loadPhoto(key) {
  return (photos[key] ||= new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = `./assets/mars/${key}.jpg`;
  }));
}

// the real image of a target, cropped as this viewpoint would see it (zoom > 1 = closer); mosaics' black edges cut
export async function photoFrame(key, zoom = 1, jitter = 0) {
  const im = await loadPhoto(key);
  const keep = 0.8 / zoom, w = im.width * keep, h = im.height * keep;
  const cx = im.width / 2 + jitter * im.width * 0.05, cy = im.height / 2;
  const c = Object.assign(document.createElement("canvas"), { width: Math.round(Math.min(768, w)), height: Math.round(Math.min(768, w) * h / w) });
  c.getContext("2d").drawImage(im, cx - w / 2, cy - h / 2, w, h, 0, 0, c.width, c.height);
  return new Promise((r) => c.toBlob(r, "image/jpeg", 0.9));
}

// ---------------------------------------------------------------- motion
class Mover {
  constructor(name, model, speed) {
    this.name = name; this.model = model; this.speed = speed;
    this.pos = new THREE.Vector3(); this.heading = 0; this.target = null; this._done = null; this.travel = 0;
  }
  get xz() { return { x: this.pos.x, z: this.pos.z }; }
  goTo(x, z) {
    this.target = new THREE.Vector2(x, z);
    return new Promise((r) => (this._done = r));
  }
  stop() { this.target = null; this._done?.(); this._done = null; }
  step(dt) {
    if (!this.target) return 0;
    const dx = this.target.x - this.pos.x, dz = this.target.y - this.pos.z, d = Math.hypot(dx, dz);
    if (d < 0.6) { this.stop(); return 0; }
    const want = Math.atan2(-dx, -dz);
    let dh = ((want - this.heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
    this.heading += Math.max(-dt * 1.6, Math.min(dt * 1.6, dh));
    const mv = Math.min(d, this.speed * dt * (Math.abs(dh) > 0.8 ? 0.25 : 1));
    this.pos.x -= Math.sin(this.heading) * mv;
    this.pos.z -= Math.cos(this.heading) * mv;
    this.travel += mv;
    return mv;
  }
}

export class Rover extends Mover {
  constructor(scene) {
    super("rover-1", roverModel(), 5.5);
    scene.add(this.model);
    this.cam = new THREE.PerspectiveCamera(62, 384 / 288, 0.3, 2500);
  }
  place(x, z, heading = 0) { this.pos.set(x, 0, z); this.heading = heading; this.update(0); }
  update(dt) {
    const mv = this.step(dt);
    const y = heightAt(this.pos.x, this.pos.z);
    this.pos.y = y;
    this.model.position.set(this.pos.x, y, this.pos.z);
    // tilt with the ground under the wheels
    const f = 1.4, fx = -Math.sin(this.heading), fz = -Math.cos(this.heading);
    const pitch = Math.atan2(heightAt(this.pos.x + fx * f, this.pos.z + fz * f) - heightAt(this.pos.x - fx * f, this.pos.z - fz * f), 2 * f);
    const roll = Math.atan2(heightAt(this.pos.x - fz * f, this.pos.z + fx * f) - heightAt(this.pos.x + fz * f, this.pos.z - fx * f), 2 * f);
    this.model.rotation.set(0, 0, 0);
    this.model.rotateY(this.heading); this.model.rotateX(pitch); this.model.rotateZ(roll);
    this.model.userData.wheels.forEach((w) => (w.rotation.x += mv / 0.36));
    // navcam on the mast, looking ahead and slightly down
    this.model.updateMatrixWorld();
    this.cam.position.copy(new THREE.Vector3(0.7, 3.25, -1.2).applyMatrix4(this.model.matrixWorld));
    this.cam.rotation.set(0, 0, 0, "YXZ");
    this.cam.rotation.y = this.heading; this.cam.rotation.x = -0.22;
  }
  headPos() { return new THREE.Vector3(0.7, 3.4, -1.1).applyMatrix4(this.model.matrixWorld); }
}

export class Scout extends Mover {
  constructor(scene) {
    super("scout-1", scoutModel(), 9);
    scene.add(this.model);
    this.alt = 0; this.wantAlt = 0;
    this.cam = new THREE.PerspectiveCamera(70, 384 / 288, 0.3, 2500);
    this.shadow = new THREE.Mesh(new THREE.CircleGeometry(2.2, 20), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35, depthWrite: false }));
    this.shadow.rotation.x = -Math.PI / 2;
    scene.add(this.shadow);
  }
  place(x, z) { this.pos.set(x, 0, z); this.update(0); }
  fly(alt) { this.wantAlt = alt; }
  async flyTo(x, z, alt = 14) { this.wantAlt = alt; await this.goTo(x, z); }
  update(dt) {
    if (this.alt > 1 || this.wantAlt > 0) this.step(dt);
    this.alt += Math.max(-dt * 4, Math.min(dt * 4, this.wantAlt - this.alt));
    const g = heightAt(this.pos.x, this.pos.z);
    const bob = this.alt > 1 ? Math.sin(performance.now() / 400) * 0.15 : 0;
    this.pos.y = g + this.alt + bob;
    this.model.position.set(this.pos.x, this.pos.y, this.pos.z);
    this.model.rotation.set(0, this.heading, 0);
    const spin = this.alt > 0.2 || this.wantAlt > 0 ? dt * 45 : 0;
    this.model.userData.rotors[0].rotation.y += spin; this.model.userData.rotors[1].rotation.y -= spin;
    this.shadow.position.set(this.pos.x, g + 0.08, this.pos.z);
    this.shadow.visible = this.alt > 0.5;
    this.cam.position.set(this.pos.x, this.pos.y + 0.5, this.pos.z);
    this.cam.rotation.set(0, 0, 0, "YXZ");
    this.cam.rotation.y = this.heading; this.cam.rotation.x = -1.0;  // looking down and ahead
  }
  headPos() { return this.pos.clone().add(new THREE.Vector3(0, 4.2, 0)); }
}

// The relay orbiter crosses the sky; while it is above the horizon the robots have a link.
export class Orbiter {
  constructor(scene) {
    this.model = orbiterModel();
    this.model.visible = false;
    scene.add(this.model);
    this.t = -1; this.duration = 26;
  }
  start(duration = 26) { this.t = 0; this.duration = duration; this.model.visible = true; }
  get active() { return this.t >= 0 && this.t <= 1; }
  get overhead() { return this.t > 0.18 && this.t < 0.82; }  // above the local horizon: link possible
  update(dt) {
    if (this.t < 0) return;
    this.t += dt / this.duration;
    if (this.t > 1) { this.t = -1; this.model.visible = false; return; }
    const a = THREE.MathUtils.lerp(-0.25, Math.PI + 0.25, this.t);  // east horizon -> west horizon
    this.model.position.set(Math.cos(a) * 1100, Math.sin(a) * 520 + 60, 900);  // southern sky, away from the sun
    this.model.rotation.set(0.3, a, 0);
  }
}
