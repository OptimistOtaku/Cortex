// The robots' real memories, drawn where they were made. Everything here mirrors /api/map on each robot:
//   pin      = an observation (colour = what the sync gate did with it; size = storage tier)
//   diamond  = a conclusion (insight / note) floating over its evidence, linked to the pictures it cites
import * as THREE from "three";
import { CSS2DObject } from "three/addons/CSS2DRenderer.js";
import { heightAt } from "./world.js";

export const COLORS = { new: 0xa89a90, queued: 0xf2a33a, held: 0x6f9cf0, sent: 0x3fbf7f, superseded: 0x8a5050 };
const TIER_SCALE = { full: 1, "thumb+full": 0.85, thumb: 0.65, embedding: 0.42 };
const ROBOT_ACCENT = { "rover-1": 0xffffff, "scout-1": 0xc79bf2 };

function label(text, cls = "lbl") {
  const el = document.createElement("div");
  el.className = cls;
  el.textContent = text;
  return new CSS2DObject(el);
}

export class MemoryViz {
  constructor(scene) {
    this.group = new THREE.Group();
    scene.add(this.group);
    this.items = new Map();  // id -> {obj, data}
    this.lines = new THREE.Group();
    this.group.add(this.lines);
    this.highlights = [];
    this.packets = [];
    this.stemGeo = new THREE.CylinderGeometry(0.06, 0.06, 1.8, 6);
    this.headGeo = new THREE.SphereGeometry(0.42, 12, 8);
    this.gemGeo = new THREE.OctahedronGeometry(1.0, 0);
  }

  colourOf(m) {
    if (m.superseded) return COLORS.superseded;
    if (m.state === "held") return COLORS.held;
    if (m.state && m.state !== "new") return COLORS.sent;
    return m.rank && m.rank <= 5 ? COLORS.queued : COLORS.new;
  }

  // lay pins made at the same spot around a small circle so they stay readable
  placeOf(m, node, seen) {
    const key = `${Math.round(m.pos.x / 3)},${Math.round(m.pos.z / 3)}`;
    const i = (seen[key] = (seen[key] ?? -1) + 1);
    const a = i * 2.4, r = i ? 1.2 + i * 0.35 : 0;
    const x = m.pos.x + Math.cos(a) * r + (node === "scout-1" ? 1.5 : 0), z = m.pos.z + Math.sin(a) * r;
    return new THREE.Vector3(x, heightAt(x, z), z);
  }

  update(node, memories) {
    const seen = {}, alive = new Set();
    const byId = {};
    for (const m of memories) {
      if (!m.pos) continue;
      alive.add(m.id);
      byId[m.id] = m;
      let it = this.items.get(m.id);
      if (!it) {
        it = { obj: m.kind === "observation" ? this.pin(node) : this.gem(node), data: m, born: performance.now() };
        this.items.set(m.id, it);
        this.group.add(it.obj);
      }
      it.data = m;
      it.node = node;
      const base = this.placeOf(m, node, seen);
      it.base = base;
      if (m.kind === "observation") {
        it.obj.position.copy(base);
        it.obj.children[1].material.color.setHex(this.colourOf(m));
        it.obj.scale.setScalar(this.scaleOf(it));
      } else {
        it.obj.scale.setScalar(this.scaleOf(it));
        it.obj.position.copy(base).add(new THREE.Vector3(0, 6.5, 0));
        const c = m.superseded ? 0x7a4a4a : node === "scout-1" ? 0xc79bf2 : 0x5cc8e6;
        it.obj.material.color.setHex(c);
        it.obj.material.emissive.setHex(m.superseded ? 0x200000 : c).multiplyScalar(0.5);
        it.obj.visible = m.kind === "insight" || m.kind === "note";
      }
    }
    for (const [id, it] of this.items) {
      if (it.node === node && !alive.has(id)) { this.group.remove(it.obj); this.items.delete(id); }
    }
    this.relink();
  }

  scaleOf(it) {
    const tier = it.data.kind === "observation" ? TIER_SCALE[it.data.tier] ?? 1 : 1;
    return tier * (it.hl ? 1.8 : 1);
  }

  pin(node) {
    const g = new THREE.Group();
    const stem = new THREE.Mesh(this.stemGeo, new THREE.MeshBasicMaterial({ color: ROBOT_ACCENT[node] }));
    stem.position.y = 0.9;
    const head = new THREE.Mesh(this.headGeo, new THREE.MeshBasicMaterial({ color: COLORS.new }));
    head.position.y = 1.9;
    g.add(stem, head);
    return g;
  }

  gem(node) {
    return new THREE.Mesh(this.gemGeo, new THREE.MeshStandardMaterial({ color: 0x5cc8e6, emissive: 0x2a6070, roughness: 0.3, metalness: 0.1 }));
  }

  // lines from each conclusion to the evidence pictures it cites
  relink() {
    this.lines.clear();
    for (const it of this.items.values()) {
      if (it.data.kind === "observation" || !it.obj.visible) continue;
      for (const e of it.data.evidence || []) {
        const ev = this.items.get(e);
        if (!ev) continue;
        const pts = [it.obj.position.clone(), ev.obj.position.clone().add(new THREE.Vector3(0, 2.1, 0))];
        const mat = new THREE.LineBasicMaterial({ color: it.data.superseded ? 0x7a4a4a : 0x5cc8e6, transparent: true, opacity: 0.75 });
        this.lines.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
      }
    }
  }

  // recall results: numbered labels over the pins, largest first
  highlight(ids) {
    this.clearHighlight();
    ids.forEach((id, i) => {
      const it = this.items.get(id);
      if (!it) return;
      const l = label(`#${i + 1} ${it.data.ref}`, "lbl rank");
      l.position.copy(it.obj.position).add(new THREE.Vector3(0, it.data.kind === "observation" ? 3.2 : 2, 0));
      this.group.add(l);
      it.hl = true;
      it.obj.scale.setScalar(this.scaleOf(it));
      this.highlights.push({ l, it });
    });
    return ids.map((id) => this.items.get(id)?.obj.position).filter(Boolean);
  }

  clearHighlight() {
    for (const { l, it } of this.highlights) { this.group.remove(l); l.element.remove(); it.hl = false; it.obj.scale.setScalar(this.scaleOf(it)); }
    this.highlights = [];
  }

  // one glowing packet per memory actually sent in a comms window, sized by its real bytes
  launch(fromPos, toObj, items) {
    items.forEach((it, i) => {
      const size = 0.5 + Math.sqrt(it.bytes / 1000) * 0.35;
      const colour = it.kind === "observation" ? 0xf2a33a : 0x5cc8e6;
      const m = new THREE.Mesh(new THREE.SphereGeometry(size, 10, 8), new THREE.MeshBasicMaterial({ color: colour, fog: false }));
      m.position.copy(fromPos);
      this.group.add(m);
      this.packets.push({ m, from: fromPos.clone(), to: toObj, t: -i * 0.18 });
    });
  }

  tick(dt) {
    const now = performance.now();
    for (const it of this.items.values()) {
      if (it.data.kind !== "observation" && it.obj.visible) { it.obj.rotation.y += dt * 1.2; }
      const age = (now - it.born) / 600;
      if (age < 1) it.obj.scale.setScalar(Math.max(0.05, age) * this.scaleOf(it));
    }
    this.packets = this.packets.filter((p) => {
      p.t += dt / 3.2;
      if (p.t < 0) return true;
      if (p.t >= 1) { this.group.remove(p.m); return false; }
      const to = p.to.position;
      p.m.position.lerpVectors(p.from, to, p.t);
      p.m.position.y += Math.sin(p.t * Math.PI) * 60;
      return true;
    });
  }

  hide(v) { this.group.visible = !v; }
}

export { label };
