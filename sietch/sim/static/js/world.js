// A Jezero-like patch of Mars: crater floor, a layered delta scarp, a dry channel, dunes, craters, the crater rim
// on the horizon, and the science targets whose real NASA imagery the robots' cameras see.
import * as THREE from "three";

// ---------------------------------------------------------------- noise
const P = new Uint8Array(512);
{
  const p = [...Array(256).keys()];
  let s = 1337;
  for (let i = 255; i > 0; i--) { s = (s * 16807) % 2147483647; const j = s % (i + 1); [p[i], p[j]] = [p[j], p[i]]; }
  for (let i = 0; i < 512; i++) P[i] = p[i & 255];
}
const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
function grad(h, x, y) { const u = h & 1 ? x : -x, v = h & 2 ? y : -y; return (h & 4 ? u + v : u - v) * 0.7; }
export function noise(x, y) {
  const X = Math.floor(x) & 255, Y = Math.floor(y) & 255;
  x -= Math.floor(x); y -= Math.floor(y);
  const u = fade(x), v = fade(y), a = P[X] + Y, b = P[X + 1] + Y;
  return THREE.MathUtils.lerp(
    THREE.MathUtils.lerp(grad(P[a], x, y), grad(P[b], x - 1, y), u),
    THREE.MathUtils.lerp(grad(P[a + 1], x, y - 1), grad(P[b + 1], x - 1, y - 1), u), v);
}
function fbm(x, y, oct) { let s = 0, a = 1, f = 1, n = 0; for (let i = 0; i < oct; i++) { s += a * noise(x * f, y * f); n += a; a *= 0.5; f *= 2; } return s / n; }
const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

// ---------------------------------------------------------------- geography (metres; x east, z south)
export const SIZE = 1000;
const DELTA = { x: -130, z: -235, r: 105 };
const CHANNEL = [[-110, -150], [-70, -95], [-20, -70], [20, -20], [40, 40], [90, 110], [140, 190]];
const DUNES = { x: 175, z: 40, r: 85 };
const CRATERS = [{ x: 60, z: 70, r: 16, d: 3.5 }, { x: -75, z: 35, r: 9, d: 2 }, { x: 215, z: -150, r: 28, d: 5 }, { x: -220, z: -60, r: 20, d: 4 }];

export const TARGETS = [
  { key: "delta_scarp", name: "Delta scarp", x: -95, z: -168, photo: "delta_scarp", close: "kodiak", look: [-130, -235] },
  { key: "streambed", name: "Ancient streambed", x: -48, z: -108, photo: "streambed_pebbles" },
  { key: "mud_cracks", name: "Cracked slab", x: 38, z: -150, photo: "mud_cracks_close", aerial: "aerial_ripples" },
  { key: "veins", name: "Veined outcrop", x: 112, z: -88, photo: "veins" },
  { key: "meteorite", name: "Metallic rock", x: 70, z: -30, photo: "meteorite" },
  { key: "leopard", name: "Spotted rock", x: -62, z: -250, photo: "leopard_spots" },
  { key: "spherules", name: "Berry field", x: 150, z: -190, photo: "spherules" },
  { key: "dunes", name: "Dark dunes", x: 150, z: 30, photo: "dune" },
  { key: "backshell", name: "Landing debris", x: -165, z: 75, aerial: "aerial_backshell" },
];

function channelDist(x, z) {
  let best = 1e9;
  for (let i = 0; i < CHANNEL.length - 1; i++) {
    const [ax, az] = CHANNEL[i], [bx, bz] = CHANNEL[i + 1];
    const dx = bx - ax, dz = bz - az, t = Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz)));
    best = Math.min(best, Math.hypot(x - ax - t * dx, z - az - t * dz));
  }
  return best;
}

function deltaMask(x, z) {
  const d = Math.hypot(x - DELTA.x, z - DELTA.z) + fbm(x * 0.02, z * 0.02, 3) * 40;
  return 1 - smooth(DELTA.r - 12, DELTA.r + 6, d);
}

export function heightAt(x, z) {
  let h = fbm(x * 0.004 + 3.1, z * 0.004 - 1.7, 5) * 12 + fbm(x * 0.035, z * 0.035, 3) * 1.1;
  const m = deltaMask(x, z);  // the delta: a plateau of stacked river sediments, cut into terraces (the strata)
  if (m > 0) { const top = 20 * m; h += Math.floor(top / 3.2) * 3.2 + smooth(0.55, 1, (top / 3.2) % 1) * 3.2; }
  h -= 2.6 * (1 - smooth(4, 13, channelDist(x, z)));  // the dry riverbed that fed the lake
  const dd = Math.hypot(x - DUNES.x, z - DUNES.z);
  if (dd < DUNES.r) { const w = Math.sin((x * 0.8 + z * 0.6) * 0.13 + noise(x * 0.02, z * 0.02) * 2); h += (w * w) * 3.2 * (1 - smooth(DUNES.r * 0.6, DUNES.r, dd)); }
  for (const c of CRATERS) {
    const r = Math.hypot(x - c.x, z - c.z) / c.r;
    if (r < 1.6) h += r < 1 ? -c.d * (1 - r * r) + c.d * 0.35 * r * r * r : c.d * 0.35 * Math.max(0, 1 - (r - 1) / 0.6) ** 2;
  }
  const edge = Math.max(Math.abs(x), Math.abs(z));  // rise toward the crater rim beyond the map edge
  h += smooth(380, 520, edge) * 45;
  return h;
}

function colourAt(x, z, h) {
  const n = fbm(x * 0.05, z * 0.05, 3), n2 = noise(x * 0.3, z * 0.3);
  let c = new THREE.Color(0.60 + n * 0.08, 0.34 + n * 0.05, 0.20 + n * 0.03);  // oxidised regolith
  const m = deltaMask(x, z);
  if (m > 0.05) {  // alternating light and dark strata on the delta
    const band = Math.sin(h * 1.9) > 0 ? new THREE.Color(0.80, 0.62, 0.46) : new THREE.Color(0.52, 0.34, 0.24);
    c.lerp(band, Math.min(1, m * 1.4) * 0.8);
  }
  const ch = 1 - smooth(3, 12, channelDist(x, z));
  c.lerp(new THREE.Color(0.74, 0.55, 0.40), ch * 0.6);  // pale channel floor
  const dd = Math.hypot(x - DUNES.x, z - DUNES.z);
  c.lerp(new THREE.Color(0.25, 0.18, 0.15), (1 - smooth(DUNES.r * 0.5, DUNES.r, dd)) * 0.85);  // dark basaltic sand
  c.multiplyScalar(0.92 + n2 * 0.1);
  return c;
}

// ---------------------------------------------------------------- scene pieces
function terrain() {
  const seg = 220;
  const g = new THREE.PlaneGeometry(SIZE, SIZE, seg, seg);
  g.rotateX(-Math.PI / 2);
  const pos = g.attributes.position, col = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), z = pos.getZ(i), h = heightAt(x, z);
    pos.setY(i, h);
    colourAt(x, z, h).toArray(col, i * 3);
  }
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0 }));
  mesh.name = "terrain";
  return mesh;
}

function rim() {  // the crater wall on the horizon, 1.2-1.8 km away
  const g = new THREE.CylinderGeometry(1500, 1500, 1, 180, 12, true);
  const pos = g.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), z = pos.getZ(i), y = pos.getY(i) + 0.5, a = Math.atan2(z, x);
    const hgt = 120 + fbm(Math.cos(a) * 3 + 10, Math.sin(a) * 3, 5) * 160;
    const r = 1500 - y * 250;
    pos.setXYZ(i, Math.cos(a) * r, y * hgt + 30, Math.sin(a) * r);
  }
  g.computeVertexNormals();
  return new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0x8f5a3c, roughness: 1, side: THREE.DoubleSide }));
}

function rocks() {
  const n = 1800, geo = new THREE.DodecahedronGeometry(1, 0);
  const mesh = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ color: 0x6e4633, roughness: 0.9, flatShading: true }), n);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), s = new THREE.Vector3();
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < n; i++) {
    const x = (rnd() - 0.5) * 760, z = (rnd() - 0.5) * 760, k = rnd() ** 3 * 1.6 + 0.12;
    if (TARGETS.some((t) => Math.hypot(t.x - x, t.z - z) < 10) || Math.hypot(x, z - 20) < 8) { m.makeScale(0, 0, 0); mesh.setMatrixAt(i, m); continue; }
    e.set(rnd() * 3, rnd() * 3, rnd() * 3); q.setFromEuler(e); s.set(k * (0.8 + rnd() * 0.6), k * 0.6, k);
    m.compose(new THREE.Vector3(x, heightAt(x, z) + k * 0.25, z), q, s);
    mesh.setMatrixAt(i, m);
    mesh.setColorAt(i, new THREE.Color().setHSL(0.05, 0.35 + rnd() * 0.2, 0.18 + rnd() * 0.14));
  }
  return mesh;
}

// each science target is an outcrop wearing its own real NASA photo, so the rover visibly drives up to "that rock"
function targetMeshes(loader) {
  const group = new THREE.Group();
  for (const t of TARGETS) {
    const tex = loader.load(`./assets/mars/${t.photo || t.aerial}.jpg`);
    tex.colorSpace = THREE.SRGBColorSpace;
    const slab = new THREE.Mesh(new THREE.CylinderGeometry(3.2, 3.8, 0.5, 9), [
      new THREE.MeshStandardMaterial({ color: 0x7a4c36, roughness: 0.9, flatShading: true }),
      new THREE.MeshStandardMaterial({ map: tex, roughness: 0.85 }),
      new THREE.MeshStandardMaterial({ color: 0x5e3a2a }),
    ]);
    slab.position.set(t.x, heightAt(t.x, t.z) + 0.05, t.z);
    slab.rotation.y = (t.x * 13 + t.z * 7) % 6;
    slab.userData.target = t;
    group.add(slab);
    t.mesh = slab;
  }
  return group;
}

function sky() {
  const u = { uSun: { value: new THREE.Vector3(0.4, 0.5, -0.7).normalize() }, uSunset: { value: 0 } };
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, uniforms: u,
    vertexShader: "varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.); }",
    fragmentShader: `varying vec3 vDir; uniform vec3 uSun; uniform float uSunset;
      void main(){
        float up = clamp(vDir.y, 0., 1.);
        vec3 day = mix(vec3(.86,.62,.42), vec3(.47,.33,.27), pow(up, .55));
        vec3 dusk = mix(vec3(.55,.38,.33), vec3(.12,.10,.14), pow(up, .5));
        vec3 col = mix(day, dusk, uSunset);
        float s = max(dot(normalize(vDir), normalize(uSun)), 0.);
        col += mix(vec3(1.,.86,.62), vec3(.45,.65,1.), uSunset) * (pow(s, 24.) * .55 + pow(s, 600.) * 1.6);
        gl_FragColor = vec4(col, 1.);
      }`,
  });
  const m = new THREE.Mesh(new THREE.SphereGeometry(4000, 32, 16), mat);
  m.userData.uniforms = u;
  return m;
}

function dust() {
  const n = 900, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { pos[i * 3] = (Math.random() - 0.5) * 400; pos[i * 3 + 1] = Math.random() * 40; pos[i * 3 + 2] = (Math.random() - 0.5) * 400; }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({ color: 0xe8b98f, size: 0.35, transparent: true, opacity: 0.45, depthWrite: false }));
}

export function buildWorld(scene) {
  const loader = new THREE.TextureLoader();
  scene.fog = new THREE.Fog(0xc99470, 260, 1700);
  const skyMesh = sky();
  scene.add(skyMesh);
  const sun = new THREE.DirectionalLight(0xffe2c4, 2.4);
  sun.position.set(400, 500, -700);
  scene.add(sun, new THREE.HemisphereLight(0xe0b48f, 0x4a2c20, 0.9));
  const ground = terrain();
  const dustPts = dust();
  scene.add(ground, rim(), rocks(), targetMeshes(loader), dustPts);
  const earth = new THREE.Mesh(new THREE.SphereGeometry(6, 12, 8), new THREE.MeshBasicMaterial({ color: 0xbfe0ff, fog: false }));
  earth.position.set(-1400, 1100, -2600);
  scene.add(earth);
  return {
    ground, sun, sky: skyMesh, earth, dust: dustPts,
    setSunset(k) {  // 0 = afternoon, 1 = Martian (blue) sunset
      skyMesh.userData.uniforms.uSunset.value = k;
      const dir = new THREE.Vector3(0.4, 0.5 - 0.46 * k, -0.7).normalize();
      skyMesh.userData.uniforms.uSun.value.copy(dir);
      sun.position.copy(dir.clone().multiplyScalar(900));
      sun.intensity = 2.4 - 1.5 * k;
      scene.fog.color.setRGB(0.79 - 0.4 * k, 0.58 - 0.3 * k, 0.44 - 0.2 * k);
    },
    tick(dt, focus) {
      const p = dustPts.geometry.attributes.position;
      for (let i = 0; i < p.count; i++) {
        let x = p.getX(i) + dt * 1.6, y = p.getY(i) + Math.sin(i + performance.now() / 900) * dt * 0.2;
        if (x > 200) x -= 400;
        p.setXY(i, x, y);
      }
      p.needsUpdate = true;
      dustPts.position.set(focus.x, heightAt(focus.x, focus.z), focus.z);
    },
  };
}
