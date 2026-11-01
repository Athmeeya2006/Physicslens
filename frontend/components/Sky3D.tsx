"use client";

import { useEffect, useImperativeHandle, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import type { SkyBody } from "@/lib/sky";

// The Solar System Lab in 3D: the Sun as the light source, each planet a lit sphere with its own colouring (painted
// here, procedurally: nothing is downloaded), its orbit traced in space, labels that follow it. Drag to turn, scroll or
// pinch to zoom, right-drag to pan. Positions are the simulation's (AU, in the target's orbital plane, with the
// plane's normal as up); sizes are not to scale, or every planet would be smaller than a pixel.

export interface Sky3DHandle {
  setTime: (days: number) => void;
  fit: () => void;
}

interface Props {
  base: SkyBody[] | null;
  modified: SkyBody[] | null;
  playing: boolean;
  daysPerSecond: number;
  follow: string | null;
  centre: string;  // "Sun" or "Jupiter"
  onTime: (days: number, end: number) => void;
  label: string;
  handle: React.Ref<Sky3DHandle>;
}

// ---------- appearance ----------

function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function paint(name: string): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = 512; c.height = 256;
  const g = c.getContext("2d")!;
  const r = rng([...name].reduce((a, ch) => a * 31 + ch.charCodeAt(0), 7));
  const bands = (colours: string[], wobble = 6) => {
    const h = c.height / colours.length;
    colours.forEach((col, i) => {
      g.fillStyle = col;
      g.beginPath();
      g.moveTo(0, i * h);
      for (let x = 0; x <= c.width; x += 16) g.lineTo(x, i * h + Math.sin(x / 40 + i) * wobble * r());
      g.lineTo(c.width, (i + 1) * h + 2);
      g.lineTo(0, (i + 1) * h + 2);
      g.fill();
    });
  };
  const blobs = (colour: string, n: number, size: number, alpha = 1) => {
    g.fillStyle = colour;
    g.globalAlpha = alpha;
    for (let i = 0; i < n; i++) {
      const x = r() * c.width, y = r() * c.height, s = size * (0.4 + r());
      g.beginPath(); g.ellipse(x, y, s * (1 + r()), s, r() * Math.PI, 0, 2 * Math.PI); g.fill();
    }
    g.globalAlpha = 1;
  };
  const craters = (base: string, dark: string, n: number) => {
    g.fillStyle = base; g.fillRect(0, 0, c.width, c.height);
    blobs(dark, n, 6, 0.35);
    blobs("#ffffff", n / 3, 3, 0.12);
  };
  switch (name) {
    case "Earth": {
      const sea = g.createLinearGradient(0, 0, 0, c.height);
      sea.addColorStop(0, "#1f3f7a"); sea.addColorStop(0.5, "#2a62b8"); sea.addColorStop(1, "#1f3f7a");
      g.fillStyle = sea; g.fillRect(0, 0, c.width, c.height);
      blobs("#3c7a3a", 14, 26); blobs("#8a7a4a", 8, 14, 0.8); blobs("#3c7a3a", 30, 9);
      g.fillStyle = "#f2f6fa"; g.fillRect(0, 0, c.width, 14); g.fillRect(0, c.height - 16, c.width, 16);
      blobs("#ffffff", 40, 10, 0.55);
      break;
    }
    case "Venus": bands(["#e8d6a8", "#f1e2b8", "#e2cc98", "#efdcae", "#e6d2a2", "#f3e6c2"], 10); blobs("#fff4d6", 30, 18, 0.25); break;
    case "Jupiter":
      bands(["#e9dcc6", "#c9a27a", "#efe4cf", "#b5835a", "#f2e8d5", "#a8744e", "#ede1c9", "#c99c72", "#e6d8bf"], 5);
      g.fillStyle = "#b5523a"; g.beginPath(); g.ellipse(330, 165, 34, 17, 0, 0, 2 * Math.PI); g.fill();
      g.fillStyle = "#d77a5c"; g.beginPath(); g.ellipse(330, 165, 22, 10, 0, 0, 2 * Math.PI); g.fill();
      break;
    case "Saturn": bands(["#f0e2b6", "#dcc690", "#f3e7c4", "#d4bb84", "#efdfb3", "#e2cd9c"], 3); break;
    case "Uranus": bands(["#a6e1e8", "#9cd9e2", "#aee5ec", "#96d3dd"], 2); break;
    case "Neptune": bands(["#3b5fd1", "#3457c7", "#4468d8", "#2f4fba"], 3); blobs("#1c2f7a", 3, 14, 0.6); break;
    case "Mars":
      g.fillStyle = "#b9542f"; g.fillRect(0, 0, c.width, c.height);
      blobs("#7a3520", 18, 22, 0.6); blobs("#d07a4a", 20, 12, 0.4);
      g.fillStyle = "#f4efe8"; g.fillRect(0, 0, c.width, 10);
      break;
    case "Io": g.fillStyle = "#e8d25a"; g.fillRect(0, 0, c.width, c.height); blobs("#b8742a", 30, 8, 0.7); blobs("#fff3a8", 20, 6, 0.6); break;
    case "Europa": g.fillStyle = "#e9e2d2"; g.fillRect(0, 0, c.width, c.height);
      g.strokeStyle = "#9b6a46"; g.globalAlpha = 0.5;
      for (let i = 0; i < 40; i++) { g.beginPath(); g.moveTo(r() * 512, r() * 256); g.lineTo(r() * 512, r() * 256); g.stroke(); }
      g.globalAlpha = 1; break;
    case "Ganymede": craters("#8f8576", "#4e4740", 60); break;
    case "Callisto": craters("#5e554b", "#2e2924", 90); break;
    case "Halley": craters("#4a4642", "#26231f", 30); break;
    case "Apophis": craters("#7a7066", "#3e3832", 30); break;
    default: craters(name === "Mercury" ? "#9b9790" : name === "Moon" ? "#a9a9a6" : name === "Pluto" ? "#c9b79c" : "#8a8580",
                     "#55524c", 70);
  }
  return c;
}

const RADIUS_KM: Record<string, number> = {
  Sun: 695700, Mercury: 2440, Venus: 6052, Earth: 6371, Moon: 1737, Mars: 3390, Jupiter: 69911, Saturn: 58232,
  Uranus: 25362, Neptune: 24622, Pluto: 1188, Ceres: 470, Io: 1822, Europa: 1561, Ganymede: 2634, Callisto: 2410,
};

function glowTexture(inner: string): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, inner); grad.addColorStop(0.25, "rgba(255,200,80,0.5)"); grad.addColorStop(1, "rgba(255,160,40,0)");
  g.fillStyle = grad; g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

function ringTexture(): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 4;
  const g = c.getContext("2d")!;
  for (let x = 0; x < 256; x++) {
    const a = x < 40 ? 0 : x > 240 ? 0 : 0.35 + 0.5 * Math.abs(Math.sin(x / 9)) * (x > 150 && x < 165 ? 0.1 : 1);
    g.fillStyle = `rgba(226,206,160,${a.toFixed(2)})`;
    g.fillRect(x, 0, 1, 4);
  }
  const t = new THREE.CanvasTexture(c);
  return t;
}

// AU in the plane (x, y) with the normal z as up, as three.js likes y up.
const toScene = (p: number[]) => new THREE.Vector3(p[1], p[3] ?? 0, -p[2]);

function at(points: number[][], t: number): THREE.Vector3 | null {
  if (!points.length) return null;
  let lo = 0, hi = points.length - 1;
  if (t <= points[0][0]) return toScene(points[0]);
  if (t >= points[hi][0]) return toScene(points[hi]);
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid][0] <= t) lo = mid; else hi = mid;
  }
  const f = (t - points[lo][0]) / (points[hi][0] - points[lo][0] || 1);
  return toScene(points[lo]).lerp(toScene(points[hi]), f);
}

// ---------- the asteroid swarm ----------

// Thousands of small bodies on Kepler orbits around the Sun, drawn as glowing dots and moved on the GPU: near-Earth
// objects (orange) and the main belt (blue), as in NASA's charts of known asteroids. Their elements are drawn at
// random from those populations' typical ranges (illustration, not a catalogue); each moves on its own fixed ellipse.
const SWARM_VERTEX = `
  attribute vec4 elems;   // a (AU), e, mean motion n (rad/day), mean anomaly at t = 0
  attribute vec3 angles;  // inclination, node, argument of perihelion
  attribute vec3 colour;
  uniform float t;
  uniform float pxScale;
  varying vec3 vColour;
  void main() {
    float a = elems.x, e = elems.y;
    float M = elems.w + elems.z * t;
    float E = M;
    for (int k = 0; k < 6; k++) E = E - (E - e * sin(E) - M) / (1.0 - e * cos(E));
    vec2 pf = vec2(a * (cos(E) - e), a * sqrt(1.0 - e * e) * sin(E));
    float ci = cos(angles.x), si = sin(angles.x), cO = cos(angles.y), sO = sin(angles.y), cw = cos(angles.z), sw = sin(angles.z);
    vec3 r = vec3((cO * cw - sO * sw * ci) * pf.x + (-cO * sw - sO * cw * ci) * pf.y,
                  (sO * cw + cO * sw * ci) * pf.x + (-sO * sw + cO * cw * ci) * pf.y,
                  (sw * si) * pf.x + (cw * si) * pf.y);
    vec4 mv = modelViewMatrix * vec4(r.x, r.z, -r.y, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = clamp(pxScale / -mv.z, 1.5, 4.0);
    vColour = colour;
  }`;
const SWARM_FRAGMENT = `
  varying vec3 vColour;
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    float a = smoothstep(0.5, 0.0, length(d));
    gl_FragColor = vec4(vColour, a);
  }`;

function swarm(count: number, gmSun: number): THREE.Points {
  const r = rng(2026);
  const elems: number[] = [], angles: number[] = [], colour: number[] = [];
  for (let i = 0; i < count; i++) {
    const near = r() < 0.45;
    const a = near ? 0.8 + r() * 1.4 : 2.1 + r() * 1.2;
    const e = near ? 0.1 + r() * 0.5 : r() * 0.2;
    elems.push(a, e, Math.sqrt(gmSun / a ** 3), r() * 2 * Math.PI);
    angles.push((near ? 25 : 12) * Math.PI / 180 * r() * r(), r() * 2 * Math.PI, r() * 2 * Math.PI);
    colour.push(...(near ? [1.0, 0.55, 0.15] : [0.25, 0.65, 1.0]));
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(new Array(count * 3).fill(0), 3));
  g.setAttribute("elems", new THREE.Float32BufferAttribute(elems, 4));
  g.setAttribute("angles", new THREE.Float32BufferAttribute(angles, 3));
  g.setAttribute("colour", new THREE.Float32BufferAttribute(colour, 3));
  const m = new THREE.ShaderMaterial({ vertexShader: SWARM_VERTEX, fragmentShader: SWARM_FRAGMENT, transparent: true,
    depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { t: { value: 0 }, pxScale: { value: 20 } } });
  const pts = new THREE.Points(g, m);
  pts.frustumCulled = false;
  return pts;
}

// ---------- the component ----------

interface Drawn { body: SkyBody; mesh: THREE.Object3D; path: THREE.Object3D; trail: THREE.Line; label: HTMLSpanElement; tail?: THREE.Mesh }

const GM_SUN = 2.9591220828559115e-4;  // AU^3/day^2, for the illustrative swarm only

export function Sky3D({ base, modified, playing, daysPerSecond, follow, centre, onTime, label, handle }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const labels = useRef<HTMLDivElement>(null);
  const live = useRef({ playing, daysPerSecond, follow, onTime });
  live.current = { playing, daysPerSecond, follow, onTime };
  const world = useRef<{
    scene: THREE.Scene; camera: THREE.PerspectiveCamera; controls: OrbitControls; renderer: THREE.WebGLRenderer;
    composer: EffectComposer; drawn: Drawn[]; group: THREE.Group; t: number; span: number;
    lines: LineMaterial[]; swarm: THREE.Points | null; size: THREE.Vector2;
  } | null>(null);

  const fit = () => {
    const w = world.current;
    if (!w) return;
    const target = w.drawn.find((d) => d.body.target && !d.body.hidden) ?? w.drawn[0];
    const pts = (target ? target.body.points : []).map(toScene);
    const box = new THREE.Box3().setFromPoints([...pts, new THREE.Vector3()]);
    const size = box.getSize(new THREE.Vector3()).length() || 1;
    const mid = box.getCenter(new THREE.Vector3());
    w.controls.target.copy(mid);
    w.camera.position.copy(mid).add(new THREE.Vector3(0, size * 0.75, size * 0.85));
    w.camera.near = size / 5000; w.camera.far = size * 200; w.camera.updateProjectionMatrix();
    w.span = size;
  };
  useImperativeHandle(handle, () => ({ setTime: (d: number) => { if (world.current) world.current.t = d; }, fit }));

  // The scene: built once.
  useEffect(() => {
    const el = host.current!;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      el.textContent = "This browser cannot draw 3D (WebGL is off); the numbers below still update.";
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    el.appendChild(renderer.domElement);
    renderer.domElement.setAttribute("role", "img");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#02040a");
    const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1e5);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    // Stars: far enough to sit behind everything, fixed to the camera's world.
    const stars = new THREE.BufferGeometry();
    const sr = rng(42), pos: number[] = [];
    for (let i = 0; i < 2500; i++) {
      const u = sr() * 2 - 1, a = sr() * 2 * Math.PI, rr = 3000;
      pos.push(rr * Math.sqrt(1 - u * u) * Math.cos(a), rr * u, rr * Math.sqrt(1 - u * u) * Math.sin(a));
    }
    stars.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    scene.add(new THREE.Points(stars, new THREE.PointsMaterial({ color: 0xcfd8ff, size: 2, sizeAttenuation: false })));
    scene.add(new THREE.AmbientLight(0xffffff, 0.08));
    const group = new THREE.Group();
    scene.add(group);
    // Bloom: the Sun, the orbits and the swarm glow like NASA's orbit charts.
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    // Only the brightest parts glow (the Sun, the cores of the orbits): a low threshold washed the inner planets out.
    const bloom = new UnrealBloomPass(new THREE.Vector2(512, 512), 0.55, 0.35, 0.55);
    // Bloom renders into float textures: where the GPU cannot render to them (some phones, software renderers) the
    // frame washes out to white, so it is used only where it works (and ?bloom=0 turns it off for testing).
    const glowOk = renderer.capabilities.isWebGL2 && renderer.extensions.has("EXT_color_buffer_float")
      && new URLSearchParams(window.location.search).get("bloom") !== "0";
    if (glowOk) composer.addPass(bloom);
    composer.addPass(new OutputPass());
    world.current = { scene, camera, controls, renderer, composer, drawn: [], group, t: 0, span: 1, lines: [], swarm: null,
                      size: new THREE.Vector2(512, 512) };

    const ro = new ResizeObserver(([e]) => {
      const { width, height } = e.contentRect;
      renderer.setSize(width, height, false);
      composer.setSize(width, height);
      bloom.resolution.set(width, height);
      renderer.domElement.style.width = `${width}px`;
      renderer.domElement.style.height = `${height}px`;
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
      const w = world.current;
      if (w) { w.size.set(width, height); w.lines.forEach((m) => m.resolution.set(width, height)); }
    });
    ro.observe(el);

    let raf = 0, last = performance.now(), told = 0;
    const v = new THREE.Vector3();
    const loop = (now: number) => {
      const w = world.current!, L = live.current;
      const end = Math.max(0, ...w.drawn.map((d) => d.body.points.at(-1)?.[0] ?? 0));
      if (L.playing && end > 0) w.t = (w.t + ((now - last) / 1000) * L.daysPerSecond) % end;
      last = now;
      const rect = el.getBoundingClientRect();
      const placed: [number, number][] = [];  // label positions this frame: a label that would overlap one is hidden
      for (const d of w.drawn) {
        const p = at(d.body.points, w.t);
        if (!p) continue;
        d.mesh.position.copy(p);
        d.mesh.rotation.y += 0.004;
        let k = d.body.points.findIndex((q) => q[0] > w.t);
        if (k < 0) k = d.body.points.length;
        d.trail.geometry.setDrawRange(Math.max(0, k - 600), Math.min(k, 600));
        if (d.tail) {  // a comet's tail points away from the Sun, longer near it
          const away = p.clone().normalize();
          d.tail.position.copy(p).addScaledVector(away, d.tail.scale.y / 2);
          d.tail.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), away);
          d.tail.scale.y = Math.min(w.span * 0.25, w.span * 0.04 / Math.max(p.length(), 0.05));
        }
        v.copy(p).project(camera);
        const sx = ((v.x + 1) / 2) * rect.width, sy = ((1 - v.y) / 2) * rect.height;
        const clear = !placed.some(([px, py]) => Math.abs(px - sx) < 46 && Math.abs(py - sy) < 14);
        const visible = v.z < 1 && Math.abs(v.x) < 1.1 && Math.abs(v.y) < 1.1 && !d.body.hidden && (clear || d.body.target);
        d.label.style.display = visible ? "block" : "none";
        if (visible) {
          placed.push([sx, sy]);
          d.label.style.transform = `translate(${sx + 8}px, ${sy - 8}px)`;
        }
      }
      if (L.follow) {
        const d = w.drawn.find((x) => x.body.name === L.follow);
        if (d) { const shift = d.mesh.position.clone().sub(controls.target); controls.target.add(shift); camera.position.add(shift); }
      }
      if (w.swarm) {
        const u = (w.swarm.material as THREE.ShaderMaterial).uniforms;
        u.t.value = w.t;
        u.pxScale.value = 2.2 * w.span;
      }
      controls.update();
      composer.render();
      if (now - told > 120) { told = now; L.onTime(w.t, end); }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      world.current = null;
    };
  }, []);

  // The bodies: rebuilt when the runs change (scenario, or a change by the user).
  useEffect(() => {
    const w = world.current;
    if (!w) return;
    for (const d of w.drawn) d.label.remove();
    w.group.clear();
    w.drawn = [];
    w.lines = [];
    w.swarm = centre === "Sun" ? swarm(5000, GM_SUN) : null;
    if (w.swarm) w.group.add(w.swarm);
    const runs: [SkyBody[] | null, boolean][] = [[base, false], [modified, true]];
    // Sizes follow the target's own orbit, the part the camera frames: scaled to every body (out to Neptune) the Sun
    // came out 2 AU wide and swallowed Apophis's whole orbit.
    const focus = [...(base ?? []), ...(modified ?? [])].filter((b) => b.target).flatMap((b) => b.points.map(toScene));
    const all = focus.length ? focus : [...(base ?? []), ...(modified ?? [])].flatMap((b) => b.points.map(toScene));
    const span = all.length ? new THREE.Box3().setFromPoints([...all, new THREE.Vector3()]).getSize(new THREE.Vector3()).length() : 1;
    w.span = span;
    const unit = span * 0.005;  // the smallest body's drawn radius; sizes grow gently with the real radius
    const size = (name: string) => unit * Math.max(1, Math.pow((RADIUS_KM[name] ?? 300) / 1000, 0.45));
    // The central body: the light source.
    const sunR = centre === "Sun" ? unit * 3 : size(centre);
    const sun = new THREE.Mesh(new THREE.SphereGeometry(sunR, 48, 32),
      centre === "Sun" ? new THREE.MeshBasicMaterial({ color: 0xffd36b })
        : new THREE.MeshStandardMaterial({ map: new THREE.CanvasTexture(paint(centre)) }));
    w.group.add(sun);
    if (centre === "Sun") {
      const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture("rgba(255,240,180,1)"), transparent: true, depthWrite: false }));
      glow.scale.setScalar(sunR * 5);
      w.group.add(glow);
      w.group.add(new THREE.PointLight(0xffffff, 3, 0, 0));
    } else {
      const light = new THREE.DirectionalLight(0xffffff, 2.2);
      light.position.set(1, 0.3, 0.4);
      w.group.add(light);
    }
    for (const [run, dashed] of runs) {
      for (const body of run ?? []) {
        if (dashed && !modified) continue;
        const pts = body.points.map(toScene);
        const colour = new THREE.Color(body.colour || "#9ec5ff");
        const pathGeo = new THREE.BufferGeometry().setFromPoints(pts);
        // The orbit: a wide soft glow under a thin bright core (the bloom pass spreads both), dashed for a changed run.
        const flat = new LineGeometry().setPositions(pts.flatMap((q) => [q.x, q.y, q.z]));
        const path = new THREE.Group();
        for (const [width, opacity] of [[6, 0.1], [1.8, 0.9]] as const) {
          const m = new LineMaterial({ color: colour.getHex(), linewidth: body.target ? width * 1.3 : width, transparent: true,
            opacity: dashed ? opacity * 0.8 : opacity, dashed, dashSize: span / 120, gapSize: span / 200,
            // Normal blending: an orbit drawn over itself many times (Io laps Jupiter 34 times in 60 days) must not
            // add up to a white haze; the bloom pass gives the glow.
            blending: THREE.NormalBlending, depthWrite: false });
          m.resolution.copy(w.size);
          w.lines.push(m);
          const line = new Line2(flat, m);
          if (dashed) line.computeLineDistances();
          path.add(line);
        }
        const trail = new THREE.Line(pathGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 }));
        const tex = new THREE.CanvasTexture(paint(body.name));
        tex.colorSpace = THREE.SRGBColorSpace;
        const r = body.target ? Math.max(size(body.name), unit * 1.4) : size(body.name);
        const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 40, 24),
          new THREE.MeshStandardMaterial({ map: tex, roughness: 0.9, metalness: 0, transparent: dashed, opacity: dashed ? 0.85 : 1 }));
        if (body.name === "Saturn") {
          const ring = new THREE.Mesh(new THREE.RingGeometry(r * 1.3, r * 2.4, 96),
            new THREE.MeshBasicMaterial({ map: ringTexture(), side: THREE.DoubleSide, transparent: true }));
          // Ring texture runs radially: map the ring's u (around) to the texture's length, from inner to outer edge.
          const uv = ring.geometry.attributes.uv as THREE.BufferAttribute, p3 = ring.geometry.attributes.position as THREE.BufferAttribute;
          for (let i = 0; i < uv.count; i++) {
            const d = Math.hypot(p3.getX(i), p3.getY(i));
            uv.setXY(i, (d - r * 1.3) / (r * 1.1), 0.5);
          }
          ring.rotation.x = Math.PI / 2 - 0.47;  // Saturn's 27 degree tilt
          mesh.add(ring);
        }
        let tail: THREE.Mesh | undefined;
        if (body.name === "Halley") {
          tail = new THREE.Mesh(new THREE.ConeGeometry(r * 0.9, 1, 16, 1, true),
            new THREE.MeshBasicMaterial({ color: 0xbfe9ff, transparent: true, opacity: 0.35, depthWrite: false, side: THREE.DoubleSide }));
          tail.rotation.x = Math.PI;
          w.group.add(tail);
        }
        if (body.hidden) { mesh.visible = false; path.visible = false; trail.visible = false; }
        w.group.add(path, trail, mesh);
        const tag = document.createElement("span");
        tag.className = "sky3d-label";
        tag.textContent = dashed ? `${body.name} (changed)` : body.name;
        labels.current?.appendChild(tag);
        w.drawn.push({ body, mesh, path, trail, label: tag, tail });
      }
    }
    fit();
  }, [base, modified, centre]);  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="sky3d" aria-label={label}>
      <div ref={host} className="sky3d-canvas" />
      <div ref={labels} className="sky3d-labels" aria-hidden="true" />
    </div>
  );
}
