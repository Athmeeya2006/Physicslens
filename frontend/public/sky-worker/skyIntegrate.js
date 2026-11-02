// The Solar System Lab's in-browser integrator: re-runs a scenario with the visitor's changes, on the page or in a
// worker (lib/skyWorker.ts) so a slider stays smooth. No imports beyond types, so a worker can load it alone.
const C_AU_PER_DAY = 173.144632674;
export const SUN_GM = 2.9591220828559115e-4; // AU^3/day^2, as the server's catalogue
export const HYPOTHETICAL = "hypothetical body";
const M_S_TO_AU_DAY = 86400 / 149597870700;
/** The state with its velocity pushed by dv (m/s) along the motion, outward and along the orbit's normal, exactly as
 * the server does it (sky/scenarios.kicked). */
export function kicked(state, dv) {
    const r = state.slice(0, 3), v = state.slice(3, 6);
    const len = (a) => Math.hypot(a[0], a[1], a[2]);
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const along = v.map((x) => x / len(v));
    const h = cross(r, v), normal = h.map((x) => x / len(h));
    let out = cross(along, normal);
    if (out[0] * r[0] + out[1] * r[1] + out[2] * r[2] < 0)
        out = out.map((x) => -x);
    return [...r, ...v.map((x, i) => x + (dv[0] * along[i] + dv[1] * out[i] + dv[2] * normal[i]) * M_S_TO_AU_DAY)];
}
/** Heliocentric position and velocity from elements in the reference plane (node and perihelion at zero), as the
 * server places a hypothetical body: Kepler's equation by Newton's method. */
export function stateFromElements(a, e, iDeg, mDeg, gm) {
    const M = (mDeg * Math.PI) / 180, inc = (iDeg * Math.PI) / 180;
    let E = e < 0.8 ? M : Math.PI;
    for (let k = 0; k < 50; k++)
        E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    const n = Math.sqrt(gm / a ** 3), q = Math.sqrt(1 - e * e), den = 1 - e * Math.cos(E);
    const x = a * (Math.cos(E) - e), y = a * q * Math.sin(E);
    const vx = (-a * n * Math.sin(E)) / den, vy = (a * n * q * Math.cos(E)) / den;
    return [x, y * Math.cos(inc), y * Math.sin(inc), vx, vy * Math.cos(inc), vy * Math.sin(inc)];
}
/** Heliocentric Newtonian N-body with the indirect term (and optional 1PN relativity and a passing star), RK4 with a
 * step sized to the fastest orbit, on typed arrays with nothing allocated per step, so it is quick enough to re-run on
 * every movement of a slider. Good to a fraction of a pixel for the lab's spans; the server's DOP853 run is the real
 * answer (and adds Jupiter's J2 for the moons). */
export function integrate(init, c, target, maxPoints = 1500, base) {
    let bodies = init.bodies.filter((b) => !c.disabled.includes(b.name)).map((b) => ({
        ...b, gm: b.massive ? b.gm * (c.mass_scale[b.name] ?? 1) : 0,
        state: c.kicks?.[b.name]?.some((v) => v !== 0) ? kicked(b.state, c.kicks[b.name]) : [...b.state],
    }));
    // The Moon circles Earth every 27 days, far faster than any step sized for the planets: Earth and Moon move as one
    // body at their centre of mass, and the Moon is drawn around Earth on its path from the scenario as written (a
    // change to the planets barely alters it). A push on the Moon alone keeps it a body of its own.
    const earth = bodies.find((b) => b.name === "Earth"), moon = bodies.find((b) => b.name === "Moon");
    const pairMoon = !!(earth && moon && base && !c.kicks?.Moon?.some((v) => v !== 0)
        && base.some((b) => b.name === "Earth") && base.some((b) => b.name === "Moon"));
    if (pairMoon && earth && moon) {
        const m = earth.gm + moon.gm;
        earth.state = earth.state.map((v, i) => (earth.gm * v + moon.gm * moon.state[i]) / (m || 1));
        earth.gm = m;
        bodies = bodies.filter((b) => b !== moon);
    }
    const x = c.extra_body;
    if (x && x.a_au > 0 && x.e >= 0 && x.e < 1 && x.gm_suns >= 0) {
        bodies.push({ name: HYPOTHETICAL, gm: SUN_GM * x.gm_suns, massive: true,
            state: stateFromElements(x.a_au, x.e, x.i_deg, x.M_deg, SUN_GM) });
    }
    const n = bodies.length;
    const gm = Float64Array.from(bodies.map((b) => b.gm));
    const mu = init.central_gm;
    const gr = c.gr ?? init.gr;
    const cScale = c.c_scale ?? init.c_scale;
    const stars = init.prescribed ?? [];
    const days = c.years ? c.years * 365.25 : init.clip_days;
    // Step: sized to the fastest passage, each body's orbit taken at its closest approach (Mercury's eccentric orbit
    // was 4% off over four years with a step from its starting distance).
    const tightest = Math.min(...bodies.map((b) => {
        const r = Math.hypot(b.state[0], b.state[1], b.state[2]), v2 = b.state[3] ** 2 + b.state[4] ** 2 + b.state[5] ** 2;
        const a = 1 / (2 / r - v2 / mu);
        const h2 = (b.state[1] * b.state[5] - b.state[2] * b.state[4]) ** 2 + (b.state[2] * b.state[3] - b.state[0] * b.state[5]) ** 2
            + (b.state[0] * b.state[4] - b.state[1] * b.state[3]) ** 2;
        const e = a > 0 ? Math.sqrt(Math.max(0, 1 - h2 / (mu * a))) : 1;
        const q = a > 0 ? a * (1 - e) : Math.min(r, h2 / (2 * mu));
        return 2 * Math.PI * Math.sqrt(q ** 3 / mu);
    }));
    const steps = Math.min(400000, Math.max(200, Math.ceil(days / (tightest / 90))));
    const j2 = init.j2;
    const h = days / steps;
    const N = 6 * n;
    // y = [x0, y0, z0, ..., vx0, vy0, vz0, ...]
    const y = new Float64Array(N), tmp = new Float64Array(N);
    const k1 = new Float64Array(N), k2 = new Float64Array(N), k3 = new Float64Array(N), k4 = new Float64Array(N);
    bodies.forEach((b, i) => { for (let d = 0; d < 3; d++) {
        y[3 * i + d] = b.state[d];
        y[3 * n + 3 * i + d] = b.state[3 + d];
    } });
    const deriv = (s, t, out) => {
        let ix = 0, iy = 0, iz = 0; // the indirect term: the bodies' pull on the Sun
        for (let j = 0; j < n; j++) {
            if (!gm[j])
                continue;
            const r = Math.hypot(s[3 * j], s[3 * j + 1], s[3 * j + 2]), r3 = r * r * r;
            ix += gm[j] * s[3 * j] / r3;
            iy += gm[j] * s[3 * j + 1] / r3;
            iz += gm[j] * s[3 * j + 2] / r3;
        }
        const sp = stars.map((p) => [p.at_start[0] + p.velocity[0] * t, p.at_start[1] + p.velocity[1] * t, p.at_start[2] + p.velocity[2] * t]);
        for (let i = 0; i < n; i++) {
            const px = s[3 * i], py = s[3 * i + 1], pz = s[3 * i + 2];
            const r = Math.hypot(px, py, pz), r3 = r * r * r;
            let ax = -mu * px / r3 - ix, ay = -mu * py / r3 - iy, az = -mu * pz / r3 - iz;
            for (let j = 0; j < n; j++) {
                if (j === i || !gm[j])
                    continue;
                const dx = s[3 * j] - px, dy = s[3 * j + 1] - py, dz = s[3 * j + 2] - pz;
                const d = Math.hypot(dx, dy, dz), d3 = d * d * d;
                ax += gm[j] * dx / d3;
                ay += gm[j] * dy / d3;
                az += gm[j] * dz / d3;
            }
            for (let k = 0; k < stars.length; k++) {
                const q = sp[k], g = stars[k].gm;
                const dx = q[0] - px, dy = q[1] - py, dz = q[2] - pz;
                const d = Math.hypot(dx, dy, dz), d3 = d * d * d, rq = Math.hypot(q[0], q[1], q[2]), rq3 = rq * rq * rq;
                ax += g * (dx / d3 - q[0] / rq3);
                ay += g * (dy / d3 - q[1] / rq3);
                az += g * (dz / d3 - q[2] / rq3);
            }
            if (j2) {
                const k = j2.pole, z = px * k[0] + py * k[1] + pz * k[2], r2 = r * r;
                const f = -1.5 * j2.j2 * j2.gm * j2.radius ** 2 / r2 ** 2.5, w = 1 - 5 * z * z / r2;
                ax += f * (w * px + 2 * z * k[0]);
                ay += f * (w * py + 2 * z * k[1]);
                az += f * (w * pz + 2 * z * k[2]);
            }
            const vx = s[3 * n + 3 * i], vy = s[3 * n + 3 * i + 1], vz = s[3 * n + 3 * i + 2];
            if (gr) {
                const k = mu * cScale / (C_AU_PER_DAY ** 2 * r3);
                const v2 = vx * vx + vy * vy + vz * vz, rv = px * vx + py * vy + pz * vz;
                ax += k * ((4 * mu / r - v2) * px + 4 * rv * vx);
                ay += k * ((4 * mu / r - v2) * py + 4 * rv * vy);
                az += k * ((4 * mu / r - v2) * pz + 4 * rv * vz);
            }
            out[3 * i] = vx;
            out[3 * i + 1] = vy;
            out[3 * i + 2] = vz;
            out[3 * n + 3 * i] = ax;
            out[3 * n + 3 * i + 1] = ay;
            out[3 * n + 3 * i + 2] = az;
        }
    };
    const every = Math.max(1, Math.floor(steps / maxPoints));
    const [e1, e2, e3] = init.basis;
    const out = bodies.map(() => []);
    const record = (t) => {
        for (let i = 0; i < n; i++) {
            const a = y[3 * i], b = y[3 * i + 1], cc = y[3 * i + 2];
            out[i].push([t, a * e1[0] + b * e1[1] + cc * e1[2], a * e2[0] + b * e2[1] + cc * e2[2], a * e3[0] + b * e3[1] + cc * e3[2]]);
        }
    };
    record(0);
    for (let st = 1; st <= steps; st++) {
        const t = (st - 1) * h;
        deriv(y, t, k1);
        for (let i = 0; i < N; i++)
            tmp[i] = y[i] + 0.5 * h * k1[i];
        deriv(tmp, t + 0.5 * h, k2);
        for (let i = 0; i < N; i++)
            tmp[i] = y[i] + 0.5 * h * k2[i];
        deriv(tmp, t + 0.5 * h, k3);
        for (let i = 0; i < N; i++)
            tmp[i] = y[i] + h * k3[i];
        deriv(tmp, t + h, k4);
        for (let i = 0; i < N; i++)
            y[i] += (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        if (st % every === 0)
            record(st * h);
    }
    const result = bodies.map((b, i) => ({ name: b.name, colour: "", hidden: (c.hidden ?? []).includes(b.name), target: b.name === target, points: out[i] }));
    if (pairMoon && base) {
        // The Moon: the new Earth plus the Moon's offset from Earth in the scenario as written, at the same times.
        const be = base.find((b) => b.name === "Earth").points, bm = base.find((b) => b.name === "Moon").points;
        const offset = (t) => {
            let k = bm.findIndex((p) => p[0] >= t);
            if (k < 0)
                k = bm.length - 1;
            return [1, 2, 3].map((d) => bm[k][d] - be[Math.min(k, be.length - 1)][d]);
        };
        const e = result.find((b) => b.name === "Earth");
        result.push({ name: "Moon", colour: "", hidden: (c.hidden ?? []).includes("Moon"), target: target === "Moon",
            points: e.points.map((p) => { const o = offset(p[0]); return [p[0], p[1] + o[0], p[2] + o[1], p[3] + o[2]]; }) });
    }
    return result;
}
