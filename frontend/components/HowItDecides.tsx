import type { ReactNode } from "react";

// The home page's "how": three real steps, each naming what did the work, then the held-out rule drawn small. The
// drawing is an illustration of the rule (a made-up damped swing), not a stored result, and says so.

const STEPS: { title: string; text: string; tags: string[] }[] = [
  { title: "Film it", tags: ["Muse Voice Transcribe"],
    text: "Anything that moves, from a phone held still: a swing, a bounce, a spin-down. Say what you know out loud, such as the string's length, and it is filled in for you." },
  { title: "Meta's models measure and propose", tags: ["SAM 3.1", "Muse Spark Open"],
    text: "SAM 3.1 tracks the object; Muse Spark Open, Meta's open-weight model, reads the scene and proposes equations, even ones it writes itself." },
  { title: "Math decides", tags: ["Numerical verification", "MuJoCo"],
    text: "Each equation must forecast motion it never saw; the winner comes with uncertainties, a ghost on the video and a MuJoCo model." },
];

// A damped swing sampled like a tracker would, and the law through it; the last quarter is held out.
const W = 560, H = 150, PAD = 14, HELD = 0.75;
const at = (u: number) => H / 2 - (H / 2 - PAD - 6) * Math.exp(-1.3 * u) * Math.cos(2 * Math.PI * 3.2 * u);
const line = Array.from({ length: 141 }, (_, i) => i / 140)
  .map((u, i) => `${i ? "L" : "M"}${(PAD + u * (W - 2 * PAD)).toFixed(1)},${at(u).toFixed(1)}`).join("");
const dots = Array.from({ length: 48 }, (_, i) => i / 47)
  .map((u, i) => ({ x: PAD + u * (W - 2 * PAD), y: at(u) + 3.2 * Math.sin(i * 7.3) }));

function HeldOutSketch() {
  const split = PAD + HELD * (W - 2 * PAD);
  return (
    <svg className="heldout-sketch" viewBox={`0 0 ${W} ${H + 22}`} role="img"
         aria-label="Illustration: measured dots and the law's line; the last quarter of the motion is shaded and held out from the fit">
      <rect x={split} y={18} width={W - PAD - split} height={H - 4} className="ho-shade" />
      <text x={PAD} y={12} className="ho-label">fitted on this part</text>
      <text x={W - PAD} y={12} textAnchor="end" className="ho-label ho-label-model">held out: never seen</text>
      <path d={line} transform="translate(0 16)" className="ho-line" />
      {dots.map((d, i) => <circle key={i} cx={d.x} cy={d.y + 16} r={2.6} className="ho-dot" />)}
    </svg>
  );
}

export function HowItDecides({ note }: { note?: ReactNode }) {
  return (
    <section className="band band-surface" aria-labelledby="how-title">
      <div className="band-inner stack" style={{ gap: 28 }}>
        <h2 id="how-title">The model proposes. The math decides.</h2>
        <ol className="steps">
          {STEPS.map((s, i) => (
            <li key={s.title}>
              <p className="step-num">Step {i + 1}</p>
              <h3>{s.title}</h3>
              <p>{s.text}</p>
              <ul className="tag-row" aria-label="Done by">
                {s.tags.map((t) => <li key={t} className="tag">{t}</li>)}
              </ul>
            </li>
          ))}
        </ol>
        <figure className="heldout-figure">
          <HeldOutSketch />
          <figcaption className="small">
            The acceptance rule: every equation is fitted on the first three quarters of the motion and must forecast the
            last quarter, which it never saw. Among the laws that pass, the simplest adequate one wins (BIC), and the
            simplest model of each family is always tested as a control. The language model never accepts an equation.
          </figcaption>
        </figure>
        {note}
      </div>
    </section>
  );
}
