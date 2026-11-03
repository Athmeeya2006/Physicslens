"use client";

import { ArrowRight, Upload } from "lucide-react";
import type { GalleryItem } from "@/lib/api";
import { Law } from "./Equation";
import { LoopVideo } from "./LoopVideo";
import { StatusChip } from "./StatusChip";

// The home page's showcase: what PhysicsLens found, two tiles to a row, each with its own clean video (ghost,
// trail and force arrows; scripts/build_stories.py), the law typeset large and the numbers it measured. Every
// number is from the stored gallery result named in `run`; the tile's credit, verdict and full description come from
// that result's gallery entry, so this page is also the list of real recordings.

interface Story {
  run: string;
  source: string;
  status: string;  // the gallery verdict, used until the gallery entry loads
  title: string;
  text: string;
  law: string;
  stats: { value: string; label: string }[];
  note?: string;
}

const STORIES: Story[] = [
  {
    run: "gallery-halley", source: "Comet 1P/Halley, the 1986 passage", status: "accepted",
    title: "A hairpin around the Sun. Left free, the exponent came out 1.9998.",
    text: "As Halley whips past the Sun its pull grows about 80-fold in a year, a hard test for any law. The agent wrote inverse-square gravity; a rival law with the exponent left free fitted it at 1.9998, Newton's 2.",
    law: "\\ddot x = -\\frac{\\mu\\,x}{(x^2+y^2)^{3/2}},\\quad \\ddot y = -\\frac{\\mu\\,y}{(x^2+y^2)^{3/2}}",
    stats: [{ value: "0.095%", label: "the Sun's GM against NASA/JPL" }, { value: "1.9998", label: "the force law's exponent, left free" }],
    note: "Carried 75 years forward with the Sun alone, the law brings Halley back in late 2063, not JPL's July 2061. Adding the giant planets' pull brings the same integration within 4 days of JPL's date; what remains is the Sun's GM, whose 0.095% error so long an orbit turns into about eight months.",
  },
  {
    run: "gallery-jupiter", source: "Jupiter and its four large moons", status: "accepted",
    title: "Jupiter, weighed by watching its moons.",
    text: "From Io's orbit alone the agent found Keplerian gravity and Jupiter's mass. Europa, Ganymede and Callisto weigh it the same way, and their periods and distances follow Kepler's third law.",
    law: "\\ddot x = -\\frac{\\mu\\,x}{(x^2+y^2)^{3/2}},\\quad \\ddot y = -\\frac{\\mu\\,y}{(x^2+y^2)^{3/2}}",
    stats: [{ value: "0.012%", label: "Jupiter's GM from all four moons, against NASA/JPL" },
            { value: "1.4991", label: "Kepler's third-law exponent (true value 1.5)" }],
  },
  {
    run: "gallery-pendulum", source: "Real footage: a pendulum on a phone", status: "ambiguous",
    title: "Damping fits better, so why isn't it accepted?",
    text: "Damping fits the whole swing slightly better, but forecasts the part it never saw worse than the simple pendulum, so verification will not confirm it: the two laws tie. They agree on g, so that is measured either way, and the agent designs the recording that would settle the rest: release from about 35 degrees and film 7 seconds.",
    law: "\\ddot\\theta = -\\frac{g}{L}\\sin\\theta \\quad\\text{or}\\quad -\\frac{g}{L}\\sin\\theta - \\gamma\\,\\dot\\theta\\,?",
    stats: [{ value: "9.91", label: "g in m/s² from a 0.5 m string (95%: 9.82 to 9.99), 0.98% above Aachen's local 9.811" },
            { value: "8.5×", label: "how far the two forecasts separate in the requested swing, against tracking noise" }],
  },
  {
    run: "gallery-paperbag", source: "Real footage: a thrown paper bag at 200 fps", status: "accepted",
    title: "Free fall, rejected.",
    text: "Of three drag laws, only drag proportional to speed forecasts the part of the flight the model never saw. No ruler in view: Earth's gravity sets the scale.",
    law: "\\dot{\\mathbf v} = -g\\,\\hat{\\mathbf y} - k\\,\\mathbf v",
    stats: [{ value: "2.42", label: "drag rate k per second (95%: 2.32 to 2.52)" }, { value: "2.9%", label: "error forecasting the unseen flight" }],
  },
  {
    run: "gallery-spring", source: "Real footage: a glider between two springs", status: "ambiguous",
    title: "Friction or damping? Fifteen seconds would tell.",
    text: "The swing shrinks by about the same amount every cycle, the signature of sliding friction, and dry friction forecasts the unseen motion best. But on this clip viscous damping is only 4 BIC behind, so neither is confirmed: both agree on the frequency, and the agent asks for a 15-second recording that would separate them. No ruler needed.",
    law: "\\ddot s = -\\omega^2 (s - s_0) - f\\,\\operatorname{sgn}(\\dot s) \\quad\\text{or}\\quad - 2\\zeta\\omega\\,\\dot s\\,?",
    stats: [{ value: "2.74", label: "natural frequency, rad/s, from both laws (95%: 2.737 to 2.746)" }, { value: "8.8%", label: "forecast error (viscous damping: 10.9%)" }],
  },
];

const VERDICT: Record<string, string> = { accepted: "Verified equation", ambiguous: "Needs one more experiment", no_fit: "No clean equation" };

export const STORY_RUNS = STORIES.map((s) => s.run);

const TIPS = [
  "Hold the phone still, side-on to the motion.",
  "Lay an A4 sheet flat in the plane of motion for real-world scale.",
  "Frequency, damping and friction are exact even without a scale reference.",
];

export function Discoveries({ items = [], offline = false }: { items?: GalleryItem[]; offline?: boolean }) {
  const byId = new Map(items.map((g) => [g.id, g]));
  return (
    <section id="results" className="band band-surface" aria-labelledby="discoveries-title">
      <div className="band-inner stack" style={{ gap: 28 }}>
        <h2 id="discoveries-title">Found from video alone, then tested on motion the model never saw.</h2>
        <ul className="story-grid">
          {STORIES.map((s) => {
            const g = byId.get(s.run);
            const status = g?.status ?? s.status;
            return (
              <li key={s.run} id={`story-${s.run}`} className="story-tile">
                <figure className="story-media">
                  <LoopVideo src={`/stories/${s.run}.mp4`} poster={`/stories/${s.run}.jpg`}
                    label={`${s.source}: the measured track, the law's ghost and its force arrow`} />
                  <span className="story-chip"><StatusChip status={status} label={VERDICT[status]} /></span>
                </figure>
                <p className="story-source">{s.source}{g?.credit && <><br /><span className="story-credit">{g.credit}</span></>}</p>
                <h3>{s.title}</h3>
                <p>{s.text}</p>
                <div className="story-law"><Law tex={s.law} /></div>
                <ul className="story-stats">
                  {s.stats.map((x) => (
                    <li key={x.label}><span className="story-number">{x.value}</span><span>{x.label}</span></li>
                  ))}
                </ul>
                {(s.note || g?.description) && (
                  <details className="story-more">
                    <summary>{s.note ? "What happens further out, and the full run" : "The full run"}</summary>
                    {s.note && <p>{s.note}</p>}
                    {g?.description && <p>{g.description}</p>}
                  </details>
                )}
                <a className="story-link" href={`/?run=${s.run}`}>See the decision trail <ArrowRight size={15} aria-hidden="true" /></a>
              </li>
            );
          })}
          <li className="story-tile story-next">
            <h3>Your recording next</h3>
            <p>Film anything that moves, 2 to 20 seconds, and it goes through the same steps and the same held-out test.</p>
            {offline ? (
              <>
                <a className="button" href="/?run=gallery-pendulum">Open a complete analysis</a>
                <p className="small">Uploads open when the analysis server is online; the header shows it.</p>
              </>
            ) : <a className="button" href="#analyze-title"><Upload size={16} aria-hidden="true" />Analyze your own clip</a>}
            <ul className="tips">{TIPS.map((t) => <li key={t}>{t}</li>)}</ul>
          </li>
        </ul>
      </div>
    </section>
  );
}
