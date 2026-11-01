"use client";

import { ArrowRight } from "lucide-react";
import { Law } from "./Equation";
import { LoopVideo } from "./LoopVideo";

// The home page's showcase: what PhysicsLens found, one discovery per row, each with its own clean video (ghost,
// trail and force arrows; scripts/build_stories.py), the law typeset large and the numbers it measured. Every
// number is from the stored gallery result named in `run`.

interface Story {
  run: string;
  eyebrow: string;
  title: string;
  text: string;
  law: string;
  stats: { value: string; label: string }[];
  note?: string;
}

const STORIES: Story[] = [
  {
    run: "gallery-halley", eyebrow: "Comet 1P/Halley, the 1986 passage",
    title: "A hairpin around the Sun. Left free, the exponent came out 1.9998.",
    text: "As Halley whips past the Sun its pull grows about 80-fold in a year, a hard test for any law. The agent wrote inverse-square gravity; a rival law with the exponent left free fitted it at 1.9998, Newton's 2.",
    law: "\\ddot x = -\\frac{\\mu\\,x}{(x^2+y^2)^{3/2}},\\quad \\ddot y = -\\frac{\\mu\\,y}{(x^2+y^2)^{3/2}}",
    stats: [{ value: "0.095%", label: "the Sun's GM against NASA/JPL" }, { value: "1.9998", label: "the force law's exponent, left free" }],
    note: "Carried 75 years forward with the Sun alone, the law brings Halley back in late 2063, not JPL's July 2061. Adding the giant planets' pull brings the same integration within 4 days of JPL's date; what remains is the Sun's GM, whose 0.095% error so long an orbit turns into about eight months.",
  },
  {
    run: "gallery-jupiter", eyebrow: "Jupiter and its four large moons",
    title: "Jupiter, weighed by watching its moons.",
    text: "From Io's orbit alone the agent found Keplerian gravity and Jupiter's mass. Europa, Ganymede and Callisto weigh it the same way, and their periods and distances follow Kepler's third law.",
    law: "\\ddot x = -\\frac{\\mu\\,x}{(x^2+y^2)^{3/2}},\\quad \\ddot y = -\\frac{\\mu\\,y}{(x^2+y^2)^{3/2}}",
    stats: [{ value: "0.012%", label: "Jupiter's GM from all four moons, against NASA/JPL" },
            { value: "1.4991", label: "Kepler's third-law exponent (true value 1.5)" }],
  },
  {
    run: "gallery-pendulum", eyebrow: "Real footage: a pendulum on a phone",
    title: "Damping fits better, so why isn't it accepted?",
    text: "Damping fits the whole swing slightly better, but forecasts the part it never saw worse than the simple pendulum, so verification will not confirm it: the two laws tie. They agree on g, so that is measured either way, and the agent designs the recording that would settle the rest: release from about 35 degrees and film 7 seconds.",
    law: "\\ddot\\theta = -\\frac{g}{L}\\sin\\theta \\quad\\text{or}\\quad -\\frac{g}{L}\\sin\\theta - \\gamma\\,\\dot\\theta\\,?",
    stats: [{ value: "9.91 ± 0.04", label: "g in m/s² from a 0.5 m string, from both laws (0.99% from 9.81)" },
            { value: "8.5×", label: "how far the two forecasts separate in the requested swing, against tracking noise" }],
  },
  {
    run: "gallery-paperbag", eyebrow: "Real footage: a thrown paper bag at 200 fps",
    title: "Free fall, rejected.",
    text: "Of three drag laws, only drag proportional to speed forecasts the part of the flight the model never saw. No ruler in view: Earth's gravity sets the scale.",
    law: "\\dot{\\mathbf v} = -g\\,\\hat{\\mathbf y} - k\\,\\mathbf v",
    stats: [{ value: "2.42 ± 0.05", label: "drag rate k per second" }, { value: "2.9%", label: "error forecasting the unseen flight" }],
  },
  {
    run: "gallery-spring", eyebrow: "Real footage: a glider between two springs",
    title: "Friction or damping? Fifteen seconds would tell.",
    text: "The swing shrinks by about the same amount every cycle, the signature of sliding friction, and dry friction forecasts the unseen motion best. But on this clip viscous damping is only 4 BIC behind, so neither is confirmed: both agree on the frequency, and the agent asks for a 15-second recording that would separate them. No ruler needed.",
    law: "\\ddot s = -\\omega^2 (s - s_0) - f\\,\\operatorname{sgn}(\\dot s) \\quad\\text{or}\\quad - 2\\zeta\\omega\\,\\dot s\\,?",
    stats: [{ value: "2.7401 ± 0.0023", label: "natural frequency, rad/s, from both laws" }, { value: "8.8%", label: "forecast error (viscous damping: 10.9%)" }],
  },
];

export function Discoveries() {
  return (
    <section className="discoveries" aria-labelledby="discoveries-title">
      <div className="discoveries-inner">
        <header className="discoveries-head">
          <p className="hero-eyebrow">What it found</p>
          <h2 id="discoveries-title">Every one of these was found from video alone, then tested on motion the model never saw.</h2>
        </header>
        {STORIES.map((s, i) => (
          <article key={s.run} className={`story${i % 2 ? " flip" : ""}`}>
            <div className="story-media"><LoopVideo src={`/stories/${s.run}.mp4`} poster={`/stories/${s.run}.jpg`}
              label={`${s.eyebrow}: the measured track, the law's ghost and its force arrow`} /></div>
            <div className="story-text">
              <p className="story-eyebrow">{s.eyebrow}</p>
              <h3>{s.title}</h3>
              <p>{s.text}</p>
              <Law tex={s.law} />
              <ul className="story-stats">
                {s.stats.map((x) => (
                  <li key={x.label}><span className="story-number">{x.value}</span><span>{x.label}</span></li>
                ))}
              </ul>
              {s.note && <p className="story-note">{s.note}</p>}
              <a className="story-link" href={`/?run=${s.run}`}>See the decision trail <ArrowRight size={15} aria-hidden="true" /></a>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
