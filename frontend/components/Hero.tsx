import { Activity, Camera, Route, Upload } from "lucide-react";
import { Law, Tex } from "./Equation";

// The first screen: what PhysicsLens found, shown working, before any form. Every number is from a stored gallery
// run (gallery-apophis, gallery-jupiter) checked against NASA/JPL values fetched at run time.

// The accepted law of gallery-apophis (exponent 1.5 written as 3/2), typeset large: the video's own overlay is too
// small to read at this size.
const LAW = "\\ddot x = -\\frac{\\mu\\,x}{(x^2+y^2)^{3/2}} - b\\,\\dot x,\\quad \\ddot y = -\\frac{\\mu\\,y}{(x^2+y^2)^{3/2}} - b\\,\\dot y";

const STATS = [
  { value: "0.023%", label: "the Sun's mass (GM), measured from pixels, against NASA/JPL" },
  { value: "0.012%", label: "Jupiter weighed from its four large moons" },
  { value: "1.4991", note: "± 0.0032", label: "Kepler's third-law exponent, found from the tracks (true value 1.5)" },
];

export function Hero() {
  return (
    <section className="hero" aria-labelledby="hero-title">
      <div className="hero-text">
        <p className="hero-eyebrow">Nobody told it the physics.</p>
        <h1 id="hero-title">It watched an asteroid and found Newton&apos;s law of gravity.</h1>
        <p className="hero-lead">
          Given only a video of a dot circling the Sun, the agent on Meta&apos;s open-weight Muse Spark Open wrote the
          inverse-square law itself, and numerical verification kept it only because it forecast motion it never saw.
          Film anything moving and it does the same: the equation that governs it, its uncertainty, and a simulation a
          robot can run.
        </p>
        <ul className="hero-stats">
          {STATS.map((s) => (
            <li key={s.label}>
              <span className="hero-number">{s.value}{s.note && <small> {s.note}</small>}</span>
              <span className="hero-label">{s.label}</span>
            </li>
          ))}
        </ul>
        <div className="hero-actions">
          <a className="button" href="/?run=gallery-apophis"><Route size={16} aria-hidden="true" />See how it decided</a>
          <a className="button secondary" href="#analyze-title"><Upload size={16} aria-hidden="true" />Try your own video</a>
          <a className="button secondary" href="/live"><Camera size={16} aria-hidden="true" />Live camera</a>
          <a className="button secondary" href="/audit"><Activity size={16} aria-hidden="true" />Catch fake physics</a>
        </div>
      </div>
      <figure className="hero-media">
        <video src="/hero/apophis-orbit.mp4" poster="/hero/apophis-orbit.jpg" autoPlay muted loop playsInline
               preload="metadata" aria-label="Asteroid Apophis around the Sun, with the law PhysicsLens found: the force arrow points at the Sun in every frame" />
        <div className="hero-law">
          <p className="hero-law-title">The law it wrote, then verified</p>
          <Law tex={LAW} />
          <ul className="hero-law-values">
            <li><Tex tex={"\\mu = (2.9598 \\pm 0.0010) \\times 10^{-4}\\ \\mathrm{AU^3/day^2}"} />
              <span>the Sun&apos;s GM: 0.023% from NASA/JPL</span></li>
            <li><Tex tex={"b = (2.4 \\pm 0.2) \\times 10^{-7}\\ \\mathrm{day^{-1}}"} />
              <span>a tiny extra term: the Sun alone is not quite the whole story (the planets pull too)</span></li>
          </ul>
        </div>
        <figcaption>
          Asteroid Apophis over four years, rendered from NASA/JPL positions. Red arrow: the force the law gives,
          pointing at the Sun in every frame; yellow: velocity. Carried 3.3 years forward, the law puts the 2029 Earth
          flyby 13 hours from JPL&apos;s time.
        </figcaption>
      </figure>
    </section>
  );
}
