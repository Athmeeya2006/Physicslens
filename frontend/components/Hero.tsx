import { Route, Upload } from "lucide-react";
import { Law, Tex } from "./Equation";
import type { GalleryItem } from "@/lib/api";
import { LoopVideo } from "./LoopVideo";

// The first screen: what PhysicsLens found, shown working, before any form. Every number is from a stored gallery
// run (gallery-apophis, gallery-jupiter) checked against NASA/JPL values fetched at run time.

// The accepted law of gallery-apophis (exponent 1.5 written as 3/2), typeset large: the video's own overlay is too
// small to read at this size.
const LAW = "\\ddot x = -\\frac{\\mu\\,x}{(x^2+y^2)^{3/2}} - b\\,\\dot x,\\quad \\ddot y = -\\frac{\\mu\\,y}{(x^2+y^2)^{3/2}} - b\\,\\dot y";

const STATS = [
  { value: "0.023%", label: "the Sun's mass (GM), measured from pixels, against NASA/JPL" },
  { value: "0.012%", label: "Jupiter weighed from its four large moons" },
  { value: "1.4991", note: "95%: 1.4929 to 1.5053", label: "Kepler's third-law exponent, found from the tracks (true value 1.5)" },
];

// `apophis` is the gallery entry, for its credit line and the run's full description.
export function Hero({ apophis }: { apophis?: GalleryItem }) {
  return (
    <>
      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-text">
          <h1 id="hero-title">It watched an asteroid and found Newton&apos;s law of gravity.</h1>
          <p className="hero-lead">
            Given only a video of a dot circling the Sun, the agent on Meta&apos;s open-weight Muse Spark Open wrote the
            inverse-square law itself, and numerical verification kept it only because it forecast motion it never saw.
          </p>
          <p className="hero-lead">
            Film anything moving and it does the same: the equation that governs it, its uncertainty, and a simulation a
            robot can run.
          </p>
          <div className="hero-actions">
            <a className="button blue" href="/?run=gallery-apophis"><Route size={16} aria-hidden="true" />See how it decided</a>
            <a className="button secondary" href="#analyze-title"><Upload size={16} aria-hidden="true" />Analyze your own clip</a>
          </div>
          <p className="hero-more">Or point the <a href="/live">live camera</a> at something moving, or{" "}
            <a href="/audit">check a video for fake physics</a>.</p>
        </div>
        <figure className="hero-media">
          <div className="hero-stage">
            <LoopVideo src="/hero/apophis-orbit.mp4" poster="/hero/apophis-orbit.jpg" preload="metadata"
                       label="Asteroid Apophis around the Sun, with the law PhysicsLens found: the force arrow points at the Sun in every frame" />
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
          </div>
          <figcaption>
            <ul className="legend hero-legend">
              <li><span className="swatch dot ink" aria-hidden="true" />measured positions</li>
              <li><span className="swatch line model" aria-hidden="true" />the law</li>
              <li><span className="swatch line force" aria-hidden="true" />force</li>
              <li><span className="swatch line velocity" aria-hidden="true" />velocity</li>
            </ul>
            <p className="small muted">Asteroid Apophis over four years, rendered from NASA/JPL positions. The force arrow points at the Sun in every frame.</p>
            <details>
              <summary>How close was the flyby forecast?</summary>
              <p className="small">Carried 3.3 years forward, the law puts the 2029 Earth flyby 13 hours from JPL&apos;s time. Given
                JPL&apos;s own Sun, planets and Moon instead, our N-body integrator lands it within a minute: the hours come from
                what one video can measure.</p>
              {apophis?.description && <p className="small">{apophis.description}</p>}
            </details>
            {apophis?.credit && <p className="small muted">{apophis.credit}</p>}
          </figcaption>
        </figure>
      </section>
      <ul className="stat-strip" aria-label="Measured from video alone">
        {STATS.map((s) => (
          <li key={s.label}>
            <span className="stat-number">{s.value}</span>
            <span className="stat-label">{s.label}{s.note ? ` (${s.note})` : ""}</span>
          </li>
        ))}
      </ul>
    </>
  );
}
