import { Activity, Camera, Route, Upload } from "lucide-react";

// The first screen: what PhysicsLens found, shown working, before any form. Every number is from a stored gallery
// run (gallery-apophis, gallery-jupiter) checked against NASA/JPL values fetched at run time.

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
        <video src="/gallery/gallery-apophis/share.mp4" poster="/gallery/gallery-apophis/poster.jpg" autoPlay muted loop
               playsInline preload="metadata" aria-label="Asteroid Apophis around the Sun, with the law PhysicsLens found: the force arrow points at the Sun in every frame" />
        <figcaption>
          Asteroid Apophis over four years, rendered from NASA/JPL positions. Red: the force from the law it found,
          pointing at the Sun in every frame; yellow: velocity. Carried 3.3 years forward, the law puts the 2029 Earth
          flyby 13 hours from JPL&apos;s time.
        </figcaption>
      </figure>
    </section>
  );
}
