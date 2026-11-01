"use client";

import { Download } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { formatUnit, withUncertainty } from "@/lib/api";

type Row = {
  id: string; object: string | null; property: string; meaning: string; value: number; std: number; units: string;
  family: string; model: string; heldout_error: number; scale_basis: string; conditions: string; run_id: string;
  clip: string; license: string;
};

// How a length scale was set, in words: a parameter that needs no scale says so instead of naming a source.
const SCALE: Record<string, string> = {
  none: "no scale needed", user: "length given by the person filming", object_size: "object size given",
  gravity: "Earth's gravity assumed as the ruler", known_scale: "scale known from how the video was drawn",
};
// Schema.org Dataset in the page itself, for dataset search engines; the full Croissant file (with each file's hash)
// is generated with the data, at /gallery/croissant.json.
const JSON_LD = JSON.stringify({
  "@context": "https://schema.org/",
  "@type": "Dataset",
  name: "PhysicsLens open dataset: physical parameters measured from video",
  description: "Physical parameters measured from video, one row per parameter, each from a law that forecast motion it was never fitted on.",
  url: "https://physicslens.vercel.app/dataset",
  license: "https://creativecommons.org/licenses/by/4.0/",
  keywords: ["system identification", "physics", "video", "parameter estimation", "sim-to-real"],
  isAccessibleForFree: true,
  creator: { "@type": "Organization", name: "PhysicsLens" },
  distribution: [
    { "@type": "DataDownload", encodingFormat: "text/csv", contentUrl: "https://physicslens.vercel.app/gallery/dataset.csv" },
    { "@type": "DataDownload", encodingFormat: "application/json", contentUrl: "https://physicslens.vercel.app/gallery/dataset.json" },
  ],
});

const scaleText = (basis: string) => SCALE[basis] ?? `reference in view (${basis.replaceAll("_", " ")})`;

export default function DatasetPage() {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [q, setQ] = useState("");

  useEffect(() => {
    fetch("/gallery/dataset.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setRows)
      .catch((e) => setError(String(e)));
  }, []);

  const shown = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    return (rows ?? []).filter((r) => {
      const text = `${r.object} ${r.property} ${r.meaning} ${r.family} ${r.conditions} ${r.clip}`.toLowerCase();
      return words.every((w) => text.includes(w));
    });
  }, [rows, q]);

  return (
    <main id="main">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON_LD }} />
      <header className="page-head">
        <h1>Open dataset</h1>
        <p className="tagline">
          Real-world physical parameters measured from video, each one verified: the law behind it forecast motion it
          never saw. Use them to calibrate a simulator, or as a reference for the same kind of object.
        </p>
      </header>
      <section className="panel stack">
        <div className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <input type="search" placeholder="Filter: damping, cart, pendulum..." value={q} onChange={(e) => setQ(e.target.value)}
            aria-label="Filter the dataset" className="filter" style={{ flex: "1 1 220px" }} />
          <a className="button secondary" href="/gallery/dataset.json" download><Download size={16} aria-hidden="true" /> JSON</a>
          <a className="button secondary" href="/gallery/dataset.csv" download><Download size={16} aria-hidden="true" /> CSV</a>
        </div>
        <p className="small muted">
          <a href="/gallery/DATASET_CARD.md">Dataset card</a> (fields, units, how each value was verified, licence) and{" "}
          <a href="/gallery/croissant.json">Croissant metadata</a>. The table is CC BY 4.0; each clip keeps its own licence.
        </p>
        {error && <p className="error">The dataset could not be loaded ({error}).</p>}
        {rows && (
          <div className="table-wrap"><table className="cards">
            <thead>
              <tr><th>Object</th><th>Property</th><th>Value</th><th>Held-out error</th><th>Scale</th><th>Source</th></tr>
            </thead>
            <tbody>
              {shown.map((r) => {
                const [v, s] = withUncertainty(r.value, r.std);
                const unit = formatUnit(r.units);
                return (
                  <tr key={r.id}>
                    <td data-label="Object">{r.object ?? ""}</td>
                    <td data-label="Property">{r.meaning}</td>
                    <td data-label="Value" className="num">{v} ± {s}{unit ? ` ${unit}` : ""}</td>
                    <td data-label="Held-out error" className="num">{(r.heldout_error * 100).toFixed(1)}%</td>
                    <td data-label="Scale" className="small">{scaleText(r.scale_basis)}</td>
                    <td data-label="Source" className="small"><a href={`/?run=${r.run_id}`}>{r.clip}</a><br /><span className="muted">{r.license}</span></td>
                  </tr>
                );
              })}
            </tbody>
          </table></div>
        )}
        {rows && shown.length === 0 && <p className="muted">Nothing matches that filter.</p>}
        <p className="muted small">
          Values hold for the conditions they were measured in (the object, the surface, the speed range), which each
          row links to. A row enters only when its result was accepted, its law forecast held-out motion within 15%,
          and the value was measured rather than given or assumed. Uncertainties are 1 standard deviation from
          bootstrap refits.
        </p>
      </section>
    </main>
  );
}
