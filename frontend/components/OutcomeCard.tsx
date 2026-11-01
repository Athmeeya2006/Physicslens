"use client";

import { useState } from "react";
import { formatUnit, type Outcome, type Readable } from "@/lib/api";
import { Tex } from "./Equation";

const STATUS: Record<Outcome["status"], [string, string]> = {
  proven: ["Proven", "checked against an independent measurement"],
  partial: ["Verified", "predicted motion it never saw; no independent measurement to compare with yet"],
  "not yet validated": ["Not yet validated", "no law passed the test"],
};

const pct = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? "n/a" : `${(v * 100).toPrecision(2)}%`);

// What the result means and what to do with it, in plain words, above the equation. Built from the numbers by
// templates (export/outcome.py), never written by the language model.
export function OutcomeCard({ o, exportHref }: { o: Outcome; exportHref?: (kind: string) => string }) {
  const [label, base] = STATUS[o.status];
  // Verified, with a truth to compare: say it was checked, and that the truth sits just outside the interval.
  const gloss = o.status === "partial" && o.checked_against
    ? "predicted motion it never saw; an independent measurement lies just outside its 95% interval"
    : base;
  return (
    <section className={`panel outcome outcome-${o.status.replaceAll(" ", "-")}`} aria-labelledby="outcome-title">
      <p className="outcome-status"><strong id="outcome-title">{label}</strong> <span className="muted small">{gloss}</span></p>
      <p className="outcome-headline">{o.headline}</p>
      <p><strong>What to do:</strong> {o.action}</p>
      {o.confidence && <p className="small"><strong>Confidence:</strong> {formatUnit(o.confidence)}</p>}
      {o.checked_against && (
        <p className="small">
          <strong>Checked against:</strong> {o.checked_against.source}: {o.checked_against.name} ={" "}
          {o.checked_against.truth.toPrecision(5)}, measured {o.checked_against.measured.toPrecision(5)}{" "}
          ({(o.checked_against.relative_error * 100).toFixed(2)}% off,{" "}
          {o.checked_against.within_interval ? "inside" : "outside"} the 95% interval)
        </p>
      )}
      {o.artifacts.length > 0 && exportHref && (
        <p className="small outcome-files">
          <strong>Files:</strong>{" "}
          {o.artifacts.map((a) => <a key={a.kind} href={exportHref(a.kind)} download>{a.kind.toUpperCase()}</a>)}
        </p>
      )}
    </section>
  );
}

// The law made readable: the fitted numbers written in, one sentence per term, and what happens without each term
// (the special case without it was fitted too, so the drift shown is measured). Then the rivals and why they lost.
export function ReadableLaw({ r }: { r: Readable }) {
  const [removed, setRemoved] = useState<number | null>(null);
  const term = removed !== null ? r.terms[removed] : null;
  return (
    <div className="stack readable" style={{ gap: 8 }}>
      <p className="small muted">With the fitted numbers:</p>
      <div className="law-scroll"><Tex tex={r.latex_numeric} display /></div>
      {r.terms.length > 0 && (
        <ul className="terms">
          {r.terms.map((t, i) => (
            <li key={t.latex}>
              {t.without ? (
                <button type="button" className={`term ${removed === i ? "off" : ""}`} aria-pressed={removed === i}
                        onClick={() => setRemoved(removed === i ? null : i)}
                        title="Remove this term and see the forecast without it">
                  <Tex tex={t.latex} />
                </button>
              ) : <span className="term fixed"><Tex tex={t.latex} /></span>}{" "}
              <span>{t.text}</span>
            </li>
          ))}
        </ul>
      )}
      {term && (
        <p className="note" role="status">
          Without this term the law is {term.without_title ?? "the simpler law"}, which forecasts the held-out quarter
          within {pct(term.without_heldout)} (with it: {pct(r.heldout)}).{" "}
          {term.without_heldout != null && r.heldout != null && term.without_heldout > r.heldout
            ? "The forecast drifts without it: the term is needed."
            : "The last quarter alone hardly needs it; it was kept because it describes the whole motion better by BIC."}
        </p>
      )}
      {r.terms.length > 0 && r.terms.some((t) => t.without) && !term && (
        <p className="small muted">Tap a term with a border to take it out and see what the forecast does without it.</p>
      )}
      {r.written_by_agent && <p className="small muted">This law was written by the agent and checked like any other.</p>}
      {r.rivals.length > 0 && (
        <details>
          <summary className="small">Rival laws and why they lost ({r.rivals.length})</summary>
          <ul className="small">
            {r.rivals.map((x) => <li key={x.title}><strong>{x.title}:</strong> {x.why}</li>)}
          </ul>
        </details>
      )}
    </div>
  );
}
