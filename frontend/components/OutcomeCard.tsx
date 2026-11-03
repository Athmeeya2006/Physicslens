"use client";

import { type ReactNode, useId, useState } from "react";
import { formatUnit, type Outcome, type Readable } from "@/lib/api";
import { Tex } from "./Equation";
import { StatusChip } from "./StatusChip";

const STATUS: Record<Outcome["status"], [string, string]> = {
  proven: ["Proven", "checked against an independent measurement"],
  partial: ["Verified", "predicted motion it never saw; no independent measurement to compare with yet"],
  "not yet validated": ["Not yet validated", "no law passed the test"],
};

// Summaries show 3 significant figures, more only where needed for the truth and the measurement to differ.
function pair(a: number, b: number): [string, string] {
  for (let p = 3; p < 7; p++) if (a.toPrecision(p) !== b.toPrecision(p)) return [a.toPrecision(p), b.toPrecision(p)];
  return [a.toPrecision(7), b.toPrecision(7)];
}

const pct = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? "n/a" : `${(v * 100).toPrecision(2)}%`);

// What the result means and what to do with it, in plain words, above the equation. Built from the numbers by
// templates (export/outcome.py), never written by the language model. `split` leaves the status and headline to the
// page (a result page puts them full width above the video) and adds the page's own rows, such as the models used.
export function OutcomeCard({ o, exportHref, split = false, tie = false, extra }: {
  o: Outcome; exportHref?: (kind: string) => string; split?: boolean; tie?: boolean; extra?: ReactNode;
}) {
  const titleId = useId();
  const gloss = outcomeGloss(o);
  return (
    <section className={`outcome${split ? " outcome-split" : ""}`} aria-labelledby={titleId}>
      {!split && (
        <div className="outcome-head">
          <p className="outcome-status"><StatusChip status={STATUS[o.status][0]} /> <span className="small">{gloss}</span></p>
          <p className="outcome-headline">{formatUnit(o.headline)}</p>
        </div>
      )}
      <div className={`callout${tie ? " callout-tie" : ""}`}>
        <p className="callout-label" id={titleId}>What to do</p>
        <p>{o.action}</p>
      </div>
      <dl className="outcome-facts">
        {split && <><dt>Status</dt><dd>{STATUS[o.status][0]}: {gloss}</dd></>}
        {o.confidence && <><dt>Confidence</dt><dd>{formatUnit(o.confidence)}</dd></>}
        {o.checked_against?.text && <><dt>Checked against</dt><dd>{o.checked_against.source}: {o.checked_against.text}</dd></>}
        {o.checked_against && !o.checked_against.text && (
          <><dt>Checked against</dt><dd>
            {o.checked_against.source}: {o.checked_against.name} ={" "}
            {pair(o.checked_against.truth, o.checked_against.measured)[0]}, measured {pair(o.checked_against.truth, o.checked_against.measured)[1]}{" "}
            ({(o.checked_against.relative_error * 100).toFixed(2)}% off,{" "}
            {o.checked_against.within_interval ? "inside" : "outside"} the 95% interval)
          </dd></>
        )}
        {o.limits && o.limits.length > 0 && <><dt>Not reproduced</dt><dd>{o.limits.join("; ")}</dd></>}
        {extra}
        {o.artifacts.length > 0 && exportHref && (
          <><dt>Files</dt><dd className="outcome-files">
            {o.artifacts.map((a) => <a key={a.kind} href={exportHref(a.kind)} download>{a.kind.toUpperCase()}</a>)}
          </dd></>
        )}
      </dl>
    </section>
  );
}

// Verified, with a truth to compare: say it was checked, and that the truth sits just outside the interval.
// With a stated limit, the limit is the reason (shown below); the gloss only says it was checked.
export function outcomeGloss(o: Outcome): string {
  const base = STATUS[o.status][1];
  return o.status === "not yet validated" && o.checked_against
    ? "the laws tie, but what they agree on was checked (below)"
    : o.status === "partial" && o.checked_against
    ? (o.limits?.length ? "checked against independent measurements; part of it is not reproduced (below)"
      : "predicted motion it never saw; an independent measurement lies just outside its 95% interval")
    : base;
}

export const outcomeLabel = (o: Outcome) => STATUS[o.status][0];

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
        <details open>
          <summary className="small">Rival laws and why they lost ({r.rivals.length})</summary>
          <ul className="small">
            {r.rivals.map((x) => <li key={x.title}><strong>{x.title}:</strong> {x.why}</li>)}
          </ul>
        </details>
      )}
    </div>
  );
}
