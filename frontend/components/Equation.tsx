"use client";

import katex from "katex";
import { useMemo } from "react";

/** A plain-text reading of a law for screen readers ("theta double-dot = minus g over L sine theta"): KaTeX's MathML
 * is read well by some readers and not at all by others, so every typeset law also carries words. */
export function speak(tex: string): string {
  const FN: Record<string, string> = { sin: "sine", cos: "cosine", tan: "tangent", tanh: "hyperbolic tangent",
    exp: "exponential", ln: "log", sqrt: "square root of", arctan: "arctangent", sgn: "sign of", pm: "plus or minus" };
  let s = tex.replace(/\\operatorname\{([^{}]*)\}/g, "\\$1").replace(/\\(mathrm|mathbf|text)\{([^{}]*)\}/g, "$2");
  // Powers first, so a fraction's braces hold no others: x^{3/2} reads "x to the power 3/2".
  s = s.replace(/\^\{2\}|\^2/g, " squared ").replace(/\^\{3\}|\^3/g, " cubed ").replace(/\^\{([^{}]*)\}/g, " to the power $1 ")
    .replace(/\^([A-Za-z0-9])/g, " to the power $1 ");
  for (let i = 0; i < 4; i++) s = s.replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, " ($1) over ($2) ");
  s = s.replace(/\\ddot\s*\{?\s*\\?([A-Za-z]+)\}?/g, " $1 double dot ").replace(/\\dot\s*\{?\s*\\?([A-Za-z]+)\}?/g, " $1 dot ");
  s = s.replace(/\\(left|right|quad|qquad)\b|\\[,;! ]/g, " ").replace(/\\odot/g, " sun ");
  s = s.replace(/\\([A-Za-z]+)/g, (_m, w: string) => ` ${FN[w] ?? w} `);  // \theta reads "theta"
  s = s.replace(/_\{([^{}]*)\}/g, " $1 ").replace(/_([A-Za-z0-9])/g, " $1 ");
  s = s.replace(/=/g, " equals ").replace(/\+/g, " plus ").replace(/(^|[\s(])-/g, "$1 minus ").replace(/-/g, " minus ")
    .replace(/[{}|]/g, " ");
  return s.replace(/\s+/g, " ").replace(/\(\s+/g, "(").replace(/\s+\)/g, ")").trim();
}

export function Tex({ tex, display = false }: { tex: string; display?: boolean }) {
  // throwOnError=false: a malformed string renders red instead of crashing the result view.
  const html = useMemo(() => katex.renderToString(tex, { displayMode: display, throwOnError: false }), [tex, display]);
  return <span role="math" aria-label={speak(tex)} dangerouslySetInnerHTML={{ __html: html }} />;
}

// A law the agent wrote: one line per axis at full size, so fractions stay readable on a phone.
export function Law({ tex }: { tex: string }) {
  return (
    <span className="law">
      {tex.split(/,\\quad\s*/).map((line, i) => <Tex key={i} tex={line} display />)}
    </span>
  );
}
