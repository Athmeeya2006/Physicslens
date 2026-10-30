"use client";

import katex from "katex";
import { useMemo } from "react";

export function Tex({ tex, display = false }: { tex: string; display?: boolean }) {
  // throwOnError=false: a malformed string renders red instead of crashing the result view.
  const html = useMemo(() => katex.renderToString(tex, { displayMode: display, throwOnError: false }), [tex, display]);
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

// A law the agent wrote: one line per axis at full size, so fractions stay readable on a phone.
export function Law({ tex }: { tex: string }) {
  return (
    <span className="law">
      {tex.split(/,\\quad\s*/).map((line, i) => <Tex key={i} tex={line} display />)}
    </span>
  );
}
