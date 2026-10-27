"use client";

import katex from "katex";
import { useMemo } from "react";

export function Tex({ tex, display = false }: { tex: string; display?: boolean }) {
  // throwOnError=false: a malformed string renders red instead of crashing the result view.
  const html = useMemo(() => katex.renderToString(tex, { displayMode: display, throwOnError: false }), [tex, display]);
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}
