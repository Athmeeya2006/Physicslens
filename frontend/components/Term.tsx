"use client";

import { useId } from "react";
import { GLOSSARY } from "@/lib/glossary";

// A technical word with its plain meaning on hover (title) and for screen readers (aria-describedby).
export function Term({ k, children }: { k: keyof typeof GLOSSARY; children?: React.ReactNode }) {
  const id = useId();
  const g = GLOSSARY[k];
  return (
    <>
      <abbr className="term-tip" title={g.text} aria-describedby={id}>{children ?? g.term}</abbr>
      <span id={id} className="visually-hidden">{g.text}</span>
    </>
  );
}
