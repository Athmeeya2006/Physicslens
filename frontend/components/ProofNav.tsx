"use client";

import { usePathname } from "next/navigation";
import { PROOF_PAGES } from "./Shell";

// The row at the top of every Proof page: the three parts of Proof, and on a single case's page the way back to all
// of them, so the cases, the audit, the dataset and each case's own page read as one section.

export function ProofNav({ caseTitle }: { caseTitle?: string }) {
  const path = (usePathname() ?? "").replace(/\.html$/, "");
  return (
    <nav className="proof-nav" aria-label="Proof">
      {PROOF_PAGES.map((p) => (
        <a key={p.href} href={p.href} aria-current={path === p.href ? "page" : undefined}>{p.label}</a>
      ))}
      {caseTitle && <span className="proof-crumb" aria-current="page">Case: {caseTitle}</span>}
    </nav>
  );
}
