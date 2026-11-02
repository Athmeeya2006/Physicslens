"use client";

import { type ReactNode, useId, useRef } from "react";

// Tabs with the keyboard behaviour screen-reader users expect: arrow keys move between tabs, Home and End jump to the
// ends, and only the chosen panel is in the page. The caller keeps which tab is open (so it can open one itself).

export interface Tab { id: string; label: ReactNode; panel: ReactNode }

export function Tabs({ tabs, active, onChange, label }: { tabs: Tab[]; active: string; onChange: (id: string) => void; label: string }) {
  const base = useId();
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const current = tabs.find((t) => t.id === active) ?? tabs[0];
  const move = (i: number) => {
    const k = (i + tabs.length) % tabs.length;
    onChange(tabs[k].id);
    refs.current[k]?.focus();
  };
  return (
    <div className="tabs">
      <div role="tablist" aria-label={label} className="tab-list">
        {tabs.map((t, i) => (
          <button key={t.id} ref={(el) => { refs.current[i] = el; }} type="button" role="tab" id={`${base}-${t.id}`}
                  aria-selected={t.id === current.id} aria-controls={`${base}-${t.id}-panel`} tabIndex={t.id === current.id ? 0 : -1}
                  className="tab" onClick={() => onChange(t.id)}
                  onKeyDown={(e) => {
                    if (e.key === "ArrowRight") { e.preventDefault(); move(i + 1); }
                    else if (e.key === "ArrowLeft") { e.preventDefault(); move(i - 1); }
                    else if (e.key === "Home") { e.preventDefault(); move(0); }
                    else if (e.key === "End") { e.preventDefault(); move(tabs.length - 1); }
                  }}>
            {t.label}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`${base}-${current.id}-panel`} aria-labelledby={`${base}-${current.id}`} className="tab-panel">
        {current.panel}
      </div>
    </div>
  );
}
