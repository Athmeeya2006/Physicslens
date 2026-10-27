"use client";

import type { GalleryItem } from "@/lib/api";
import { posterUrl } from "@/lib/api";

const STATUS: Record<string, { text: string; cls: string }> = {
  accepted: { text: "Verified equation", cls: "accepted" },
  ambiguous: { text: "Asks for an experiment", cls: "tied" },
  no_fit: { text: "No clean equation", cls: "rejected" },
};

export function Gallery({ items, onOpen, activeId }: { items: GalleryItem[]; onOpen: (id: string) => void; activeId?: string }) {
  if (!items.length) return null;
  return (
    <section aria-labelledby="gallery-title" className="stack">
      <div>
        <h2 id="gallery-title">Real recordings, analyzed</h2>
        <p className="muted small">Each one went through the full pipeline with Meta&apos;s models. Open one to see the ghost, the decision trail and the exports.</p>
      </div>
      <ul className="gallery-list">
        {items.map((it) => {
          const status = it.status ? STATUS[it.status] : undefined;
          return (
            <li key={it.id}>
              <button className="gallery-item" onClick={() => onOpen(it.id)} aria-current={activeId === it.id ? "true" : undefined}>
                <img src={posterUrl(it.id)} alt={`Frame from the ${it.object ?? "analyzed"} clip with the model ghost overlaid`} loading="lazy" />
                <span>
                  <span className="g-title">{it.title}</span>
                  <span className="g-desc" style={{ display: "block" }}>{it.description}</span>
                  <span className="g-meta">
                    {status && <span className={`verdict ${status.cls}`}>{status.text}</span>}
                    <span>{it.credit}</span>
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
