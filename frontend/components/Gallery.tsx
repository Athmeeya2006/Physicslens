"use client";

import type { GalleryItem } from "@/lib/api";

const STATUS: Record<string, string> = { accepted: "verified", ambiguous: "needs more data", no_fit: "no clean equation" };

export function Gallery({ items, onOpen }: { items: GalleryItem[]; onOpen: (id: string) => void }) {
  if (!items.length) return null;
  return (
    <section>
      <h2 className="section-title">Gallery</h2>
      <p className="muted small">Real recordings analyzed end to end by PhysicsLens. Open one to see the ghost, the agent&apos;s reasoning and the exports.</p>
      <div className="gallery">
        {items.map((it) => (
          <button key={it.id} className="gallery-card" onClick={() => onOpen(it.id)}>
            <strong>{it.title}</strong>
            <span className="muted small">{it.description}</span>
            {it.status && <span className={`chip ${it.status === "accepted" ? "accepted" : it.status === "ambiguous" ? "tied" : "rejected"}`}>{STATUS[it.status] ?? it.status}</span>}
            <span className="credit">{it.credit}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
