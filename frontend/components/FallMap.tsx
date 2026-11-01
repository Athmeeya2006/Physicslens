"use client";

import "leaflet/dist/leaflet.css";
import { useEffect, useRef } from "react";

// The strewn field on OpenStreetMap: the bright flight's ground track, the simulated stones (coloured by mass), the
// median spot per mass joined into the fall line, and the stones actually found.

export interface FallMapData {
  track: [number, number][]; // [lat, lon] from the first light to the luminous end
  points: [number, number, number, number][]; // lon, lat, mass kg, shape
  line: { mass_kg: number; lat: number; lon: number }[];
  finds: { lat: number; lon: number; mass_g: number; place: string; explained_by: string | null }[];
}

const MASS_COLOURS: [number, string][] = [[0.01, "#93c5fd"], [0.03, "#60a5fa"], [0.08, "#3b82f6"], [0.2, "#a78bfa"], [0.4, "#f472b6"], [Infinity, "#f59e0b"]];
const colourFor = (kg: number) => MASS_COLOURS.find(([lim]) => kg <= lim)?.[1] ?? "#f59e0b";

export function FallMap({ data, label }: { data: FallMapData; label: string }) {
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let map: import("leaflet").Map | null = null;
    let cancelled = false;
    import("leaflet").then((L) => {
      if (cancelled || !box.current) return;
      map = L.map(box.current, { scrollWheelZoom: false });
      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 18, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).addTo(map);
      L.polyline(data.track, { color: "#ef4444", weight: 3, opacity: 0.8 }).addTo(map).bindTooltip("bright flight (ground track)");
      for (const [lon, lat, kg] of data.points) {
        L.circleMarker([lat, lon], { radius: 2.5, stroke: false, fillColor: colourFor(kg), fillOpacity: 0.55 }).addTo(map);
      }
      L.polyline(data.line.map((b) => [b.lat, b.lon] as [number, number]), { color: "#e5e7eb", weight: 2, dashArray: "6 4" })
        .addTo(map).bindTooltip("fall line: the most likely spot for each mass");
      for (const f of data.finds) {
        L.circleMarker([f.lat, f.lon], { radius: 7, color: "#111827", weight: 2, fillColor: f.explained_by ? "#22c55e" : "#ef4444", fillOpacity: 0.95 })
          .addTo(map).bindTooltip(`${f.mass_g} g found (${f.place})${f.explained_by ? "" : ": not reproduced"}`);
      }
      // Frame the stones (found and simulated) and the end of the bright flight; the track runs off to the west.
      const all = [data.track[data.track.length - 1], ...data.finds.map((f) => [f.lat, f.lon] as [number, number]),
        ...data.line.map((b) => [b.lat, b.lon] as [number, number])];
      map.fitBounds(L.latLngBounds(all), { padding: [30, 30] });
    });
    return () => {
      cancelled = true;
      map?.remove();
    };
  }, [data]);

  return (
    <figure className="fall-map">
      <div ref={box} className="fall-map-box" role="img" aria-label={label} />
      <figcaption className="small">
        <span className="key"><span className="swatch-line" style={{ background: "#ef4444" }} />bright flight</span>
        <span className="key"><span className="swatch-line dashed" />fall line</span>
        {MASS_COLOURS.map(([lim, c], i) => (
          <span key={c} className="key"><span className="dot" style={{ background: c }} />
            {i === MASS_COLOURS.length - 1 ? `over ${MASS_COLOURS[i - 1][0] * 1000} g` : `up to ${lim * 1000} g`}</span>
        ))}
        <span className="key"><span className="dot ring" style={{ background: "#22c55e" }} />stone found</span>
      </figcaption>
    </figure>
  );
}
