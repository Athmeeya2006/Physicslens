"use client";

import { useEffect, useRef, useState } from "react";

// A muted looping clip that plays only while on screen: several looping videos at once would make a phone stutter.
// Visitors whose system asks for reduced motion get the poster and play controls instead of autoplay.
export function LoopVideo({ src, poster, label, preload = "none" }: {
  src: string; poster: string; label: string; preload?: "none" | "metadata";
}) {
  const ref = useRef<HTMLVideoElement>(null);
  const [still, setStill] = useState(false);
  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    setStill(reduce);
    if (reduce) return;
    const io = new IntersectionObserver(([e]) => (e.isIntersecting ? v.play().catch(() => {}) : v.pause()), { threshold: 0.3 });
    io.observe(v);
    return () => io.disconnect();
  }, []);
  return <video ref={ref} src={src} poster={poster} muted loop playsInline preload={preload} controls={still} aria-label={label} />;
}
