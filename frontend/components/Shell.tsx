"use client";

import { FlaskConical, Menu, Moon, ScanSearch, ShieldCheck, Sun, X } from "lucide-react";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { accessCode } from "@/lib/api";

// The mark: a lens with a swing arc and a bob, drawn once in the shell's stroke weight.
function Mark() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <path d="M12 4.5 L8.2 15.6" strokeLinecap="round" />
      <path d="M5.6 13.4 A 9 9 0 0 0 18.4 13.4" strokeDasharray="1.6 2.2" strokeLinecap="round" />
      <circle cx="8.2" cy="15.6" r="2" fill="currentColor" stroke="none" />
    </svg>
  );
}

// The main pages, always in view on a wide screen (nobody should have to open a menu to find them); the phone menu
// lists every page under three labels.
const FLAT = [
  { href: "/", label: "Analyze", also: [] as string[] },
  { href: "/live", label: "Live", also: [] },
  { href: "/study", label: "Studies", also: [] },
  { href: "/cases", label: "Real-world cases", also: ["/robot", "/timing", "/fireball", "/sun", "/satellites", "/occultation", "/asteroids"] },
  { href: "/predictions", label: "Predictions", also: [] },
  { href: "/audit", label: "Physics audit", also: [] },
  { href: "/health", label: "Machine health", also: [] },
  { href: "/sky", label: "Solar System Lab", also: [] },
  { href: "/dataset", label: "Dataset", also: [] },
];

// Three groups for the phone menu: what you can do, what has been shown, and the tools behind it.
const NAV = [
  { id: "analyze", label: "Analyze", icon: ScanSearch, links: [
    { href: "/", label: "Upload a video" }, { href: "/live", label: "Live camera" }, { href: "/study", label: "Studies" },
  ] },
  { id: "results", label: "Results", icon: ShieldCheck, links: [
    { href: "/cases", label: "Real-world cases" }, { href: "/predictions", label: "Predictions" },
    { href: "/#gallery-title", label: "Gallery" }, { href: "/dataset", label: "Open dataset" },
  ] },
  { id: "tools", label: "Tools", icon: FlaskConical, links: [
    { href: "/robot", label: "Robot arm" }, { href: "/audit", label: "Physics audit" }, { href: "/health", label: "Machine health" }, { href: "/sky", label: "Solar System Lab" },
    { href: "/fireball", label: "Fireballs" }, { href: "/timing", label: "Transit timing" }, { href: "/sun", label: "The Sun" },
    { href: "/satellites", label: "Satellites" }, { href: "/occultation", label: "Occultations" }, { href: "/asteroids", label: "Asteroids" },
  ] },
];

/** The sub-pages of Proof, shown as a row under the page title on each of them. */
export const PROOF_PAGES = [
  { href: "/cases", label: "Real-world cases" },
  { href: "/predictions", label: "Predictions" },
  { href: "/audit", label: "Physics audit" },
  { href: "/dataset", label: "Open dataset" },
];

function ThemeToggle() {
  // Dark by default; light is the one alternative, remembered on this device. The whole site follows it.
  const [light, setLight] = useState(false);
  useEffect(() => setLight(document.documentElement.dataset.theme === "light"), []);
  const flip = () => {
    const next = !light;
    setLight(next);
    if (next) document.documentElement.dataset.theme = "light";
    else delete document.documentElement.dataset.theme;
    try {
      localStorage.setItem("physicslens-theme", next ? "light" : "dark");
    } catch {
      /* storage blocked: the choice lasts for this page only */
    }
  };
  return (
    <button type="button" className="theme-toggle" onClick={flip} aria-pressed={light}
            title={light ? "Switch to dark" : "Switch to light"} aria-label={light ? "Switch to dark theme" : "Switch to light theme"}>
      {light ? <Moon size={16} aria-hidden="true" /> : <Sun size={16} aria-hidden="true" />}
    </button>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const path = (usePathname() ?? "/").replace(/\.html$/, "").replace(/^\/index$/, "/");
  // Reading the code moves it from the URL into session storage, so plain links keep it.
  useEffect(() => void accessCode(), []);
  // The bar slims down and lifts once the page scrolls; on a narrow screen the sections fold into a menu.
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);
  const here = (href: string) => href === path;
  const groupHere = (g: (typeof NAV)[number]) => g.links.some((l) => here(l.href));
  // Phone: every group as a labelled list, never icons alone.
  const menu = NAV.map((g) => (
    <div key={g.id} className="menu-group">
      <p className="menu-group-label">{g.label}</p>
      {g.links.map((l) => (
        <a key={l.href} href={l.href} aria-current={here(l.href) ? "page" : undefined} onClick={() => setOpen(false)}>{l.label}</a>
      ))}
    </div>
  ));

  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <header className={`shell-bar${scrolled ? " scrolled" : ""}${open ? " menu-open" : ""}`}>
        <div className="shell-inner">
          <a className="wordmark" href="/">
            <Mark />
            <span className="wordmark-text">PhysicsLens</span>
          </a>
          <nav className="shell-nav" aria-label="Main">
            {FLAT.map((l) => (
              <a key={l.href} href={l.href} aria-current={here(l.href) || l.also.includes(path) ? "page" : undefined}>{l.label}</a>
            ))}
          </nav>
          <div className="shell-spacer" />
          <ThemeToggle />
          <button type="button" className="menu-toggle" aria-expanded={open} aria-controls="shell-menu"
                  aria-label={open ? "Close the menu" : "Open the menu"} onClick={() => setOpen((o) => !o)}>
            {open ? <X size={20} aria-hidden="true" /> : <Menu size={20} aria-hidden="true" />}
          </button>
        </div>
        <nav id="shell-menu" className="shell-menu" aria-label="Main" hidden={!open}>{menu}</nav>
      </header>
      {children}
      <footer className="shell-foot">
        Reasoning on Meta&apos;s open-weight models: Muse Spark Open (served by Baseten) and Muse Glimmer 30B. Tracking by
        SAM 3.1. Every result names the model or the numerical method behind each step.{" "}
        <a href="https://github.com/Athmeeya2006/Physicslens/blob/main/OPEN_WEIGHTS.md">Open-weight evidence</a> · <a href="/glossary">Glossary</a>
      </footer>
    </>
  );
}
