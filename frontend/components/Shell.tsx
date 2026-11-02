"use client";

import { Activity, Camera, ClipboardList, Moon, Orbit, ScanSearch, ShieldCheck, Sun } from "lucide-react";
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

const NAV = [
  { href: "/", label: "Analyze", icon: ScanSearch, also: [] as string[] },
  { href: "/live", label: "Live", icon: Camera, also: [] },
  { href: "/study", label: "Studies", icon: ClipboardList, also: [] },
  { href: "/health", label: "Machine health", icon: Activity, also: [] },
  { href: "/sky", label: "Solar System Lab", icon: Orbit, also: [] },
  // Proof: every result checked against an independent answer, the physics audit and the open dataset.
  { href: "/cases", label: "Proof", icon: ShieldCheck,
    also: ["/audit", "/dataset", "/timing", "/fireball", "/sun", "/satellites", "/occultation", "/asteroids"] },
];

/** The sub-pages of Proof, shown as a row under the page title on each of them. */
export const PROOF_PAGES = [
  { href: "/cases", label: "Real-world cases" },
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
  const path = usePathname();
  // Reading the code moves it from the URL into session storage, so plain links keep it.
  useEffect(() => void accessCode(), []);

  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="shell-bar">
        <div className="shell-inner">
          <a className="wordmark" href="/">
            <Mark />
            <span className="wordmark-text">PhysicsLens</span>
          </a>
          <nav className="shell-nav" aria-label="Main">
            {NAV.map(({ href, label, icon: Icon, also }) => (
              <a key={href} href={href} title={label}
                 aria-current={path === href || also.some((a) => path === a || path === `${a}.html`) ? "page" : undefined}>
                <Icon size={16} strokeWidth={1.8} aria-hidden="true" />
                <span>{label}</span>
              </a>
            ))}
          </nav>
          <div className="shell-spacer" />
          <ThemeToggle />
        </div>
      </header>
      {children}
      <footer className="shell-foot">
        Reasoning on Meta&apos;s open-weight models: Muse Spark Open (served by Baseten) and Muse Glimmer 30B. Tracking by
        SAM 3.1. Every result names the model or the numerical method behind each step.{" "}
        <a href="https://github.com/Athmeeya2006/Physicslens/blob/main/OPEN_WEIGHTS.md">Open-weight evidence</a>
      </footer>
    </>
  );
}
