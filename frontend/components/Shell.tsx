"use client";

import {
  Activity, Camera, ChevronDown, ClipboardList, Database, FileCheck2, Menu, Moon, ScanSearch, ShieldCheck, Sun, Telescope, X,
  type LucideIcon,
} from "lucide-react";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { accessCode, useBackend } from "@/lib/api";

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

interface Item { href: string; label: string; icon: LucideIcon; text: string }

// Evidence: what has been shown and how. Tools: what you can run yourself.
const EVIDENCE: Item[] = [
  { href: "/cases", label: "Real-world cases", icon: Telescope, text: "Each answer checked against an independent one" },
  { href: "/predictions", label: "Predictions", icon: FileCheck2, text: "Registered before the answer exists, scored after" },
  { href: "/audit", label: "Physics audit", icon: ShieldCheck, text: "Flags physically impossible motion when it can judge" },
  { href: "/dataset", label: "Open dataset", icon: Database, text: "Every gallery result as data, with its licence" },
];
const TOOLS: Item[] = [
  { href: "/#analyze-title", label: "Analyze a clip", icon: ScanSearch, text: "Upload a video: the equation, verified" },
  { href: "/live", label: "Live camera", icon: Camera, text: "The law converging as you film" },
  { href: "/study", label: "Studies", icon: ClipboardList, text: "Give a goal; the agent plans the recordings" },
  { href: "/health", label: "Machine health", icon: Activity, text: "Two recordings of one part: did it drift?" },
];
// Pages that belong to Evidence (the case pages too), for the header's current-section mark.
export const EVIDENCE_PATHS = ["/cases", "/predictions", "/audit", "/dataset", "/robot", "/flight", "/timing", "/fireball", "/sun",
  "/satellites", "/occultation", "/asteroids"];
const TOOL_PATHS = ["/live", "/study", "/health"];

/** The sub-pages of Evidence, shown as a slim underline tab bar under the header on each of them. */
export const PROOF_PAGES = EVIDENCE.map(({ href, label }) => ({ href, label }));

function ThemeToggle() {
  // Light by default; dark is the one alternative, remembered on this device. The whole site follows it.
  const [dark, setDark] = useState(false);
  useEffect(() => setDark(document.documentElement.dataset.theme === "dark"), []);
  const flip = () => {
    const next = !dark;
    setDark(next);
    if (next) document.documentElement.dataset.theme = "dark";
    else delete document.documentElement.dataset.theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", next ? "#12110f" : "#f6f3ec");
    try {
      localStorage.setItem("physicslens-theme", next ? "dark" : "light");
    } catch {
      /* storage blocked: the choice lasts for this page only */
    }
  };
  return (
    <button type="button" className="theme-toggle" onClick={flip} aria-pressed={dark}
            title={dark ? "Switch to light" : "Switch to dark"} aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}>
      {dark ? <Sun size={17} aria-hidden="true" /> : <Moon size={17} aria-hidden="true" />}
    </button>
  );
}

function ServerStatus() {
  const online = useBackend();
  if (online === null) return <span className="server-status" aria-live="polite"><span className="dot wait" aria-hidden="true" />Looking for the server</span>;
  return (
    <span className="server-status" aria-live="polite">
      <span className={`dot ${online ? "on" : "off"}`} aria-hidden="true" />
      {online ? "Analysis server online" : "Server between sessions"}
    </span>
  );
}

// Between server sessions the main button opens a complete analysis instead of an upload form that cannot run.
function PrimaryAction({ className, onClick }: { className: string; onClick?: () => void }) {
  const online = useBackend();
  return online === false
    ? <a className={className} href="/?run=gallery-apophis" onClick={onClick}>See a full analysis</a>
    : <a className={className} href="/#analyze-title" onClick={onClick}>Analyze a clip</a>;
}

function DropMenu({ id, label, items, current, open, setOpen }: {
  id: string; label: string; items: Item[]; current: boolean; open: string | null; setOpen: (v: string | null) => void;
}) {
  const isOpen = open === id;
  const listRef = useRef<HTMLUListElement>(null);
  const onKey = (e: React.KeyboardEvent) => {
    const links = Array.from(listRef.current?.querySelectorAll("a") ?? []);
    const at = links.indexOf(document.activeElement as HTMLAnchorElement);
    if (e.key === "ArrowDown") { e.preventDefault(); if (!isOpen) setOpen(id); (links[at + 1] ?? links[0])?.focus(); }
    if (e.key === "ArrowUp") { e.preventDefault(); (links[at - 1] ?? links[links.length - 1])?.focus(); }
  };
  return (
    <div className="nav-group" onKeyDown={onKey}>
      <button type="button" className="nav-group-btn" aria-expanded={isOpen} aria-controls={`nav-${id}`}
              aria-current={current ? "page" : undefined} onClick={() => setOpen(isOpen ? null : id)}>
        {label} <ChevronDown size={15} aria-hidden="true" />
      </button>
      <ul id={`nav-${id}`} ref={listRef} className="nav-drop" hidden={!isOpen}>
        {items.map(({ href, label: l, icon: Icon, text }) => (
          <li key={href}>
            <a href={href} onClick={() => setOpen(null)}>
              <Icon size={18} aria-hidden="true" />
              <span><strong>{l}</strong><span className="nav-drop-text">{text}</span></span>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const path = (usePathname() ?? "/").replace(/\.html$/, "").replace(/^\/index$/, "/");
  // Reading the code moves it from the URL into session storage, so plain links keep it.
  useEffect(() => void accessCode(), []);
  const [sheet, setSheet] = useState(false);
  const [drop, setDrop] = useState<string | null>(null);
  const navRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") { setDrop(null); setSheet(false); } };
    const onClick = (e: MouseEvent) => { if (!navRef.current?.contains(e.target as Node)) setDrop(null); };
    window.addEventListener("keydown", onKey);
    window.addEventListener("mousedown", onClick);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("mousedown", onClick); };
  }, []);
  const cur = (href: string) => (path === href ? "page" : undefined);

  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="shell-bar">
        <div className="shell-inner">
          <a className="wordmark" href="/">
            <Mark />
            <span className="wordmark-text">PhysicsLens</span>
          </a>
          <nav ref={navRef} className="shell-nav" aria-label="Main">
            <a href="/#results" className="nav-link" aria-current={cur("/")}>Results</a>
            <DropMenu id="evidence" label="Evidence" items={EVIDENCE} current={EVIDENCE_PATHS.includes(path)} open={drop} setOpen={setDrop} />
            <a href="/sky" className="nav-link" aria-current={cur("/sky")}>Solar System Lab</a>
            <DropMenu id="tools" label="Tools" items={TOOLS} current={TOOL_PATHS.includes(path)} open={drop} setOpen={setDrop} />
          </nav>
          <div className="shell-spacer" />
          <ServerStatus />
          <ThemeToggle />
          <PrimaryAction className="button primary-dark shell-cta" />
          <button type="button" className="menu-toggle" aria-expanded={sheet} aria-controls="shell-sheet"
                  aria-label={sheet ? "Close the menu" : "Open the menu"} onClick={() => setSheet((o) => !o)}>
            {sheet ? <X size={22} aria-hidden="true" /> : <Menu size={22} aria-hidden="true" />}
          </button>
        </div>
      </header>
      <nav id="shell-sheet" className="shell-sheet" aria-label="Main" hidden={!sheet}>
        <a href="/#results" onClick={() => setSheet(false)}>Results</a>
        <a href="/sky" onClick={() => setSheet(false)}>Solar System Lab</a>
        {([["Evidence", EVIDENCE], ["Tools", TOOLS]] as [string, Item[]][]).map(([title, items]) => (
          <div key={title} className="sheet-group">
            <p className="sheet-label">{title}</p>
            {items.map(({ href, label, icon: Icon }) => (
              <a key={href} href={href} onClick={() => setSheet(false)}><Icon size={18} aria-hidden="true" /> {label}</a>
            ))}
          </div>
        ))}
        <PrimaryAction className="button primary-dark" onClick={() => setSheet(false)} />
      </nav>
      {children}
      <footer className="shell-foot">
        <span>Reasoning on Meta&apos;s open-weight models: Muse Spark Open (served by Baseten) and Muse Glimmer 30B; tracking by SAM 3.1.
          Every result names the model or the numerical method behind each step.</span>
        <span className="foot-links">
          <a href="https://github.com/Athmeeya2006/Physicslens/blob/main/OPEN_WEIGHTS.md">Open-weight evidence</a>
          <a href="/glossary">Glossary</a>
          <a href="https://github.com/Athmeeya2006/Physicslens">Source on GitHub</a>
        </span>
      </footer>
    </>
  );
}
