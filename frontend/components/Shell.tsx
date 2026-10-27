"use client";

import { Activity, Camera, ScanSearch } from "lucide-react";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

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
  { href: "/", label: "Analyze", icon: ScanSearch },
  { href: "/live", label: "Live", icon: Camera },
  { href: "/health", label: "Machine health", icon: Activity },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const [code, setCode] = useState<string | null>(null);
  useEffect(() => setCode(new URLSearchParams(window.location.search).get("code")), []);
  const withCode = (href: string) => (code ? `${href}?code=${encodeURIComponent(code)}` : href);

  return (
    <>
      <header className="shell-bar">
        <div className="shell-inner">
          <a className="wordmark" href={withCode("/")}>
            <Mark />
            PhysicsLens
          </a>
          <nav className="shell-nav" aria-label="Main">
            {NAV.map(({ href, label, icon: Icon }) => (
              <a key={href} href={withCode(href)} aria-current={path === href ? "page" : undefined}>
                <Icon size={16} strokeWidth={1.8} aria-hidden="true" />
                <span>{label}</span>
              </a>
            ))}
          </nav>
          <div className="shell-spacer" />
          <span className="shell-note">System identification from a phone camera</span>
        </div>
      </header>
      {children}
    </>
  );
}
