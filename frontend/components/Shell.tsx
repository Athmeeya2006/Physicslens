"use client";

import { Activity, Camera, ClipboardList, ScanSearch } from "lucide-react";
import { usePathname } from "next/navigation";
import { useEffect } from "react";
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
  { href: "/", label: "Analyze", icon: ScanSearch },
  { href: "/study", label: "Studies", icon: ClipboardList },
  { href: "/live", label: "Live", icon: Camera },
  { href: "/health", label: "Machine health", icon: Activity },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  // Reading the code moves it from the URL into session storage, so plain links keep it.
  useEffect(() => void accessCode(), []);

  return (
    <>
      <header className="shell-bar">
        <div className="shell-inner">
          <a className="wordmark" href="/">
            <Mark />
            PhysicsLens
          </a>
          <nav className="shell-nav" aria-label="Main">
            {NAV.map(({ href, label, icon: Icon }) => (
              <a key={href} href={href} aria-current={path === href ? "page" : undefined}>
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
