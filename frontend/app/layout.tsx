import type { Metadata, Viewport } from "next";
import { Schibsted_Grotesk } from "next/font/google";
import "katex/dist/katex.min.css";
import { Shell } from "@/components/Shell";
import "./globals.css";

// Self-hosted at build time by next/font; a grotesk with real character and tabular figures for data.
const ui = Schibsted_Grotesk({ subsets: ["latin"], variable: "--font-ui", display: "swap" });

export const metadata: Metadata = {
  metadataBase: new URL("https://physicslens.vercel.app"),
  title: "PhysicsLens",
  description: "Film something moving and get the equation that governs it, verified against the real motion and exportable to MuJoCo.",
  openGraph: {
    title: "PhysicsLens",
    description: "Film something moving. Get its equation, verified against motion it never saw, and export it to MuJoCo.",
    images: [{ url: "/og.jpg", width: 1200, height: 630 }],
  },
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#000000" };

// Applied before the first paint, so a visitor who chose light never sees a dark flash (dark is the default).
const THEME_SCRIPT = `try{if(localStorage.getItem("physicslens-theme")==="light")document.documentElement.dataset.theme="light"}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={ui.variable} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
