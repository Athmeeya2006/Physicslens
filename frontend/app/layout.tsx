import type { Metadata, Viewport } from "next";
import { Geologica } from "next/font/google";
import "katex/dist/katex.min.css";
import { Shell } from "@/components/Shell";
import "./globals.css";

// Self-hosted at build time by next/font: Geologica for all UI and headings (KaTeX keeps the math).
const ui = Geologica({ subsets: ["latin"], weight: ["300", "400", "500", "600", "700"], variable: "--font-ui", display: "swap" });

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

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#eef1ee" };

// Applied before the first paint, so a visitor who chose dark never sees a light flash (light is the default).
const THEME_SCRIPT = `try{if(localStorage.getItem("physicslens-theme")==="dark")document.documentElement.dataset.theme="dark"}catch(e){}`;

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
