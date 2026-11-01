"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// The phone's tilt from its accelerometer, for filming side-on and level (the measured limits assume it).
// Camera axes match the backend: X right, Y up, Z away from the camera (the back camera looks out of the phone's back).

export const LEVEL_WARN_DEG = 3;
const MOVING_MS2 = 0.6; // spread of |a| over the last half second above this: the phone is being moved

export interface Level {
  roll: number;   // degrees: gravity's angle from the image's vertical
  pitch: number;  // degrees: the view axis above (+) or below (-) level
  moving: boolean;
  up: [number, number, number];  // "up" in camera axes, m/s^2
}

type MotionPermission = { requestPermission?: () => Promise<"granted" | "denied"> };

function screenAngle(): number {
  try {
    return (screen.orientation?.angle ?? (window as unknown as { orientation?: number }).orientation ?? 0) * Math.PI / 180;
  } catch {
    return 0;
  }
}

export function useLevel() {
  const [level, setLevel] = useState<Level | null>(null);
  const [state, setState] = useState<"unsupported" | "needs-permission" | "listening" | "denied" | "idle">("idle");
  const smooth = useRef<[number, number, number] | null>(null);
  const recent = useRef<{ t: number; m: number }[]>([]);

  const onMotion = useCallback((e: DeviceMotionEvent) => {
    const a = e.accelerationIncludingGravity;
    if (a?.x == null || a.y == null || a.z == null) return;
    const ang = screenAngle();
    // Device axes (x right, y up the screen in portrait, z out of the screen) to image axes for the current rotation.
    let x = a.x * Math.cos(ang) - a.y * Math.sin(ang);
    let y = a.x * Math.sin(ang) + a.y * Math.cos(ang);
    let z = -a.z;
    // Platforms disagree on the sign of this reading; a phone filming is never upside down, so up is +Y-ish.
    if (y < 0) { x = -x; y = -y; z = -z; }
    const k = 0.15;
    const s = smooth.current ?? [x, y, z];
    smooth.current = [s[0] + k * (x - s[0]), s[1] + k * (y - s[1]), s[2] + k * (z - s[2])];
    const now = performance.now();
    recent.current = [...recent.current.filter((r) => now - r.t < 500), { t: now, m: Math.hypot(x, y, z) }];
    const ms = recent.current.map((r) => r.m);
    const moving = Math.max(...ms) - Math.min(...ms) > MOVING_MS2;
    const [sx, sy, sz] = smooth.current;
    setLevel({
      roll: (Math.atan2(sx, sy) * 180) / Math.PI,
      pitch: (Math.atan2(sz, Math.hypot(sx, sy)) * 180) / Math.PI,
      moving,
      up: smooth.current,
    });
  }, []);

  const listen = useCallback(() => {
    window.addEventListener("devicemotion", onMotion);
    setState("listening");
  }, [onMotion]);

  // iOS asks for permission, and only from a tap; elsewhere listening starts at once.
  const request = useCallback(async () => {
    const DM = (typeof DeviceMotionEvent !== "undefined" ? DeviceMotionEvent : undefined) as unknown as MotionPermission | undefined;
    if (!DM?.requestPermission) return listen();
    try {
      (await DM.requestPermission()) === "granted" ? listen() : setState("denied");
    } catch {
      setState("denied");
    }
  }, [listen]);

  useEffect(() => {
    // Only phones and tablets: a laptop has no useful accelerometer for this.
    const touch = typeof window !== "undefined" && window.matchMedia?.("(pointer: coarse)").matches;
    if (!touch || typeof DeviceMotionEvent === "undefined") {
      setState("unsupported");
      return;
    }
    if ((DeviceMotionEvent as unknown as MotionPermission).requestPermission) setState("needs-permission");
    else listen();
    return () => window.removeEventListener("devicemotion", onMotion);
  }, [listen, onMotion]);

  return { level, state, request };
}

export const isLevel = (l: Level | null) =>
  !!l && !l.moving && Math.abs(l.roll) <= LEVEL_WARN_DEG && Math.abs(l.pitch) <= 15;
