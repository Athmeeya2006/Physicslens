"""Shareable MP4: the analyzed clip with the measured trail, the simulated ghost and the verified equation burned in."""

from __future__ import annotations

import io
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")  # headless: render math to images, never open windows
from matplotlib.mathtext import math_to_image

GHOST_BGR = (238, 211, 34)  # same cyan as the web overlay
TRAIL_BGR = (255, 255, 255)
TRAIL_FRAMES = 20
STATUS_TEXT = {"accepted": "Verified equation", "ambiguous": "More data needed", "no_fit": "No clean governing equation found"}
STATUS_RGB = {"accepted": (52, 211, 153), "ambiguous": (251, 191, 36), "no_fit": (248, 113, 113)}
FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf")


class ExportError(RuntimeError):
    pass


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _math_raw(tex: str) -> Image.Image:
    buf = io.BytesIO()
    # math_to_image paints an opaque white background, so render black ink and turn its darkness into alpha.
    math_to_image(f"${tex}$", buf, dpi=300, format="png", color="black")
    ink = 255 - np.array(Image.open(io.BytesIO(buf.getvalue())).convert("L"))
    rgba = np.zeros((*ink.shape, 4), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = ink
    return Image.fromarray(rgba, "RGBA")


_REFERENCE_HEIGHT = None


def _math(tex: str, em_px: int) -> Image.Image:
    """White LaTeX on transparent. Every formula gets the same scale (set by a reference glyph), so type
    size stays consistent whether or not a line has fractions or superscripts."""
    global _REFERENCE_HEIGHT
    if _REFERENCE_HEIGHT is None:
        _REFERENCE_HEIGHT = _math_raw("M").height
    img = _math_raw(tex)
    scale = em_px / _REFERENCE_HEIGHT
    return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)


def _sig(v: float, digits: int = 4) -> str:
    return f"{v:.{digits}g}"


def _unit_tex(unit: str) -> str:
    return rf"\ \mathrm{{{unit}}}" if unit else ""


def build_panel(result: dict[str, Any], frame_width: int) -> Image.Image:
    """The text block (status, equation, parameters, footer) as one RGBA image."""
    s = max(frame_width / 1600, 0.4)  # everything scales with the video size
    pad = int(12 * s)
    rows: list[Image.Image] = []
    status = result.get("status", "no_fit")

    head = _font(int(30 * s))
    title = Image.new("RGBA", (int(900 * s), int(42 * s)), (0, 0, 0, 0))
    ImageDraw.Draw(title).text((0, 0), STATUS_TEXT.get(status, status), font=head, fill=STATUS_RGB.get(status, (230, 230, 230)))
    rows.append(title)

    eq = result.get("equation")
    if eq:
        rows.append(_math(eq["latex"], int(30 * s)))
        for p in result.get("parameters") or []:
            tex = rf"{p['latex']} = {_sig(p['value'])} \pm {_sig(p['std'], 2)}{_unit_tex(p['unit'])}"
            rows.append(_math(tex, int(22 * s)))
    elif result.get("regimes"):
        for r in result["regimes"]:
            if not r.get("params"):
                continue
            bits = ", ".join(f"{k} = {_sig(v)}" for k, v in r["params"].items())
            length = r.get("geometry", {}).get("length_m")
            text = f"{r['start_s']:.1f} to {r['end_s']:.1f} s: {bits}" + (f", L = {length:.3g} m" if length else "")
            rows.append(_text_row(text, int(24 * s), int(900 * s)))
    tied = [i["title"] for t in result.get("trail") or [] if t.get("step") == "verification"
            for i in t.get("items") or [] if i.get("status") == "tied"]
    if status == "ambiguous" and tied:
        rows.append(_text_row("Still tied: " + ", ".join(tied), int(20 * s), int(900 * s), fill=(251, 191, 36)))
    message = (result.get("agent") or {}).get("message")
    if status != "accepted" and message:
        lines = textwrap.wrap(message, 62)
        if len(lines) > 3:
            lines = [*lines[:2], textwrap.shorten(" ".join(lines[2:]), 60, placeholder="...")]
        for line in lines:
            rows.append(_text_row(line, int(22 * s), int(900 * s)))

    rows.append(_text_row("Dashed circle: simulation of the fitted equation. White trail: measured by SAM 3.1.",
                          int(18 * s), int(900 * s), fill=(180, 190, 205)))
    rows.append(_text_row("PhysicsLens", int(18 * s), int(900 * s), fill=(34, 211, 238)))

    width = max(r.width for r in rows) + 2 * pad
    height = sum(r.height for r in rows) + pad * (len(rows) + 1)
    panel = Image.new("RGBA", (width, height), (11, 15, 23, 190))
    y = pad
    for r in rows:
        panel.alpha_composite(r, (pad, y))
        y += r.height + pad
    return panel


def _text_row(text: str, size: int, max_w: int, fill: tuple[int, int, int] = (230, 234, 242)) -> Image.Image:
    font = _font(size)
    w = min(max_w, int(ImageDraw.Draw(Image.new("RGBA", (1, 1))).textlength(text, font=font)) + 4)
    img = Image.new("RGBA", (w, int(size * 1.35)), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((0, 0), text, font=font, fill=fill)
    return img


def _dashed_circle(frame: np.ndarray, center: tuple[int, int], radius: int, thickness: int) -> None:
    overlay = frame.copy()
    cv2.circle(overlay, center, radius, GHOST_BGR, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, dst=frame)
    for start in range(0, 360, 30):
        cv2.ellipse(frame, center, (radius, radius), 0, start, start + 18, GHOST_BGR, thickness, cv2.LINE_AA)


def _quiet_corner(points: list[tuple[float, float]], w: int, h: int, pw: int, ph: int, margin: int = 10) -> tuple[int, int]:
    """The corner where the panel covers the fewest track points, so it never hides the motion."""
    corners = [(margin, margin), (w - pw - margin, margin), (margin, h - ph - margin), (w - pw - margin, h - ph - margin)]
    pts = np.array(points) if points else np.empty((0, 2))

    def covered(c: tuple[int, int]) -> int:
        x, y = c
        if not len(pts):
            return 0
        inside = (pts[:, 0] >= x - margin) & (pts[:, 0] <= x + pw + margin) & (pts[:, 1] >= y - margin) & (pts[:, 1] <= y + ph + margin)
        return int(inside.sum())

    x, y = min(corners, key=covered)
    return max(0, x), max(0, y)


def render_share_video(result: dict[str, Any], clip: Path, out: Path) -> Path:
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise ExportError(f"Cannot open {clip.name}.")
    fps = float(result["clip"]["fps"])
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real = {int(f): (x, y) for f, x, y in result.get("real_track_px", [])}
    ghosts = result.get("ghosts") or []
    ghost = {int(f): (x, y) for f, x, y in ghosts[0]["points"]} if ghosts else {}
    radius = max(4, int((result.get("object_size_px") or min(w, h) / 20) / 2))
    thickness = max(2, w // 400)

    panel = build_panel(result, w)
    max_w = int(0.46 * w)
    if panel.width > max_w:  # keep the video, not the caption, the main thing
        k = max_w / panel.width
        panel = panel.resize((int(panel.width * k), int(panel.height * k)), Image.LANCZOS)
    panel_rgba = np.array(panel)
    pb, pa = panel_rgba[:, :, 2::-1].astype(np.float32), panel_rgba[:, :, 3:4].astype(np.float32) / 255.0
    ph, pw = panel_rgba.shape[:2]
    px, py = _quiet_corner(list(real.values()) + list(ghost.values()), w, h, pw, ph)
    change_frame = int(result["regime_change"]["t_estimate"] * fps) if result.get("regime_change") else None

    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:g}",
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "veryfast", "-movflags", "+faststart",
         str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert enc.stdin is not None
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            trail = [real[f] for f in range(idx - TRAIL_FRAMES, idx + 1) if f in real]
            if len(trail) >= 2:
                cv2.polylines(frame, [np.array(trail, np.int32)], False, TRAIL_BGR, max(1, thickness - 1), cv2.LINE_AA)
            if idx in ghost:
                gx, gy = ghost[idx]
                _dashed_circle(frame, (round(gx), round(gy)), radius, thickness)
            region = frame[py:py + ph, px:px + pw].astype(np.float32)
            frame[py:py + ph, px:px + pw] = (pb * pa + region * (1 - pa)).astype(np.uint8)
            if change_frame is not None and change_frame <= idx < change_frame + int(1.5 * fps):
                label_y = py + ph + int(30 * w / 1280) if py < h // 2 else py - int(12 * w / 1280)
                cv2.putText(frame, "Dynamics changed", (px, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1280), GHOST_BGR, thickness, cv2.LINE_AA)
            enc.stdin.write(frame.tobytes())
            idx += 1
    finally:
        cap.release()
        enc.stdin.close()
        err = enc.stderr.read().decode() if enc.stderr else ""
        enc.wait()
    if enc.returncode != 0 or idx == 0:
        raise ExportError(f"Encoding failed: {err.strip()[:200] or 'no frames'}")
    return out
