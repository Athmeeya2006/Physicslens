"""A share image for one result (1200 x 630, the size link previews use): the clip's frame, the law, one number.

Rendered with Pillow and matplotlib's mathtext (no browser). A law mathtext cannot typeset falls back to its title,
never to a broken image.
"""

from __future__ import annotations

import html
import io
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
BG = (5, 7, 10)
INK = (242, 245, 248)
MUTED = (158, 171, 183)
ACCENT = (82, 156, 255)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _law_image(latex: str, width: int) -> Image.Image | None:
    """The law as white-on-transparent mathtext, scaled to fit width; None when mathtext cannot read it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tex = latex.replace(r"\operatorname{sgn}", r"\mathrm{sgn}").replace(r"\quad", r"\quad ").replace(r"\,", " ")
    fig = plt.figure(figsize=(12, 1.6), dpi=100)
    fig.patch.set_alpha(0)
    try:
        fig.text(0.0, 0.5, f"${tex}$", color="white", fontsize=34, va="center")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.05)
    except (ValueError, RuntimeError):
        return None
    finally:
        plt.close(fig)
    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGBA")
    if img.width > width:
        img = img.resize((width, max(1, int(img.height * width / img.width))), Image.LANCZOS)
    return img


def headline(result: dict[str, Any]) -> str:
    params = result.get("parameters") or []
    if not params:
        return {"no_fit": "No clean equation: the honest answer", "ambiguous": "Two laws tie: an experiment is asked for"}.get(
            result.get("status", ""), result.get("reason", "")[:70])
    p = params[0]
    return f"{p['name']} = {p['value']:.5g} ± {p['std']:.2g} {p.get('unit', '')}".strip()


def share_page(gid: str, result: dict[str, Any], site: str) -> str:
    """A tiny page for one gallery result: link previews read its metadata, people are sent on to the result."""
    title = html.escape(((result.get("equation") or {}).get("title") or "PhysicsLens result") + " | PhysicsLens")
    desc = html.escape(headline(result) + ". " + (result.get("reason") or "")[:180])
    target = f"/?run={gid}"
    image = f"{site}/gallery/{gid}/og.jpg"
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n<title>{title}</title>\n'
            f'<meta name="description" content="{desc}">\n<meta property="og:title" content="{title}">\n'
            f'<meta property="og:description" content="{desc}">\n<meta property="og:image" content="{image}">\n'
            f'<meta property="og:image:width" content="{W}">\n<meta property="og:image:height" content="{H}">\n'
            f'<meta property="og:url" content="{site}/r/{gid}.html">\n<meta name="twitter:card" content="summary_large_image">\n'
            f'<meta http-equiv="refresh" content="0; url={target}">\n<link rel="canonical" href="{site}{target}">\n'
            f'</head><body><p><a href="{target}">Open the result</a></p></body></html>\n')


def render(result: dict[str, Any], poster: Path | None, out: Path) -> Path:
    img = Image.new("RGB", (W, H), BG)
    if poster is not None and poster.is_file():
        frame = Image.open(poster).convert("RGB")
        # Below the title band, so a long title never runs over the picture.
        frame.thumbnail((W // 2 - 20, H - 220))
        img.paste(frame, (W - frame.width - 40, 175 + (H - 215 - frame.height) // 2))
    d = ImageDraw.Draw(img)
    d.text((48, 40), "PhysicsLens", font=_font(34), fill=ACCENT)
    title = ((result.get("equation") or {}).get("title") or {"accepted": "Verified equation"}.get(result.get("status", ""),
                                                                                                     "Measured motion"))
    d.text((48, 100), title[:46], font=_font(40), fill=INK)
    law = (result.get("equation") or {}).get("latex")
    y = 180
    if law:
        typeset = _law_image(law, W // 2 - 60)
        if typeset is not None:
            img.paste(typeset, (48, y), typeset)
            y += typeset.height + 30
    d.text((48, max(y, 300)), headline(result), font=_font(30), fill=INK)
    verdict = {"accepted": "Verified on motion it never saw", "ambiguous": "A tie, settled by the next experiment",
               "no_fit": "No law forecasts it: said plainly"}.get(result.get("status", ""), "")
    d.text((48, H - 70), verdict, font=_font(24), fill=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=88)
    return out
