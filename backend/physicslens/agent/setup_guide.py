"""Setup guides: the requested experiment drawn onto the user's own frame.

The written instruction (from experiment design, i.e. from simulation) stays primary; the picture only illustrates
it. Muse Glimmer plans where the marks go, code turns the plan into a fixed edit prompt with no numbers in it,
Muse Image draws, math checks that nothing but the marks changed, and code writes every label and number on top.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..costguard import BudgetExceeded, CostGuardClient, CostGuardError
from ..perception import media
from ..perception.guide_check import Box, check_edit
from ..settings import IMAGE_MODEL
from .llm import LLMCall, LLMError, ask_json
from .schemas import Box1000, SetupGuidePlan, SetupMark

log = logging.getLogger("physicslens.agent")

GUIDE_JPG = "setup_guide.jpg"
GUIDE_JSON = "setup_guide.json"
GUIDE_FRAME = "setup_guide_frame.jpg"  # the untouched frame, shown beside the drawing
PAUSED = "Picture guides are paused for today."
CAPTION = "Illustration by Muse Image. A drawing of the setup, not a measurement."
MAX_ATTEMPTS = 2  # one retry after a rejected picture, then the written guide alone


class SetupGuideError(ValueError):
    """The request can't have a guide (wrong step, no clip). Nothing was spent."""


SETUP_GUIDE_SYSTEM = """You plan a picture guide for an experiment a measurement tool has asked the user to record.
You see one frame of the user's own scene and the requested experiment. Choose one to four marks that would be drawn
on this frame to show how to set it up:
- place_object: where the object should be put or held;
- camera_position: where the phone should be, only if it should move;
- release_point: where to let go of the object;
- push_here: where to push or pull;
- keep_clear: an area that must stay free while recording.
Each mark has a label of one to four plain words with no digits (the tool writes the numbers itself, from the verified
instruction) and a box on the 0-1000 grid of this image (x right, y down) on a real place in this frame.
instruction: the requested experiment restated in one plain sentence, keeping its numbers exactly.
Do not change or add to the experiment."""

# Every word of the edit prompt comes from these fixed phrases or from the plan's labels; nothing else gets in.
PROMPT_HEAD = ("Edit this photo of a physics experiment. Keep the scene exactly as it is: the same objects, "
               "positions, lighting, colors and framing. Only draw simple marks on top, with clean bright lines:")
PROMPT_TAIL = ("Add no text, no numbers and no letters anywhere. Do not move, add or remove anything in the scene. "
               "Leave every other part of the photo unchanged.")
DRAWING = {
    "place_object": "a dashed outline showing where to put the object",
    "camera_position": "a simple phone outline showing where to hold the camera",
    "release_point": "a bright ring showing where to let go",
    "push_here": "a bold arrow showing where to push",
    "keep_clear": "a light shaded patch showing an area to keep clear",
}
SIZE_WORDS = ("small", "medium", "large")
COLUMN_WORDS = ("left", "center", "right")
ROW_WORDS = ("upper", "middle", "lower")
_LABEL_CHARS = re.compile(r"[^A-Za-z' ]+")


def position_words(box: Box1000) -> str:
    """Plain-language place on the picture, e.g. "lower left part"; the image model gets no coordinates."""
    cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    col = COLUMN_WORDS[min(int(cx * 3 / 1000), 2)]
    row = ROW_WORDS[min(int(cy * 3 / 1000), 2)]
    if (row, col) == ("middle", "center"):
        return "center"
    if col == "center":
        return f"{row} center part"
    if row == "middle":
        return f"middle {col} part"
    return f"{row} {col} part"


def size_word(box: Box1000) -> str:
    area = (box.x2 - box.x1) * (box.y2 - box.y1) / 1e6
    return SIZE_WORDS[0 if area < 0.04 else 1 if area < 0.2 else 2]


def clean_label(label: str) -> str:
    return " ".join(_LABEL_CHARS.sub(" ", label).split()).lower()


def mark_phrase(mark: SetupMark) -> str:
    return f"{DRAWING[mark.kind]}, {size_word(mark.box)}, in the {position_words(mark.box)} of the picture, for {clean_label(mark.label)}"


def build_edit_prompt(plan: SetupGuidePlan) -> str:
    prompt = f"{PROMPT_HEAD} {'; '.join(mark_phrase(m) for m in plan.marks)}. {PROMPT_TAIL}"
    if any(c.isdigit() for c in prompt):
        # Labels are validated digit-free and the template has none; a digit here means a bug, not a model whim.
        raise SetupGuideError("The edit prompt would contain a number; refusing to send it.")
    return prompt


def template_vocabulary() -> set[str]:
    """All words the fixed template can contribute (tests check the prompt holds nothing else but labels)."""
    text = " ".join([PROMPT_HEAD, PROMPT_TAIL, *DRAWING.values(), *SIZE_WORDS, *COLUMN_WORDS, *ROW_WORDS,
                     "part in the of the picture for"])
    return set(_words(text))


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


# ---------- which instruction, which frame ----------

def source_instruction(result: dict[str, Any], step_ref: int) -> str:
    """The exact instruction the guide illustrates: the designed experiment, or the agent's request for one."""
    trail = result.get("trail") or []
    if not 0 <= step_ref < len(trail):
        raise SetupGuideError("No such step in this analysis.")
    step = trail[step_ref]
    if step.get("step") == "design":
        rec = ((result.get("experiment_design") or {}).get("recommended") or {}).get("label")
        if rec:
            return rec[0].upper() + rec[1:] + "."
        items = step.get("items") or []
        if items:
            return str(items[0]["title"]) + "."
    if step.get("step") == "agent" and step.get("action") == "request_experiment" and step.get("text"):
        return str(step["text"])
    raise SetupGuideError("Setup guides are only for experiment requests.")


def sharpness(jpeg: bytes) -> float:
    gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray is not None else 0.0


def clearest_keyframe(clip: Path, media_cfg: dict[str, Any]) -> bytes:
    """The sharpest of the keyframes the reasoning model already saw, at the same reduced size."""
    if not clip.is_file():
        raise SetupGuideError("This analysis has no stored clip to draw on.")
    frames = media.keyframes(clip, int(media_cfg.get("llm_keyframes_max", 4)), int(media_cfg.get("llm_image_long_side_px", 768)))
    return max(frames, key=sharpness)


# ---------- the exact text, drawn by code ----------

def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def overlay_text(image_bgr: np.ndarray, plan: SetupGuidePlan, instruction: str) -> tuple[bytes, list[str]]:
    """Labels at their marks and the verified instruction (with its numbers) as a caption band. Returns JPEG + texts."""
    rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    w, h = rgb.size
    size = max(12, round(min(w, h) / 28))
    font = _font(size)
    lines = textwrap.wrap(instruction, width=max(20, int(w / (size * 0.55))))[:3]
    band = (size + 8) * len(lines) + 12
    canvas = Image.new("RGB", (w, h + band), (17, 17, 17))
    canvas.paste(rgb, (0, 0))
    draw = ImageDraw.Draw(canvas)
    drawn: list[str] = []
    for mark in plan.marks:
        x1, y1, _, _ = mark.box.to_pixels(w, h)
        text = mark.label
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        tx = min(max(x1, 0), max(w - tw - 8, 0))
        ty = y1 - th - 10 if y1 - th - 10 >= 0 else min(y1 + 4, h - th - 8)
        draw.rectangle((tx, ty, tx + tw + 8, ty + th + 6), fill=(17, 17, 17))
        draw.text((tx + 4, ty + 2), text, font=font, fill=(255, 255, 255))
        drawn.append(text)
    y = h + 6
    for line in lines:
        draw.text((8, y), line, font=font, fill=(255, 255, 255))
        y += size + 8
    drawn.append(" ".join(lines))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), drawn


def decode_image(data: bytes) -> np.ndarray | None:
    """BGR array from whatever the image model returned (WebP by default) or our own JPEG."""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is not None:
        return arr
    try:
        return cv2.cvtColor(np.array(Image.open(io.BytesIO(data)).convert("RGB")), cv2.COLOR_RGB2BGR)
    except OSError:
        return None


# ---------- the whole step ----------

def plan_guide(client: CostGuardClient, keyframe: bytes, instruction: str, scene: dict[str, Any], run_id: str,
               audience: str) -> tuple[SetupGuidePlan, LLMCall]:
    payload = {"requested_experiment": instruction, "moving_object": scene.get("moving_object"),
               "observations": scene.get("observations")}
    return ask_json(client, task="setup_guide", system=SETUP_GUIDE_SYSTEM, text=json.dumps(payload),
                    schema_model=SetupGuidePlan, run_id=run_id, images=[keyframe], audience=audience,
                    meta_effort="minimal", max_output_tokens=1000)


def trail_item(guide: dict[str, Any]) -> dict[str, Any]:
    picture = guide["status"] == "picture"
    text = ("Drew the requested setup on a frame of this clip; math confirmed the scene is unchanged outside the marks."
            if picture else f"Written guide only: {guide['reason']}")
    return {"step": "setup_guide", "provider": "meta_api", "model": IMAGE_MODEL, "text": text,
            "status": "accepted" if picture else "rejected", "step_ref": guide["step_ref"],
            "planned_by": guide.get("planned_by")}


def stored_guide(run_dir: Path, step_ref: int) -> dict[str, Any] | None:
    path = run_dir / GUIDE_JSON
    if not path.is_file() or not (run_dir / GUIDE_JPG).is_file():
        return None
    guide = json.loads(path.read_text())
    if guide.get("step_ref") != step_ref or guide.get("status") != "picture":
        return None
    return guide | {"cost_usd": 0.0, "cached": True}


def make_setup_guide(client: CostGuardClient, run_dir: Path, result: dict[str, Any], step_ref: int, run_id: str,
                     audience: str) -> dict[str, Any]:
    """Returns the guide record (also stored as setup_guide.json; the picture as setup_guide.jpg)."""
    instruction = source_instruction(result, step_ref)
    hit = stored_guide(run_dir, step_ref)
    if hit is not None:
        return hit
    media_cfg = client.limits.get("media") or {}
    keyframe = clearest_keyframe(run_dir / "clip.mp4", media_cfg)
    guide: dict[str, Any] = {"step_ref": step_ref, "instruction": instruction, "caption": CAPTION,
                             "keyframe_sha256": hashlib.sha256(keyframe).hexdigest(), "provider": "meta_api",
                             "model": IMAGE_MODEL, "cost_usd": 0.0, "images_billed": 0, "checks": [],
                             "status": "text_only"}
    try:
        plan, call = plan_guide(client, keyframe, instruction, result.get("scene") or {}, run_id, audience)
    except (CostGuardError, LLMError) as e:
        log.warning("Setup guide plan failed for %s: %s", run_id, e)
        reason = str(e) if audience == "dev" else "The planning model is busy right now."
        return _finish(run_dir, guide | {"reason": reason}, persist=False)
    guide |= {"plan": plan.model_dump(), "planned_by": {"provider": call.provider, "model": call.model}}
    prompt = build_edit_prompt(plan)
    guide["prompt"] = prompt
    original = decode_image(keyframe)
    assert original is not None  # our own JPEG
    h, w = original.shape[:2]
    boxes: list[Box] = [m.box.to_pixels(w, h) for m in plan.marks]
    cfg = client.limits.get("setup_guide") or {}
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = client.meta_image_edit(image_jpeg=keyframe, prompt=prompt, run_id=run_id, audience=audience, attempt=attempt)
        except BudgetExceeded as e:
            log.warning("Setup guide paused: %s", e)
            return _finish(run_dir, guide | {"reason": PAUSED}, persist=False)
        except CostGuardError as e:
            log.warning("Setup guide image failed for %s: %s", run_id, e)
            reason = str(e) if audience == "dev" else "The picture could not be made right now."
            return _finish(run_dir, guide | {"reason": reason}, persist=False)
        guide["cost_usd"] += r.cost_usd
        guide["images_billed"] += 0 if r.cached else r.usage.images
        images = r.raw.get("images_b64") or []
        edited = decode_image(base64.b64decode(images[0])) if images else None
        if edited is None:
            guide["checks"].append({"attempt": attempt, "accepted": False, "reason": "No picture came back."})
            continue
        check, edited = check_edit(original, edited, boxes, ssim_min=float(cfg.get("ssim_outside_min", 0.80)),
                                   change_min=float(cfg.get("change_inside_min", 6.0)),
                                   pad_fraction=float(cfg.get("box_padding_fraction", 0.05)))
        guide["checks"].append({"attempt": attempt, "cached": r.cached, **check.as_dict()})
        if check.accepted:
            jpeg, drawn = overlay_text(edited, plan, instruction)
            (run_dir / GUIDE_JPG).write_bytes(jpeg)
            (run_dir / GUIDE_FRAME).write_bytes(keyframe)
            return _finish(run_dir, guide | {"status": "picture", "reason": check.reason, "overlay_text": drawn})
    return _finish(run_dir, guide | {"reason": _last_reason(guide["checks"])})


def _last_reason(checks: list[dict[str, Any]]) -> str:
    return (checks[-1].get("reason") if checks else None) or "The picture did not pass the check."


def _finish(run_dir: Path, guide: dict[str, Any], persist: bool = True) -> dict[str, Any]:
    guide["trail_item"] = trail_item(guide)
    if persist:
        # The stored record says what was billed (images) but holds no money: a gallery run's folder is public,
        # and dollar amounts live only in the private ledger.
        stored = {k: v for k, v in guide.items() if k != "cost_usd"}
        (run_dir / GUIDE_JSON).write_text(json.dumps(stored, indent=1))
    return guide

