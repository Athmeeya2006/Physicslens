"""Generate samples/smoke_ball.mp4: a 2 s synthetic pendulum (60 frames, H.264) for the paid smoke test.

60 frames sits just above SAM's 50-frame billing minimum, so the smoke call stays as cheap as possible.
"""

import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECONDS = 640, 360, 30, 2
OUT = Path(__file__).resolve().parents[1] / "samples" / "smoke_ball.mp4"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pivot, length = (W // 2, 40), 250
        for i in range(FPS * SECONDS):
            theta = 0.5 * math.cos(2 * math.pi * i / FPS / 1.0)
            x = int(pivot[0] + length * math.sin(theta))
            y = int(pivot[1] + length * math.cos(theta))
            frame = np.full((H, W, 3), 245, np.uint8)
            cv2.line(frame, pivot, (x, y), (60, 60, 60), 2)
            cv2.circle(frame, (x, y), 22, (30, 30, 220), -1)
            cv2.imwrite(f"{tmp}/{i:04d}.png", frame)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{tmp}/%04d.png",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)],
            check=True,
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
