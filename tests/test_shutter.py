import numpy as np
from physicslens.sigproc.shutter import WARN_PX, note, skew_px


def test_a_slow_swing_is_fine_and_a_fast_throw_is_flagged():
    t = np.arange(60) / 30
    slow = np.column_stack([300 + 20 * np.sin(t), np.full(60, 200.0)])  # 20 px/s at most
    fast = np.column_stack([2000 * t, np.full(60, 200.0)])  # 2000 px/s across the frame
    assert skew_px(slow, 30, 40, 720) < WARN_PX and note(skew_px(slow, 30, 40, 720)) is None
    s = skew_px(fast, 30, 40, 720)
    assert s > WARN_PX and "rolling shutter" in note(s)
    assert abs(s - 2000 * 0.03 * 40 / 720) < 1e-6  # speed x readout x object height / frame height
