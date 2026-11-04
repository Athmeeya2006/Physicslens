"""Occultation edge timing against PyOTE, the community tool, on the same light curves with known edges.

  python scripts/occultation_pyote_compare.py        # needs PyOTE in its own Python 3.10 env (PYOTE_PYTHON, default conda env pyote310)

The light curves are synthetic with known edge times (the same model and noise as the edge trials of
scripts/occultation_demo.py: each frame integrates its exposure, noise 0.08 of the drop), so both tools are scored
against the truth, not against each other. PyOTE (bob-anderson-ok/py-ote, MIT) runs headless: its event search
(iterative_logl_functions.find_best_event_from_min_max_size) and its sub-frame adjustment (solverUtils.subFrameAdjusted)
are called directly, as its GUI does. PyOTE needs Python < 3.11, so it runs in a separate interpreter on the same file.
Free, local. Writes data/astro/occultation/pyote_compare.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.astro import occultation as O

OUT = ROOT / "data" / "astro" / "occultation" / "pyote_compare.json"
EXPOSURES = (0.13, 0.5, 1.0)
TRIALS = 40
NOISE = 0.08

PYOTE_SIDE = r'''
import json, os, sys
import numpy as np
import pyoteapp
sys.path.insert(0, os.path.dirname(pyoteapp.__file__))  # PyOTE's modules import each other by bare name
from pyoteapp.iterative_logl_functions import find_best_event_from_min_max_size
from pyoteapp.solverUtils import subFrameAdjusted
d = np.load(sys.argv[1], allow_pickle=True)
out = []
for flux, exp in zip(d["flux"], d["exposure"]):
    y = np.asarray(flux, float)
    n = len(y)
    last = None
    for item in find_best_event_from_min_max_size(y, 0, n - 1, 3, n - 6):
        last = item
    flag, _, D, R, B, A, sB, sA, _ = last
    if flag < 0:
        out.append(None)
        continue
    (aD, aR), *_ = subFrameAdjusted(eventType="DandR", cand=(int(D), int(R)), B=B, A=A, sigmaA=sA, sigmaB=sB,
                                    yValues=y, left=0, right=n - 1)
    # Frame i spans [t_i - exp/2, t_i + exp/2] with t_i = i * exp: a fractional frame index f is the time f*exp - exp/2.
    out.append([float(aD) * exp - exp / 2, float(aR) * exp - exp / 2])
print(json.dumps(out))
'''


def main() -> None:
    rng = np.random.default_rng(1)
    curves, truth, ours = [], [], []
    for exposure in EXPOSURES:
        for _ in range(TRIALS):
            td = rng.uniform(10, 11)
            tr = td + rng.uniform(20, 40)
            t = np.arange(0, 60, exposure)
            flux = O._model(t, exposure, td, tr, 1.0, 0.1) + rng.normal(0, NOISE, len(t))
            e = O.edge_times(t, flux, exposure, NOISE)
            curves.append((flux, exposure))
            truth.append((td, tr))
            ours.append((e["disappear"], e["reappear"]))
    with tempfile.TemporaryDirectory() as tmp:
        npz = Path(tmp) / "curves.npz"
        np.savez(npz, flux=np.array([c[0] for c in curves], dtype=object), exposure=np.array([c[1] for c in curves]))
        side = Path(tmp) / "pyote_side.py"
        side.write_text(PYOTE_SIDE)
        py = os.environ.get("PYOTE_PYTHON", os.path.expanduser("~/miniforge3/envs/pyote310/bin/python"))
        res = subprocess.run([py, str(side), str(npz)], capture_output=True, text=True, check=False)
        if res.returncode:
            sys.exit("PyOTE side failed:\n" + res.stderr[-3000:])
        pyote = json.loads(res.stdout.strip().splitlines()[-1])
    rows = []
    for k, exposure in enumerate(EXPOSURES):
        sl = slice(k * TRIALS, (k + 1) * TRIALS)
        tru, us, them = np.array(truth[sl]), np.array(ours[sl]), pyote[sl]
        ok = [i for i, p in enumerate(them) if p is not None]
        e_us = (us - tru).ravel()
        e_them = (np.array([them[i] for i in ok]) - tru[ok]).ravel()
        rows.append({"exposure_s": exposure, "trials": TRIALS, "pyote_found": len(ok),
                     "physicslens_rms_s": float(np.sqrt(np.mean(e_us ** 2))), "pyote_rms_s": float(np.sqrt(np.mean(e_them ** 2))),
                     "physicslens_rms_frames": float(np.sqrt(np.mean(e_us ** 2)) / exposure),
                     "pyote_rms_frames": float(np.sqrt(np.mean(e_them ** 2)) / exposure)})
        print(rows[-1], flush=True)
    out = {"generated": "scripts/occultation_pyote_compare.py",
           "tools": {"physicslens": "astro/occultation.edge_times (exposure-integrated edge model, grid scan)",
                     "pyote": "PyOTE (bob-anderson-ok/py-ote, MIT): event search and sub-frame adjustment, run headless"},
           "curves": f"synthetic, known edges, noise {NOISE} of the drop, {TRIALS} per exposure", "rows": rows}
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
