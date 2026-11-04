"""A real quadrotor: its thrust and drag laws from flight logs, against the textbook and a neural hybrid model.

  python scripts/drone_demo.py       # after downloading NeuroBEM into var/drone/ (see below); free, local math

Data: NeuroBEM (Bauersfeld, Kaufmann, Loquercio, Scaramuzza, RSS 2021, University of Zurich), 96 real flights of a
0.772 kg racing quadrotor in a motion-capture arena at up to 65 km/h: per sample (400 per second) the four motor
speeds, the body-frame velocity and the specific force (thrust and aerodynamic force over mass) from motion capture.
Download https://download.ifi.uzh.ch/rpg/NeuroBEM/ processed_data.zip and predictions.tar.xz into var/drone/; no
licence is stated, so no data is redistributed here, only results. The authors' test flights (testset.txt) are held
out; their hybrid model's predictions for them (blade-element momentum theory plus a neural network, the bem+nn
folder) are the baseline to beat, beside the textbook law every simulator uses (thrust k * sum of squared rotor
speeds along the body axis, nothing sideways).

Candidate laws for the body-frame force (N), simplest first, each linear in its parameters:
  thrust   z: k S2  |  + climb loss c1 vz S1  |  + forward inflow c2 |vxy| S1  |  + spin-up c3 sum(w dw/dt)
  sideways x, y: 0  |  rotor drag -d S1 v (one d per axis)  |  + quadratic drag -q |v| v
with S1 and S2 the sums of rotor speeds and of their squares. The training flights are split by date: the earlier 80%
fit, the later 20% choose (the simplest law whose extra term improves their error by more than 2%), the winner is
refitted on all training flights and scored once on the test flights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "var" / "drone"
OUT = ROOT / "data" / "drone" / "result.json"
MASS = 0.772  # kg, from the dataset's readme
STEP = 4  # every fourth sample (100 per second): neighbouring samples at 400 per second are not independent
IMPROVE = 0.02


def segments() -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted((RAW / "processed_data").glob("merged_*.csv")):
        out[f.stem.replace("merged_", "")] = pd.read_csv(f).iloc[::STEP].reset_index(drop=True)
    return out


def features(d: pd.DataFrame) -> dict[str, np.ndarray]:
    w = d[["mot 1", "mot 2", "mot 3", "mot 4"]].to_numpy()
    v = d[["vel x", "vel y", "vel z"]].to_numpy()
    s1, s2 = w.sum(1), (w ** 2).sum(1)
    sp = np.linalg.norm(v[:, :2], axis=1)
    dw = d[["dmot 1", "dmot 2", "dmot 3", "dmot 4"]].to_numpy()
    return {"S1": s1, "S2": s2, "vx": v[:, 0], "vy": v[:, 1], "vz": v[:, 2], "vxy2": sp ** 2, "vxy": sp,
            "W": (w * dw).sum(1),
            "F": MASS * d[["acc x", "acc y", "acc z"]].to_numpy()}


def design(law: str, f: dict) -> np.ndarray:
    """Columns of the linear model for one axis and law."""
    if law == "thrust":
        return f["S2"][:, None]
    if law == "thrust_climb":
        return np.column_stack([f["S2"], f["vz"] * f["S1"]])
    if law == "thrust_climb_inflow":
        return np.column_stack([f["S2"], f["vz"] * f["S1"], f["vxy"] * f["S1"]])
    if law == "thrust_climb_inflow_spinup":
        return np.column_stack([f["S2"], f["vz"] * f["S1"], f["vxy"] * f["S1"], f["W"]])
    raise ValueError(law)


def design_side(law: str, f: dict, axis: str) -> np.ndarray | None:
    v = f[axis]
    if law == "none":
        return None
    if law == "rotor_drag":
        return (-f["S1"] * v)[:, None]
    if law == "rotor_drag_quadratic":
        speed = np.sqrt(f["vx"] ** 2 + f["vy"] ** 2)
        return np.column_stack([-f["S1"] * v, -speed * v])
    raise ValueError(law)


def stack(segs: dict, names: list[str]) -> dict:
    fs = [features(segs[n]) for n in names]
    return {k: np.concatenate([f[k] for f in fs]) for k in fs[0]}


def fit_axis(X: np.ndarray | None, y: np.ndarray) -> np.ndarray:
    return np.zeros(0) if X is None else np.linalg.lstsq(X, y, rcond=None)[0]


def predict(X: np.ndarray | None, p: np.ndarray, n: int) -> np.ndarray:
    return np.zeros(n) if X is None else X @ p


def rms(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def choose(laws: list[str], err: dict[str, float]) -> str:
    chosen = laws[0]
    for law in laws[1:]:
        if err[law] < err[chosen] * (1 - IMPROVE):
            chosen = law
    return chosen


def main() -> None:
    segs = segments()
    test = [n for n in (RAW / "testset.txt").read_text().split() if n in segs]
    train = sorted(n for n in segs if n not in test)
    cut = int(len(train) * 0.8)
    fit_n, sel_n = train[:cut], train[cut:]
    F, S, T, A = stack(segs, fit_n), stack(segs, sel_n), stack(segs, test), stack(segs, train)
    zl = ["thrust", "thrust_climb", "thrust_climb_inflow", "thrust_climb_inflow_spinup"]
    sl = ["none", "rotor_drag", "rotor_drag_quadratic"]
    sel_err = {"z": {}, "x": {}, "y": {}}
    for law in zl:
        p = fit_axis(design(law, F), F["F"][:, 2])
        sel_err["z"][law] = rms(predict(design(law, S), p, len(S["S1"])) - S["F"][:, 2])
    for i, ax in ((0, "x"), (1, "y")):
        for law in sl:
            p = fit_axis(design_side(law, F, "v" + ax), F["F"][:, i])
            sel_err[ax][law] = rms(predict(design_side(law, S, "v" + ax), p, len(S["S1"])) - S["F"][:, i])
    chosen = {"z": choose(zl, sel_err["z"]), "x": choose(sl, sel_err["x"]), "y": choose(sl, sel_err["y"])}

    # Refit on every training flight; score once on the test flights, against the textbook and NeuroBEM's hybrid.
    params, test_err, textbook_err = {}, {}, {}
    pz_text = fit_axis(design("thrust", A), A["F"][:, 2])
    for i, ax in ((0, "x"), (1, "y"), (2, "z")):
        if ax == "z":
            X, Xt = design(chosen["z"], A), design(chosen["z"], T)
            text = predict(design("thrust", T), pz_text, len(T["S1"]))
        else:
            X, Xt = design_side(chosen[ax], A, "v" + ax), design_side(chosen[ax], T, "v" + ax)
            text = np.zeros(len(T["S1"]))
        p = fit_axis(X, A["F"][:, i])
        params[ax] = p.tolist()
        test_err[ax] = rms(predict(Xt, p, len(T["S1"])) - T["F"][:, i])
        textbook_err[ax] = rms(text - T["F"][:, i])

    nn = []
    for n in test:
        f = RAW / "bem+nn" / f"bem+nn_{n}.csv"
        if f.exists():
            d = pd.read_csv(f, header=None).iloc[::STEP]
            nn.append((MASS * d.iloc[:, 11:14].to_numpy(), d.iloc[:, 29:32].to_numpy()))
    nn_err = {ax: rms(np.concatenate([m[:, i] - p[:, i] for m, p in nn])) for i, ax in enumerate("xyz")} if nn else None

    def total(e: dict) -> float:
        return float(np.sqrt(np.mean([e[a] ** 2 for a in "xyz"])))

    out = {"generated": "scripts/drone_demo.py",
           "data": {"dataset": "NeuroBEM (Bauersfeld et al., RSS 2021), University of Zurich",
                    "url": "https://download.ifi.uzh.ch/rpg/NeuroBEM/", "licence": "none stated; no data redistributed here",
                    "segments": len(segs), "train_segments": len(train), "test_segments": len(test),
                    "test_samples": len(T["S1"]), "mass_kg": MASS, "rate_hz": 400 / STEP},
           "chosen": chosen, "selection_rms_n": sel_err, "params": params,
           "thrust_coefficient_n_per_rad2_s2": params["z"][0],
           "heldout_rms_n": {"physicslens": test_err, "textbook": textbook_err, "neurobem_bem_nn": nn_err},
           "heldout_total_rms_n": {"physicslens": total(test_err), "textbook": total(textbook_err),
                                   "neurobem_bem_nn": total(nn_err) if nn_err else None},
           "neurobem_test_segments_with_predictions": len(nn),
           "hover_thrust_n": MASS * 9.80665}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("chosen", "heldout_rms_n", "heldout_total_rms_n", "thrust_coefficient_n_per_rad2_s2")}, indent=1))


if __name__ == "__main__":
    main()
