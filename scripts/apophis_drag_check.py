"""Is Apophis's small "drag" term the planets' pull? Fit the same two laws to simulated Apophis tracks with and
without the planets.

  python scripts/apophis_drag_check.py

The video's winning law (gallery-apophis) is the Sun's gravity plus a tiny velocity term, b = 2.4e-7 per day, which
the reasoning model named Stokes drag. There is no air in space; if the term stands in for the planets, it must come
out near that value on a track with planets and vanish on a Sun-only track. Free and local: no model calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from physicslens.physics.coords import Coordinates
from physicslens.physics.custom import MAX_PARAMS_ORBIT, FreeParam, build, register
from physicslens.physics.fit import fit_family
from physicslens.sky import scenarios as S
from physicslens.sky.discover import kepler


def kepler_drag(gm: float):
    return build("plane", "Kepler plus linear drag", ["-mu*x/(x^2+y^2)^1.5 - b*vx", "-mu*y/(x^2+y^2)^1.5 - b*vy"],
                 [FreeParam("mu", "AU^3/day^2", gm * 0.5, gm * 2, gm), FreeParam("b", "1/day", -1e-4, 1e-4, 0.0)],
                 max_params=MAX_PARAMS_ORBIT)


def main() -> None:
    for name in ("apophis_sun_only", "apophis_with_planets"):
        sr = S.run(S.load(name))
        t, xy, _ = S.track(sr, 0.5, seed=0)
        c = Coordinates("plane", t, xy, ("x", "y"), {})
        plain, drag = fit_family([register(kepler(sr.central_gm)), register(kepler_drag(sr.central_gm))], c)
        winner = "Kepler plus drag" if drag.bic < plain.bic else "Kepler alone"
        print(f"{name}: b = {drag.params['b']:.2e} per day; {winner} wins by {abs(drag.bic - plain.bic):.0f} BIC")


if __name__ == "__main__":
    main()
