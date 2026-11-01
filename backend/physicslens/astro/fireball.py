"""Fireballs: from a triangulated bright flight to "did anything survive, and where did it land".

Three steps, each checked the PhysicsLens way (fit the earlier part of the flight, forecast the rest):

1. Alpha-beta (Gritsevich 2007; Sansom et al. 2019, ApJ 885, 115, eq. 1, checked in the paper's source):
     y = ln(alpha) + beta - ln(Delta / 2),   Delta = Ei(beta) - Ei(beta v^2)
   with y = height / h0 (h0 = 7160 m, the homogeneous atmosphere) and v = V / V0. alpha is the ballistic coefficient
   (drag against mass per area), beta the mass-loss parameter. Fitted by the Q4 least squares of the Desert Fireball
   Network's alpha_beta_modules (MIT), whose example event reproduces here.
   Triage (Sansom et al. 2019, eqs. 10 and 11, a 50 g final mass, rho_m = 3500 kg/m^3, cd A = 1.5):
     mu = 0     ln beta = ln(13.20 - 3 ln(alpha sin gamma))   above it: unlikely to drop a meteorite
     mu = 2/3   ln beta = ln(4.4 - ln(alpha sin gamma))       below it: a likely meteorite dropper
2. Single body (the classical drag and ablation equations, in height so that height and speed are enough):
     dv/dt = -Gamma0 mu^(-1/3) rho_air v^2 + g sin(gamma),  dmu/dt = -sigma Gamma0 mu^(2/3) rho_air v^3
   mu = m / m_entry, Gamma0 = cd A rho_m^(-2/3) m_entry^(-1/3) / 2, sigma the ablation coefficient. Fragmentation is a
   dynamics change: a sudden jump in drag per unit mass, accepted only when it beats the single body by a decisive BIC.
3. Dark flight: below the luminous end the stone only falls through the wind, drag on a sphere-like body, in a local
   east-north-up frame. Monte Carlo over the end state, the mass, the shape and the wind gives the strewn field.

The atmosphere is NRLMSIS 2.1 (pymsis) for the date and place when available, else the US Standard Atmosphere 1976.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.special import expi

H0_M = 7160.0  # homogeneous atmosphere scale height used by alpha-beta
RHO0 = 1.29  # sea-level air density in the alpha-beta mass formula (DFN alpha_beta_modules), kg/m^3
G0 = 9.80665
EARTH_R_M = 6371000.0
TRAIN_FRACTION = 0.75
DECISIVE_DELTA_BIC = 10.0

# Sansom et al. 2019 eqs. (10), (11): 50 g final mass, rho_m 3500 kg/m^3, cd A 1.5.
MU0_LINE = 13.20
MU23_LINE = 4.4


# --- atmosphere ------------------------------------------------------------------------------------------------------

# US Standard Atmosphere 1976 below 86 km: (base geopotential height m, base temperature K, lapse K/m).
_US76 = ((0.0, 288.15, -0.0065), (11000.0, 216.65, 0.0), (20000.0, 216.65, 0.001), (32000.0, 228.65, 0.0028),
         (47000.0, 270.65, 0.0), (51000.0, 270.65, -0.0028), (71000.0, 214.65, -0.002))
_R_AIR = 287.053  # J/(kg K)


def _us76_layers() -> list[tuple[float, float, float, float]]:
    out, p = [], 101325.0
    for i, (hb, tb, lapse) in enumerate(_US76):
        out.append((hb, tb, lapse, p))
        if i + 1 < len(_US76):
            h1 = _US76[i + 1][0]
            if lapse == 0:
                p = p * math.exp(-G0 * (h1 - hb) / (_R_AIR * tb))
            else:
                p = p * (1 + lapse * (h1 - hb) / tb) ** (-G0 / (_R_AIR * lapse))
    return out


_LAYERS = _us76_layers()


def us76_temperature(h_m: np.ndarray | float) -> np.ndarray:
    """Air temperature (K) of the US Standard Atmosphere 1976 (held at the 84.852 km value above)."""
    h = np.atleast_1d(np.asarray(h_m, float))
    z = np.clip(6356766.0 * h / (6356766.0 + h), 0.0, 84852.0)
    out = np.empty_like(z)
    for k, zk in enumerate(z):
        hb, tb, lapse, _ = next(layer for layer in reversed(_LAYERS) if zk >= layer[0])
        out[k] = tb + lapse * (zk - hb)
    return out


def us76_density(h_m: np.ndarray | float) -> np.ndarray:
    """Air density (kg/m^3) of the US Standard Atmosphere 1976, geometric height in meters; an exponential tail above
    86 km (scale height of the top layer)."""
    h = np.atleast_1d(np.asarray(h_m, float))
    z = 6356766.0 * h / (6356766.0 + h)  # geopotential height
    out = np.empty_like(z)
    for k, zk in enumerate(z):
        zk = min(max(zk, 0.0), 84852.0)
        hb, tb, lapse, pb = next(layer for layer in reversed(_LAYERS) if zk >= layer[0])
        if lapse == 0:
            t, p = tb, pb * math.exp(-G0 * (zk - hb) / (_R_AIR * tb))
        else:
            t = tb + lapse * (zk - hb)
            p = pb * (t / tb) ** (-G0 / (_R_AIR * lapse))
        out[k] = p / (_R_AIR * t)
    top = h > 84852.0
    if top.any():
        out[top] *= np.exp(-(h[top] - 84852.0) / 5800.0)
    return out


@dataclass
class Atmosphere:
    """Density and wind against height. Winds are (east, north) in m/s, interpolated in height; none means calm."""

    heights_m: np.ndarray
    density: np.ndarray
    wind_e: np.ndarray | None = None
    wind_n: np.ndarray | None = None
    source: str = "US Standard Atmosphere 1976"
    temperature: np.ndarray | None = None  # K; the standard atmosphere's when not given

    def __post_init__(self) -> None:
        if self.temperature is None:
            self.temperature = us76_temperature(self.heights_m)

    def sound_speed(self, h: float) -> float:
        return math.sqrt(1.4 * _R_AIR * float(np.interp(h, self.heights_m, self.temperature)))

    def rho(self, h: float) -> float:
        return float(np.exp(np.interp(h, self.heights_m, np.log(self.density))))

    def wind(self, h: float) -> tuple[float, float]:
        if self.wind_e is None or self.wind_n is None:
            return 0.0, 0.0
        return float(np.interp(h, self.heights_m, self.wind_e)), float(np.interp(h, self.heights_m, self.wind_n))

    def with_wind(self, heights_m, east, north, source: str) -> Atmosphere:
        h = np.asarray(heights_m, float)
        order = np.argsort(h)
        return Atmosphere(self.heights_m, self.density,
                          np.interp(self.heights_m, h[order], np.asarray(east, float)[order]),
                          np.interp(self.heights_m, h[order], np.asarray(north, float)[order]),
                          f"{self.source}; winds: {source}", self.temperature)


def standard_atmosphere(top_m: float = 120000.0) -> Atmosphere:
    h = np.linspace(0.0, top_m, 241)
    return Atmosphere(h, us76_density(h))


def msis_atmosphere(when_utc: str, lat: float, lon: float, top_m: float = 120000.0) -> Atmosphere:
    """NRLMSIS 2.1 total mass density for the date and place (pymsis; it fetches the space-weather indices once)."""
    import pymsis

    h = np.linspace(0.0, top_m, 241)
    out = pymsis.calculate(np.datetime64(when_utc.replace("Z", "")), lon, lat, h / 1000.0)
    rho = np.asarray(out[..., pymsis.Variable.MASS_DENSITY], float).ravel()
    temp = np.asarray(out[..., pymsis.Variable.TEMPERATURE], float).ravel()
    return Atmosphere(h, rho, source=f"NRLMSIS 2.1 at {lat:.2f}, {lon:.2f}, {when_utc}", temperature=temp)


def parse_wyoming_sounding(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """University of Wyoming upper-air TEXT:LIST (plain text or its HTML page): (height m, wind east m/s, wind north
    m/s), sorted by height.

    Columns as the page labels them, each seven characters wide: PRES HGHT TEMP DWPT RELH MIXR DRCT and either SKNT
    (knots, the older pages) or SPED (m/s, the newer ones); DRCT is where the wind blows from. Levels with a missing
    wind are skipped."""
    import html as _html
    import re

    text = _html.unescape(re.sub(r"<[^>]+>", "\n", text)) if "<" in text else text
    lines = text.splitlines()
    try:
        head = next(i for i, ln in enumerate(lines) if ln.split()[:2] == ["PRES", "HGHT"])
    except StopIteration as e:
        raise ValueError("Not a Wyoming TEXT:LIST sounding (no PRES HGHT header).") from e
    names = lines[head].split()
    if "SPED" in names:
        speed_col, to_ms = names.index("SPED"), 1.0
    elif "SKNT" in names:
        speed_col, to_ms = names.index("SKNT"), 0.514444
    else:
        raise ValueError("The sounding has no wind speed column (SPED or SKNT).")
    h_col, d_col = names.index("HGHT"), names.index("DRCT")
    width = 7
    rows = []
    for ln in lines[head + 1:]:
        if not ln.strip() or set(ln.strip()) == {"-"} or ln.split()[0] in ("hPa",):
            if rows:
                break
            continue
        cells = [ln[k * width:(k + 1) * width].strip() for k in range(len(names))]
        try:
            hgt, d, sp = float(cells[h_col]), float(cells[d_col]), float(cells[speed_col]) * to_ms
        except (ValueError, IndexError):
            if rows and not ln[:7].strip().replace(".", "").isdigit():
                break
            continue
        rows.append((hgt, -sp * math.sin(math.radians(d)), -sp * math.cos(math.radians(d))))
    if len(rows) < 3:
        raise ValueError("The sounding has fewer than three levels with winds.")
    a = np.array(sorted(rows))
    return a[:, 0], a[:, 1], a[:, 2]


# --- alpha-beta ------------------------------------------------------------------------------------------------------

def _q4(theta, v, y):
    alpha, beta = theta
    return 2 * alpha * np.exp(-y) - (expi(beta) - expi(beta * v ** 2)) * np.exp(-beta)


def alpha_beta_height(alpha: float, beta: float, v: np.ndarray) -> np.ndarray:
    """Height (m) where the normalised speed is v (Sansom et al. 2019 eq. 1)."""
    v = np.clip(np.asarray(v, float), 1e-6, 1 - 1e-12)
    return H0_M * (math.log(alpha) + beta - np.log((expi(beta) - expi(beta * v ** 2)) / 2))


def alpha_beta_speed(alpha: float, beta: float, h_m: np.ndarray) -> np.ndarray:
    """Inverse of alpha_beta_height: the normalised speed at each height (bisection; height falls as v falls)."""
    h = np.atleast_1d(np.asarray(h_m, float))
    lo, hi = np.full(h.shape, 1e-4), np.full(h.shape, 1 - 1e-9)
    for _ in range(60):
        mid = (lo + hi) / 2
        above = alpha_beta_height(alpha, beta, mid) > h  # at speed mid it would still be higher: slow down more
        lo, hi = np.where(above, lo, mid), np.where(above, mid, hi)
    return (lo + hi) / 2


@dataclass
class AlphaBeta:
    alpha: float
    beta: float
    v0: float
    slope_deg: float
    alpha_sigma: float = float("nan")
    beta_sigma: float = float("nan")
    heldout_rms_kms: float = float("nan")
    train_rms_kms: float = float("nan")
    n: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> float:
        return math.log(self.alpha * math.sin(math.radians(self.slope_deg)))

    def triage(self) -> dict[str, Any]:
        """Sansom et al. 2019 eqs. (10), (11): where the event sits against the two 50 g lines."""
        x, b = self.x, self.beta
        mu0 = MU0_LINE - 3 * x
        mu23 = MU23_LINE - x
        if mu23 > 0 and b < mu23:
            verdict, words = "likely", "below the spinning-body (mu = 2/3) line: a likely meteorite dropper"
        elif mu0 <= 0 or b > mu0:
            verdict, words = "unlikely", "above the non-spinning (mu = 0) line: unlikely to have dropped 50 g or more"
        else:
            verdict, words = "possible", "between the two lines: a possible dropper, depending on how it spun"
        return {"verdict": verdict, "words": words, "ln_alpha_sin_gamma": x, "ln_beta": math.log(b),
                "line_mu0_ln_beta": math.log(mu0) if mu0 > 0 else None,
                "line_mu23_ln_beta": math.log(mu23) if mu23 > 0 else None}

    def masses_kg(self, cd_a: float = 1.5, rho_m: float = 3500.0, mu: float = 2 / 3, v_end: float | None = None) -> tuple[float, float]:
        """Entry and final mass (Sansom et al. 2019 eqs. 3, 4 with the DFN module's constants); v_end defaults to
        the last normalised speed seen."""
        m0 = (0.5 * cd_a * RHO0 * H0_M / (self.alpha * math.sin(math.radians(self.slope_deg)) * rho_m ** (2 / 3))) ** 3
        ve = self.extra.get("v_end", 0.0) if v_end is None else v_end
        return m0, m0 * math.exp(-self.beta / (1 - mu) * (1 - ve ** 2))


def _ab_fit(v: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    b0 = 1.0
    a0 = math.exp(y[-1]) / (2 * b0)
    res = least_squares(lambda th: _q4(th, v, y), [a0, b0], bounds=([0.001, 1e-5], [1e4, 50.0]), x_scale="jac")
    return float(res.x[0]), float(res.x[1])


def fit_alpha_beta(height_m, speed_ms, slope_deg: float, v0: float | None = None, n_boot: int = 200,
                   seed: int = 0) -> AlphaBeta:
    """Alpha and beta from heights and speeds (any order). v0 defaults to the mean of the first ten speeds (as the
    DFN module does). Uncertainty by bootstrap over the points; the held-out check fits the upper 75% of the flight
    and forecasts the speed over the lower 25%."""
    h = np.asarray(height_m, float)
    s = np.asarray(speed_ms, float)
    ok = np.isfinite(h) & np.isfinite(s) & (s > 1.0)
    h, s = h[ok], s[ok]
    order = np.argsort(-h)
    h, s = h[order], s[order]
    if len(h) < 8:
        raise ValueError("At least eight heights with speeds are needed.")
    v0 = float(np.mean(s[:10])) if v0 is None else float(v0)
    v, y = s / v0, h / H0_M
    alpha, beta = _ab_fit(v, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        k = rng.integers(0, len(v), len(v))
        try:
            boots.append(_ab_fit(v[k], y[k]))
        except ValueError:
            continue
    boots_a = np.array(boots) if boots else np.full((1, 2), np.nan)
    cut = int(len(h) * TRAIN_FRACTION)
    a_tr, b_tr = _ab_fit(v[:cut], y[:cut])
    pred = alpha_beta_speed(a_tr, b_tr, h[cut:]) * v0
    fit_tr = alpha_beta_speed(a_tr, b_tr, h[:cut]) * v0
    # Half the 16 to 84 percentile range: the bootstrap is often lopsided (alpha and beta trade off along a valley).
    spread = (np.nanpercentile(boots_a, 84, axis=0) - np.nanpercentile(boots_a, 16, axis=0)) / 2
    return AlphaBeta(alpha, beta, v0, slope_deg, float(spread[0]), float(spread[1]),
                     float(np.sqrt(np.mean((pred - s[cut:]) ** 2)) / 1000), float(np.sqrt(np.mean((fit_tr - s[:cut]) ** 2)) / 1000),
                     len(h), {"v_end": float(v[-1]), "train_heldout_cut_m": float(h[cut])})


# --- single body with ablation, and fragmentation as a dynamics change --------------------------------------------

@dataclass
class BodyFit:
    v0: float
    gamma0: float  # m^2/kg: cd A rho_m^(-2/3) m_entry^(-1/3) / 2
    sigma: float  # s^2/m^2
    jump_h: float | None = None  # a fragmentation: drag per unit mass multiplied by `jump` below this height
    jump: float = 1.0
    rss: float = float("nan")
    bic: float = float("nan")
    heldout_rms_kms: float = float("nan")
    k: int = 3

    def entry_mass_kg(self, cd_a: float = 1.5, rho_m: float = 3500.0) -> float:
        return (0.5 * cd_a / (self.gamma0 * rho_m ** (2 / 3))) ** 3


def body_speeds(fit: BodyFit, h_eval: np.ndarray, slope_deg: float, atm: Atmosphere, h_top: float) -> tuple[np.ndarray, np.ndarray]:
    """(speed, mass fraction) at each height, integrating down from h_top."""
    sg = math.sin(math.radians(slope_deg))

    def rhs(h, st):
        v, mu = st
        if v <= 10 or mu <= 1e-9:
            return [0.0, 0.0]
        gam = fit.gamma0 * (fit.jump if fit.jump_h is not None and h < fit.jump_h else 1.0)
        rho = atm.rho(h)
        dv = (gam * mu ** (-1 / 3) * rho * v * v - G0 * sg) / (v * sg)
        dmu = fit.sigma * gam * mu ** (2 / 3) * rho * v * v / sg
        return [dv, dmu]

    h_eval = np.asarray(h_eval, float)
    lo = float(min(h_eval.min(), h_top))
    sol = solve_ivp(rhs, (h_top, lo - 1), [fit.v0, 1.0], t_eval=np.sort(h_eval)[::-1], rtol=1e-7, atol=[1e-3, 1e-10],
                    max_step=(h_top - lo) / 50 + 1)
    if not sol.success or sol.y.shape[1] != len(h_eval):
        return np.full(len(h_eval), np.nan), np.full(len(h_eval), np.nan)
    order = np.argsort(-h_eval)
    v, mu = np.empty(len(h_eval)), np.empty(len(h_eval))
    v[order], mu[order] = sol.y[0], sol.y[1]
    return v, mu


def _fit_body(h, s, slope_deg, atm, jump_h=None, start=None) -> BodyFit:
    h_top = float(h.max())
    # start: [v0, log10 gamma0, log10 sigma] from an earlier fit (one descent); else three drag starts.
    p0 = start or [float(np.mean(s[:5])), -4.0, -8.0]

    def unpack(th):
        return BodyFit(th[0], 10 ** th[1], 10 ** th[2], jump_h, 10 ** th[3] if jump_h is not None else 1.0)

    def resid(th):
        v, _ = body_speeds(unpack(th), h, slope_deg, atm, h_top)
        r = v - s
        return np.where(np.isfinite(r), r, 1e4)

    lo = [p0[0] * 0.8, -7.0, -11.0] + ([0.0] if jump_h is not None else [])
    hi = [p0[0] * 1.2, 0.0, -5.0] + ([1.5] if jump_h is not None else [])
    best = None
    for g_start in ((p0[1],) if start is not None else (-5.0, -3.5, -2.0)):
        x0 = [p0[0], g_start, p0[2]] + ([0.2] if jump_h is not None else [])
        try:
            res = least_squares(resid, x0, bounds=(lo, hi), x_scale=[100.0, 1.0, 1.0] + ([1.0] if jump_h is not None else []))
        except ValueError:
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        raise ValueError("The single-body fit failed.")
    f = unpack(best.x)
    f.k = len(best.x) + (1 if jump_h is not None else 0)  # the jump height counts as a fitted parameter
    f.rss = float(2 * best.cost)
    n = len(h)
    f.bic = n * math.log(max(f.rss / n, 1e-12)) + f.k * math.log(n)
    return f


def fit_single_body(height_m, speed_ms, slope_deg: float, atm: Atmosphere | None = None,
                    look_for_fragmentation: bool = True) -> dict[str, Any]:
    """The drag and ablation law fitted to the upper 75% of the flight and checked on the rest; then, on the whole
    flight, a fragmentation (one jump in drag per unit mass) accepted only with a decisive BIC lead."""
    atm = atm or standard_atmosphere()
    h = np.asarray(height_m, float)
    s = np.asarray(speed_ms, float)
    ok = np.isfinite(h) & np.isfinite(s) & (s > 1.0)
    h, s = h[ok], s[ok]
    order = np.argsort(-h)
    h, s = h[order], s[order]
    cut = int(len(h) * TRAIN_FRACTION)
    train = _fit_body(h[:cut], s[:cut], slope_deg, atm)
    v_pred, _ = body_speeds(train, h[cut:], slope_deg, atm, float(h[0]))
    train.heldout_rms_kms = float(np.sqrt(np.nanmean((v_pred - s[cut:]) ** 2)) / 1000)
    whole = _fit_body(h, s, slope_deg, atm)
    out: dict[str, Any] = {"single": whole, "train_only": train, "fragmentation": None, "n": len(h),
                           "atmosphere": atm.source}
    if look_for_fragmentation and len(h) >= 20:
        best = None
        for q in np.linspace(0.2, 0.8, 7):
            jh = float(np.quantile(h, 1 - q))
            try:
                f = _fit_body(h, s, slope_deg, atm, jump_h=jh, start=[whole.v0, math.log10(whole.gamma0), math.log10(whole.sigma)])
            except ValueError:
                continue
            if best is None or f.bic < best.bic:
                best = f
        if best is not None and whole.bic - best.bic >= DECISIVE_DELTA_BIC and best.jump > 1.05:
            out["fragmentation"] = best
    final = out["fragmentation"] or whole
    _, mu = body_speeds(final, h[-1:], slope_deg, atm, float(h[0]))
    out["end_mass_fraction"] = float(mu[0])
    return out


# --- dark flight and the strewn field ---------------------------------------------------------------------------------

def drag_coefficient(mach: float, shape_a: float) -> float:
    """Drag coefficient against Mach number for a sphere (A = 1.21) and a cube (A = 2.18), interpolated (and
    extrapolated) linearly in the shape factor A. The two curves are Carter et al. (2011) as written in the Desert
    Fireball Network's dark-flight code (DFN_darkflight, atm_functions.dragcoefff, MIT licence)."""
    m = max(float(mach), 0.0)
    sphere = 0.45 * m * m + 0.424 if m < 0.722 else 2.1 * math.exp(-1.2 * (m + 0.35)) - 8.9 * math.exp(-2.2 * (m + 0.35)) + 0.92
    cube = 0.60 * m * m + 1.04 if m < 1.15 else 2.1 * math.exp(-1.16 * (m + 0.35)) - 6.5 * math.exp(-2.23 * (m + 0.35)) + 1.67
    return float(sphere + (cube - sphere) * (shape_a - 1.21) / (2.18 - 1.21))


@dataclass(frozen=True)
class EndState:
    """The luminous end: place, height, speed, the direction it travels (azimuth from north, degrees, and the slope
    below the horizontal), with one-sigma uncertainties."""

    lat: float
    lon: float
    h_m: float
    speed_ms: float
    azimuth_deg: float
    slope_deg: float
    sigma_pos_m: float = 100.0
    sigma_h_m: float = 100.0
    sigma_speed_ms: float = 300.0
    sigma_angle_deg: float = 0.5
    ground_m: float = 0.0


def dark_flight(end: EndState, mass_kg: float, atm: Atmosphere, shape_a: float = 1.21, rho_m: float = 3500.0,
                cd_scale: float = 1.0, wind_scale: float = 1.0, dt_max: float = 0.5) -> dict[str, Any]:
    """One stone from the luminous end to the ground. Drag F = cd rho v_rel^2 S / 2 with S = A (m/rho_m)^(2/3)
    (A = 1.21 is a sphere) and cd from the Mach number (drag_coefficient) times cd_scale; gravity falls with height;
    no ablation (it has stopped glowing)."""
    area = shape_a * (mass_kg / rho_m) ** (2 / 3)
    k = 0.5 * area / mass_kg
    az, sl = math.radians(end.azimuth_deg), math.radians(end.slope_deg)
    v = end.speed_ms * np.array([math.sin(az) * math.cos(sl), math.cos(az) * math.cos(sl), -math.sin(sl)])

    def rhs(t, st):
        z, vx, vy, vz = st[2], st[3], st[4], st[5]
        we, wn = atm.wind(z)
        rel = np.array([vx - wind_scale * we, vy - wind_scale * wn, vz])
        sp = float(np.linalg.norm(rel))
        cd = cd_scale * drag_coefficient(sp / atm.sound_speed(z), shape_a)
        g = G0 * (EARTH_R_M / (EARTH_R_M + z)) ** 2
        a = -k * cd * atm.rho(z) * sp * rel
        return [vx, vy, vz, a[0], a[1], a[2] - g]

    def ground(t, st):
        return st[2] - end.ground_m
    ground.terminal, ground.direction = True, -1
    sol = solve_ivp(rhs, (0, 3600), [0.0, 0.0, end.h_m, *v], events=ground, max_step=dt_max, rtol=1e-6, atol=1e-3)
    if not sol.t_events[0].size:
        raise ValueError("The stone did not reach the ground within an hour.")
    x, y, _, vx, vy, vz = sol.y_events[0][0]
    lat, lon = offset_latlon(end.lat, end.lon, x, y)
    return {"lat": lat, "lon": lon, "east_m": float(x), "north_m": float(y), "time_s": float(sol.t_events[0][0]),
            "impact_speed_ms": float(math.sqrt(vx * vx + vy * vy + vz * vz)), "mass_kg": mass_kg, "shape_a": shape_a,
            "path": [[*offset_latlon(end.lat, end.lon, a, b), float(c)] for a, b, c in zip(sol.y[0][::20], sol.y[1][::20], sol.y[2][::20], strict=True)]}


def offset_latlon(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    """A local east/north offset as latitude and longitude (spherical Earth; meters-level over tens of km)."""
    dlat = north_m / EARTH_R_M
    dlon = east_m / (EARTH_R_M * math.cos(math.radians(lat)))
    return float(lat + math.degrees(dlat)), float(lon + math.degrees(dlon))


def release_state(end: EndState, h_m: float, speed_ms: float) -> EndState:
    """A stone that separated earlier, at height h_m on the same straight bright-flight line, moving at speed_ms."""
    back = (h_m - end.h_m) / math.sin(math.radians(end.slope_deg))  # meters back along the track
    horiz = back * math.cos(math.radians(end.slope_deg))
    az = math.radians(end.azimuth_deg)
    lat, lon = offset_latlon(end.lat, end.lon, -horiz * math.sin(az), -horiz * math.cos(az))
    return EndState(lat, lon, h_m, speed_ms, end.azimuth_deg, end.slope_deg, end.sigma_pos_m, end.sigma_h_m,
                    end.sigma_speed_ms, end.sigma_angle_deg, end.ground_m)


def strewn_field(end: EndState, atm: Atmosphere, masses_kg: list[float], n_per_mass: int = 200, seed: int = 0,
                 cd_sigma: float = 0.1, shape_range: tuple[float, float] = (1.21, 1.55),
                 wind_sigma: float = 0.2, rho_m: float = 3500.0,
                 release: tuple[float, float, Any] | None = None) -> dict[str, Any]:
    """Monte Carlo landing points: the end state within its errors, the shape within its range, the drag curve scaled
    by 1 +- cd_sigma, the wind profile by 1 +- wind_sigma. With release = (lowest, highest height, speed(height)),
    each stone separates at a height drawn between the two (a fragmentation zone seen in the light curve) instead
    of at the luminous end. Returns the points and, per mass, the median spot and the 1 and 2
    sigma ellipses."""
    rng = np.random.default_rng(seed)
    pts = []
    for m in masses_kg:
        for _ in range(n_per_mass):
            start = end
            if release is not None:
                h_rel = rng.uniform(release[0], release[1])
                start = release_state(end, h_rel, float(release[2](h_rel)))
            dn, de = rng.normal(0, start.sigma_pos_m, 2)
            lat, lon = offset_latlon(start.lat, start.lon, de, dn)
            e = EndState(lat, lon, start.h_m + rng.normal(0, start.sigma_h_m), max(500.0, start.speed_ms + rng.normal(0, start.sigma_speed_ms)),
                         start.azimuth_deg + rng.normal(0, start.sigma_angle_deg), start.slope_deg + rng.normal(0, start.sigma_angle_deg),
                         ground_m=start.ground_m)
            try:
                r = dark_flight(e, m, atm, shape_a=rng.uniform(*shape_range), rho_m=rho_m,
                                cd_scale=max(0.5, 1 + rng.normal(0, cd_sigma)), wind_scale=1 + rng.normal(0, wind_sigma))
            except ValueError:
                continue
            r.pop("path")
            pts.append(r)
    by_mass = []
    for m in masses_kg:
        mine = [p for p in pts if p["mass_kg"] == m]
        if len(mine) < 5:
            continue
        en = np.array([[p["east_m"], p["north_m"]] for p in mine])
        mid = np.median(en, axis=0)
        cov = np.cov(en.T)
        by_mass.append({"mass_kg": m, "lat": offset_latlon(end.lat, end.lon, *mid)[0], "lon": offset_latlon(end.lat, end.lon, *mid)[1],
                        "east_m": float(mid[0]), "north_m": float(mid[1]), "ellipse_1s": _ellipse(end, mid, cov, 1.0),
                        "ellipse_2s": _ellipse(end, mid, cov, 2.0), "n": len(mine)})
    return {"points": pts, "by_mass": by_mass, "end": end.__dict__, "atmosphere": atm.source}


def _ellipse(end: EndState, mid: np.ndarray, cov: np.ndarray, k: float, n: int = 48) -> list[list[float]]:
    w, vecs = np.linalg.eigh(cov)
    w = np.sqrt(np.maximum(w, 0))
    out = []
    for a in np.linspace(0, 2 * math.pi, n + 1):
        d = mid + k * (vecs @ (w * np.array([math.cos(a), math.sin(a)])))
        lat, lon = offset_latlon(end.lat, end.lon, d[0], d[1])
        out.append([lon, lat])
    return out


def to_geojson(field_: dict[str, Any], found: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Points (every simulated stone), the median mass line, the 2-sigma ellipses and any recovered meteorites."""
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
              "properties": {"kind": "simulated", "mass_kg": p["mass_kg"]}} for p in field_["points"]]
    if field_["by_mass"]:
        feats.append({"type": "Feature", "geometry": {"type": "LineString",
                                                      "coordinates": [[b["lon"], b["lat"]] for b in field_["by_mass"]]},
                      "properties": {"kind": "mass line", "masses_kg": [b["mass_kg"] for b in field_["by_mass"]]}})
    for b in field_["by_mass"]:
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [b["ellipse_2s"]]},
                      "properties": {"kind": "2-sigma area", "mass_kg": b["mass_kg"]}})
    for f in found or []:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [f["lon"], f["lat"]]},
                      "properties": {"kind": "recovered", **{k: v for k, v in f.items() if k not in ("lat", "lon")}}})
    return {"type": "FeatureCollection", "features": feats}


def to_kml(field_: dict[str, Any], name: str) -> str:
    from xml.sax.saxutils import escape
    marks = []
    for b in field_["by_mass"]:
        ring = " ".join(f"{lon:.6f},{lat:.6f},0" for lon, lat in b["ellipse_2s"])
        marks.append(f"<Placemark><name>{b['mass_kg']:g} kg: 2-sigma area</name><Polygon><outerBoundaryIs><LinearRing>"
                     f"<coordinates>{ring}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")
        marks.append(f"<Placemark><name>{b['mass_kg']:g} kg: most likely spot</name><Point><coordinates>"
                     f"{b['lon']:.6f},{b['lat']:.6f},0</coordinates></Point></Placemark>")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            f"<name>{escape(name)}</name>{''.join(marks)}</Document></kml>\n")


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_M * math.asin(math.sqrt(a))


# --- the orbit before it met the Earth ----------------------------------------------------------------------------

GM_EARTH = 3.986004418e14  # m^3/s^2
GM_SUN = 1.32712440018e20  # m^3/s^2
AU_M = 1.495978707e11
OBLIQUITY_J2000_DEG = 23.4392911  # 84381.448 arcsec, IAU 1976


def pre_entry_orbit(when_utc: str, lat: float, lon: float, h_m: float, ra_deg: float, dec_deg: float,
                    v_inf_ms: float) -> dict[str, Any]:
    """Heliocentric orbit from the apparent radiant (right ascension and declination of date, as seen in an
    Earth-centred inertial frame, so Earth's rotation is already included), the speed before the atmosphere slowed
    it, and a point on the trajectory.

    The Earth's pull is undone exactly: the geocentric two-body hyperbola through that point is followed back to its
    incoming asymptote (this is what the zenith-attraction correction approximates), then the Earth's heliocentric
    velocity (JPL ephemeris via astropy) is added and the elements are taken in the J2000 ecliptic."""
    import astropy.units as u
    from astropy.coordinates import GCRS, TETE, EarthLocation, SkyCoord, get_body_barycentric_posvel
    from astropy.time import Time

    t = Time(when_utc, scale="utc")
    loc = EarthLocation.from_geodetic(lon * u.deg, lat * u.deg, h_m * u.m)
    pos, _ = loc.get_gcrs_posvel(t)
    r = pos.xyz.to_value(u.m)
    # The meteoroid moves away from its radiant: the opposite direction, turned from the frame of date to GCRS axes.
    rad = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame=TETE(obstime=t)).transform_to(GCRS(obstime=t))
    v = -v_inf_ms * rad.cartesian.xyz.value
    # Geocentric hyperbola through (r, v): its incoming asymptote's velocity.
    energy = v @ v / 2 - GM_EARTH / np.linalg.norm(r)
    if energy <= 0:
        raise ValueError("Slower than escape speed: this was not on a heliocentric orbit.")
    v_g = math.sqrt(2 * energy)
    hvec = np.cross(r, v)
    evec = np.cross(v, hvec) / GM_EARTH - r / np.linalg.norm(r)
    e = np.linalg.norm(evec)
    p_hat = evec / e
    q_hat = np.cross(hvec / np.linalg.norm(hvec), p_hat)
    nu_inf = math.acos(-1 / e)
    # Velocity direction as the true anomaly goes to -nu_inf (the far past), from v(nu) = (mu/h)(-sin nu p + (e + cos nu) q).
    d = math.sin(nu_inf) * p_hat + (e - 1 / e) * q_hat
    vg_vec = v_g * d / np.linalg.norm(d)
    geo = SkyCoord(x=-vg_vec[0], y=-vg_vec[1], z=-vg_vec[2], representation_type="cartesian", frame=GCRS(obstime=t))
    # Earth's heliocentric state (barycentric Earth minus barycentric Sun), ICRS axes = GCRS axes to well under an arcsec.
    pe, ve = get_body_barycentric_posvel("earth", t.tdb)
    ps, vs = get_body_barycentric_posvel("sun", t.tdb)
    R = (pe - ps).xyz.to_value(u.m) + r
    V = (ve - vs).xyz.to_value(u.m / u.s) + vg_vec
    eps = math.radians(OBLIQUITY_J2000_DEG)
    rot = np.array([[1, 0, 0], [0, math.cos(eps), math.sin(eps)], [0, -math.sin(eps), math.cos(eps)]])
    R, V = rot @ R, rot @ V
    el = elements(R, V, GM_SUN)
    gsph = geo.spherical
    el.update({"v_g_ms": v_g, "geocentric_ra_deg": float(gsph.lon.to_value(u.deg)) % 360,
               "geocentric_dec_deg": float(gsph.lat.to_value(u.deg)), "v_h_ms": float(np.linalg.norm(V))})
    return el


def elements(r: np.ndarray, v: np.ndarray, mu: float) -> dict[str, float]:
    """Keplerian elements (a, e, i, node, argument of perihelion, q, aphelion, Tisserand with Jupiter) in the frame
    of r and v; angles in degrees, distances in AU."""
    rn = np.linalg.norm(r)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    evec = np.cross(v, h) / mu - r / rn
    e = float(np.linalg.norm(evec))
    a = 1 / (2 / rn - v @ v / mu)
    inc = math.degrees(math.acos(h[2] / hn))
    n = np.array([-h[1], h[0], 0.0])
    nn = np.linalg.norm(n)
    node = math.degrees(math.atan2(n[1], n[0])) % 360
    peri = math.degrees(math.acos(np.clip(n @ evec / (nn * e), -1, 1)))
    if evec[2] < 0:
        peri = 360 - peri
    a_au = a / AU_M
    q = a_au * (1 - e)
    a_jup = 5.2026
    tj = a_jup / a_au + 2 * math.cos(math.radians(inc)) * math.sqrt(a_au / a_jup * (1 - e * e)) if e < 1 else float("nan")
    return {"a_au": float(a_au), "e": e, "i_deg": float(inc), "node_deg": float(node), "peri_deg": float(peri),
            "q_au": float(q), "Q_au": float(a_au * (1 + e)) if e < 1 else float("inf"), "tisserand_jupiter": float(tj)}
