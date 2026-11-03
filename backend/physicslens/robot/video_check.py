"""Does the arm's logged joint path match what the top camera saw?

Forward kinematics of the logged joint angles gives a 3D point on the gripper in the simulator's metric frame; a
pinhole camera maps it into the image. The camera (pose, focal length) and the tracked point's place on the gripper
are fitted on training episodes, then frozen and checked against SAM 3.1's track of the gripper on held-out episodes.

Pure functions except fk_body (MuJoCo, local and free). No network, no model calls.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize, stats

# ---------- geometry ----------


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rotation matrix of an axis-angle vector."""
    r = np.asarray(rvec, float)
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * kx @ kx


def rotation_to_rvec(rot: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix (inverse of rodrigues away from 180 degrees)."""
    c = np.clip((np.trace(rot) - 1) / 2, -1.0, 1.0)
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]) / (2 * np.sin(th))
    return w * th


def tracked_points(pos: np.ndarray, rot: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """World points [N, 3] of a point fixed in a moving body: pos [N, 3] + rot [N, 3, 3] @ offset."""
    return pos + np.einsum("nij,j->ni", rot, np.asarray(offset, float))


@dataclass(frozen=True)
class Camera:
    """Pinhole: square pixels, no skew, no lens distortion, principal point fixed at the image centre."""
    rvec: np.ndarray  # world to camera rotation (axis-angle)
    tvec: np.ndarray  # world to camera translation (m)
    f: float  # focal length (px)
    c: tuple[float, float]  # principal point (px)

    def to_camera(self, x: np.ndarray) -> np.ndarray:
        return x @ rodrigues(self.rvec).T + self.tvec

    def project(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates [N, 2] (col, row) and depth [N] (m, along the optical axis)."""
        xc = self.to_camera(np.asarray(x, float))
        z = xc[:, 2]
        uv = self.f * xc[:, :2] / z[:, None] + np.asarray(self.c)
        return uv, z

    def as_dict(self) -> dict:
        r = rodrigues(self.rvec)
        centre = -r.T @ self.tvec  # camera position in the world frame
        return {"rvec": self.rvec.tolist(), "tvec_m": self.tvec.tolist(), "focal_px": self.f,
                "principal_point_px": list(self.c), "camera_position_m": centre.tolist(),
                "optical_axis_world": r[2].tolist()}


def dlt(x: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """3x4 projection matrix from 3D-2D pairs (normalised direct linear transform, least squares)."""
    x, uv = np.asarray(x, float), np.asarray(uv, float)
    mx, sx = x.mean(0), np.sqrt(3) / np.mean(np.linalg.norm(x - x.mean(0), axis=1))
    mu, su = uv.mean(0), np.sqrt(2) / np.mean(np.linalg.norm(uv - uv.mean(0), axis=1))
    tx = np.diag([sx, sx, sx, 1.0])
    tx[:3, 3] = -sx * mx
    tu = np.diag([su, su, 1.0])
    tu[:2, 2] = -su * mu
    xn = (np.c_[x, np.ones(len(x))] @ tx.T)
    un = (np.c_[uv, np.ones(len(uv))] @ tu.T)[:, :2]
    rows = []
    for (X, Y, Z, W), (u, v) in zip(xn, un, strict=True):
        p = [X, Y, Z, W]
        rows.append([*p, 0, 0, 0, 0, *(-u * np.array(p))])
        rows.append([0, 0, 0, 0, *p, *(-v * np.array(p))])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    pn = vt[-1].reshape(3, 4)
    return np.linalg.inv(tu) @ pn @ tx


def camera_from_projection(p: np.ndarray, c: tuple[float, float], x_sample: np.ndarray) -> Camera:
    """A Camera close to a general 3x4 projection: RQ decomposition, sign fixed so the sample points lie in front."""
    p = np.asarray(p, float)
    # p is defined up to scale and sign: choose the sign that puts the points in front (third row positive).
    if np.median(np.c_[x_sample, np.ones(len(x_sample))] @ p[2]) < 0:
        p = -p
    k, r = linalg.rq(p[:, :3])
    d = np.diag(np.sign(np.diag(k)))
    k, r = k @ d, d @ r
    t = np.linalg.solve(k, p[:, 3])
    if np.linalg.det(r) < 0:  # a reflection from noise: take the nearest rotation
        u, _, vt = np.linalg.svd(r)
        r = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    f = float((k[0, 0] + k[1, 1]) / 2 / k[2, 2])
    return Camera(rotation_to_rvec(r), t, f, c)


@dataclass(frozen=True)
class CameraFit:
    camera: Camera
    offset: np.ndarray  # tracked point in the body frame (m)
    rms_px: float
    median_px: float
    n: int
    params: int = 10


def _unpack(z: np.ndarray, c: tuple[float, float]) -> tuple[Camera, np.ndarray]:
    return Camera(z[0:3], z[3:6], float(np.exp(z[6])), c), z[7:10]


def residuals(z: np.ndarray, pos: np.ndarray, rot: np.ndarray, uv: np.ndarray, c: tuple[float, float]) -> np.ndarray:
    cam, off = _unpack(z, c)
    pred, depth = cam.project(tracked_points(pos, rot, off))
    r = (pred - uv).ravel()
    # A point behind the camera is not an image point: a large residual keeps the fit away from that branch.
    return np.where(np.repeat(depth, 2) > 1e-3, r, 1e4)


def _project_p(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    h = np.c_[x, np.ones(len(x))] @ p.T
    return h[:, :2] / h[:, 2:3]


def _offset_given_projection(p: np.ndarray, pos: np.ndarray, rot: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """The body-frame point that best fits a fixed projection: linear, since (p1 - u p3) . [pos + R o, 1] = 0."""
    a_rows, b_rows = [], []
    for row, coord in ((p[0], uv[:, 0]), (p[1], uv[:, 1])):
        a = row[None, :] - coord[:, None] * p[2][None, :]  # [N, 4]
        a_rows.append(np.einsum("ni,nij->nj", a[:, :3], rot))
        b_rows.append(-(np.einsum("ni,ni->n", a[:, :3], pos) + a[:, 3]))
    sol, *_ = np.linalg.lstsq(np.concatenate(a_rows), np.concatenate(b_rows), rcond=None)
    return sol


def linear_start(pos: np.ndarray, rot: np.ndarray, uv: np.ndarray, offset0: np.ndarray,
                 iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Alternate a DLT for the projection and a linear solve for the offset, refitting each time on the frames
    within three times the median error (at least 3 px), so a few mis-tracked frames do not set the start."""
    off = np.asarray(offset0, float)
    keep = np.ones(len(uv), bool)
    p = None
    for _ in range(iters):
        p = dlt(tracked_points(pos[keep], rot[keep], off), uv[keep])
        off = _offset_given_projection(p, pos[keep], rot[keep], uv[keep])
        err = pixel_errors(_project_p(p, tracked_points(pos, rot, off)), uv)
        keep = err <= max(3.0 * float(np.median(err)), 3.0)
        if keep.sum() < 12:
            keep = np.ones(len(uv), bool)
    return p, off


def fit_camera(pos: np.ndarray, rot: np.ndarray, uv: np.ndarray, size: tuple[int, int],
               loss_scale_px: float = 5.0, focal_px: tuple[float, float] = (100.0, 1e5)) -> CameraFit:
    """Pinhole pose (6), log focal length (1) and the tracked point's offset in the body frame (3): ten numbers.

    Started from linear_start (from seven starting offsets: zero and 5 cm along each body axis), refined with a
    soft-L1 loss (residuals beyond loss_scale_px count less), so occluded or mis-tracked frames pull the camera less
    than clean ones. The start with the lowest final cost wins. focal_px bounds the focal length; equal bounds fix
    it (a near-overhead view of a shallow workspace barely tells perspective from a far, long-lens camera, so the
    caller profiles it).
    """
    c = (size[0] / 2.0, size[1] / 2.0)
    starts = [np.zeros(3)] + [s * 0.05 * np.eye(3)[k] for k in range(3) for s in (1.0, -1.0)]
    best = None
    for off0 in starts:
        try:
            p, off = linear_start(pos, rot, uv, off0)
            x = tracked_points(pos, rot, off)
            cam0 = camera_from_projection(p, c, x)
        except (np.linalg.LinAlgError, ValueError):
            continue
        lo, hi = np.log(focal_px[0]), np.log(focal_px[1]) + 1e-9
        lf = float(np.clip(np.log(max(cam0.f, 1.0)), lo, hi))
        # Keep the start's image scale when the focal length is clipped: move the camera along its axis instead.
        tvec = cam0.tvec * np.r_[1.0, 1.0, 0.0] + np.r_[0.0, 0.0, cam0.tvec[2] * np.exp(lf) / max(cam0.f, 1e-9)]
        z0 = np.r_[cam0.rvec, tvec, lf, off]
        bounds = (np.r_[np.full(6, -np.inf), lo, np.full(3, -np.inf)], np.r_[np.full(6, np.inf), hi, np.full(3, np.inf)])
        sol = optimize.least_squares(residuals, z0, args=(pos, rot, uv, c), loss="soft_l1", bounds=bounds,
                                     f_scale=loss_scale_px, x_scale="jac", max_nfev=4000)
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise ValueError("no camera start could be computed from these points")
    cam, off = _unpack(best.x, c)
    err = pixel_errors(cam.project(tracked_points(pos, rot, off))[0], uv)
    return CameraFit(cam, off, float(np.sqrt(np.mean(err**2))), float(np.median(err)), len(uv))


# ---------- comparison ----------


def pixel_errors(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Euclidean distance in pixels per frame."""
    return np.linalg.norm(np.asarray(pred, float) - np.asarray(obs, float), axis=1)


def mm_errors(err_px: np.ndarray, depth_m: np.ndarray, f_px: float) -> np.ndarray:
    """A pixel error as a distance across the line of sight at the point's own depth: err * Z / f, in mm.
    The metric scale comes from the simulator's link lengths, through the fitted camera."""
    return 1000.0 * np.asarray(err_px) * np.asarray(depth_m) / f_px


def shift_frames(q: np.ndarray, shift: float) -> np.ndarray:
    """q [N, J] read `shift` frames later (q_shifted[i] = q[i + shift]), linear between frames, held at the ends."""
    n = len(q)
    i = np.clip(np.arange(n) + shift, 0, n - 1)
    return np.stack([np.interp(i, np.arange(n), q[:, j]) for j in range(q.shape[1])], axis=1)


def summary(err: np.ndarray) -> dict[str, float]:
    e = np.asarray(err, float)
    return {"n": len(e), "median": float(np.median(e)), "mean": float(np.mean(e)),
            "rms": float(np.sqrt(np.mean(e**2))), "p90": float(np.percentile(e, 90)), "max": float(np.max(e))}


def t_interval(values: list[float] | np.ndarray, level: float = 0.95) -> dict[str, float]:
    """Mean over episodes and its Student-t interval (episodes are the independent units, not frames)."""
    v = np.asarray(values, float)
    m = float(v.mean())
    if len(v) < 2:
        return {"mean": m, "lo": m, "hi": m, "n": len(v)}
    h = float(stats.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return {"mean": m, "lo": m - h, "hi": m + h, "n": len(v)}


def binned(err: np.ndarray, factor: np.ndarray, edges: list[float]) -> list[dict[str, float]]:
    """Median error in each bin of a factor (left-closed bins; the last is closed on both sides)."""
    out = []
    for a, b in itertools.pairwise(edges):
        sel = (factor >= a) & ((factor < b) | (b == edges[-1]) & (factor <= b))
        if sel.any():
            out.append({"from": float(a), "to": float(b), "frames": int(sel.sum()),
                        "median_err": float(np.median(err[sel])), "p90_err": float(np.percentile(err[sel], 90))})
    return out


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman's rho; None when either series is constant (no ranking to correlate)."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(stats.spearmanr(a, b).statistic)


# ---------- forward kinematics (MuJoCo, local) ----------


def fk_body(model, joint_map, q_deg: np.ndarray, body: str) -> tuple[np.ndarray, np.ndarray]:
    """World position [N, 3] and rotation [N, 3, 3] of a body at each logged pose (degrees in the log's convention)."""
    import mujoco

    d = mujoco.MjData(model)
    adr = [model.joint(n).qposadr[0] for n in joint_map.names]
    bid = model.body(body).id
    pos, rot = np.empty((len(q_deg), 3)), np.empty((len(q_deg), 3, 3))
    for i, q in enumerate(q_deg):
        d.qpos[adr] = joint_map.to_model(q)
        mujoco.mj_kinematics(model, d)
        pos[i] = d.xpos[bid]
        rot[i] = d.xmat[bid].reshape(3, 3)
    return pos, rot
