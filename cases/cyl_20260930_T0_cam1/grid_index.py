"""Grid indices (i, j) of every dot of every plane, carried from plane to plane by tracking.

i grows along the rows to the image right (world +X); j grows along the columns *upwards* (world +Y), so a dot
(i, j) of plane k sits at (pitch_x i, pitch_y j, z_k) in the world frame.

Why tracking.  The tube's horizontal magnification falls as the board approaches the camera, so the grid
contracts towards the leftmost column, which hardly moves, and a new column enters on the right (a 9th complete
column from z = 14 mm).  Counting columns from either edge would therefore shift the indices of some planes.
Dots move less than ~7 px between planes 2 mm apart, against a pitch of 115-155 px, so the origin dot is followed
from plane to plane, each plane is indexed on its own lattice anchored there, and every dot's index is then checked
against its nearest neighbour in the previous plane.
"""
from __future__ import annotations

import numpy as np


def lattice_vectors(P: np.ndarray):
    """Median nearest-neighbour steps: a along a row (image right) and b down a column (image down)."""
    D = P[None, :, :] - P[:, None, :]
    dist = np.hypot(D[..., 0], D[..., 1])
    np.fill_diagonal(dist, np.inf)
    k = min(4, len(P) - 1)
    nn = np.argsort(dist, axis=1)[:, :k]
    vecs = D[np.arange(len(P))[:, None], nn].reshape(-1, 2)
    horiz = vecs[np.abs(vecs[:, 0]) > np.abs(vecs[:, 1])].copy()
    horiz[horiz[:, 0] < 0] *= -1
    vert = vecs[np.abs(vecs[:, 1]) >= np.abs(vecs[:, 0])].copy()
    vert[vert[:, 1] < 0] *= -1
    return np.median(horiz, axis=0), np.median(vert, axis=0)


def fit_lattice(P: np.ndarray, ij: np.ndarray):
    """Affine lattice P = p0 + A [i, jd] by least squares.  Returns (p0, A, residual norms)."""
    G = np.column_stack([np.ones(len(P)), ij[:, 0], ij[:, 1]])
    coef, *_ = np.linalg.lstsq(G, P, rcond=None)
    p0, A = coef[0], coef[1:].T
    res = np.linalg.norm(P - (p0 + ij @ A.T), axis=1)
    return p0, A, res


def index_plane(P_fit: np.ndarray, seed, a, b, iters: int = 6):
    """Lattice coordinates (i, jd) of P_fit relative to `seed` (jd grows downwards in the image).

    The affine lattice is refitted to all points until the assignment no longer changes.
    Returns (ij, p0, A, residuals).
    """
    A = np.column_stack([a, b])
    p0 = np.asarray(seed, float)
    ij = None
    for _ in range(iters):
        new = np.rint(np.linalg.solve(A, (P_fit - p0).T).T).astype(int)
        p0, A, res = fit_lattice(P_fit, new)
        if ij is not None and np.array_equal(new, ij):
            break
        ij = new
    return ij, p0, A, res


def assign(P: np.ndarray, p0, A) -> np.ndarray:
    """Nearest lattice coordinates (i, jd) of arbitrary points under a fitted lattice."""
    return np.rint(np.linalg.solve(A, (np.asarray(P, float) - p0).T).T).astype(int)


def index_planes(planes: list, origin_row_v: float, gate: float, tol_frac: float) -> dict:
    """Index every plane.

    planes: list (increasing z) of dicts {"z": float, "P": (N,2) centres, "ok": (N,) bool usable for the lattice}.
    Returns {"ij": [ (N,2) int arrays with j upwards ], "lattice": [(p0, A)], "seed": [index into P],
             "checks": {...}} where the origin dot of plane 0 is (0, 0).
    """
    out_ij, lattices, seeds = [], [], []
    checks = {"lattice_max_res_px": [], "lattice_bad": [], "track_mismatch": [], "track_pairs": [],
              "duplicates": []}
    prev_seed_xy, prev = None, None
    for k, pl in enumerate(planes):
        P, ok = np.asarray(pl["P"], float), np.asarray(pl["ok"], bool)
        Pok = P[ok]
        a, b = lattice_vectors(Pok)
        if k == 0:
            # provisional indexing from the dot nearest the image centre, then the origin: the leftmost complete
            # column, on the row nearest origin_row_v
            c0 = Pok[np.argmin(np.hypot(Pok[:, 0] - Pok[:, 0].mean(), Pok[:, 1] - Pok[:, 1].mean()))]
            ij0, p0, A, _ = index_plane(Pok, c0, a, b)
            i_min = ij0[:, 0].min()
            col = np.where(ij0[:, 0] == i_min)[0]
            s = col[np.argmin(np.abs(Pok[col, 1] - origin_row_v))]
            seed_xy = Pok[s]
        else:
            d = np.hypot(Pok[:, 0] - prev_seed_xy[0], Pok[:, 1] - prev_seed_xy[1])
            s = int(np.argmin(d))
            if d[s] > gate:
                raise RuntimeError(f"plane {k}: origin dot lost (nearest {d[s]:.1f} px > gate {gate})")
            seed_xy = Pok[s]
        ij_ok, p0, A, res = index_plane(Pok, seed_xy, a, b)
        # make the seed exactly (0, 0) (index_plane is relative to it, but guard against a rounding shift)
        ij_ok = ij_ok - ij_ok[s]
        p0, A, res = fit_lattice(Pok, ij_ok)
        tol = tol_frac * min(np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1]))
        checks["lattice_max_res_px"].append(float(res.max()))
        checks["lattice_bad"].append(int(np.sum(res > tol)))
        ij_all = assign(P, p0, A)
        ij_all[ok] = ij_ok
        ij_up = ij_all.copy()
        ij_up[:, 1] = -ij_up[:, 1]                      # j grows upwards (world +Y)
        # duplicates among usable dots
        keys = [tuple(v) for v in ij_up[ok]]
        checks["duplicates"].append(len(keys) - len(set(keys)))
        # tracking cross-check against the previous plane
        mism, pairs = 0, 0
        if prev is not None:
            Pp, okp, ijp = prev
            Q = Pp[okp]
            for p, v in zip(P[ok], ij_up[ok]):
                d = np.hypot(Q[:, 0] - p[0], Q[:, 1] - p[1])
                m = int(np.argmin(d))
                if d[m] <= gate:
                    pairs += 1
                    if tuple(ijp[okp][m]) != tuple(v):
                        mism += 1
        checks["track_mismatch"].append(mism)
        checks["track_pairs"].append(pairs)
        out_ij.append(ij_up)
        lattices.append((p0, A))
        seeds.append(int(np.where(ok)[0][s]))
        prev_seed_xy, prev = seed_xy, (P, ok, ij_up)
    return {"ij": out_ij, "lattice": lattices, "seed": seeds, "checks": checks}


def expected_complete(p0, A, ij_range, width: int, height: int, half_w: float, half_h: float, margin: float):
    """Lattice points whose whole dot (half sizes half_w, half_h plus margin) lies on the sensor, as (i, j_up)."""
    out = []
    for i in range(ij_range[0][0], ij_range[0][1] + 1):
        for jd in range(ij_range[1][0], ij_range[1][1] + 1):
            u, v = p0 + A @ np.array([i, jd], float)
            if (u - half_w - margin >= 0 and u + half_w + margin <= width - 1
                    and v - half_h - margin >= 0 and v + half_h + margin <= height - 1):
                out.append((i, -jd))
    return out
