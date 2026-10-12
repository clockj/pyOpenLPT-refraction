"""
Image correction files: the optional "correction" of a camera in config.json.

A camera's correction c = (du, dv), in pixels, says where its images sit relative to where its model puts them: a
detection at pixel d has the model's image point d - c(d). The field VSC (field_vsc.py) fits it from long tracks,
and the library (ImageCorrection and CorrectedCamera in inc/libMath/Camera.h) applies it to every projection and
line of sight.

c is held at the nodes of a regular grid, read bilinearly between them and as the nearest edge value beyond the
grid. The file is text, '#' starting a comment:

    VERSION 1
    IMAGE_SIZE n_col n_row     the camera's image size
    GRID u0 v0 step nu nv      node (i, j) at pixel (u0 + i step, v0 + j step)
    DU                         then nv rows of nu values (v outer)
    DV                         the same for dv

`Correction` reads, writes and evaluates it with the library's arithmetic, operation for operation, so that both give
the same numbers.
"""
from __future__ import annotations

import math
import os
from typing import Callable, Iterable, Optional

import numpy as np

STEP_PX = 8.0       # node spacing of the corrections the VSC writes
SLOPE_MAX = 0.5     # the library refuses steeper fields (its fixed point must contract)


class Correction:
    """A camera's image correction on a grid: du, dv of shape (nv, nu)."""

    def __init__(self, n_col: int, n_row: int, u0: float, v0: float, step: float, du, dv,
                 source: str = ""):
        self.n_col, self.n_row = int(n_col), int(n_row)
        self.u0, self.v0, self.step = float(u0), float(v0), float(step)
        self.du = np.array(du, dtype=float)
        self.dv = np.array(dv, dtype=float)
        self.source = source
        if self.du.ndim != 2 or self.du.shape != self.dv.shape or min(self.du.shape) < 2:
            raise ValueError("du and dv need the same shape (nv, nu), at least 2 x 2")
        if not (np.isfinite(self.du).all() and np.isfinite(self.dv).all()):
            raise ValueError("a correction value is not finite")
        if not (self.step > 0 and math.isfinite(self.step)):
            raise ValueError("the grid step must be positive")

    @property
    def nu(self) -> int:
        return self.du.shape[1]

    @property
    def nv(self) -> int:
        return self.du.shape[0]

    # ------------------------------------------------------------------ building
    @staticmethod
    def grid_for(n_col: int, n_row: int, step: float = STEP_PX):
        """The node layout covering an image: from pixel 0 to past its last pixel. (u0, v0, nu, nv)."""
        nu = int(math.ceil((n_col - 1) / step)) + 1
        nv = int(math.ceil((n_row - 1) / step)) + 1
        return 0.0, 0.0, max(nu, 2), max(nv, 2)

    @classmethod
    def zeros(cls, n_col: int, n_row: int, step: float = STEP_PX, source: str = "") -> "Correction":
        u0, v0, nu, nv = cls.grid_for(n_col, n_row, step)
        return cls(n_col, n_row, u0, v0, step, np.zeros((nv, nu)), np.zeros((nv, nu)), source)

    @classmethod
    def from_function(cls, n_col: int, n_row: int, f: Callable[[np.ndarray], np.ndarray],
                      step: float = STEP_PX, source: str = "") -> "Correction":
        """The field f(uv) -> (n, 2) sampled at the nodes covering the image."""
        u0, v0, nu, nv = cls.grid_for(n_col, n_row, step)
        uu, vv = np.meshgrid(u0 + step * np.arange(nu), v0 + step * np.arange(nv))
        c = np.asarray(f(np.column_stack([uu.ravel(), vv.ravel()])), float)
        return cls(n_col, n_row, u0, v0, step, c[:, 0].reshape(nv, nu), c[:, 1].reshape(nv, nu), source)

    def nodes(self) -> np.ndarray:
        """The node pixels, (nv * nu, 2), v outer."""
        uu, vv = np.meshgrid(self.u0 + self.step * np.arange(self.nu), self.v0 + self.step * np.arange(self.nv))
        return np.column_stack([uu.ravel(), vv.ravel()])

    # ------------------------------------------------------------------ evaluation
    def at(self, uv) -> np.ndarray:
        """c at pixels uv (n, 2): bilinear between the nodes, the edge value beyond (NaN for non-finite pixels)."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        out = np.full(uv.shape, np.nan)
        ok = np.isfinite(uv).all(axis=1)
        if not ok.any():
            return out
        s = (uv[ok, 0] - self.u0) / self.step
        t = (uv[ok, 1] - self.v0) / self.step
        s = np.where(s < 0.0, 0.0, np.where(s > self.nu - 1, float(self.nu - 1), s))
        t = np.where(t < 0.0, 0.0, np.where(t > self.nv - 1, float(self.nv - 1), t))
        i = np.minimum(s.astype(np.int64), self.nu - 2)
        j = np.minimum(t.astype(np.int64), self.nv - 2)
        fs, ft = s - i, t - j
        w00, w10 = (1.0 - fs) * (1.0 - ft), fs * (1.0 - ft)
        w01, w11 = (1.0 - fs) * ft, fs * ft
        for k, a in enumerate((self.du, self.dv)):
            out[ok, k] = w00 * a[j, i] + w10 * a[j, i + 1] + w01 * a[j + 1, i] + w11 * a[j + 1, i + 1]
        return out

    def to_model(self, uv) -> np.ndarray:
        """The model's image points of detections at uv: uv - c(uv) (non-finite rows unchanged)."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        out = uv.copy()
        ok = np.isfinite(uv).all(axis=1)
        out[ok] = uv[ok] - self.at(uv[ok])
        return out

    def to_image(self, q, tol: float = 1e-11, iters: int = 60) -> np.ndarray:
        """The pixels d with d - c(d) = q, by the library's fixed point d <- q + c(d)."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        out = q.copy()
        ok = np.isfinite(q).all(axis=1)
        if not ok.any():
            return out
        qq = q[ok]
        d = qq + self.at(qq)
        active = np.ones(len(qq), bool)
        for _ in range(iters):
            if not active.any():
                break
            nd = qq[active] + self.at(d[active])
            change = np.abs(nd - d[active]).sum(axis=1)
            d[active] = nd
            idx = np.flatnonzero(active)
            active[idx[change <= tol]] = False
        out[ok] = d
        return out

    def slope_bound(self) -> float:
        """The library's bound on the norm of c's derivative (it must stay below 1/2)."""
        g = 0.0
        for a in (self.du, self.dv):
            gu = np.maximum(np.abs(a[:-1, 1:] - a[:-1, :-1]), np.abs(a[1:, 1:] - a[1:, :-1])) / self.step
            gv = np.maximum(np.abs(a[1:, :-1] - a[:-1, :-1]), np.abs(a[1:, 1:] - a[:-1, 1:])) / self.step
            g = g + gu ** 2 + gv ** 2
        return float(np.sqrt(g).max())

    def magnitude(self) -> np.ndarray:
        """|c| at the nodes, (nv, nu)."""
        return np.hypot(self.du, self.dv)

    # ------------------------------------------------------------------ files
    def write(self, path: str, comments: Iterable[str] = ()) -> None:
        """Write the file (values with 17 significant digits, so that a read gives the same doubles)."""
        if self.slope_bound() >= SLOPE_MAX:
            raise ValueError(f"the correction is too steep for the library (slope bound {self.slope_bound():.3f})")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        lines = ["# OpenLPT image correction: a detection at pixel d has the camera model's image point d - c(d)",
                 "# c = (du, dv) in pixels at the grid's nodes, bilinear between them, the edge value beyond"]
        lines += ["# " + str(c) for c in comments]
        lines += ["VERSION 1",
                  f"IMAGE_SIZE {self.n_col} {self.n_row}",
                  f"GRID {self.u0!r} {self.v0!r} {self.step!r} {self.nu} {self.nv}"]
        for name, a in (("DU", self.du), ("DV", self.dv)):
            lines.append(name)
            lines += [" ".join(repr(float(x)) for x in row) for row in a]
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")

    @classmethod
    def read(cls, path: str) -> "Correction":
        """Read a file in the library's format (the same checks of its layout)."""
        toks = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                toks += line.split("#", 1)[0].split()
        it = iter(toks)

        def expect(word):
            got = next(it, None)
            if got != word:
                raise ValueError(f"{path}: expected {word}, found {got}")

        try:
            expect("VERSION")
            if next(it) != "1":
                raise ValueError(f"{path}: only VERSION 1 is known")
            expect("IMAGE_SIZE")
            n_col, n_row = int(next(it)), int(next(it))
            expect("GRID")
            u0, v0, step = float(next(it)), float(next(it)), float(next(it))
            nu, nv = int(next(it)), int(next(it))
            expect("DU")
            du = np.array([float(next(it)) for _ in range(nu * nv)]).reshape(nv, nu)
            expect("DV")
            dv = np.array([float(next(it)) for _ in range(nu * nv)]).reshape(nv, nu)
        except StopIteration:
            raise ValueError(f"{path}: the file ends early") from None
        rest = next(it, None)
        if rest is not None:
            raise ValueError(f"{path}: unexpected text after DV: {rest}")
        return cls(n_col, n_row, u0, v0, step, du, dv, source=path)


def compose(c1: Optional[Correction], c2: Correction) -> Correction:
    """The total correction of two passes. c1 is the camera's correction before the pass (None for none), and c2
    the field the pass fitted against the camera corrected by c1: a detection d has c1's corrected model point
    d - c2(d), whose model point is that minus c1 there. So c(d) = c2(d) + c1(d - c2(d)), at c2's nodes."""
    if c1 is None:
        return c2
    if (c1.n_col, c1.n_row) != (c2.n_col, c2.n_row):
        raise ValueError("the two corrections are for different image sizes")
    d = c2.nodes()
    c = c2.at(d) + c1.at(d - c2.at(d))
    return Correction(c2.n_col, c2.n_row, c2.u0, c2.v0, c2.step, c[:, 0].reshape(c2.nv, c2.nu),
                      c[:, 1].reshape(c2.nv, c2.nu), source="")
