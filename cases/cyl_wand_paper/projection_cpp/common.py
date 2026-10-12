"""Shared by the tests of this folder: the C++ module, the paper's Python run of E1 (code/paper_baselines.py: scenes,
point sets, ground truth, scoring and the Python baselines), the run log and the provenance record.

Every script runs in the refract env, one thread, at below-normal priority, without writing bytecode:

    PYTHONDONTWRITEBYTECODE=1 C:\\Users\\zcloc\\.conda\\envs\\refract\\python.exe <script>.py
"""
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True                  # no __pycache__ in this folder or in the paper's code/

HERE = Path(__file__).resolve().parent
OPENLPT = HERE.parents[2]
REFRACTION = Path(os.environ.get("REFRACTION_ROOT", str(HERE.parents[3])))
CODE = REFRACTION / "code"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE / "build" / "Release"))

import numpy as np                              # noqa: E402

import paper_baselines as e1                    # noqa: E402  the Python run of the paper's E1
import projbench as cpp                         # noqa: E402  this folder's module

SCENES = {sc.key: sc for sc in e1.SCENES}       # ref: the reference window; pa: Paolillo & Astarita; sph: the sphere
NAMES = ("afrt", "gn", "lm", "shoot")           # the order of the module's methods (pb::Method)
STATUS_NAMES = {0: "list", 1: "zero", 2: "stationary", 3: "cap", 4: "breakdown"}


def args(sc):
    """The scene as the module takes it."""
    return (list(sc.n), list(sc.R), float(sc.rP), bool(sc.sphere))


def below_normal_priority():
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def _git(*a):
    p = subprocess.run(["git", *a], cwd=OPENLPT, capture_output=True, text=True,
                       env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    return p.stdout.strip()


def provenance():
    """What ran: the module and its build record, the paper repository's E1 script and the pyOpenLPT commit."""
    build = json.loads((HERE / "build_info.json").read_text(encoding="utf-8"))
    mod = Path(cpp.__file__)
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "module": mod.name,
        "module_sha256": hashlib.sha256(mod.read_bytes()).hexdigest(),
        "module_as_built": hashlib.sha256(mod.read_bytes()).hexdigest() == build["module_sha256"],
        "module_info": dict(cpp.info()),
        "refract_curved_sources": build["sources"],
        "paper_baselines_sha256": hashlib.sha256((CODE / "paper_baselines.py").read_bytes()).hexdigest(),
        "project_cylinder_sha256": hashlib.sha256((CODE / "project_cylinder.py").read_bytes()).hexdigest(),
        "pyopenlpt_head": _git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


class Log:
    """A test's output: <name>_out.txt (every line also printed) and <name>.json, with the provenance at the start."""

    def __init__(self, name):
        below_normal_priority()
        self.name = name
        self.t0 = time.time()
        self.out = open(HERE / f"{name}_out.txt", "w", encoding="utf-8")
        self.data = {"provenance": provenance()}
        p = self.data["provenance"]
        self(f"{name}: started {p['time']}; module {p['module']} (SHA-256 {p['module_sha256'][:12]}, "
             f"{'as built' if p['module_as_built'] else 'NOT the built one'}; {p['module_info']['compiler']}, "
             f"{p['module_info']['release_flags']}); pyOpenLPT {p['pyopenlpt_head'][:7]}; RefractCurved sources "
             f"{'unchanged since the commit' if all(s['same_as_head'] for s in p['refract_curved_sources']) else 'MODIFIED'}")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.out.write(msg + "\n")
        self.out.flush()

    def close(self, **extra):
        self.data.update(extra)
        self.data["seconds"] = round(time.time() - self.t0, 1)
        self(f"{self.name}: done in {self.data['seconds']} s")
        (HERE / f"{self.name}.json").write_text(json.dumps(self.data, indent=1, default=_jsonable) + "\n",
                                                encoding="utf-8")
        self.out.close()


def _jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def angle(a, b):
    """The angle between two directions, resolved down to rounding (atan2 of the cross and dot products; the arccos
    of e1._angle cannot resolve angles below about 2e-8 rad)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), float(a @ b)))
