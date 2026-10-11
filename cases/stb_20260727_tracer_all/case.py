"""Settings and paths of the stb_20260727_tracer_all case: tracer tracking (STB) of every run of 20260727 with the
flat-window wand calibration.

Everything case-specific lives in config.json.  For one run (a test case such as "15Hz_r_b_2"):
  - case_dir(cfg, name):    the run's folder on X: (Results\\<name>);
  - output_dir(cfg, name):  <case_dir>\\Results_tracer, where the results end up;
  - stage_dir(cfg, name):   the run's folder on D: (staging_root\\<name>), where STB writes while it runs.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))          # pyOpenLPT root
CONFIG_PATH = os.path.join(HERE, "config.json")
TESTS = os.path.join(HERE, "tests")
BUILD = os.path.join(REPO, "build-py", os.environ.get("CONFIG", "Release"))
for _p in (BUILD, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def case_dir(cfg: dict, name: str) -> str:
    return os.path.join(cfg["results_root"], name)


def output_dir(cfg: dict, name: str) -> str:
    return os.path.join(case_dir(cfg, name), cfg["output_folder"])


def stage_dir(cfg: dict, name: str) -> str:
    return os.path.join(cfg["staging_root"], name)


def image_path(cfg: dict, name: str, cam: int, frame: int) -> str:
    return os.path.join(case_dir(cfg, name), cfg["image_folder"], f"cam{cam}", cfg["image_name"].format(frame))


def import_lpt():
    import pyopenlpt as lpt  # noqa: WPS433 (the local build, put on sys.path above)
    return lpt


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
