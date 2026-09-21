# OpenLPT Development Guide

This document defines the development environment, build instructions, and mandatory coding protocols for the pyOpenLPT project.

## Development Environment
- **Operating System**: Windows (Primary), Linux/Mac (Support)
- **Language**: Python 3.10+, C++ 17 (C++ Core)
- **Primary Entry Point**: `python GUI.py`

## Build & Installation
- Install dependencies: `pip install -r requirements.txt`
- Build C++ bindings: `python build_local.py` or `python setup.py install`
- Run GUI: `python GUI.py`

## Python Environment Rule
- For this repo, agents must use the `refract` conda environment for Python compilation, script execution, and verification commands.
- Preferred order:
  1. use `python ...` only if the active shell is already confirmed to be `refract`
  2. otherwise call the interpreter by absolute path (below)
  3. `conda run -n refract python ...` is a last resort: on the current Windows box it swallows output and exits 127 even when the command succeeds
- Interpreter paths seen so far:
  - `C:\Users\zcloc\.conda\envs\refract\python.exe` (current machine)
- Apply this rule to commands such as `py_compile`, `--help` checks, calibration script runs, and other Python-based validation in this project.
- Do not assume the default `python` points to the correct environment.
- `build_local.py` passes `sys.executable` to CMake, so run it with the `refract`
  interpreter or the extension will be built against the wrong Python ABI.

## Running the tests
- `pytest` from the repository root. `conftest.py` puts the repo root and
  `build-py/<CONFIG>/` on `sys.path`, so `import pyopenlpt` works against a
  `build_local.py` build without installing it. `CONFIG=Debug pytest` picks up a
  Debug build.
- Known pre-existing breakage, unrelated to any current work (baseline
  2026-09-17: 48 passed, 1 failed, 7 skipped):
  - `tests/test_p0_bootstrap_failure.py` and `tests/test_plane_pipeline.py` fail
    to import - they need a `scripts.case_013_bootstrap_debug_loop` /
    `scripts.case_026_plane_debug_loop` package that is not in this repo.
  - `tests/test_p1_round4_radius_frame_mismatch.py::TestRadiusEstimationFrameMismatch::test_radius_estimation_frame_mismatch_inflates`
    fails.

## Mandatory Coding Protocols
All AI agents and developers **MUST** adhere to the following protocols defined in the `.agent/skills/` directory:

0. **Check the Bindings First!** Always verify available methods and attributes in `src/pybind_OpenLPT/` before writing code for C++ objects. Do not assume methods exist.
1. **C++ Binding Interface**: When modifying or accessing `lpt.Camera` objects or any C++ parameters from Python, you **MUST** follow the **[cpp-protocol](file:///d:/0.Code/OpenLPTGUI/OpenLPT/.agent/skills/cpp_protocol/SKILL.md)**. Improper access will cause `AttributeError` or binary crashes.
2. **Internal Data Structures**: When interacting with `dataset`, `window_media`, or `window_planes` dictionaries, you **MUST** follow the **[python-protocol](file:///d:/0.Code/OpenLPTGUI/OpenLPT/.agent/skills/python_protocol/SKILL.md)**. Incorrect key access causes `KeyError`.

## Design Guidelines
- **Math/Physics First**: For refractive calibration, ensure coordinate shifts (Closest vs Farthest) are handled correctly.
- **Authority**: The C++ kernel (`lpt.Camera`) is the source of truth for ray tracing and triangulation. Always sync Python state to C++ objects before physical computation.
- **Fail-Fast**: Validate input consistency (e.g., wand length, camera mapping) at the start of every module.

## Workflow Commands
- **Testing**: Run scripts in `test/` directory.
- **Calibration**: Use `modules/camera_calibration/wand_calibration/`.
