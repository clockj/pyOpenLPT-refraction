"""G3's debugging aid: why the offscreen tracking view exits (code 127) after the VSC's worker finishes.

    python g3_debug.py make NAME       # the service's result on projects/NAME, pickled to data/g3_debug_<NAME>.pkl
    python g3_debug.py view NAME       # that result handed to the view's _on_vsc_finished, each step traced

The trace (entry and exit of the view's methods that _on_vsc_finished calls) goes to stdout, unbuffered.
"""
from __future__ import annotations

import os
import pickle
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import vcf_common as vc  # noqa: E402


def make(name):
    vc.lib("vcf")
    from modules.vsc.vsc_service import VSCService
    proj = os.path.join(vc.HERE, "projects", name)
    svc = VSCService(proj, log_callback=None)
    svc.set_params(15, 20000, 2000, "default", 5.0)
    ok, msg, data = svc.run()
    with open(os.path.join(vc.DATA, f"g3_debug_{name}.pkl"), "wb") as fh:
        pickle.dump((ok, msg, data), fh)
    print("saved", ok, msg[:100], flush=True)


def view(name):
    vc.lib("vcf")
    import faulthandler
    faulthandler.enable(all_threads=True)
    with open(os.path.join(vc.DATA, f"g3_debug_{name}.pkl"), "rb") as fh:
        ok, msg, data = pickle.load(fh)
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import gui.views.tracking_view as tv
    from g3_gui import Recorder
    tv.QMessageBox = Recorder
    v = tv.TrackingView(settings_view=None)
    v.resize(1700, 1050)
    v.show()
    app.processEvents()
    v.proj_path_edit.setText(os.path.join(vc.HERE, "projects", name))
    app.processEvents()
    for meth in ("_discover_cameras", "_update_vsc_visualization", "_show_vsc_3d_plot", "_show_vsc_correction_plot",
                 "_append_log", "_busy_end"):
        orig = getattr(v, meth)

        def traced(*a, _orig=orig, _m=meth, **k):
            print(f"-> {_m}", flush=True)
            r = _orig(*a, **k)
            print(f"<- {_m}", flush=True)
            return r
        setattr(v, meth, traced)
    print("calling _on_vsc_finished", flush=True)
    v._on_vsc_finished(ok, msg, data)
    print("returned", flush=True)
    app.processEvents()
    print("events processed; messages:", [m["title"] for m in Recorder.shown], flush=True)


if __name__ == "__main__":
    {"make": make, "view": view}[sys.argv[1]](sys.argv[2])
