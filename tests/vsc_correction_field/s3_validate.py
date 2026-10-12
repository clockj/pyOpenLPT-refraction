"""S3: the tracking settings page's Validate on a tube project (decision 2, 2026-10-09-plan.md).

    python s3_validate.py

T1 after a VSC pass (projects/s1_tube as s1_save.py builds it: the line detector, the OTF table, the depth cue, the
cameras' corrections; T1's preprocessed frames, read only), with today's page (commit cffed7d's file) and with the
change, each on its own copy: the project loaded as the page does when it is shown, the 2-D tolerance set to 1.0 px,
Validate Settings pressed. Validate saves the configuration first, then detects and matches on the start frame with
the saved config. Qt offscreen, build vcf's pyopenlpt, message boxes recorded.
Recorded:
- config.json after Validate: the tracer's settings, the 2-D tolerance, the corrections;
- the per-camera 2-D counts the page prints, against the tracking run's own for the same frame (T1's 15 px run's
  log, its first IPR pass at frame 0: 1067 and 985; before the detection fix of 9 Oct Validate got 1047 and 949);
- the result box.
Pass: the change keeps the tracer's settings and counts as the tracking run does; today's page is recorded for
comparison. Writes s3_validate_out.txt and .json here.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import vcf_common as vc  # noqa: E402
from h1_vsc_thread import Immediate  # noqa: E402
from s1_save import TUBE_EXTRAS, clone, make_t1  # noqa: E402


class Tee(io.StringIO):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        return super().write(s)


def validate(mod, app, proj):
    Immediate.shown = []
    mod.QMessageBox = Immediate
    view = mod.TrackingSettingsView()
    view.project_path.setText(proj)
    app.processEvents()
    view._update_paths()
    app.processEvents()
    view.ipr_2d_tol.setValue(1.0)
    buf = Tee(__import__("sys").stdout)
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        view._validate_settings()
        app.processEvents()
    dt = time.time() - t0
    counts = [int(m.group(2)) for m in re.finditer(r"Camera (\d+): found (\d+) 2D objects", buf.getvalue())]
    matches = [int(m.group(1)) for m in re.finditer(r"found (\d+) 3D objects", buf.getvalue())]
    boxes = list(Immediate.shown)
    form_tol = view.ipr_2d_tol.value()
    view.deleteLater()
    app.processEvents()
    return {"seconds": round(dt, 1), "counts_2d": counts, "matches_3d": matches, "boxes": boxes,
            "form_tol_2d_px": form_tol}


def main():
    lpt = vc.lib("vcf")
    log = vc.Log("s3_validate")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    today = vc.committed_module("gui/views/tracking_settings_view.py", "tracking_settings_today")
    import gui.views.tracking_settings_view as change
    log(f"S3: today's page {today.__file__}, the change {change.__file__}; pyopenlpt {lpt.__file__}")
    src = make_t1("s3_tube", "PINCYL")
    before = vc.load_json(os.path.join(src, "config.json"))
    from d1_detection import log_counts
    run_counts = log_counts(os.path.join(vc.CASE_RUNS, "s3_final_m15", "log.txt"), [0])[0]
    out = {"run_counts": run_counts}
    for name, mod in (("today", today), ("change", change)):
        proj = clone(src, f"s3_run_{name}")
        r = validate(mod, app, proj)
        after = vc.load_json(os.path.join(proj, "config.json"))
        tr = after["objects"][0].get("tracer", {})
        r["tracer_keys_kept"] = sorted(k for k in TUBE_EXTRAS if k in tr)
        r["tracer_keys_lost"] = sorted(k for k in TUBE_EXTRAS if k not in tr)
        r["tol_2d_px_saved"] = after["objects"][0]["ipr"]["tol_2d_px"]
        r["corrections"] = [c.get("correction") for c in after["cameras"]]
        r["corrections_kept"] = r["corrections"] == [c.get("correction") for c in before["cameras"]]
        r["counts_as_run"] = r["counts_2d"] == run_counts
        out[name] = r
        res = next((b for b in r["boxes"] if b["title"].startswith("Validation")), None)
        log(f"{name}'s page: Validate {r['seconds']} s; 2-D counts {r['counts_2d']} (the tracking run on this frame: "
            f"{run_counts}); 3-D matches {r['matches_3d']}; config.json after: tracer settings kept "
            f"{len(r['tracer_keys_kept'])} of {len(TUBE_EXTRAS)}"
            + (f" (lost: {', '.join(r['tracer_keys_lost'])})" if r["tracer_keys_lost"] else "")
            + f", 2-D tolerance {r['tol_2d_px_saved']} px, corrections kept {r['corrections_kept']}")
        if res:
            log(f"  box '{res['title']}': " + res["text"].replace("\n", " | "))
        for b in r["boxes"]:
            if b is not res and b["title"] not in ("Success",):
                log(f"  other box '{b['title']}': {b['text'][:200]}")
    c = out["change"]
    ok = (not c["tracer_keys_lost"] and c["counts_as_run"] and c["corrections_kept"]
          and any(b["title"].startswith("Validation") for b in c["boxes"]))
    out["pass"] = bool(ok)
    log(f"S3: {'PASS' if ok else 'FAIL'}")
    log.save(out)


if __name__ == "__main__":
    main()
