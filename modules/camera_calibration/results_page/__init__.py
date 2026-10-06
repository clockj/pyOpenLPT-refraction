"""The results page of a wand calibration (one self-contained HTML file; plan: tests/cylinder_wand_gui/2026-10-06-plan.md).

    from modules.camera_calibration.results_page import page
    report = page.report_from_cylinder(result, data, folder)      # or report_from_pinhole / report_from_plate
    page.write_page(report, os.path.join(folder, "calibration_report.html"))
"""
