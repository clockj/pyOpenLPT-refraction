"""T5b: the T5 comparison for the column/row board model M2c (added to T5's list after T5 had run).

Same protocol as T5: fit on all planes and on the even planes, test on the odd planes.  Starts from the delivered
rig without its board corrections.  Writes t5b_out.txt beside this script.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(HERE)
sys.path.insert(0, CASE)
sys.path.insert(0, HERE)
import case                                   # noqa: E402
import cylfit as CF                           # noqa: E402
import run_calibration as RC                  # noqa: E402
from t5_models import cyl_case                # noqa: E402


def main():
    cfg = case.load_config()
    st = RC.load_state(os.path.join(cfg["out_dir"], "fit_state.pkl"))
    plain = st["rig"].copy()
    plain.board, plain.board_ij = None, None
    r = cyl_case("M2c (column/row board)", st["obs"], plain, CF.MODELS["M2c"])
    line = (f"{r['label']:24s} fit {r['fit']:.4f}  train {r['train']:.4f}  test nom-z {r['test_nom']:.4f}  "
            f"test own-z {r['test_own']:.4f}   {r['params']}")
    print(line)
    with open(os.path.join(HERE, "t5b_out.txt"), "w") as f:
        f.write("T5b model comparison, M2c (rms in px; train = even planes, test = odd planes)\n" + line + "\n")


if __name__ == "__main__":
    main()
