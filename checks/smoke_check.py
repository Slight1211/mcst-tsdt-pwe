"""Small algebraic/software checks; never start a research scan or COMSOL."""
from pathlib import Path
import argparse
import ast
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
WAVE = ROOT / "research/wavemotion_surface_20260920"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kernel", choices=("current", "frozen"), default="current")
args = parser.parse_args()
sys.dont_write_bytecode = True
sys.path[:0] = [str(WAVE / "src"), str(ROOT / "research/msse_20260919/python")]
if args.kernel == "frozen":
    sys.path.insert(0, str(WAVE / "runs/scale_addendum_20260927/sources"))

import numpy as np
from threadpoolctl import threadpool_limits
from inverse_circle_bending import regression
from node_shape_bending import AREA, boundary, signed_area


def main():
    paths = sorted(ROOT.rglob("*.py"))
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    with threadpool_limits(limits=1):
        algebra = regression()
        vertices = boundary(np.zeros(6))
        assert vertices.shape == (48, 2)
        area_error = abs(signed_area(vertices) - AREA)
        assert area_error < 1e-12
    print(json.dumps({"status": "PASS", "python_sources_parsed": len(paths),
                      "kernel": args.kernel,
                      "basis_order_for_algebra_check": 2, "algebra": algebra,
                      "nodal_fixed_area_absolute_error": area_error,
                      "full_scans": "NOT_RUN", "optimization": "NOT_RUN",
                      "COMSOL": "NOT_RUN"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
