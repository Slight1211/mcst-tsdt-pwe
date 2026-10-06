"""One small-basis illustrative eigensolve, not manuscript validation data."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WAVE = ROOT / "research/wavemotion_surface_20260920"
sys.dont_write_bytecode = True
sys.path[:0] = [str(WAVE / "src"), str(ROOT / "research/msse_20260919/python")]

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from inverse_circle_bending import prepare, classical_inverse
from real_circle_bending import assemble_real


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=2, help="PWE cutoff; default is a small software example")
    args = parser.parse_args()
    if args.N < 1:
        parser.error("N must be at least 1")
    k = np.array([np.pi, 0.0])
    a_um, h_um, ell_um = 500.0, 20.0, 1.0
    rows = []
    with threadpool_limits(limits=1):
        for theory in ("FSDT", "TSDT"):
            cache = prepare(args.N, h_um / a_um, 0.3, theory)
            parts = assemble_real(args.N, k, h_um / a_um, 0.3, theory=theory)
            K = classical_inverse(cache, k) + (ell_um / h_um)**2 * parts["KB"]
            M = parts["M"]
            lam, V = eigh(K, M, subset_by_index=[0, 5], check_finite=False)
            assert np.all(lam > 0)
            kv, mv = K @ V, M @ V
            residual = np.linalg.norm(kv - mv * lam, axis=0) / (
                np.linalg.norm(kv, axis=0) + abs(lam) * np.linalg.norm(mv, axis=0))
            assert max(residual) < 1e-7
            factor = np.sqrt(4.35e9 / 1180) / (a_um * 1e-6 * 2 * np.pi) / 1e6
            rows.append({"theory": theory, "frequency_MHz": (np.sqrt(lam) * factor).tolist(),
                         "max_algebraic_residual": float(max(residual))})
    print(json.dumps({"status": "PASS_SMALL_BASIS_EXAMPLE", "N": args.N,
                      "wavevector": "X", "a_um": a_um, "h_um": h_um,
                      "radius_over_a": 0.3, "ell_epoxy_um": ell_um,
                      "ell_steel_um": 0.0, "results": rows,
                      "convergence": "NOT_TESTED"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
