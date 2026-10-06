"""Compute and audit a denser, same-geometry MCST-TSDT inverse-PWE path scan.

All missing integer Fourier orders from N=7 through 21 are solved here. The
previously audited N=9, 15, and 21 spectra are reused at identical 12 k-points.
No interpolation or continuous-Brillouin-zone claim is made.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from run_verified_inverse_scans import verified_solve
from run_inverse_scan_bands import assemble_real, classical_inverse, prepare
import report_pwe_path_convergence_20260927 as previous


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/pwe_path_convergence_dense_20260927"
RAW = OUT / "raw"
NEW_ORDERS = tuple(n for n in range(7, 22) if n not in (9, 15, 21))
ALL_ORDERS = tuple(range(7, 22))
A_UM = 75.0
H_UM = 6.0
ELL_UM = 1.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_file(n: int, point: int) -> Path:
    return RAW / f"N{n}_a75_h6_p{point}_TSDT_l1.npz"


def inspect_raw(n: int, point: int, expected_k: np.ndarray) -> tuple[np.ndarray, float]:
    with np.load(raw_file(n, point)) as data:
        frequency = np.asarray(data["frequency_MHz"], dtype=float)
        residual = np.asarray(data["residual"], dtype=float)
        k = np.asarray(data["k"], dtype=float)
        agreement = float(data["seed_agreement"])
        orthogonality = float(data["mass_orthogonality"])
    assert frequency.shape == (8,) and residual.shape[0] >= 8
    assert np.isfinite(frequency).all() and frequency[1:8].min() > 0
    assert np.max(abs(k - expected_k)) < 1e-12
    assert max(residual[1:8]) < 1e-5
    assert agreement < 1e-7 and orthogonality < 1e-7
    return frequency, float(max(residual[3:5]))


def compute() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    path = previous.path()
    assert path.shape == (12, 2)
    with threadpool_limits(limits=1):
        for n in NEW_ORDERS:
            tick = time.perf_counter()
            cache = prepare(n, H_UM / A_UM, .3, "TSDT")
            for point, k in enumerate(path):
                if raw_file(n, point).is_file():
                    try:
                        inspect_raw(n, point, k)
                        print(f"REUSED N={n} point={point}", flush=True)
                        continue
                    except (AssertionError, KeyError, OSError, ValueError, EOFError):
                        raise RuntimeError(f"Invalid existing checkpoint: {raw_file(n, point)}")
                section = assemble_real(n, k, H_UM / A_UM, .3, theory="TSDT")
                classical_k = classical_inverse(cache, k)
                stiffness = classical_k + (ELL_UM / H_UM) ** 2 * section["KB"]
                eigenvalues, vectors, residual, raw, agreement, orth = verified_solve(
                    stiffness, section["M"], k, n
                )
                frequency = (
                    np.sqrt(eigenvalues)
                    * math.sqrt(4.35e9 / 1180)
                    / (A_UM * 1e-6 * 2 * math.pi)
                    / 1e6
                )
                np.savez_compressed(
                    raw_file(n, point),
                    frequency_MHz=frequency[:8],
                    residual=residual,
                    k=k,
                    ij=section["ij"],
                    raw_eigenvalue=raw,
                    vectors=vectors[:, :8],
                    seed_agreement=agreement,
                    mass_orthogonality=orth,
                    N=n,
                    a_um=A_UM,
                    h_um=H_UM,
                    r_over_a=.3,
                    ell_epoxy_um=ELL_UM,
                    theory="TSDT",
                )
                inspect_raw(n, point, k)
                print(f"DONE N={n} point={point} elapsed_s={time.perf_counter()-tick:.1f}", flush=True)
            print(f"COMPLETE N={n} elapsed_s={time.perf_counter()-tick:.1f}", flush=True)
    print("PASS_DENSE_RAW_12_POINT_PATH", flush=True)


def audited_matrix(n: int, path: np.ndarray) -> tuple[np.ndarray, float]:
    if n == 9:
        return previous.spectrum_n9_n15(9)
    if n == 15:
        return previous.spectrum_n9_n15(15)
    if n == 21:
        return previous.spectrum_n21()
    matrix = np.empty((12, 6), dtype=float)
    max_residual = 0.0
    for point, k in enumerate(path):
        frequency, residual = inspect_raw(n, point, k)
        matrix[point] = frequency[:6]
        max_residual = max(max_residual, residual)
    return matrix, max_residual


def report() -> None:
    path = previous.path()
    assert path.shape == (12, 2)
    old_audit = json.loads((previous.OUT / "audit.json").read_text(encoding="utf-8"))
    assert old_audit["status"] == "PASS_SAME_GEOMETRY_12_POINT_PATH_N9_N15_N21"
    old_rows = {int(row["N"]): row for row in previous.rows(
        previous.OUT / "pwe_path_convergence.csv"
    )}
    matrices = {}
    results = []
    for n in ALL_ORDERS:
        matrix, max_residual = audited_matrix(n, path)
        assert np.isfinite(matrix).all()
        assert matrix[:, 3:5].min() > 0 and max_residual < 1e-5
        lower_point = int(np.argmax(matrix[:, 3]))
        upper_point = int(np.argmin(matrix[:, 4]))
        lower = float(matrix[lower_point, 3])
        upper = float(matrix[upper_point, 4])
        gap = 200 * (upper - lower) / (upper + lower)
        matrices[n] = matrix
        if n in old_rows:
            old = old_rows[n]
            for key, value in (("lower_MHz", lower), ("upper_MHz", upper),
                               ("J45_percent", gap)):
                assert abs(float(old[key]) - value) < 1e-8
        results.append({
            "N": n,
            "DOF_per_k": 3 * (2 * n + 1) ** 2,
            "a_um": A_UM,
            "h_um": H_UM,
            "r_over_a": .3,
            "ell_epoxy_um": ELL_UM,
            "path_points": 12,
            "lower_MHz": lower,
            "upper_MHz": upper,
            "lower_point": lower_point,
            "upper_point": upper_point,
            "J45_percent": gap,
            "max_bands4_5_residual": max_residual,
            "source": "previously_audited" if n in old_rows else "new_full_path",
        })
    reference = results[-1]
    for row in results:
        n = row["N"]
        row["lower_difference_to_N21_percent"] = 100 * abs(
            row["lower_MHz"] / reference["lower_MHz"] - 1
        )
        row["upper_difference_to_N21_percent"] = 100 * abs(
            row["upper_MHz"] / reference["upper_MHz"] - 1
        )
        row["J_difference_to_N21_pp"] = row["J45_percent"] - reference["J45_percent"]
        row["max_bands4_5_path_difference_to_N21_percent"] = float(
            100 * np.max(abs(matrices[n][:, 3:5] / matrices[21][:, 3:5] - 1))
        )
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "pwe_path_convergence_dense.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    sources = {
        str(path): sha(path)
        for path in (
            Path(__file__),
            ROOT / "run_verified_inverse_scans.py",
            ROOT / "run_inverse_scan_bands.py",
            ROOT / "src/inverse_circle_bending.py",
            ROOT / "src/real_circle_bending.py",
            previous.OUT / "pwe_path_convergence.csv",
        )
    }
    audit = {
        "status": "PASS_SAME_GEOMETRY_12_POINT_PATH_EVERY_N7_TO_N21",
        "scope": "MCST-TSDT inverse-PWE; sampled Gamma-X-M-Gamma only",
        "orders": list(ALL_ORDERS),
        "new_orders": list(NEW_ORDERS),
        "reused_orders": [9, 15, 21],
        "new_raw_files": len(list(RAW.glob("*.npz"))),
        "all_edge_locations_X": all(
            row["lower_point"] == row["upper_point"] == 4 for row in results
        ),
        "max_residual": max(row["max_bands4_5_residual"] for row in results),
        "sources_sha256": sources,
        "NOT_RUN": ["continuous-k extrema certification", "N>21 convergence",
                    "joint PWE and 3D FEM convergence"],
    }
    assert audit["new_raw_files"] == 12 * len(NEW_ORDERS)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"audit": audit, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["compute"]:
        compute()
    elif sys.argv[1:] == ["report"]:
        report()
    else:
        raise SystemExit("usage: run_pwe_path_convergence_dense_20260927.py compute|report")
