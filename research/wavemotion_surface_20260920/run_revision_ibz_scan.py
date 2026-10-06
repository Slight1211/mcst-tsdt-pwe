"""Single-process, checkpointed finite-IBZ search for the existing a=75 um case.

This is a numerical grid search, not a proof of a continuous-domain band gap.
The only inverse-factorized term is the classical generalized stiffness.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from run_verified_inverse_scans import (
    assemble_real, classical_inverse, prepare, verified_solve,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/manuscript_revision_20260925"
A_UM = 75.0
RADIUS = 0.3
ELL_UM = 1.0
RATIOS = (0.02, 0.08, 0.20)
N_VALUES = (9, 15)
GRID_DIVISIONS = 4
FIELDNAMES = (
    "N", "a_um", "h_over_a", "h_um", "ix", "iy", "kx_a", "ky_a",
    "band", "frequency_MHz", "residual", "seed_agreement",
    "mass_orthogonality",
)


def frequency_factor() -> float:
    return np.sqrt(4.35e9 / 1180.0) / (A_UM * 1e-6 * 2 * np.pi) / 1e6


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = {
        "a_um": A_UM, "radius_over_a": RADIUS, "ell_epoxy_um": ELL_UM,
        "ell_steel_um": 0.0, "h_over_a": RATIOS, "N": N_VALUES,
        "grid_divisions": GRID_DIVISIONS, "theory": "MCST-TSDT",
        "method": "inverse classical generalized stiffness, direct MCST and mass",
        "ibz": f"0<=ky_a<=kx_a<=pi, triangular {GRID_DIVISIONS+1}x{GRID_DIVISIONS+1} grid",
        "threads": 1, "workers": 1, "bands_stored": 8,
        "zero_branch_rule": "keep Gamma zero as band 1",
        "scope": "finite-grid IBZ estimates, not continuous extrema or 3D FEM",
    }
    cfg_path = OUT / "ibz_scan_config.json"
    if cfg_path.exists():
        prior = json.loads(cfg_path.read_text(encoding="utf-8"))
        obsolete_label = "0<=ky_a<=kx_a<=pi, triangular 5x5 grid"
        if (GRID_DIVISIONS == 8 and prior.get("ibz") == obsolete_label
                and dict(prior, ibz=config["ibz"]) == json.loads(json.dumps(config))):
            # Only correct a descriptive label; no numerical configuration changes.
            cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        elif prior != json.loads(json.dumps(config)):
            raise RuntimeError("Existing IBZ configuration differs; use a new run identifier")
    else:
        cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    start = time.perf_counter()
    rows: list[dict] = []
    with threadpool_limits(limits=1):
        for N in N_VALUES:
            for ratio in RATIOS:
                h_um = A_UM * ratio
                cache = prepare(N, ratio, RADIUS, "TSDT")
                for ix in range(GRID_DIVISIONS + 1):
                    for iy in range(ix + 1):
                        k = np.array([np.pi * ix / GRID_DIVISIONS,
                                      np.pi * iy / GRID_DIVISIONS])
                        tag = f"ibz_N{N}_tau{ratio:.2f}_ix{ix}_iy{iy}"
                        saved = OUT / (tag + ".npz")
                        if saved.exists():
                            with np.load(saved) as prior:
                                freq = prior["frequency_MHz"]
                                residual = prior["residual"]
                                agreement = float(prior["seed_agreement"])
                                orth = float(prior["mass_orthogonality"])
                                assert len(freq) == 8 and np.isfinite(freq).all()
                                assert agreement < 1e-7 and orth < 1e-7
                        else:
                            s = assemble_real(N, k, ratio, RADIUS, theory="TSDT")
                            kc = classical_inverse(cache, k)
                            K = kc + (ELL_UM / h_um) ** 2 * s["KB"]
                            lam, V, residual, raw, agreement, orth = verified_solve(
                                K, s["M"], k, N
                            )
                            freq = np.sqrt(lam[:8]) * frequency_factor()
                            np.savez_compressed(
                                saved, frequency_MHz=freq, residual=residual[:8],
                                raw_eigenvalue=raw[:8], vectors=V[:, :8], k=k,
                                seed_agreement=agreement, mass_orthogonality=orth,
                            )
                        for band, value in enumerate(freq, 1):
                            rows.append({
                                "N": N, "a_um": A_UM, "h_over_a": ratio,
                                "h_um": h_um, "ix": ix, "iy": iy,
                                "kx_a": float(k[0]), "ky_a": float(k[1]),
                                "band": band, "frequency_MHz": float(value),
                                "residual": float(residual[band - 1]),
                                "seed_agreement": float(agreement),
                                "mass_orthogonality": float(orth),
                            })
                        print("PASS", tag, "seconds", round(time.perf_counter() - start, 1), flush=True)
    spectra = OUT / "ibz_grid_spectra.csv"
    with spectra.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for N in N_VALUES:
        for ratio in RATIOS:
            subset = [row for row in rows if row["N"] == N and row["h_over_a"] == ratio]
            lower = max((row for row in subset if row["band"] == 4),
                        key=lambda row: row["frequency_MHz"])
            upper = min((row for row in subset if row["band"] == 5),
                        key=lambda row: row["frequency_MHz"])
            L, U = lower["frequency_MHz"], upper["frequency_MHz"]
            summary.append({
                "N": N, "a_um": A_UM, "h_over_a": ratio, "h_um": A_UM * ratio,
                "L45_MHz": L, "L_ix": lower["ix"], "L_iy": lower["iy"],
                "L_kx_a": lower["kx_a"], "L_ky_a": lower["ky_a"],
                "U45_MHz": U, "U_ix": upper["ix"], "U_iy": upper["iy"],
                "U_kx_a": upper["kx_a"], "U_ky_a": upper["ky_a"],
                "J45_percent": 200 * (U - L) / (U + L),
                "point_count": len(subset) // 8,
                "scope": f"{GRID_DIVISIONS+1}x{GRID_DIVISIONS+1} triangular finite-grid IBZ estimate",
            })
    edges = OUT / "band_edge_results.csv"
    with edges.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    metadata = {
        "status": "PASS_FINITE_GRID_ONLY",
        "command": sys.argv, "pid": os.getpid(),
        "elapsed_seconds": time.perf_counter() - start,
        "peak_working_set_mb": psutil.Process().memory_info().peak_wset / (1024 ** 2),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), Path(sys.argv[0]),
                                    ROOT / "run_verified_inverse_scans.py",
                                    ROOT / "src/inverse_circle_bending.py",
                                    ROOT / "src/real_circle_bending.py")},
        "records": len(rows), "edge_records": len(summary),
        "NOT_RUN": ["local k refinement",
                    "higher-N full IBZ" if max(N_VALUES) < 21 else "N>21 full IBZ",
                    "full-IBZ 3D FEM", "band-edge energy and sensitivity"],
    }
    (OUT / "ibz_scan_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "manifest": metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
