"""Freeze and independently evaluate X-edge perturbation predictions at a=75 um.

This script predicts the two X-point modes, not a continuous-IBZ band gap.
Run --stage freeze before --stage evaluate; evaluation refuses a missing freeze.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/manuscript_revision_20260925"
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "msse_20260919/python")]
from inverse_circle_bending import classical_inverse, prepare  # noqa: E402
from real_circle_bending import assemble_real  # noqa: E402
from run_verified_inverse_scans import verified_solve  # noqa: E402

N, A_UM, RADIUS, H_UM, ELL_UM = 15, 75., .3, 6., 1.
KPOINT = np.array([np.pi, 0.])
INPUT = OUT / "ibz_N15_tau0.08_ix4_iy0.npz"
FROZEN = OUT / "edge_prediction_frozen.json"
OUTPUT = OUT / "prediction_checks.csv"


def factor() -> float:
    return np.sqrt(4.35e9 / 1180.)/(A_UM*1e-6*2*np.pi)/1e6


def matrix(h_um: float, ell_um: float):
    ratio = h_um/A_UM
    cache = prepare(N, ratio, RADIUS, "TSDT")
    s = assemble_real(N, KPOINT, ratio, RADIUS, theory="TSDT")
    kc = classical_inverse(cache, KPOINT)
    kmc = (ell_um/h_um)**2*s["KB"]
    return kc+kmc, s["M"], kmc


def gap(f4: float, f5: float) -> float:
    return 200*(f5-f4)/(f5+f4)


def freeze():
    if FROZEN.exists():
        raise RuntimeError("Frozen predictions already exist; do not overwrite them")
    with np.load(INPUT) as data:
        V = data["vectors"][:, 3:5].copy()
        frequencies = data["frequency_MHz"][3:5].copy()
        assert np.max(data["residual"][3:5]) < 1e-5
    K, M, Kmc = matrix(H_UM, ELL_UM)
    slopes = []
    for step in (.001, .002, .005):
        kh, mh, _ = matrix(H_UM*np.exp(step), ELL_UM)
        kl, ml, _ = matrix(H_UM*np.exp(-step), ELL_UM)
        dk, dm = (kh-kl)/(2*step), (mh-ml)/(2*step)
        values = []
        for column in range(2):
            q = V[:, column]
            lam = float(q @ K @ q)/float(q @ M @ q)
            assert abs(np.sqrt(lam)*factor()/frequencies[column]-1) < 1e-7
            sh = float(q @ (dk-lam*dm) @ q)/(2*lam*float(q @ M @ q))
            values.append(sh)
        slopes.append(values)
    spread = np.max(np.ptp(np.array(slopes), axis=0))
    assert spread < 1e-3, spread
    rmc = [float(V[:, j] @ Kmc @ V[:, j])/float(V[:, j] @ K @ V[:, j])
           for j in range(2)]
    sh = slopes[1]
    rows = []
    for parameter in ("h", "ell"):
        sensitivities = sh if parameter == "h" else rmc
        for change in (-.03, -.01, .01, .03):
            predicted = [frequencies[j]*np.exp(sensitivities[j]*np.log1p(change))
                         for j in range(2)]
            rows.append({
                "parameter": parameter, "fractional_change": change,
                "predicted_f4_MHz": predicted[0], "predicted_f5_MHz": predicted[1],
                "predicted_X_J45_percent": gap(*predicted),
                "predicted_delta_ln_f4": sensitivities[0]*np.log1p(change),
                "predicted_delta_ln_f5": sensitivities[1]*np.log1p(change),
            })
    frozen = {
        "status": "FROZEN_BEFORE_PERTURBED_EIGENSOLVES",
        "case": "a=75um h=6um r/a=.3 ell_epoxy=1um MCST-TSDT inverse N15 X",
        "baseline_f4_f5_MHz": frequencies.tolist(),
        "baseline_X_J45_percent": gap(*frequencies),
        "R_mc_f4_f5": rmc, "S_h_f4_f5": sh,
        "S_h_step_values": slopes, "S_h_spread": float(spread),
        "changes": rows,
        "scope": "fixed X only, not a full-IBZ extremum verification",
    }
    FROZEN.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(json.dumps(frozen, indent=2))


def evaluate():
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert frozen["status"] == "FROZEN_BEFORE_PERTURBED_EIGENSOLVES"
    with np.load(INPUT) as data:
        Vbase = data["vectors"][:, 3:5].copy()
    K0, M0, _ = matrix(H_UM, ELL_UM)
    rows = []
    for case in frozen["changes"]:
        parameter, change = case["parameter"], case["fractional_change"]
        h = H_UM*(1+change) if parameter == "h" else H_UM
        ell = ELL_UM*(1+change) if parameter == "ell" else ELL_UM
        K, M, _ = matrix(h, ell)
        lam, V, residual, _, agreement, orth = verified_solve(K, M, KPOINT, N)
        freq = np.sqrt(lam[:8])*factor()
        actual_j = gap(float(freq[3]), float(freq[4]))
        for column, band in enumerate((4, 5)):
            base = Vbase[:, column]
            projection = base @ M0 @ V[:, :8]
            base_norm = float(base @ M0 @ base)
            new_norm = np.einsum("ij,ij->j", V[:, :8], M0 @ V[:, :8])
            mac = abs(projection)**2/(base_norm*new_norm)
            chosen = int(np.argmax(mac))
            predicted_delta = case[f"predicted_delta_ln_f{band}"]
            actual_delta = float(np.log(freq[chosen]/frozen["baseline_f4_f5_MHz"][column]))
            rows.append({
                "parameter": parameter, "fractional_change": change,
                "baseline_band": band, "matched_band": chosen+1,
                "mass_MAC": float(mac[chosen]),
                "baseline_f_MHz": frozen["baseline_f4_f5_MHz"][column],
                "predicted_f_MHz": case[f"predicted_f{band}_MHz"],
                "actual_f_MHz": float(freq[chosen]),
                "predicted_delta_ln_f": predicted_delta,
                "actual_delta_ln_f": actual_delta,
                "error_delta_ln_f": actual_delta-predicted_delta,
                "baseline_X_J45_percent": frozen["baseline_X_J45_percent"],
                "predicted_X_J45_percent": case["predicted_X_J45_percent"],
                "actual_X_J45_percent": actual_j,
                "error_X_J45_percentage_points":
                    actual_j-case["predicted_X_J45_percent"],
                "max_residual": float(np.max(residual[1:8])),
                "seed_agreement": agreement,
                "mass_orthogonality": orth,
                "scope": "fixed X sorted 4/5 gap; full IBZ not re-searched",
            })
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "PASS_FIXED_X_PREDICTION_RECOMPUTED",
        "rows": len(rows),
        "min_mass_MAC": min(row["mass_MAC"] for row in rows),
        "max_abs_change_error": max(abs(row["error_delta_ln_f"]) for row in rows),
        "max_abs_gap_prediction_error_percentage_points":
            max(abs(row["error_X_J45_percentage_points"]) for row in rows),
        "NOT_RUN": ["perturbed full-IBZ search", "3D perturbation prediction"],
    }
    (OUT / "prediction_assessment.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        (freeze if args.stage == "freeze" else evaluate)()
