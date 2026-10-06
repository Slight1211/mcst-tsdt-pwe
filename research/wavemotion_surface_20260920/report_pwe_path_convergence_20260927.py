"""Same-geometry inverse-PWE path-edge convergence at a=75 um, h=6 um.

N=9 is recomputed at all 12 path points. N=15 uses the completed thickness
scan; N=21 uses the matching boundary points of a completed 9x9 IBZ scan.
No 3D FEM or continuous-Brillouin-zone claim follows from this comparison.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from run_verified_inverse_scans import worker


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/pwe_path_convergence_20260927"
BASE9_CONTROLS = ROOT / "runs/thickness_ratio75_20260925/pwe/N9"
BASE15 = ROOT / "runs/thickness_ratio75_20260925/pwe/N15"
BASE21 = ROOT / "runs/manuscript_revision_20260925/ibz_N21_tau008"
PATH21 = (
    (0, 0), (2, 0), (4, 0), (6, 0), (8, 0),
    (8, 2), (8, 4), (8, 6), (8, 8), (6, 6), (4, 4), (2, 2),
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_n9() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "N9_raw"
    raw.mkdir(exist_ok=True)
    _, a, h, spectra, timings = worker((9, 75.0, 6.0, str(raw), 4))
    assert a == 75.0 and h == 6.0 and len(spectra) == 384
    assert len(list(raw.glob("*.npz"))) == 48
    with (OUT / "N9_spectra.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=spectra[0].keys())
        writer.writeheader()
        writer.writerows(spectra)
    np.savetxt(
        OUT / "path.csv",
        np.array(
            [(1 - t) * np.array(((0., 0.), (np.pi, 0.), (np.pi, np.pi), (0., 0.)))[j]
             + t * np.array(((0., 0.), (np.pi, 0.), (np.pi, np.pi), (0., 0.)))[j + 1]
             for j in range(3) for t in np.arange(4) / 4]
        ),
        delimiter=",", header="kx_a,ky_a", comments=""
    )
    (OUT / "N9_completion.json").write_text(
        json.dumps({
            "status": "PASS_N9_FULL_SAMPLED_PATH",
            "a_um": a, "h_um": h, "r_over_a": 0.3,
            "ell_epoxy_um": [0, 1], "models": ["FSDT", "TSDT"],
            "path_points": 12, "raw_spectra": 48, "stored_bands": 8,
            "timings": timings,
            "source_sha256": {
                str(p): digest(p) for p in (
                    Path(__file__), ROOT / "run_verified_inverse_scans.py",
                    ROOT / "run_inverse_scan_bands.py",
                    ROOT / "src/inverse_circle_bending.py",
                    ROOT / "src/real_circle_bending.py",
                )
            },
        }, indent=2), encoding="utf-8"
    )
    print("PASS_N9_FULL_SAMPLED_PATH", flush=True)


def spectrum_n9_n15(order: int) -> tuple[np.ndarray, float]:
    source = OUT / "N9_spectra.csv" if order == 9 else BASE15 / "spectra.csv"
    records = [
        r for r in rows(source)
        if float(r["a_um"]) == 75 and float(r["h_um"]) == 6
        and r["theory"] == "TSDT" and float(r["ell_um"]) == 1
        and 1 <= int(r["band"]) <= 6
    ]
    assert len(records) == 72
    rawdir = OUT / "N9_raw" if order == 9 else BASE15
    matrix = np.empty((12, 6))
    residuals = []
    for r in records:
        point, band = int(r["point"]), int(r["band"])
        matrix[point, band - 1] = float(r["frequency_MHz"])
        if band in (4, 5):
            residuals.append(float(r["residual"]))
        saved = rawdir / f"N{order}_a75_h6_p{point}_TSDT_l1.npz"
        with np.load(saved) as data:
            assert abs(float(data["frequency_MHz"][band - 1])
                       - matrix[point, band - 1]) < 1e-8
            assert abs(float(data["k"][0]) - path()[point, 0]) < 1e-12
            assert abs(float(data["k"][1]) - path()[point, 1]) < 1e-12
    return matrix, max(residuals)


def path() -> np.ndarray:
    return np.loadtxt(OUT / "path.csv", delimiter=",", skiprows=1)


def spectrum_n21() -> tuple[np.ndarray, float]:
    records = [
        r for r in rows(BASE21 / "ibz_grid_spectra.csv")
        if int(r["N"]) == 21 and float(r["h_over_a"]) == .08
        and 1 <= int(r["band"]) <= 6
    ]
    assert len(records) == 270
    lookup = {(int(r["ix"]), int(r["iy"]), int(r["band"])): r for r in records}
    matrix = np.empty((12, 6))
    residuals = []
    for point, (ix, iy) in enumerate(PATH21):
        for band in range(1, 7):
            r = lookup[ix, iy, band]
            matrix[point, band - 1] = float(r["frequency_MHz"])
            if band in (4, 5):
                residuals.append(float(r["residual"]))
            saved = BASE21 / f"ibz_N21_tau0.08_ix{ix}_iy{iy}.npz"
            with np.load(saved) as data:
                assert abs(float(data["frequency_MHz"][band - 1])
                           - matrix[point, band - 1]) < 1e-8
                assert np.max(abs(data["k"] - path()[point])) < 1e-12
    return matrix, max(residuals)


def report() -> None:
    assert json.loads((OUT / "N9_completion.json").read_text(encoding="utf-8")
                      )["status"] == "PASS_N9_FULL_SAMPLED_PATH"
    assert json.loads((BASE21 / "assessment.json").read_text(encoding="utf-8")
                      )["status"] == "PASS_FINITE_N21_IBZ_ONLY"
    assert json.loads((BASE15 / "completion.json").read_text(encoding="utf-8")
                      )["status"] == "PASS_VERIFIED_EXTRACTION_NOT_N_CONVERGENCE"
    assert path().shape == (12, 2)
    old9 = {
        (int(r["point"]), int(r["band"])): float(r["frequency_MHz"])
        for r in rows(BASE9_CONTROLS / "spectra.csv")
        if float(r["a_um"]) == 75 and float(r["h_um"]) == 6
        and r["theory"] == "TSDT" and float(r["ell_um"]) == 1
        and int(r["band"]) in (4, 5)
    }
    assert len(old9) == 6
    all_rows = []
    for order, extractor in ((9, lambda: spectrum_n9_n15(9)),
                             (15, lambda: spectrum_n9_n15(15)),
                             (21, spectrum_n21)):
        matrix, residual = extractor()
        assert np.isfinite(matrix).all() and matrix[:, 3:5].min() > 0
        assert residual < 1e-6
        if order == 9:
            for old_point, new_point in ((0, 0), (1, 4), (2, 8)):
                for band in (4, 5):
                    assert abs(matrix[new_point, band - 1]
                               - old9[old_point, band]) < 1e-8
        lower_point, upper_point = int(np.argmax(matrix[:, 3])), int(np.argmin(matrix[:, 4]))
        lower, upper = float(matrix[lower_point, 3]), float(matrix[upper_point, 4])
        all_rows.append({
            "N": order, "DOF_per_k": 3 * (2 * order + 1) ** 2,
            "a_um": 75, "h_um": 6, "ell_epoxy_um": 1,
            "path_points": 12, "lower_MHz": lower, "upper_MHz": upper,
            "lower_point": lower_point, "upper_point": upper_point,
            "J45_percent": 200 * (upper - lower) / (upper + lower),
            "max_bands4_5_residual": residual,
        })
        if order == 21:
            reference_spectrum = matrix.copy()
    ref = all_rows[-1]
    for r in all_rows:
        r["lower_difference_to_N21_percent"] = 100 * abs(
            r["lower_MHz"] / ref["lower_MHz"] - 1
        )
        r["upper_difference_to_N21_percent"] = 100 * abs(
            r["upper_MHz"] / ref["upper_MHz"] - 1
        )
        r["J_difference_to_N21_pp"] = r["J45_percent"] - ref["J45_percent"]
    for r, order in zip(all_rows, (9, 15, 21)):
        matrix = (spectrum_n21()[0] if order == 21 else
                  spectrum_n9_n15(order)[0])
        r["max_bands4_5_path_difference_to_N21_percent"] = float(
            100 * np.max(abs(matrix[:, 3:5] / reference_spectrum[:, 3:5] - 1))
        )
    with (OUT / "pwe_path_convergence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    audit = {
        "status": "PASS_SAME_GEOMETRY_12_POINT_PATH_N9_N15_N21",
        "scope": "MCST-TSDT inverse-PWE; sampled Gamma-X-M-Gamma only",
        "N9_source": str(OUT / "N9_spectra.csv"),
        "N9_old_control_crosscheck": str(BASE9_CONTROLS / "spectra.csv"),
        "N15_source": str(BASE15 / "spectra.csv"),
        "N21_source": str(BASE21 / "ibz_grid_spectra.csv"),
        "N15_source_sha256": digest(BASE15 / "spectra.csv"),
        "N21_source_sha256": digest(BASE21 / "ibz_grid_spectra.csv"),
        "NOT_RUN": ["continuous-k extrema certification", "N>21 convergence",
                    "joint PWE and 3D FEM convergence"],
    }
    (OUT / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(all_rows, indent=2), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["compute-n9"]:
        compute_n9()
    elif sys.argv[1:] == ["report"]:
        report()
    else:
        raise SystemExit("usage: report_pwe_path_convergence_20260927.py compute-n9|report")
