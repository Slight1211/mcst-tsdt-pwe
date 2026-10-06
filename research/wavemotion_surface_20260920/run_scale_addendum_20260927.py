"""Add three fixed-ratio scales without modifying the completed original runs.

PWE uses the existing N=15 inverse solver on 12 distinct path points.
The 3D reference is the same-geometry MCST M2L4 COMSOL model on 13 points.
Run the stages sequentially (pwe, fem) to keep memory use bounded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from comsol_runtime import comsol_bin, preferences_file, require_file

import numpy as np

import run_fixed_ratio_scale as base
from run_ratio_scale_fullpath_m2l4 import validated_spectrum

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
OUT = ROOT / "runs/scale_addendum_20260927"
SCALES = (150, 750, 1500)
N = 15
H_OVER_A = 0.08
R_OVER_A = 0.30
OLD_FEM = ROOT / "runs/ratio_scale_fullpath_M2L4_20260926"
PATH_VERTICES = np.array([[0., 0.], [np.pi, 0.], [np.pi, np.pi], [0., 0.]])
PATH = np.array([
    (1 - t) * PATH_VERTICES[j] + t * PATH_VERTICES[j + 1]
    for j in range(3) for t in np.arange(4) / 4
])


def prepare_output() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("pwe", "prefs", "sources"):
        (OUT / name).mkdir(exist_ok=True)
    config = {
        "a_um": list(SCALES), "h_over_a": H_OVER_A,
        "r_over_a": R_OVER_A, "ell_host_um": [0, 1],
        "ell_steel_um": 0, "pwe_order": N,
        "pwe_distinct_path_points": 12, "fem_path_points": 13,
        "fem_theory": "mcst", "fem_mesh": "M2L4",
        "comsol_threads": 4, "time_limit_seconds": None,
    }
    config_path = OUT / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding="utf-8")) == config
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    sources = (
        Path(__file__), ROOT / "run_fixed_ratio_scale.py",
        ROOT / "run_verified_inverse_scans.py",
        ROOT / "src/inverse_circle_bending.py",
        ROOT / "src/real_circle_bending.py",
        PROJECT / "comsol/BuildTwoPhaseValidation.java",
        PROJECT / "comsol/BuildTwoPhaseValidation.class",
    )
    for source in sources:
        target = OUT / "sources" / source.name
        if target.exists():
            assert target.read_bytes() == source.read_bytes(), source
        else:
            shutil.copy2(source, target)
    provenance = OUT / "provenance.json"
    info = {
        "command": sys.argv, "python": sys.version,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "sources_sha256": {
            source.name: hashlib.sha256(source.read_bytes()).hexdigest()
            for source in sources
        },
    }
    if not provenance.exists():
        provenance.write_text(json.dumps(info, indent=2), encoding="utf-8")
    np.savetxt(OUT / "path.csv", PATH, delimiter=",",
               header="kx_a,ky_a", comments="")


def pwe_file(a: int, point: int, theory: str, ell: int) -> Path:
    return OUT / "pwe" / f"N15_a{a}_p{point}_{theory}_l{ell}.npz"


def checked_pwe_file(path: Path, a: int, point: int, ell: int) -> np.ndarray:
    with np.load(path) as record:
        freq = record["frequency_MHz"]
        residual = record["residual"]
        assert int(record["a_um"]) == a
        assert abs(float(record["h_um"]) - H_OVER_A * a) < 1e-10
        assert int(record["ell_um"]) == ell
        assert np.allclose(record["k"], PATH[point], atol=1e-12, rtol=0)
        assert len(freq) == 8 and np.all(np.isfinite(freq))
        start = 1 if point == 0 else 0
        assert min(freq[start:]) > 0
        assert max(residual[start:8]) < 1e-5
        assert float(record["seed_agreement"]) < 1e-7
        assert float(record["mass_orthogonality"]) < 1e-7
        return np.array(freq)


def run_pwe() -> None:
    prepare_output()
    completed = 0
    with base.threadpool_limits(limits=2):
        for theory in ("FSDT", "TSDT"):
            cache = base.prepare(N, H_OVER_A, R_OVER_A, theory)
            for point, k in enumerate(PATH):
                stiffness = base.assemble_real(
                    N, k, H_OVER_A, R_OVER_A, theory=theory
                )
                classical = base.classical_inverse(cache, k)
                for a in SCALES:
                    h = H_OVER_A * a
                    for ell in (0, 1):
                        path = pwe_file(a, point, theory, ell)
                        try:
                            checked_pwe_file(path, a, point, ell)
                            print("PWE_REUSE", path.name, flush=True)
                            completed += 1
                            continue
                        except (OSError, ValueError, AssertionError, KeyError):
                            pass
                        K = classical + (ell / h) ** 2 * stiffness["KB"]
                        lam, vectors, residual, raw, agree, orth = base.verified_solve(
                            K, stiffness["M"], k, N
                        )
                        freq = (np.sqrt(lam[:8]) * math.sqrt(4.35e9 / 1180)
                                / (a * 1e-6 * 2 * math.pi) / 1e6)
                        temporary = path.with_name(path.name + ".tmp.npz")
                        np.savez_compressed(
                            temporary, frequency_MHz=freq,
                            vectors=vectors[:, :8], raw=raw, residual=residual,
                            seed_agreement=agree, mass_orthogonality=orth,
                            a_um=a, h_um=h, ell_um=ell, k=k,
                        )
                        checked_pwe_file(temporary, a, point, ell)
                        os.replace(temporary, path)
                        print("PWE_DONE", path.name, flush=True)
                        completed += 1
    assert completed == 2 * 12 * 3 * 2 == 144
    gaps = []
    for a in SCALES:
        for theory in ("FSDT", "TSDT"):
            for ell in (0, 1):
                spectra = np.array([
                    checked_pwe_file(pwe_file(a, point, theory, ell),
                                     a, point, ell)[:6]
                    for point in range(12)
                ])
                lower_point = int(np.argmax(spectra[:, 3]))
                upper_point = int(np.argmin(spectra[:, 4]))
                lower = float(spectra[lower_point, 3])
                upper = float(spectra[upper_point, 4])
                gaps.append({
                    "a_um": a, "h_um": H_OVER_A * a,
                    "theory": theory, "ell_um": ell,
                    "lower_MHz": lower, "upper_MHz": upper,
                    "lower_point_index": lower_point,
                    "upper_point_index": upper_point,
                    "J45_percent": 200 * (upper - lower) / (upper + lower),
                })
    with (OUT / "pwe_gaps.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=gaps[0])
        writer.writeheader()
        writer.writerows(gaps)
    (OUT / "pwe_completion.json").write_text(json.dumps({
        "status": "PASS_N15_TWELVE_POINT_SPECTRA",
        "spectra": completed, "models": len(gaps), "scales": list(SCALES),
    }, indent=2), encoding="utf-8")
    print("PWE_COMPLETE", OUT, flush=True)


def run_fem() -> None:
    BIN = comsol_bin()
    profile = preferences_file(ROOT)
    require_file(PROJECT / "comsol/BuildTwoPhaseValidation.class", "compiled COMSOL builder")
    prepare_output()
    assert json.loads((OUT / "pwe_completion.json").read_text())["status"] == \
        "PASS_N15_TWELVE_POINT_SPECTRA"
    prefs = OUT / "prefs/comsol.prefs"
    if not prefs.exists():
        shutil.copy2(profile, prefs)
    scratch = Path(tempfile.gettempdir()) / "mcst_scale_addendum_20260927"
    scratch.mkdir(parents=True, exist_ok=True)
    assert scratch.is_dir()
    base_manifest = json.loads(
        (OLD_FEM / "mcst_a100_M2L4/manifest.json").read_text()
    )
    assert base_manifest["status"] == "PASS_FULL_PATH_EXPORT"
    for a in SCALES:
        folder = OUT / f"mcst_a{a}_M2L4"
        folder.mkdir(exist_ok=True)
        manifest_path = folder / "manifest.json"
        try:
            print("FEM_REUSE", a, validated_spectrum(folder), flush=True)
            continue
        except (OSError, ValueError, AssertionError, KeyError):
            pass
        settings = dict(base_manifest["settings"])
        settings.update({
            "COMSOL_LATTICE_UM": str(a),
            "COMSOL_THICKNESS_UM": str(H_OVER_A * a),
            "COMSOL_CORE_RADIUS_UM": str(0.15 * a),
            "COMSOL_RING_RADIUS_UM": str(R_OVER_A * a),
            "COMSOL_ELECTRODE_SPLIT_UM": str(0.225 * a),
            "COMSOL_MCST_LENGTH_UM": "1",
            "COMSOL_MESH_LEVEL": "2",
            "COMSOL_DRY_SWEEP_LAYERS": "4",
            "COMSOL_PATH_POINTS_PER_SEGMENT": "4",
            "COMSOL_IBZ_SUBDIVISIONS": "0",
            "COMSOL_CONTROL_XM_ONLY": "0",
            "COMSOL_EIGEN_COUNT": "20",
            "COMSOL_EIGEN_SHIFT_MHZ": str(0.0001 * 25 / a),
            "COMSOL_RESULT_CSV": str(folder / "frequencies.csv"),
            "COMSOL_TIMING_CSV": str(folder / "timing.csv"),
        })
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("COMSOL_", "VALIDATION_"))}
        env.update(settings)
        command = [
            str(BIN / "comsolbatch.exe"), "-np", "4",
            "-prefsdir", str(OUT / "prefs"),
            "-tmpdir", str(scratch), "-recoverydir", str(scratch),
            "-autosave", "off",
            "-inputfile", str(OUT / "sources/BuildTwoPhaseValidation.class"),
            "-outputfile", str(folder / "model.mph"),
            "-batchlog", str(folder / "batch.log"),
        ]
        manifest = {"status": "RUNNING", "settings": settings, "command": command}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        started = time.perf_counter()
        print("FEM_START", a, flush=True)
        with (folder / "console.log").open("w", encoding="utf-8") as stream:
            code = subprocess.run(
                command, cwd=PROJECT / "comsol", env=env,
                stdout=stream, stderr=subprocess.STDOUT,
            ).returncode
        manifest.update(exit_code=code, seconds=time.perf_counter() - started)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        try:
            manifest.update(validated_spectrum(folder))
        except (OSError, ValueError, AssertionError, KeyError) as exc:
            manifest.update(status="FAIL", reason=repr(exc))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("FEM_DONE", a, manifest["status"], round(manifest["seconds"], 1),
              flush=True)
        if manifest["status"] != "PASS_FULL_PATH_EXPORT":
            raise RuntimeError(f"a={a}: {manifest.get('reason', 'failed')}")
    (OUT / "fem_completion.json").write_text(json.dumps({
        "status": "PASS_THREE_MCST_M2L4_PATHS",
        "scales": list(SCALES), "path_points": 13,
    }, indent=2), encoding="utf-8")
    print("FEM_COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("pwe", "fem"):
        raise SystemExit("usage: run_scale_addendum_20260927.py {pwe|fem}")
    {"pwe": run_pwe, "fem": run_fem}[sys.argv[1]]()
