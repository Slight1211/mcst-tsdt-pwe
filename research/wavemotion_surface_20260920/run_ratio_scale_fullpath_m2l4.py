"""Sequential same-geometry 3D full-path runs for the fixed h/a=0.08 scan.

No X/M-only data are accepted as band edges. This script is resumable and keeps
every raw COMSOL export and manifest; it does not create manuscript figures.
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
from comsol_runtime import comsol_batch, preferences_file, require_file

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = ROOT / "runs/fixed_ratio_scale_20260922T175850279921Z"
OUT = ROOT / "runs/ratio_scale_fullpath_M2L4_20260926"
SCALES = (75, 10, 15, 25, 40, 100, 200, 500, 1000, 2000)
CASES = tuple((a, theory) for a in SCALES for theory in ("classic", "mcst"))


def validated_spectrum(folder: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("exit_code") == 0
    assert (folder / "model_Model.mph").is_file()
    rows = list(csv.DictReader((folder / "frequencies.csv").open(newline="")))
    assert sorted({int(row["point_index"]) for row in rows}) == list(range(13))
    minimum_vertical = 1.0
    maximum_imaginary = 0.0
    for point in range(13):
        bending = sorted(
            (row for row in rows if int(row["point_index"]) == point
             and float(row["frequency_mhz"]) > 1e-5
             and float(row["vertical_fraction"]) > 0.8),
            key=lambda row: float(row["frequency_mhz"]),
        )
        assert len(bending) >= (5 if point in (0, 12) else 6), (point, len(bending))
        for row in bending[:6]:
            frequency = float(row["frequency_mhz"])
            imag = abs(float(row["imag_frequency_mhz"])) / frequency
            assert math.isfinite(frequency) and math.isfinite(imag) and imag < 1e-7
            minimum_vertical = min(minimum_vertical, float(row["vertical_fraction"]))
            maximum_imaginary = max(maximum_imaginary, imag)
    return {
        "status": "PASS_FULL_PATH_EXPORT",
        "path_points": 13,
        "minimum_vertical_fraction": minimum_vertical,
        "maximum_relative_imaginary": maximum_imaginary,
    }


def main() -> None:
    BIN = comsol_batch()
    profile = preferences_file(ROOT)
    require_file(SOURCE / "audit.json")
    require_file(PROJECT / "comsol/BuildTwoPhaseValidation.class", "compiled COMSOL builder")
    assert json.loads((SOURCE / "audit.json").read_text())["status"] == "COMPLETE_FIXED_MESH_CHECKS"
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("prefs", "sources"):
        (OUT / name).mkdir(exist_ok=True)
    config = {
        "a_um": list(SCALES), "h_over_a": 0.08, "r_over_a": 0.3,
        "theories": ["classic", "mcst"], "mesh": "M2L4",
        "path_segments": 4, "path_points": 13, "threads": 4,
        "time_limit_seconds": None,
        "reference": str(SOURCE),
    }
    config_file = OUT / "config.json"
    if config_file.exists():
        assert json.loads(config_file.read_text()) == config
    else:
        config_file.write_text(json.dumps(config, indent=2))
    source_class = PROJECT / "comsol/BuildTwoPhaseValidation.class"
    source_java = PROJECT / "comsol/BuildTwoPhaseValidation.java"
    for source in (source_class, source_java, Path(__file__)):
        target = OUT / "sources" / source.name
        if target.exists():
            assert target.read_bytes() == source.read_bytes()
        else:
            shutil.copy2(source, target)
    prefs = OUT / "prefs/comsol.prefs"
    if not prefs.exists():
        shutil.copy2(profile, prefs)
    scratch = Path(tempfile.gettempdir()) / "mcst_ratio_scale_fullpath_scratch_20260926"
    scratch.mkdir(parents=True, exist_ok=True)
    assert scratch.is_dir()
    (OUT / "provenance.json").write_text(json.dumps({
        "command": sys.argv, "python": sys.version,
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "sources": {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                    for item in (OUT / "sources").iterdir()},
    }, indent=2))
    print("OUTPUT", OUT, flush=True)
    for a, theory in CASES:
        tag = f"{theory}_a{a}_M2L4"
        folder = OUT / tag
        folder.mkdir(exist_ok=True)
        try:
            print("REUSE", tag, validated_spectrum(folder), flush=True)
            continue
        except (OSError, ValueError, AssertionError, KeyError):
            pass
        base_path = SOURCE / f"mcst_a{a}" / "manifest.json"
        base = json.loads(base_path.read_text())["settings"]
        assert float(base["COMSOL_LATTICE_UM"]) == a
        assert abs(float(base["COMSOL_THICKNESS_UM"]) / a - 0.08) < 1e-12
        assert abs(float(base["COMSOL_RING_RADIUS_UM"]) / a - 0.3) < 1e-12
        settings = dict(base)
        settings.update({
            "COMSOL_MCST_LENGTH_UM": "1" if theory == "mcst" else "0",
            "COMSOL_MESH_LEVEL": "2",
            "COMSOL_DRY_SWEEP_LAYERS": "4",
            "COMSOL_PATH_POINTS_PER_SEGMENT": "4",
            "COMSOL_IBZ_SUBDIVISIONS": "0",
            "COMSOL_CONTROL_XM_ONLY": "0",
            "COMSOL_EIGEN_COUNT": "20",
            "COMSOL_RESULT_CSV": str(folder / "frequencies.csv"),
            "COMSOL_TIMING_CSV": str(folder / "timing.csv"),
        })
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("COMSOL_", "VALIDATION_"))}
        env.update(settings)
        command = [
            str(BIN), "-np", "4", "-prefsdir", str(OUT / "prefs"),
            "-tmpdir", str(scratch), "-recoverydir", str(scratch),
            "-autosave", "off", "-inputfile", str(OUT / "sources" / source_class.name),
            "-outputfile", str(folder / "model.mph"),
            "-batchlog", str(folder / "batch.log"),
        ]
        manifest = {"status": "RUNNING", "settings": settings, "command": command}
        manifest_file = folder / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2))
        started = time.perf_counter()
        print("START", tag, flush=True)
        with (folder / "console.log").open("w") as stream:
            code = subprocess.run(command, cwd=PROJECT / "comsol", env=env,
                                  stdout=stream, stderr=subprocess.STDOUT).returncode
        manifest.update(exit_code=code, seconds=time.perf_counter() - started)
        manifest_file.write_text(json.dumps(manifest, indent=2))
        try:
            manifest.update(validated_spectrum(folder))
        except (OSError, ValueError, AssertionError, KeyError) as exc:
            manifest.update(status="FAIL", reason=repr(exc))
        manifest_file.write_text(json.dumps(manifest, indent=2))
        print("DONE", tag, manifest["status"], round(manifest["seconds"], 1), flush=True)
        if manifest["status"] != "PASS_FULL_PATH_EXPORT":
            raise RuntimeError(tag + ": " + manifest["reason"])
    (OUT / "completion.json").write_text(
        json.dumps({"status": "ALL_TWENTY_PATHS_VALIDATED",
                    "cases": [f"{theory}_a{a}_M2L4" for a, theory in CASES]}, indent=2))
    print("COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    main()
