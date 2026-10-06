"""Queue actual 3D M2L4 full paths for the seven a=75 um thicknesses.

The controller waits for the ten-scale COMSOL run to finish before starting,
so the two memory-intensive batches never run concurrently.
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

import psutil

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
BASE = ROOT / "runs/thickness_ratio75_20260925"
CONTROL = BASE / "comsol_M2L4"
SCALE = ROOT / "runs/ratio_scale_fullpath_M2L4_20260926"
OUT = ROOT / "runs/thickness_ratio75_fullpath_M2L4_20260926"
RATIOS = (0.02, 0.04, 0.06, 0.08, 0.12, 0.20, 0.32)
CASES = tuple((ratio, theory) for ratio in RATIOS
              for theory in ("classic", "mcst"))
CUTOFF = 0.7
SCALE_CONTROLLER_PID = int(os.environ.get("SCALE_CONTROLLER_PID", "0"))


def folder_for(ratio: float, theory: str) -> Path:
    if ratio == 0.08:
        return SCALE / f"{theory}_a75_M2L4"
    return OUT / f"{theory}_ratio{ratio:g}_M2L4"


def validate(folder: Path, ratio: float, theory: str) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest.get("exit_code") == 0
    assert (folder / "model_Model.mph").is_file()
    settings = manifest["settings"]
    assert float(settings["COMSOL_LATTICE_UM"]) == 75
    assert abs(float(settings["COMSOL_THICKNESS_UM"]) - 75 * ratio) < 1e-9
    assert abs(float(settings["COMSOL_CORE_RADIUS_UM"]) - 11.25) < 1e-9
    assert abs(float(settings["COMSOL_RING_RADIUS_UM"]) - 22.5) < 1e-9
    assert float(settings["COMSOL_MCST_LENGTH_UM"]) == (1 if theory == "mcst" else 0)
    assert settings["COMSOL_MESH_LEVEL"] == "2"
    assert settings["COMSOL_DRY_SWEEP_LAYERS"] == "4"
    assert settings["COMSOL_PATH_POINTS_PER_SEGMENT"] == "4"
    assert settings["COMSOL_IBZ_SUBDIVISIONS"] == "0"
    assert settings["COMSOL_CONTROL_XM_ONLY"] == "0"
    rows = list(csv.DictReader((folder / "frequencies.csv").open(newline="")))
    assert sorted({int(row["point_index"]) for row in rows}) == list(range(13))
    min_vertical, max_imaginary = 1.0, 0.0
    for point in range(13):
        selected = sorted(
            (row for row in rows if int(row["point_index"]) == point
             and float(row["frequency_mhz"]) > 1e-5
             and float(row["vertical_fraction"]) > CUTOFF),
            key=lambda row: float(row["frequency_mhz"]),
        )
        needed = 5 if point in (0, 12) else 6
        assert len(selected) >= needed, (folder, point, len(selected))
        for row in selected[:needed]:
            frequency = float(row["frequency_mhz"])
            imaginary = abs(float(row["imag_frequency_mhz"])) / frequency
            assert math.isfinite(imaginary) and imaginary < 1e-7
            min_vertical = min(min_vertical, float(row["vertical_fraction"]))
            max_imaginary = max(max_imaginary, imaginary)
    return {
        "status": "PASS_FULL_PATH_EXPORT", "path_points": 13,
        "minimum_vertical_fraction": min_vertical,
        "maximum_relative_imaginary": max_imaginary,
    }


def wait_for_scale() -> None:
    while not (SCALE / "completion.json").is_file():
        if not SCALE_CONTROLLER_PID:
            raise RuntimeError("Historical scale completion is absent. Restore its inputs and set SCALE_CONTROLLER_PID for a local scale controller before queueing this archived batch.")
        if not psutil.pid_exists(SCALE_CONTROLLER_PID):
            raise RuntimeError("scale controller exited without completion.json")
        print("WAIT_SCALE_COMPLETION", flush=True)
        time.sleep(60)
    completion = json.loads((SCALE / "completion.json").read_text())
    assert completion["status"] == "ALL_TWENTY_PATHS_VALIDATED"


def wait_for_memory(tag: str) -> None:
    for _ in range(60):
        free = psutil.virtual_memory().available
        if free >= 5 * 1024 ** 3:
            return
        print("WAIT_MEMORY_GIB", tag, round(free / 1024 ** 3, 2), flush=True)
        time.sleep(10)
    raise MemoryError("Less than 5 GiB available before " + tag)


def main() -> None:
    BIN = comsol_batch()
    profile = preferences_file(ROOT)
    require_file(BASE / "assessment.json")
    require_file(PROJECT / "comsol/BuildTwoPhaseValidation.class", "compiled COMSOL builder")
    assert json.loads((BASE / "assessment.json").read_text())["status"] == "PASS_ACTUAL_A75_PWE_AND_M2L4_XM"
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("prefs", "sources"):
        (OUT / name).mkdir(exist_ok=True)
    config = {
        "a_um": 75, "h_over_a": list(RATIOS), "r_over_a": 0.3,
        "theories": ["classic", "mcst"], "mesh": "M2L4",
        "path_segments": 4, "path_points": 13, "threads": 4,
        "vertical_fraction_cutoff": CUTOFF, "time_limit_seconds": None,
        "reused_ratio": 0.08, "scale_source": str(SCALE),
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
    scratch = Path(tempfile.gettempdir()) / "mcst_ratio75_fullpath_scratch_20260926"
    scratch.mkdir(parents=True, exist_ok=True)
    assert scratch.is_dir()
    (OUT / "provenance.json").write_text(json.dumps({
        "command": sys.argv, "python": sys.version,
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "sources": {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                    for item in (OUT / "sources").iterdir()},
    }, indent=2))
    wait_for_scale()
    print("OUTPUT", OUT, flush=True)
    for ratio, theory in CASES:
        tag = f"{theory}_ratio{ratio:g}_M2L4"
        folder = folder_for(ratio, theory)
        if ratio == 0.08:
            print("REUSE_SCALE", tag, validate(folder, ratio, theory), flush=True)
            continue
        folder.mkdir(exist_ok=True)
        try:
            print("REUSE", tag, validate(folder, ratio, theory), flush=True)
            continue
        except (OSError, ValueError, AssertionError, KeyError):
            pass
        wait_for_memory(tag)
        template = json.loads((CONTROL / f"{theory}_ratio{ratio:g}" /
                               "manifest.json").read_text())["settings"]
        assert abs(float(template["COMSOL_THICKNESS_UM"]) - 75 * ratio) < 1e-9
        settings = dict(template)
        settings.update({
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
            manifest.update(validate(folder, ratio, theory))
        except (OSError, ValueError, AssertionError, KeyError) as exc:
            manifest.update(status="FAIL", reason=repr(exc))
        manifest_file.write_text(json.dumps(manifest, indent=2))
        print("DONE", tag, manifest["status"], round(manifest["seconds"], 1), flush=True)
        if manifest["status"] != "PASS_FULL_PATH_EXPORT":
            raise RuntimeError(tag + ": " + manifest["reason"])
    (OUT / "completion.json").write_text(json.dumps({
        "status": "ALL_FOURTEEN_PATHS_VALIDATED",
        "cases": [f"{theory}_ratio{ratio:g}_M2L4" for ratio, theory in CASES],
    }, indent=2))
    print("COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    main()
