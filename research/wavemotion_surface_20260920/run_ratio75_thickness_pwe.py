"""Verified inverse-PWE thickness scan at fixed a=75 um; no FEM is run.

The h/a grid matches the preceding a=25 um comparison, so only the
in-plane scale changes. Existing solver routines and extraction checks are
reused without modifying their constitutive or Fourier implementations.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv
import hashlib
import json
import sys
import time

import numpy as np
import scipy

from run_verified_inverse_scans import worker
from run_large_scale_3d import inverse_checks


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/thickness_ratio75_20260925/pwe"
A_UM = 75.0
RATIOS = (0.02, 0.04, 0.06, 0.08, 0.12, 0.20, 0.32)


def save_rows(directory, rows, timings):
    if rows:
        with (directory / "spectra.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (directory / "timings.json").write_text(json.dumps(timings, indent=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = inverse_checks()
    config = dict(a_um=A_UM, h_over_a=list(RATIOS),
                  h_um=[A_UM * ratio for ratio in RATIOS], r_over_a=0.3,
                  ell_host_um=1, ell_steel_um=0,
                  theories=["FSDT", "TSDT"], ell_values_um=[0, 1],
                  path="Gamma-X-M-Gamma", N_path=15, segments_path=4,
                  N_controls=9, segments_controls=1,
                  extraction="verified k16, ncv40/80, two independent starts; N9 dense",
                  scope="first eight sorted bending-related branches; path only")
    config_file = OUT / "config.json"
    if config_file.exists():
        assert json.loads(config_file.read_text()) == config
    else:
        config_file.write_text(json.dumps(config, indent=2))
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2))

    started = time.perf_counter()
    for N, segments in ((9, 1), (15, 4)):
        directory = OUT / f"N{N}"
        directory.mkdir(exist_ok=True)
        vertices = np.array(((0., 0.), (np.pi, 0.), (np.pi, np.pi), (0., 0.)))
        path = np.array([(1-t) * vertices[j] + t * vertices[j+1]
                         for j in range(3) for t in np.arange(segments)/segments])
        np.savetxt(directory / "path.csv", path, delimiter=",",
                   header="kx_a,ky_a", comments="")
        jobs = [(N, A_UM, A_UM * ratio, str(directory), segments)
                for ratio in RATIOS]
        rows, timings = [], []
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for future in as_completed(futures):
                n, a, h, result, elapsed = future.result()
                assert n == N and a == A_UM and h in config["h_um"]
                rows.extend(result)
                timings.extend(elapsed)
                save_rows(directory, rows, timings)
                print("DONE", N, "a", a, "h", h,
                      "rows", len(rows), flush=True)
        assert len(rows) == 7 * 3 * segments * 4 * 8
        assert len(timings) == 7 * 2
        (directory / "completion.json").write_text(json.dumps(
            dict(status="PASS_VERIFIED_EXTRACTION_NOT_N_CONVERGENCE",
                 frequency_count=len(rows)), indent=2))

    sources = [Path(__file__), ROOT / "run_verified_inverse_scans.py",
               ROOT / "run_inverse_scan_bands.py",
               ROOT / "src/inverse_circle_bending.py",
               ROOT / "src/real_circle_bending.py"]
    provenance = dict(status="PASS_PWE_COMPUTED", command=sys.argv,
                      python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                      wall_seconds=time.perf_counter() - started,
                      config_sha256=hashlib.sha256(config_file.read_bytes()).hexdigest(),
                      source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in sources},
                      NOT_RUN=["3D FEM", "full Brillouin zone",
                               "high-N full-path convergence"])
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("COMPLETE", OUT, "wall_seconds", provenance["wall_seconds"], flush=True)


if __name__ == "__main__":
    main()
