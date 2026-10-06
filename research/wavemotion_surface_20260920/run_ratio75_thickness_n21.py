"""N=21 X/M convergence controls for the a=75 um thickness scan."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import time

from run_inverse_scan_bands import *
from run_verified_inverse_scans import verified_solve


BASE = ROOT / "runs/thickness_ratio75_20260925"
OUT = BASE / "pwe/N21"
EXISTING = ROOT / "runs/fixed_ratio_scale_20260922T175850279921Z/pwe"
RATIOS = (0.02, 0.04, 0.06, 0.08, 0.12, 0.20, 0.32)
A_UM = 75.0
N = 21


def compute_ratio(ratio):
    h = A_UM * ratio
    rows = []
    path = (("X", 4, np.array((np.pi, 0.))),
            ("M", 8, np.array((np.pi, np.pi))))
    with threadpool_limits(limits=1):
        for plate in ("FSDT", "TSDT"):
            cache = prepare(N, ratio, .3, plate)
            for point_name, point, k in path:
                stiffness = assemble_real(N, k, ratio, .3, theory=plate)
                del stiffness["Kcl"]
                classical = classical_inverse(cache, k)
                for ell in (0., 1.):
                    filename = OUT / f"N21_a75_h{h:g}_p{point}_{plate}_l{ell:g}.npz"
                    old = EXISTING / f"N21_a75_p{point}_{plate}_l{ell:g}.npz"
                    if ratio == .08 and not filename.exists():
                        with np.load(old) as prior:
                            assert float(prior["a_um"]) == A_UM
                            assert abs(float(prior["h_um"]) - h) < 1e-10
                            assert float(prior["ell_um"]) == ell
                            assert float(prior["seed_agreement"]) < 1e-7
                        shutil.copy2(old, filename)
                    if filename.exists():
                        with np.load(filename) as record:
                            frequency = record["frequency_MHz"].copy()
                            residual = record["residual"].copy()
                            agreement = float(record["seed_agreement"])
                            orthogonality = float(record["mass_orthogonality"])
                    else:
                        K = classical + (ell/h)**2 * stiffness["KB"]
                        eigenvalues, vectors, residual, raw, agreement, orthogonality = \
                            verified_solve(K, stiffness["M"], k, N)
                        frequency = (np.sqrt(eigenvalues[:8]) *
                                     np.sqrt(4.35e9/1180)/(A_UM*1e-6*2*np.pi)/1e6)
                        np.savez_compressed(filename, frequency_MHz=frequency,
                                            residual=residual, raw=raw,
                                            vectors=vectors[:,:8], a_um=A_UM, h_um=h,
                                            ell_um=ell, k=k, seed_agreement=agreement,
                                            mass_orthogonality=orthogonality)
                        del K, vectors
                    assert len(frequency) == 8 and np.min(frequency) > 0
                    assert np.max(residual[:8]) < 1e-5
                    assert agreement < 1e-7 and orthogonality < 1e-7
                    for band in range(8):
                        rows.append(dict(a_um=A_UM, h_um=h, h_over_a=ratio,
                                         point=point_name, plate=plate, ell_um=ell,
                                         band=band+1, frequency_MHz=float(frequency[band]),
                                         residual=float(residual[band]),
                                         seed_agreement=agreement,
                                         mass_orthogonality=orthogonality,
                                         reused_validated_a75_h6=ratio == .08))
                del stiffness, classical
            del cache
    return ratio, rows


def main():
    assert json.loads((BASE / "pwe/provenance.json").read_text())["status"] == "PASS_PWE_COMPUTED"
    OUT.mkdir(exist_ok=True)
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(compute_ratio, ratio) for ratio in RATIOS]
        for future in as_completed(futures):
            ratio, local = future.result()
            assert len(local) == 64
            rows.extend(local)
            with (OUT / "spectra.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print("DONE_N21", ratio, "rows", len(rows), flush=True)
    assert len(rows) == 448
    sources = [Path(__file__), ROOT / "run_verified_inverse_scans.py",
               ROOT / "src/inverse_circle_bending.py",
               ROOT / "src/real_circle_bending.py"]
    audit = dict(status="PASS_N21_XM_ONLY", rows=len(rows), a_um=A_UM,
                 h_over_a=list(RATIOS), reused_validated_a75_h6_problems=8,
                 wall_seconds=time.perf_counter()-started,
                 command=sys.argv,
                 source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sources},
                 NOT_RUN=["N21 full path", "3D FEM", "full Brillouin zone"])
    (OUT / "completion.json").write_text(json.dumps(audit, indent=2))
    print("COMPLETE_N21", OUT, flush=True)


if __name__ == "__main__":
    main()
