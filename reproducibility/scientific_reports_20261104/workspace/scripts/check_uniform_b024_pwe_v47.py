"""Fresh spectra, modal energy fractions and N21 checks of selected b=.24 shapes."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, eigsh
from threadpoolctl import threadpool_limits
import run_lengthscale_optimization as base
from inverse_circle_bending import prepare, classical_inverse
from real_circle_bending import assemble_real

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '05_validation/UNIFORM_B024_PWE_V47_20261103'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
POINTS = {'Gamma': np.array([0., 0.]), 'X': np.array([np.pi, 0.]), 'M': np.array([np.pi, np.pi])}


def save(p, value):
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(p)


def solve(F, cache, N, k, ell):
    tick = time.perf_counter()
    parts = assemble_real(N, k, h=.08, theory='TSDT', indicator_matrix=F)
    mass, kb = parts['M'], parts['KB']
    del parts
    kmc = (ell / 6.) ** 2 * kb
    K = classical_inverse(cache, k) + kmc
    factor = cho_factor(K + .001 * mass, lower=True, check_finite=False)
    inverse = LinearOperator(K.shape, matvec=lambda v: cho_solve(factor, v, check_finite=False), dtype=float)
    val, vec = eigsh(K, k=12, M=mass, sigma=-.001, OPinv=inverse, tol=1e-10, ncv=60,
                    v0=np.random.default_rng(20261030).normal(size=len(K)))
    ix = np.argsort(val)
    val, vec = val[ix], vec[:, ix]
    kv, mv = K @ vec, mass @ vec
    residual = np.linalg.norm(kv - mv * val, axis=0) / (np.linalg.norm(kv, axis=0) + abs(val) * np.linalg.norm(mv, axis=0) + 1e-30)
    first = 1 if max(abs(k)) < 1e-14 else 0
    assert min(val[first:]) > 0
    assert first == 0 or abs(val[0]) < 1e-7
    orth = float(np.max(abs(vec.T @ mv - np.eye(12))))
    assert max(residual[first:6]) < 1e-7 and orth < 1e-7
    rmc = [float(vec[:, b] @ (kmc @ vec[:, b]) / (vec[:, b] @ kv[:, b])) for b in [3, 4]]
    check = dict(max_nonrigid_first6_residual=float(max(residual[first:6])), mass_orthogonality=orth,
                 Rmc4=rmc[0], Rmc5=rmc[1], solve_seconds=time.perf_counter() - tick)
    return np.sqrt(np.maximum(val[:6], 0)), dict(vectors=vec, omega=np.sqrt(np.maximum(val, 0)), ij=cache['ij'], k=k, residual=residual), check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape-ell', type=float, required=True)
    ap.add_argument('--ell', type=float, required=True)
    ap.add_argument('--N', type=int, choices=[15, 21], default=15)
    ap.add_argument('--grid', choices=['45', '153', 'GXM'], default='45')
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    summary = load(ROOT / 'output/uniform_b024_v47/summary.json')
    case = next(c for c in summary['cases'] if c['ell_um'] == args.shape_ell)
    assert case['bound'] == .24 and sha(ROOT / case['manifest']) == case['manifest_sha256']
    folder = OUT / args.tag
    if folder.exists():
        m = load(folder / 'manifest.json')
        assert m['status'] == 'PASS_EXACT_B024_GEOMETRY_PWE' and m['shape_vertices_sha256'] == case['vertices_sha256']
        print('REUSED_COMPLETED', args.tag, flush=True)
        return
    folder.mkdir(parents=True, exist_ok=False)
    y = np.asarray(case['design'])
    kpoints = base.ibz(8 if args.grid == '45' else 16) if args.grid != 'GXM' else np.array(list(POINTS.values()))
    inputs = {p: sha(ROOT / p) for p in [case['manifest'], case['vertices']]}
    sources = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)), 'scripts/run_lengthscale_optimization.py': sha(ROOT / 'scripts/run_lengthscale_optimization.py')}
    meta = dict(status='RUNNING', pid=os.getpid(), N=args.N, a_um=75, h_um=6, shape_bound=.24,
                shape_ell_um=args.shape_ell, evaluation_ell_um=args.ell, design=y.tolist(),
                shape_manifest=case['manifest'], shape_vertices_sha256=case['vertices_sha256'],
                points=len(kpoints), grid=args.grid, eigen_count=12, shift=-.001, tolerance=1e-10,
                threads=2, random_seed=20261030, source_sha256=sources, input_sha256=inputs,
                new_optimization=False, new_COMSOL_runs=False, human_verified=False)
    save(folder / 'manifest.json', meta)
    tick = time.perf_counter()
    values, checks = [], []
    try:
        with threadpool_limits(2):
            F = base.fourier(y, args.N)
            cache = prepare(args.N, .08, theory='TSDT', indicator_matrix=F)
            with (folder / 'spectra.csv').open('x', encoding='utf-8', newline='') as f, (folder / 'diagnostics.csv').open('x', encoding='utf-8', newline='') as d:
                w = csv.writer(f)
                w.writerow(['point', 'kx_a', 'ky_a'] + [f'Omega{i}' for i in range(1, 7)])
                auditor = None
                for i, k in enumerate(kpoints):
                    freq, fields, check = solve(F, cache, args.N, k, args.ell)
                    values.append(freq)
                    rec = dict(point=i, kx_a=float(k[0]), ky_a=float(k[1]), **check)
                    checks.append(rec)
                    w.writerow([i, *k, *freq])
                    f.flush()
                    if auditor is None:
                        auditor = csv.DictWriter(d, fieldnames=list(rec))
                        auditor.writeheader()
                    auditor.writerow(rec)
                    d.flush()
                    for tag, q in POINTS.items():
                        if max(abs(k - q)) < 1e-13:
                            np.savez_compressed(folder / f'fields_{tag}.npz', **fields)
                    if (i + 1) % 10 == 0 or i + 1 == len(kpoints):
                        print(args.tag, i + 1, '/', len(kpoints), flush=True)
                    meta['progress'] = i + 1
                    save(folder / 'manifest.json', meta)
        gap = base.sampled_gap(np.array(values))
        gap['kL'] = kpoints[gap['active_L']].tolist()
        gap['kU'] = kpoints[gap['active_U']].tolist()
        assert all(sha(ROOT / p) == digest for p, digest in {**inputs, **sources}.items())
        meta.update(status='PASS_EXACT_B024_GEOMETRY_PWE', gap=gap,
                    max_nonrigid_residual=max(c['max_nonrigid_first6_residual'] for c in checks),
                    max_mass_orthogonality=max(c['mass_orthogonality'] for c in checks),
                    elapsed_seconds=time.perf_counter() - tick,
                    output_sha256={p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != 'manifest.json'})
        save(folder / 'manifest.json', meta)
        print('COMPLETE', args.tag, json.dumps(gap), flush=True)
    except BaseException:
        meta.update(status='FAILED', traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - tick)
        save(folder / 'manifest.json', meta)
        raise


if __name__ == '__main__':
    main()
