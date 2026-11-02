"""Frozen-geometry 2x2 order comparison after independent N21 reoptimization.

The single-order21 mode can prepare g15/N21 before g21 is selected. The final
2x2 mode reuses completed diagonals and optionally the hash-bound frozen g15/N21
cell; all new solves are frozen-geometry evaluations on the same153-point grid.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / '04_code/research/wavemotion_surface_20260920'
CORE = CODE.parent / 'msse_20260919/python'
sys.path[:0] = [str(CODE / 'src'), str(CORE)]
import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import eigsh, LinearOperator
from threadpoolctl import threadpool_info, threadpool_limits
from node_shape_bending import AREA, boundary, fourier, signed_area
from inverse_circle_bending import prepare, classical_inverse
from real_circle_bending import assemble_real


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def save(path, data):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def gap(values, k):
    L, U = float(np.max(values[:, 3])), float(np.min(values[:, 4]))
    lower = list(map(int, np.flatnonzero(abs(values[:, 3] - L) <= 1e-8)))
    upper = list(map(int, np.flatnonzero(abs(values[:, 4] - U) <= 1e-8)))
    return dict(L=L, U=U, J=200 * (U - L) / (U + L),
                lower_tied_indices=lower, upper_tied_indices=upper,
                lower_tied_k_a=k[lower].tolist(), upper_tied_k_a=k[upper].tolist())


def sources_now():
    paths = [Path(__file__), CODE / 'src/node_shape_bending.py',
             CODE / 'src/inverse_circle_bending.py', CODE / 'src/real_circle_bending.py',
             CORE / 'msse_core.py', CORE / 'two_phase.py']
    return {rel(p): sha(p) for p in paths}


def read_grid_spectrum(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 153
    assert [int(r['point']) for r in rows] == list(range(153))
    k = np.array([[float(r['kx_a']), float(r['ky_a'])] for r in rows])
    expected = np.array([(i, j) for i in range(17) for j in range(i + 1)])
    assert np.max(abs(k * 16 / np.pi - expected)) < 1e-10
    values = np.array([[float(r[f'Omega{b}']) for b in range(1, 7)] for r in rows])
    assert np.isfinite(values).all() and np.all(np.diff(values, axis=1) >= -1e-9)
    assert np.all(values[:, 1:] > 0) and np.all(values[1:, 0] > 0)
    return k, values


def read_case(path, N):
    path = path.resolve()
    assert path.is_relative_to(ROOT)
    worker = json.loads(path.read_text(encoding='utf-8'))
    assert worker['status'].startswith('PASS') and worker['N'] == N
    assert (worker['a_um'], worker['h_um'], worker['ell_epoxy_um']) == (75, 6, 0)
    assert worker.get('ell_steel_um', 0) == 0
    y = np.asarray(worker['design'], float)
    assert y.shape == (6,) and np.isfinite(y).all() and np.max(abs(y)) <= .16 + 1e-10
    vertices = boundary(y)
    area = float(signed_area(vertices))
    assert vertices.shape == (48, 2) and abs(area - AREA) < 1e-13
    spectrum = path.parent / 'final_ibz153.csv'
    if 'output_sha256' in worker:
        assert sha(spectrum) == worker['output_sha256'][spectrum.name]
    k, values = read_grid_spectrum(spectrum)
    result = gap(values, k)
    assert all(abs(result[key] - worker['final_ibz153'][key]) < 1e-9 for key in ['L', 'U', 'J'])
    meta = dict(shape_label=f'g{N}', optimization_N=N, design=y.tolist(),
                source_manifest=rel(path), source_manifest_sha256=sha(path),
                own_order_spectrum=rel(spectrum), own_order_spectrum_sha256=sha(spectrum),
                area_fraction=area, area_error=area - AREA, own_order_gap=result)
    return meta, y, k, values, {rel(p): sha(p) for p in [path, spectrum]}


def fresh_cross(case, y, k, N, out, meta):
    label = f"{case['shape_label']}_N{N}"
    F = fourier(y, N)
    cache = prepare(N, .08, theory='TSDT', indicator_matrix=F)
    frequencies, checks = [], []
    columns = ['point', 'kx_a', 'ky_a'] + [f'Omega{b}' for b in range(1, 7)]
    check_columns = ['point', 'kx_a', 'ky_a', 'max_first6_nonrigid_relative_residual',
                     'mass_orthogonality_max', 'Gamma_rigid_lambda', 'solve_seconds']
    with (out / f'{label}_ibz153.csv').open('x', encoding='utf-8', newline='') as file, \
            (out / f'{label}_checks.csv').open('x', encoding='utf-8', newline='') as audit:
        writer, auditor = csv.writer(file), csv.DictWriter(audit, fieldnames=check_columns)
        writer.writerow(columns)
        auditor.writeheader()
        for index, wave in enumerate(k):
            tick = time.perf_counter()
            parts = assemble_real(N, wave, h=.08, theory='TSDT', indicator_matrix=F)
            M = parts['M']
            del parts  # ell=0: no MCST term, same classical inverse factorization.
            K = classical_inverse(cache, wave)
            factor = cho_factor(K + .001 * M, lower=True, check_finite=False)
            inverse = LinearOperator(K.shape,
                matvec=lambda v: cho_solve(factor, v, check_finite=False), dtype=float)
            values, vectors = eigsh(K, k=12, M=M, sigma=-.001, OPinv=inverse,
                tol=1e-10, ncv=60, v0=np.random.default_rng(20261030).normal(size=len(K)))
            order = np.argsort(values)
            values, vectors = values[order], vectors[:, order]
            gamma = np.max(abs(wave)) < 1e-14
            first = 1 if gamma else 0
            assert np.all(values[first:] > 0)
            if gamma:
                assert abs(values[0]) < 1e-7
            Kv, Mv = K @ vectors, M @ vectors
            residual = np.linalg.norm(Kv - Mv * values, axis=0) / (
                np.linalg.norm(Kv, axis=0) + abs(values) * np.linalg.norm(Mv, axis=0) + 1e-30)
            maxres = float(np.max(residual[first:6]))
            orth = float(np.max(abs(vectors.T @ Mv - np.eye(12))))
            assert maxres < 1e-7 and orth < 1e-7, (label, index, maxres, orth)
            frequency = np.sqrt(np.maximum(values[:6], 0))
            frequencies.append(frequency)
            writer.writerow([index, *wave, *frequency])
            check = dict(point=index, kx_a=float(wave[0]), ky_a=float(wave[1]),
                max_first6_nonrigid_relative_residual=maxres, mass_orthogonality_max=orth,
                Gamma_rigid_lambda=float(values[0]) if gamma else None,
                solve_seconds=time.perf_counter() - tick)
            checks.append(check)
            auditor.writerow(check)
            file.flush()
            audit.flush()
            meta['progress'] = dict(cell=label, completed_points=index + 1, total_points=len(k))
            save(out / 'manifest.json', meta)
            print('CROSS', label, index + 1, '/', len(k), 'maxres', maxres, flush=True)
            del K, M, Kv, Mv, factor, inverse, vectors
    result = dict(shape_label=case['shape_label'], optimization_N=case['optimization_N'], evaluation_N=N,
        provenance='FRESH_FROZEN_GEOMETRY_CROSS_EVALUATION', spectrum=f'{label}_ibz153.csv',
        point_count=len(frequencies), **gap(np.asarray(frequencies), k),
        max_first6_nonrigid_relative_residual=max(c['max_first6_nonrigid_relative_residual'] for c in checks),
        max_mass_orthogonality_error=max(c['mass_orthogonality_max'] for c in checks))
    return result


def read_reusable_single(path, g15, source_hashes):
    path = path.resolve()
    if path.is_dir():
        path = path / 'manifest.json'
    assert path.is_relative_to(ROOT)
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved['status'] == 'PASS_FROZEN_ELL0_G15_N21_SINGLE'
    assert saved['mode'] == 'single_g15_order21'
    for key, value in [('a_um', 75.), ('h_um', 6.), ('ell_epoxy_um', 0.),
                       ('ell_steel_um', 0.), ('optimization_points', 153), ('optimization_grid', 16),
                       ('sparse_eigen_count', 12), ('sparse_shift', -.001),
                       ('sparse_tolerance', 1e-10), ('ncv', 60), ('seed', 20261030)]:
        assert saved[key] == value, ('Frozen parameter mismatch', key)
    assert saved['shapes'] == [g15[0]], 'Frozen cell uses a different g15 geometry or source'
    assert saved['source_hash_check'] == saved['input_hash_check'] == 'PASS'
    assert saved['source_sha256'] == saved['source_sha256_post'] == source_hashes
    assert saved['input_sha256'] == saved['input_sha256_post'] == g15[4]
    assert all(sha(ROOT / p) == digest for p, digest in saved['input_sha256'].items())
    assert len(saved['cells']) == 1
    cell = saved['cells'][0]
    assert (cell['shape_label'], cell['optimization_N'], cell['evaluation_N'], cell['point_count']) == ('g15', 15, 21, 153)
    assert cell['provenance'] == 'FRESH_FROZEN_GEOMETRY_CROSS_EVALUATION'
    # Every declared output is bound; recheck the actual full spectrum and
    # per-point numerical gates rather than relying on the PASS string alone.
    bound = {rel(path): sha(path)}
    for name, digest in saved['output_sha256'].items():
        artifact = path.parent / name
        assert artifact.parent == path.parent and sha(artifact) == digest
        bound[rel(artifact)] = digest
    spectrum = path.parent / cell['spectrum']
    assert spectrum.name in saved['output_sha256']
    k, values = read_grid_spectrum(spectrum)
    assert np.max(abs(k - g15[2])) < 1e-14
    result = gap(values, k)
    assert all(abs(result[key] - cell[key]) < 1e-9 for key in ['L', 'U', 'J'])
    for key in ['lower_tied_indices', 'upper_tied_indices', 'lower_tied_k_a', 'upper_tied_k_a']:
        assert result[key] == cell[key]
    audit = path.parent / 'g15_N21_checks.csv'
    assert audit.name in saved['output_sha256']
    with audit.open(encoding='utf-8-sig', newline='') as handle:
        checks = list(csv.DictReader(handle))
    assert len(checks) == 153 and [int(r['point']) for r in checks] == list(range(153))
    assert np.max(abs(np.array([[float(r['kx_a']), float(r['ky_a'])] for r in checks]) - k)) < 1e-14
    maxres = max(float(r['max_first6_nonrigid_relative_residual']) for r in checks)
    orth = max(float(r['mass_orthogonality_max']) for r in checks)
    assert maxres < 1e-7 and orth < 1e-7
    assert abs(float(checks[0]['Gamma_rigid_lambda'])) < 1e-7
    assert all(r['Gamma_rigid_lambda'] == '' for r in checks[1:])
    assert maxres == cell['max_first6_nonrigid_relative_residual']
    assert orth == cell['max_mass_orthogonality_error']
    return dict(**cell, reused_manifest=rel(path)), bound, spectrum, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n15-manifest', type=Path, default=ROOT / '05_validation/ELL0_MULTISTART_20261101/lhs_05/manifest.json')
    parser.add_argument('--n21-manifest', type=Path)
    parser.add_argument('--single-order21', action='store_true')
    parser.add_argument('--geometry-manifest', type=Path,
                        help='N15 geometry manifest for --single-order21; defaults to --n15-manifest')
    parser.add_argument('--reuse-g15-n21', type=Path,
                        help='Completed single-order21 directory or manifest to reuse in final2x2')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.single_order21:
        assert args.n21_manifest is None and args.reuse_g15_n21 is None
    else:
        assert args.n21_manifest is not None and args.geometry_manifest is None
    g15 = read_case(args.geometry_manifest or args.n15_manifest, 15)
    g21 = None if args.single_order21 else read_case(args.n21_manifest, 21)
    if g21:
        assert np.max(abs(g15[2] - g21[2])) < 1e-14
    geometries = [g15] if g21 is None else [g15, g21]
    inputs = {name: digest for data in geometries for name, digest in data[4].items()}
    source_hashes = sources_now()
    reused = read_reusable_single(args.reuse_g15_n21, g15, source_hashes) if args.reuse_g15_n21 else None
    if reused:
        inputs.update(reused[1])
    plans = [] if reused else [dict(shape_label='g15', evaluation_N=21)]
    if g21:
        plans.append(dict(shape_label='g21', evaluation_N=15))
    if args.plan_only:
        print(json.dumps(dict(status='PLAN_ONLY_NO_EIGENSOLVES', fresh_crosses=plans,
                              reused_g15_n21=reused[0] if reused else None,
                              geometries=[d[0] for d in geometries], input_sha256=inputs), indent=2))
        return
    out = args.out.resolve()
    assert out.is_relative_to(ROOT)
    assert not any(Path(ROOT / c[0]['source_manifest']).parent == out or
                   Path(ROOT / c[0]['source_manifest']).parent in out.parents for c in geometries)
    if reused:
        reuse_dir = reused[2].parent
        assert reuse_dir != out and reuse_dir not in out.parents
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(status='RUNNING', mode='single_g15_order21' if args.single_order21 else 'full2x2',
        pid=os.getpid(), N_values=[21] if args.single_order21 else [15, 21], a_um=75., h_um=6.,
        ell_epoxy_um=0., ell_steel_um=0., threads=2, optimization_points=153, optimization_grid=16,
        source_sha256=source_hashes, input_sha256=inputs, shapes=[d[0] for d in geometries],
        sparse_eigen_count=12, sparse_shift=-.001, sparse_tolerance=1e-10, ncv=60, seed=20261030,
        construction='K=classical_inverse(prepare(N,.08,indicatorF),k); M=assemble_real(...)[M]; ell=0',
        scope='Single mode freezes g15 atN21; full2x2 reuses completed diagonals and any validated frozen cross. '
              'Frozen order comparison does not constitute additional optimization or global/BZ certification.',
        new_optimization=False, new_dense_reference=False, new_COMSOL_runs=False,
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()),
        cells=[])
    save(out / 'manifest.json', meta)
    clock = time.perf_counter()
    try:
        if not args.single_order21:
            for case, _, _, _, _ in geometries:
                filename = f"{case['shape_label']}_N{case['optimization_N']}_ibz153.csv"
                shutil.copyfile(ROOT / case['own_order_spectrum'], out / filename)
                meta['cells'].append(dict(shape_label=case['shape_label'], optimization_N=case['optimization_N'],
                    evaluation_N=case['optimization_N'], provenance='HASH_BOUND_SAVED_OWN_ORDER_DIAGONAL',
                    spectrum=filename, point_count=153, **case['own_order_gap']))
        if reused:
            cell = dict(reused[0])
            cell['provenance'] = 'HASH_BOUND_REUSED_FROZEN_CROSS_EVALUATION'
            shutil.copyfile(reused[2], out / cell['spectrum'])
            shutil.copyfile(reused[3], out / reused[3].name)
            meta['cells'].append(cell)
        fresh_jobs = [] if reused else [(g15, 21)]
        if g21:
            fresh_jobs.append((g21, 15))
        with threadpool_limits(2):
            meta['threadpool'] = threadpool_info()
            for data, N in fresh_jobs:
                meta['cells'].append(fresh_cross(data[0], data[1], data[2], N, out, meta))
                save(out / 'manifest.json', meta)
        source_post = {p: sha(ROOT / p) for p in source_hashes}
        input_post = {p: sha(ROOT / p) for p in inputs}
        assert source_post == source_hashes and input_post == inputs
        cells = sorted(meta['cells'], key=lambda c: (c['optimization_N'], c['evaluation_N']))
        with (out / 'order_cross_summary.csv').open('x', encoding='utf-8', newline='') as handle:
            fields = ['shape_label', 'optimization_N', 'evaluation_N', 'provenance', 'L', 'U', 'J', 'spectrum']
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(cells)
        save(out / 'order_cross_summary.json', dict(scope=meta['scope'], cells=cells))
        outputs = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'manifest.json')
        meta.update(status='PASS_FROZEN_ELL0_G15_N21_SINGLE' if args.single_order21 else 'PASS_FROZEN_ELL0_2X2_ORDER_CROSS', cells=cells,
            source_hash_check='PASS', input_hash_check='PASS', source_sha256_post=source_post,
            input_sha256_post=input_post, output_sha256={p.name: sha(p) for p in outputs},
            fresh_point_count=153 * len(fresh_jobs), reused_diagonal_point_count=0 if args.single_order21 else 306,
            reused_frozen_cross_point_count=153 if reused else 0, elapsed_seconds=time.perf_counter() - clock)
        save(out / 'manifest.json', meta)
        print('PASS_CROSS', json.dumps(cells), flush=True)
    except BaseException:
        meta.update(status='FAILED_FROZEN_ORDER_CROSS', error=traceback.format_exc(),
                    elapsed_seconds=time.perf_counter() - clock)
        save(out / 'manifest.json', meta)
        raise


if __name__ == '__main__':
    main()
