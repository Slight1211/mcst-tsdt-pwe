"""Four fresh N21 ell=0 searches in the unchanged six-variable design space."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '05_validation/ELL0_N21_REOPT_20261101'
WORKER = ROOT / 'scripts/run_ell0_resolution_case.py'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def geometry(y):
    H = np.zeros((7, 6))
    for j in range(6):
        i = j + 1
        H[:i, j] = 1 / math.sqrt(i * (i + 1))
        H[i, j] = -i / math.sqrt(i * (i + 1))
    indices = np.minimum(np.arange(48) % 12, 12 - np.arange(48) % 12)
    theta = np.arange(48) * math.pi / 24
    v = np.exp(H @ y)[indices, None] * np.column_stack([np.cos(theta), np.sin(theta)])
    area = .5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1) - v[:, 1] * np.roll(v[:, 0], -1))
    return v * math.sqrt(.09 * math.pi / area)


def prepare():
    BASE.mkdir(parents=True, exist_ok=False)
    (BASE / 'seeds').mkdir()
    definitions = [
        ('n15_best', '05_validation/ELL0_MULTISTART_20261101/lhs_05/manifest.json'),
        ('regular48gon', None),
        ('ell1_diamond', '02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1/manifest.json'),
        ('lhs07_broad', '05_validation/ELL0_MULTISTART_20261101/seeds/lhs_07.json'),
    ]
    seeds, inputs = [], {}
    for label, relative in definitions:
        parent = load(ROOT / relative) if relative else None
        y = np.asarray(parent['design'] if parent else np.zeros(6), dtype=float)
        assert y.shape == (6,) and np.isfinite(y).all() and np.max(abs(y)) <= .16 + 1e-12
        v = geometry(y)
        assert np.max(abs(v)) < .49
        assert all(np.max(abs(v - geometry(np.asarray(s['design'])))) > 1e-8 for s in seeds)
        if relative:
            inputs[relative] = sha(ROOT / relative)
        record = dict(label=label, design=y.tolist(), N=21, ell_epoxy_um=0, ell_steel_um=0,
                      a_um=75, h_um=6, shape_bounds=[-.16, .16], fixed_area_fraction=.09 * math.pi,
                      status='GEOMETRY_ONLY_NOT_A_N21_RESULT', source=relative or 'zero six-vector',
                      source_sha256=inputs.get(relative),
                      semantics='Only geometry reused; optimization, frequency evaluation and derivative gates are freshly computed at N21.')
        sp = BASE / 'seeds' / f'{label}.json'
        save(sp, record)
        record.update(path=sp.relative_to(ROOT).as_posix(), sha256=sha(sp))
        seeds.append(record)
    baseline = ROOT / definitions[0][1]
    bm = load(baseline)
    assert bm['status'].startswith('PASS') and bm['N'] == 15 and bm['ell_epoxy_um'] == 0
    for p in [baseline.parent / 'final_vertices.csv', baseline.parent / 'final_ibz153.csv']:
        inputs[p.relative_to(ROOT).as_posix()] = sha(p)
    source_paths = [Path(__file__), WORKER, ROOT / 'scripts/ell0_resolution_base.py',
                    ROOT / 'scripts/ell0_resolution_sensitivity.py']
    meta = dict(status='PREPARED', N=21, a_um=75, h_um=6, ell_epoxy_um=0, ell_steel_um=0,
                vertices=48, shape_variables=6, shape_bounds=[-.16, .16], fixed_area_fraction=.09 * math.pi,
                optimization_points=153, optimization_grid=16, initial_active_indices=[0, 27, 136, 152],
                starts=4, threads_per_worker=2, seeds=seeds, runs=[], input_sha256=inputs,
                source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths},
                n15_baseline_manifest=baseline.relative_to(ROOT).as_posix(),
                n15_baseline_J153=bm['final_ibz153']['J'],
                no_global_optimality_claim=True, no_continuous_zone_certification=True,
                comparison_scope='Two Fourier orders, same153-point grid and same geometry space; not an asymptotic convergence certificate.',
                new_COMSOL_runs=False)
    save(BASE / 'manifest.json', meta)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--parallel', type=int, default=2)
    args = parser.parse_args()
    assert args.parallel in [1, 2]
    meta = load(BASE / 'manifest.json') if BASE.exists() else prepare()
    assert meta['status'] == 'PREPARED', 'Do not overwrite a launched study.'
    assert all(sha(ROOT / p) == h for p, h in meta['source_sha256'].items())
    assert all(sha(ROOT / p) == h for p, h in meta['input_sha256'].items())
    assert all(sha(ROOT / s['path']) == s['sha256'] for s in meta['seeds'])
    if args.prepare_only:
        print('PREPARED_N21', len(meta['seeds']), 'distinct geometry seeds', flush=True)
        return
    env = os.environ.copy()
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[name] = '2'
    pending, running = list(meta['seeds']), []
    meta.update(status='RUNNING', pid=os.getpid(), max_parallel=args.parallel)
    while pending or running:
        while pending and len(running) < args.parallel:
            seed = pending.pop(0)
            out = BASE / seed['label']
            command = [sys.executable, '-u', '-X', 'utf8', str(WORKER), '--ell-um', '0', '--N', '21',
                       '--out', str(out), '--seed-manifest', str(ROOT / seed['path'])]
            handle = (BASE / f'{seed["label"]}.console.log').open('x', encoding='utf-8')
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            record = dict(label=seed['label'], status='RUNNING', pid=child.pid, command=command,
                          out=out.relative_to(ROOT).as_posix(), seed_sha256=seed['sha256'])
            meta['runs'].append(record)
            running.append((child, handle, record))
            print('LAUNCHED_N21', seed['label'], child.pid, flush=True)
        for child, handle, record in running[:]:
            if child.poll() is None:
                continue
            handle.close()
            p = ROOT / record['out'] / 'manifest.json'
            m = load(p) if p.exists() else {}
            record.update(exit_code=child.returncode, status=m.get('status', 'MISSING_MANIFEST'),
                          manifest_sha256=sha(p) if p.exists() else None)
            if m.get('status', '').startswith('PASS'):
                record.update(J153=m['final_ibz153']['J'], best_evaluated_J153=m['best_evaluated153']['J'])
            running.remove((child, handle, record))
            print('FINISHED_N21', record['label'], record['status'], record.get('J153'), flush=True)
        meta['pending_labels'] = [s['label'] for s in pending]
        save(BASE / 'manifest.json', meta)
        if pending or running:
            time.sleep(5)
    meta['status'] = ('COMPLETED_ALL_N21_SEARCHES' if all(r['status'].startswith('PASS') for r in meta['runs'])
                      else 'COMPLETED_WITH_FAILED_STARTS_REQUIRES_REVIEW')
    assert all(sha(ROOT / p) == h for p, h in meta['source_sha256'].items())
    assert all(sha(ROOT / p) == h for p, h in meta['input_sha256'].items())
    save(BASE / 'manifest.json', meta)
    print(meta['status'], flush=True)


if __name__ == '__main__':
    main()
