"""Sixteen new reproducible starts at ell=0; retain the two previous searches."""
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
BASE = ROOT / '05_validation/ELL0_MULTISTART_20261101'
WORKER = ROOT / 'scripts/run_ell0_multistart_case.py'
PREVIOUS = ROOT / '05_validation/LOW_ELL_SHAPES_20261101/ell_0'
RNG_SEED = 20261101
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))


def save(p, data):
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(p)


def helmert():
    H = np.zeros((7, 6))
    for j in range(6):
        i = j + 1
        H[:i, j] = 1 / math.sqrt(i * (i + 1))
        H[i, j] = -i / math.sqrt(i * (i + 1))
    return H


def geometry(y):
    radial = np.exp(helmert() @ y)
    indices = np.minimum(np.arange(48) % 12, 12 - np.arange(48) % 12)
    theta = np.arange(48) * math.pi / 24
    v = radial[indices, None] * np.column_stack([np.cos(theta), np.sin(theta)])
    area = .5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1) - v[:, 1] * np.roll(v[:, 0], -1))
    return v * math.sqrt(.09 * math.pi / area)


def prepare():
    BASE.mkdir(parents=True, exist_ok=False)
    (BASE / 'seeds').mkdir()
    inputs = {}
    seeds = []

    def from_manifest(relative):
        p = ROOT / relative
        m = load(p)
        assert m['status'].startswith('PASS') and m['N'] == 15
        assert m['a_um'] == 75 and m['h_um'] == 6
        inputs[relative] = sha(p)
        return np.asarray(m['design'], dtype=float)

    old = from_manifest('05_validation/LOW_ELL_SHAPES_20261101/ell_0/regular/manifest.json')

    def add(label, values, description):
        raw = np.asarray(values, dtype=float)
        assert raw.shape == (6,) and np.isfinite(raw).all()
        scale = min(1., .16 / max(float(np.max(abs(raw))), 1e-30))
        y = raw * scale
        v = geometry(y)
        assert np.max(abs(y)) <= .16 + 1e-12 and np.max(abs(v)) < .49
        assert all(np.max(abs(v - geometry(np.asarray(s['design'])))) > 1e-8 for s in seeds)
        seed = dict(label=label, design=y.tolist(), raw_design=raw.tolist(),
                    log_radius_contrast_scale=scale, description=description,
                    N=15, a_um=75, h_um=6, ell_epoxy_um=0, ell_steel_um=0,
                    shape_bounds=[-.16, .16], fixed_area_fraction=.09 * math.pi,
                    coordinate_max=float(np.max(abs(v))))
        path = BASE / 'seeds' / f'{label}.json'
        save(path, seed)
        seed.update(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))
        seeds.append(seed)

    for label, path in [
        ('legacy_ell0p5', '05_validation/LOW_ELL_SHAPES_20261101/ell_0p5/regular/manifest.json'),
        ('legacy_ell2', '05_validation/V07_ell2_reoptimization_20261030/restart/manifest.json'),
        ('legacy_ell5', '05_validation/GAP_CLOSURE_SCAN_20261031/ell_5/warm/manifest.json'),
    ]:
        add(label, from_manifest(path), 'Existing geometry only; all frequencies freshly solved at ell=0.')
    add('opposite_old', -old, 'Reverse all six log-radius contrast coordinates of the previous ell=0 solution.')
    H = helmert()
    add('rotated45_reduced_contrast', H.T @ (H @ old)[::-1],
        'Rotate sector log radii by 45 degrees, then reduce log-radius contrast uniformly to fit the same box; not an exact rigid rotation after reduction.')
    theta = np.arange(7) * math.pi / 24
    for sign, name in [(1, 'smooth_cos4_positive'), (-1, 'smooth_cos4_negative')]:
        z = sign * .14 * np.cos(4 * theta)
        add(name, H.T @ (z - np.mean(z)), 'Smooth fourfold log-radius seed; fit the same six-coordinate box before fixed-area normalization.')
    add('alternating_bounds', [.16, -.16, .16, -.16, .16, -.16], 'Alternating coordinate-box corner.')
    rng = np.random.default_rng(RNG_SEED)
    lhs = np.column_stack([(rng.permutation(8) + rng.random(8)) / 8 for _ in range(6)])
    for i, y in enumerate(-.16 + .32 * lhs, 1):
        add(f'lhs_{i:02d}', y, 'Randomized Latin hypercube: eight strata per coordinate; fixed generator seed 20261101.')
    assert len(seeds) == 16
    baseline = []
    for name in ['regular', 'ell1_seed']:
        p = PREVIOUS / name / 'manifest.json'
        m = load(p)
        assert m['status'].startswith('PASS') and m['ell_epoxy_um'] == 0
        baseline.append(dict(label='previous_' + name, manifest=p.relative_to(ROOT).as_posix(),
                             manifest_sha256=sha(p), J153=m['final_ibz153']['J']))
    meta = dict(status='PREPARED', N=15, a_um=75, h_um=6, ell_epoxy_um=0, ell_steel_um=0,
                shape_bounds=[-.16, .16], vertices=48, shape_variables=6, fixed_area_fraction=.09 * math.pi,
                optimization_points=153, active_start_indices=[0, 27, 136, 152],
                active_start_semantics='Known valid subset of the unchanged153 constraints; every terminal geometry gets a complete153-point check.',
                random_seed=RNG_SEED, new_starts=16, previous_starts=2, seeds=seeds, baseline=baseline,
                worker_sha256=sha(WORKER), controller_sha256=sha(Path(__file__)), input_sha256=inputs,
                no_global_optimality_claim=True, new_COMSOL_runs=False, runs=[])
    save(BASE / 'manifest.json', meta)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--parallel', type=int, default=6)
    args = parser.parse_args()
    assert 1 <= args.parallel <= 6
    meta = load(BASE / 'manifest.json') if BASE.exists() else prepare()
    assert meta['status'] == 'PREPARED', 'Do not rerun or overwrite a launched study.'
    assert sha(WORKER) == meta['worker_sha256'] and sha(Path(__file__)) == meta['controller_sha256']
    assert all(sha(ROOT / p) == h for p, h in meta['input_sha256'].items())
    assert all(sha(ROOT / s['path']) == s['sha256'] for s in meta['seeds'])
    if args.prepare_only:
        print('PREPARED', len(meta['seeds']), 'new seeds; previous2 retained', flush=True)
        return
    env = os.environ.copy()
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[name] = '2'
    pending = list(meta['seeds'])
    running = []
    meta.update(status='RUNNING', pid=os.getpid(), max_parallel=args.parallel, threads_per_worker=2)
    while pending or running:
        while pending and len(running) < args.parallel:
            seed = pending.pop(0)
            out = BASE / seed['label']
            cmd = [sys.executable, '-u', '-X', 'utf8', str(WORKER), '--ell-um', '0', '--out', str(out),
                   '--seed-manifest', str(ROOT / seed['path'])]
            handle = (BASE / (seed['label'] + '.console.log')).open('x', encoding='utf-8')
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            record = dict(label=seed['label'], pid=child.pid, status='RUNNING', command=cmd,
                          out=out.relative_to(ROOT).as_posix(), seed_sha256=seed['sha256'])
            meta['runs'].append(record)
            running.append((child, handle, record))
            print('LAUNCHED', seed['label'], child.pid, flush=True)
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
            print('FINISHED', record['label'], record['status'], record.get('J153'), flush=True)
            running.remove((child, handle, record))
        meta['pending_labels'] = [s['label'] for s in pending]
        save(BASE / 'manifest.json', meta)
        if pending or running:
            time.sleep(5)
    meta['status'] = ('COMPLETED_ALL_STARTS' if all(r['status'].startswith('PASS') for r in meta['runs'])
                      else 'COMPLETED_WITH_FAILED_STARTS_REQUIRES_REVIEW')
    assert all(sha(ROOT / p) == h for p, h in meta['input_sha256'].items())
    save(BASE / 'manifest.json', meta)
    print(meta['status'], flush=True)


if __name__ == '__main__':
    main()
