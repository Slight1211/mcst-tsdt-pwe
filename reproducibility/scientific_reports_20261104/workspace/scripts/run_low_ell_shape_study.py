"""Two-start N15 shape searches at ell=0 and 0.5 micrometres."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '05_validation/LOW_ELL_SHAPES_20261101'
SEED = ROOT / '02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1/manifest.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def main():
    BASE.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[name] = '2'
    meta = {'status': 'RUNNING', 'N': 15, 'a_um': 75, 'h_um': 6,
            'ell_steel_um': 0, 'optimization_points': 153, 'starts_per_ell': 2,
            'length_scales_um': [0, .5], 'pid': os.getpid(), 'runs': [],
            'worker_sha256': sha(ROOT / 'scripts/run_shape_comparison_case.py'),
            'controller_sha256': sha(Path(__file__)),
            'no_global_optimality_claim': True, 'no_new_COMSOL': True}
    jobs = []
    for ell, slug in [(0, 'ell_0'), (.5, 'ell_0p5')]:
        folder = BASE / slug
        folder.mkdir()
        for label in ['regular', 'ell1_seed']:
            out = folder / label
            cmd = [sys.executable, '-u', '-X', 'utf8', str(ROOT / 'scripts/run_shape_comparison_case.py'),
                   '--ell-um', str(ell), '--out', str(out)]
            if label == 'ell1_seed':
                cmd += ['--seed-manifest', str(SEED)]
            handle = (folder / f'{label}.console.log').open('x', encoding='utf-8')
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            record = {'ell_um': ell, 'label': label, 'out': out.relative_to(ROOT).as_posix(),
                      'pid': child.pid, 'status': 'RUNNING', 'command': cmd}
            meta['runs'].append(record)
            jobs.append((child, handle, record))
            print('LAUNCHED', ell, label, child.pid, flush=True)
    save(BASE / 'manifest.json', meta)
    while any(child.poll() is None for child, _, _ in jobs):
        time.sleep(5)
        for child, _, record in jobs:
            if record['status'] == 'RUNNING' and child.poll() is not None:
                record.update(status='EXITED', exit_code=child.returncode)
                print('EXITED', record['ell_um'], record['label'], child.returncode, flush=True)
        save(BASE / 'manifest.json', meta)
    for child, handle, record in jobs:
        handle.close()
        mp = ROOT / record['out'] / 'manifest.json'
        worker = json.loads(mp.read_text(encoding='utf-8')) if mp.exists() else {}
        record.update(exit_code=child.returncode, status=worker.get('status', 'MISSING_MANIFEST'),
                      manifest_sha256=sha(mp) if mp.exists() else None)
        if worker.get('status', '').startswith('PASS'):
            record['J153'] = worker['final_ibz153']['J']
            record['best_evaluated_J153'] = worker['best_evaluated153']['J']
    meta['status'] = ('COMPLETED_ALL_SEARCHES' if all(r['status'].startswith('PASS') for r in meta['runs'])
                      else 'REQUIRES_FAILURE_REVIEW')
    meta['cases'] = []
    for ell in [0, .5]:
        runs = [r for r in meta['runs'] if r['ell_um'] == ell and r['status'].startswith('PASS')]
        if len(runs) == 2:
            best = max(runs, key=lambda r: r['J153'])
            meta['cases'].append({'ell_um': ell, 'selected_run': best['out'], 'J153': best['J153'],
                                  'best_evaluated_J153': max(r['best_evaluated_J153'] for r in runs)})
    save(BASE / 'manifest.json', meta)
    print(meta['status'], meta['cases'], flush=True)


if __name__ == '__main__':
    main()
