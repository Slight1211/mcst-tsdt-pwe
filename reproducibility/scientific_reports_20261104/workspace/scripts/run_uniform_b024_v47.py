"""Complete five length-scale searches with b=.24, retaining immutable old runs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '05_validation/UNIFORM_B024_V47_20261103'
REVIEW = ROOT / '03_review/uniform_b024_v47'
SNAP = ROOT / '03_review/revisions/pre_uniform_b024_v46'
WORKER = ROOT / 'scripts/run_relaxed_shape_case.py'
OLD = {
    1.: '02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1/manifest.json',
    2.: '05_validation/V07_ell2_reoptimization_20261030/restart/manifest.json',
    5.: '05_validation/GAP_CLOSURE_SCAN_20261031/ell_5/warm/manifest.json',
}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    REVIEW.mkdir(parents=True, exist_ok=False)
    SNAP.mkdir(parents=True, exist_ok=False)
    paths = [ROOT / 'README_CN.txt']
    paths += [ROOT / '03_review' / f for f in ['DELIVERY_SUMMARY.json', 'WORKSPACE_SCOPE.json', 'revision_build_inputs.json', 'VALIDATION_BACKLOG.csv']]
    paths += [ROOT / 'latex' / f'{name}.{ext}' for name in ['main', 'main_zh', 'supplementary', 'supplementary_zh'] for ext in ['tex', 'aux', 'log']]
    paths += list((ROOT / 'output/pdf/v46').glob('*.pdf'))
    for p in paths:
        target = SNAP / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    oldstudy = load(ROOT / '05_validation/SHAPE_BOUND_SENSITIVITY_20261103/manifest.json')
    reuse = [j for j in oldstudy['jobs'] if j['bound'] == .24]
    assert len(reuse) == 4 and all(j['status'] == 'PASS' for j in reuse)
    jobs = []
    for ell in [1., 2., 5.]:
        for seed in ['regular', 'warm']:
            jobs.append(dict(ell_um=ell, seed=seed, bound=.24, solver='slsqp', status='PENDING',
                             seed_manifest=OLD[ell] if seed == 'warm' else None,
                             label=f'ell_{ell:g}_{seed}', attempts=[]))
    state = dict(status='PREPARED', revision='v47', date='2026-11-03', N=15, a_um=75, h_um=6,
                 lengths_um=[0, .5, 1, 2, 5], shape_bound=.24, fixed_area_fraction=0.09 * 3.141592653589793,
                 vertices=48, shape_variables=6, starts_per_length_scale=2,
                 optimization_points=153, active_start_indices=[0, 27, 136, 152],
                 existing_low_ell_b024_jobs=reuse, new_jobs=jobs,
                 worker_sha256=sha(WORKER), controller_sha256=sha(Path(__file__)),
                 archived_seed_sha256={p: sha(ROOT / p) for p in OLD.values()},
                 protocol='Two starts: regular 48-gon and corresponding prior local design; expanded b=.24 box; unchanged area and geometric guards; complete 153-point constraint-exchange checks.',
                 selection='Highest terminal J on the full 153-point grid among successful starts satisfying original gradient, feasibility and KKT gates.',
                 solid_comparisons='Must match selected vertices; historical solid checks are not relabeled for changed geometries.',
                 human_verified=False, submission_ready=False)
    save(OUT / 'manifest.json', state)
    save(REVIEW / 'DECLARED_PROTOCOL.json', state)
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parallel', type=int, default=3)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    assert 1 <= args.parallel <= 3
    state = load(OUT / 'manifest.json') if OUT.exists() else prepare()
    assert state['status'] == 'PREPARED'
    if args.prepare_only:
        print('PREPARED', flush=True)
        return
    env = os.environ.copy()
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[key] = '2'
    state.update(status='RUNNING', pid=os.getpid(), threads_per_job=2, parallel=args.parallel)
    pending = list(state['new_jobs'])
    running = []
    while pending or running:
        while pending and len(running) < args.parallel:
            job = pending.pop(0)
            folder = OUT / (job['label'] + ('_fsqp' if job['solver'] == 'fsqp' else ''))
            command = [sys.executable, '-u', '-B', '-X', 'utf8', str(WORKER), '--bound', '.24',
                       '--ell-um', str(job['ell_um']), '--solver', job['solver'], '--out', str(folder)]
            if job['seed_manifest']:
                command += ['--seed-manifest', str(ROOT / job['seed_manifest'])]
            handle = (OUT / (folder.name + '.console.log')).open('x', encoding='utf-8')
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            attempt = dict(out=folder.relative_to(ROOT).as_posix(), pid=child.pid, solver=job['solver'], command=command, status='RUNNING')
            job['attempts'].append(attempt)
            job['status'] = 'RUNNING'
            running.append((child, handle, job, attempt))
            print('LAUNCHED', folder.name, child.pid, flush=True)
        for child, handle, job, attempt in running[:]:
            if child.poll() is None:
                continue
            handle.close()
            mp = ROOT / attempt['out'] / 'manifest.json'
            result = load(mp) if mp.exists() else {}
            attempt.update(exit_code=child.returncode, status=result.get('status', 'NO_MANIFEST'))
            if child.returncode == 0 and result.get('status', '').startswith('PASS'):
                job.update(status='PASS', final_manifest=mp.relative_to(ROOT).as_posix(),
                           manifest_sha256=sha(mp), J153=result['final_ibz153']['J'], design=result['design'])
            elif job['solver'] == 'slsqp':
                job.update(status='PENDING_FSQP_RETRY', solver='fsqp')
                pending.append(job)
            else:
                job.update(status='FAILED_BOTH_SOLVERS')
            print('FINISHED', job['label'], attempt['solver'], attempt['status'], job.get('J153'), flush=True)
            running.remove((child, handle, job, attempt))
        save(OUT / 'manifest.json', state)
        if pending or running:
            time.sleep(3)
    assert sha(WORKER) == state['worker_sha256'] and sha(Path(__file__)) == state['controller_sha256']
    assert all(sha(ROOT / p) == digest for p, digest in state['archived_seed_sha256'].items())
    state['status'] = 'COMPLETED_ALL_STARTS' if all(j['status'] == 'PASS' for j in state['new_jobs']) else 'COMPLETED_WITH_FAILURES'
    save(OUT / 'manifest.json', state)
    print(state['status'], flush=True)


if __name__ == '__main__':
    main()
