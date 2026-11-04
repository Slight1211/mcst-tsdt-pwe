"""Schedule geometry-bound PWE and solid checks as two-start selections finish."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '05_validation/UNIFORM_B024_CHECK_QUEUE_V47_20261103'


def main():
    OUT.mkdir(exist_ok=False)
    env = os.environ.copy()
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[key] = '2'
    jobs, scheduled, running = [], set(), []
    while True:
        subprocess.run([sys.executable, '-B', '-X', 'utf8', str(ROOT / 'scripts/summarize_uniform_b024_v47.py')], cwd=ROOT, env=env, check=True, capture_output=True)
        summary = json.loads((ROOT / 'output/uniform_b024_v47/summary.json').read_text(encoding='utf-8'))
        ready = {c['ell_um'] for c in summary['cases']}
        def add(tag, kind, cmd):
            if tag not in scheduled:
                scheduled.add(tag)
                jobs.append(dict(tag=tag, kind=kind, command=[sys.executable, '-u', '-B', '-X', 'utf8'] + cmd, status='PENDING'))
        for ell in [1, 2]:
            if ell not in ready:
                continue
            add(f'g{ell}_N21_own', 'pwe', [str(ROOT / 'scripts/check_uniform_b024_pwe_v47.py'), '--shape-ell', str(ell), '--ell', str(ell), '--N', '21', '--tag', f'g{ell}_N21_own'])
            add(f'g{ell}_N15_own_GXM', 'pwe', [str(ROOT / 'scripts/check_uniform_b024_pwe_v47.py'), '--shape-ell', str(ell), '--ell', str(ell), '--grid', 'GXM', '--tag', f'g{ell}_N15_own_GXM'])
            for kind in ['base', 'mesh', 'count', 'thickness']:
                add(f'solid_ell{ell}_{kind}', 'solid', [str(ROOT / 'scripts/run_uniform_b024_solid_v47.py'), '--ell', str(ell), '--kind', kind])
        if 1 in ready:
            for ell in [.5, 1.5, 2]:
                tag = 'g1_at' + str(ell).replace('.', 'p')
                add(tag, 'pwe', [str(ROOT / 'scripts/check_uniform_b024_pwe_v47.py'), '--shape-ell', '1', '--ell', str(ell), '--tag', tag])
        if 5 in ready:
            for ell, grid in [(0, '153'), (5, 'GXM')]:
                tag = f'g5_at{ell}'
                add(tag, 'pwe', [str(ROOT / 'scripts/check_uniform_b024_pwe_v47.py'), '--shape-ell', '5', '--ell', str(ell), '--grid', grid, '--tag', tag])
        for child, handle, job in running[:]:
            if child.poll() is not None:
                handle.close()
                job.update(status='PASS' if child.returncode == 0 else 'FAILED', exit_code=child.returncode)
                running.remove((child, handle, job))
                print('FINISHED', job['tag'], job['status'], flush=True)
        if any(j['status'] == 'FAILED' for j in jobs):
            # Other running children may finish and retain their complete outputs.
            # Do not discard numerical failures or launch more jobs after one.
            state = dict(status='STOPPED_LAUNCHING_AFTER_FAILURE', jobs=jobs)
            (OUT / 'manifest.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
            for child, handle, job in running:
                job['exit_code'] = child.wait()
                handle.close()
                job['status'] = 'PASS' if job['exit_code'] == 0 else 'FAILED'
            (OUT / 'manifest.json').write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
            raise RuntimeError('A declared check failed; inspect durable case outputs.')
        for kind, maximum in [('pwe', 2), ('solid', 2 if len(ready) == 5 else 1)]:
            available = maximum - sum(j['kind'] == kind for _, _, j in running)
            for job in [j for j in jobs if j['kind'] == kind and j['status'] == 'PENDING'][:max(0, available)]:
                handle = (OUT / (job['tag'] + '.console.log')).open('x', encoding='utf-8')
                child = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                job.update(status='RUNNING', pid=child.pid)
                running.append((child, handle, job))
                print('LAUNCHED', job['tag'], child.pid, flush=True)
        done = len(ready) == 5 and not running and all(j['status'] == 'PASS' for j in jobs)
        state = dict(status='COMPLETED_ALL_CHECKS' if done else 'RUNNING', ready_lengths=sorted(ready), jobs=jobs)
        temp = OUT / 'manifest.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        temp.replace(OUT / 'manifest.json')
        if done:
            print('COMPLETED_ALL_CHECKS', flush=True)
            return
        time.sleep(6)


if __name__ == '__main__':
    main()
