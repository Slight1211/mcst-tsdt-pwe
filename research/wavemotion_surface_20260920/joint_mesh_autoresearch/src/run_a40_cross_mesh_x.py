"""One preregistered M2L4 X solve, preserving all original data and settings."""
import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from run_joint_baseline import require_prefs
from run_joint_baseline import ROOT, OUT, HERE, WORK, PROJECT, BIN, preflight, sha, utc, write_json
from run_baseline_cross_mesh_x import selected

EXP=WORK/'experiments/a40_cross_mesh'
JOINT=ROOT/'runs/endpoint_a40_joint_M1L6_20261001_v2'
OLD=ROOT/'runs/ratio_scale_fullpath_M2L4_20260926/mcst_a40_M2L4'
DEST=JOINT/'cross_mesh_X_v1'
SOURCE=OLD/'model_Model.mph'
FILES=('SolveA40X.java','SolveA40X.class','run_a40_cross_mesh_x.py','run_joint_baseline.py','run_baseline_cross_mesh_x.py')


def freeze():
    assert not (EXP/'inputs.json').exists()
    audit=json.loads((JOINT/'report_v1/audit.json').read_text())
    assert audit['status']=='PASS_RAW_PATH_AND_PWE_RECOMPUTATION'
    assert all(a['L_point']==a['U_point']==4 for a in audit['audits'].values())
    paths=[SOURCE,OLD/'manifest.json',OLD/'frequencies.csv',JOINT/'report_v1/audit.json',JOINT/'checkpoints/point_4.mph']
    write_json(EXP/'inputs.json',dict(frozen_utc=utc(),protocol_sha256=sha(EXP/'protocol.md'),
        input_hashes={str(p):sha(p) for p in paths},source_hashes={n:sha(HERE/n) for n in FILES}))
    print('PASS_FROZEN_A40_X')


def verify(frozen):
    assert sha(EXP/'protocol.md')==frozen['protocol_sha256']
    assert all(sha(Path(p))==h for p,h in frozen['input_hashes'].items())
    assert all(sha(HERE/n)==h for n,h in frozen['source_hashes'].items())


def run():
    frozen=json.loads((EXP/'inputs.json').read_text());verify(frozen)
    machine=preflight();old=selected(OLD/'frequencies.csv')
    DEST.mkdir(exist_ok=False)
    for name in FILES:shutil.copy2(HERE/name,DEST/name)
    scratch=Path(tempfile.mkdtemp(prefix='mcst_a40_X_'));assert scratch.is_dir()
    env={k:v for k,v in os.environ.items() if not k.startswith(('COMSOL_','MCST_X_'))}
    env.update(MCST_X_MODEL=str(SOURCE),MCST_X_CSV=str(DEST/'frequencies.csv'))
    cmd=[str(BIN),'-np','4','-prefsdir',str(require_prefs(OUT/'prefs')),'-tmpdir',str(scratch),'-recoverydir',str(scratch),
         '-autosave','off','-inputfile',str(DEST/'SolveA40X.class'),'-outputfile',str(DEST/'model.mph'),'-batchlog',str(DEST/'batch.log')]
    rec=dict(status='RUNNING',started_utc=utc(),controller_pid=os.getpid(),machine=machine,
        inputs_sha256=sha(EXP/'inputs.json'),protocol_sha256=frozen['protocol_sha256'],
        source_hashes=frozen['source_hashes'],command=cmd,cross_mesh_MAC='NOT_RUN',time_limit_seconds=None)
    write_json(DEST/'manifest.json',rec);start=time.perf_counter()
    try:
        with (DEST/'console.log').open('w') as stream:
            proc=subprocess.Popen(cmd,cwd=PROJECT/'comsol',env=env,stdout=stream,stderr=subprocess.STDOUT)
            rec['comsol_pid']=proc.pid;write_json(DEST/'manifest.json',rec);rec['exit_code']=proc.wait()
        verify(frozen)
        assert rec['exit_code']==0 and (DEST/'model_A40X.mph').is_file()
        log=(DEST/'console.log').read_text(errors='replace')
        assert 'A40X_SETTINGS a40 h3.2 M2L4' in log and 'A40X_SOLVE_COMPLETE' in log
        new=selected(DEST/'frequencies.csv')
        comparisons=[dict(branch=i+1,old_MHz=float(a['frequency_mhz']),new_MHz=float(b['frequency_mhz']),
            difference_Hz=1e6*abs(float(a['frequency_mhz'])-float(b['frequency_mhz'])),
            passed=math.isclose(float(a['frequency_mhz']),float(b['frequency_mhz']),rel_tol=1e-8,abs_tol=1e-8))
            for i,(a,b) in enumerate(zip(old,new))]
        rec['comparisons']=comparisons
        assert all(c['passed'] for c in comparisons),'X frequency repeatability'
        rec.update(status='PASS_X_SPECTRUM_REPEAT',finished_utc=utc(),seconds=time.perf_counter()-start,
            raw_sha256=sha(DEST/'frequencies.csv'),model_sha256=sha(DEST/'model_A40X.mph'))
        write_json(DEST/'manifest.json',rec);write_json(DEST/'completion.json',rec)
        print('PASS_A40_X',rec['seconds'],flush=True)
    except Exception as exc:
        rec.update(status='FAIL_EXECUTION_OR_AUDIT',reason=repr(exc),finished_utc=utc(),seconds=time.perf_counter()-start)
        write_json(DEST/'manifest.json',rec);write_json(DEST/'failure.json',rec);raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');args=ap.parse_args()
    freeze() if args.freeze else run()
