"""H2 selected-point prerequisite; one serial solve, immutable old model."""
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from run_joint_baseline import require_prefs
from run_joint_baseline import OUT, OLD, HERE, WORK, PROJECT, BIN, preflight, sha, utc, write_json

DEST = OUT/'cross_mesh_X_v1'
SOURCE = OLD/'mcst_M1_L4/model_Model.mph'


def selected(path, point=4):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    rows = [r for r in rows if int(r['point_index']) == point]
    assert len(rows) == 20 and len({r['mode_index'] for r in rows}) == 20
    assert all(math.isfinite(float(v)) for r in rows for v in r.values())
    assert all(float(r['kx_pi_over_a']) == 1 and float(r['ky_pi_over_a']) == 0 for r in rows)
    chosen = sorted((r for r in rows if float(r['frequency_mhz'])>1e-5
                     and float(r['vertical_fraction'])>.8), key=lambda r:float(r['frequency_mhz']))[:6]
    assert len(chosen) == 6
    assert all(abs(float(r['imag_frequency_mhz']))/float(r['frequency_mhz'])<1e-7 for r in chosen)
    return chosen


def main():
    machine = preflight()
    original = json.loads((OLD/'mcst_M1_L4/manifest.json').read_text())
    assert original['status'] == 'PASS_EXPORT_AND_PATH_SPECTRUM'
    old = selected(OLD/'mcst_M1_L4/frequencies.csv')
    DEST.mkdir(exist_ok=False)
    for name in ('SolveJointX.java','SolveJointX.class','run_baseline_cross_mesh_x.py'):
        shutil.copy2(HERE/name, DEST/name)
    scratch=Path(tempfile.mkdtemp(prefix='mcst_crossmesh_X_'))
    assert scratch.is_dir()
    digest=sha(SOURCE)
    env={k:v for k,v in os.environ.items() if not k.startswith(('COMSOL_','MCST_X_'))}
    env.update(MCST_X_MODEL=str(SOURCE),MCST_X_CSV=str(DEST/'frequencies.csv'))
    cmd=[str(BIN),'-np','4','-prefsdir',str(require_prefs(OUT/'prefs')),'-tmpdir',str(scratch),
         '-recoverydir',str(scratch),'-autosave','off','-inputfile',str(DEST/'SolveJointX.class'),
         '-outputfile',str(DEST/'model.mph'),'-batchlog',str(DEST/'batch.log')]
    rec=dict(status='RUNNING',controller_pid=os.getpid(),started_utc=utc(),machine=machine,
        source_model=str(SOURCE),source_model_sha256=digest,settings=original['settings'],
        protocol_sha256=sha(WORK/'experiments/baseline_cross_mesh/protocol.md'),
        source_hashes={p.name:sha(p) for p in DEST.iterdir()},command=cmd,
        source_frequency_sha256=sha(OLD/'mcst_M1_L4/frequencies.csv'),cross_mesh_MAC='NOT_RUN')
    write_json(DEST/'manifest.json',rec)
    start=time.perf_counter()
    try:
        with (DEST/'console.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=PROJECT/'comsol',env=env,stdout=log,stderr=subprocess.STDOUT)
            rec['comsol_pid']=proc.pid; write_json(DEST/'manifest.json',rec)
            rec['exit_code']=proc.wait()
        assert rec['exit_code']==0 and (DEST/'model_JointX.mph').is_file()
        log=(DEST/'console.log').read_text(errors='replace')
        assert all(t in log for t in ('UNCHANGED_RTOL=1.0E-6','UNCHANGED_SHIFT=1.0E-4[MHz]',
                                      'UNCHANGED_NEIGS=20','X_SOLVE_COMPLETE'))
        assert sha(SOURCE)==digest, 'Original model changed'
        new=selected(DEST/'frequencies.csv')
        comparisons=[dict(branch=i+1,old_MHz=float(a['frequency_mhz']),new_MHz=float(b['frequency_mhz']),
            difference_Hz=1e6*abs(float(a['frequency_mhz'])-float(b['frequency_mhz'])),
            passed=math.isclose(float(a['frequency_mhz']),float(b['frequency_mhz']),rel_tol=1e-8,abs_tol=1e-8))
            for i,(a,b) in enumerate(zip(old,new))]
        rec.update(comparisons=comparisons,raw_sha256=sha(DEST/'frequencies.csv'))
        assert all(c['passed'] for c in comparisons), 'X repeatability criterion failed'
        rec.update(status='PASS_X_SPECTRUM_REPEAT',finished_utc=utc(),seconds=time.perf_counter()-start)
        write_json(DEST/'manifest.json',rec); write_json(DEST/'completion.json',rec)
        print(json.dumps(rec,indent=2),flush=True)
    except Exception as exc:
        rec.update(status='FAIL_EXECUTION_OR_AUDIT',reason=repr(exc),finished_utc=utc(),seconds=time.perf_counter()-start)
        write_json(DEST/'manifest.json',rec); write_json(DEST/'failure.json',rec)
        raise


if __name__=='__main__':
    main()
