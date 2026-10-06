"""Thick endpoint joint path only after audited shifted Gamma prerequisites."""
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from run_joint_baseline import require_prefs
from run_joint_baseline import ROOT,HERE,WORK,PROJECT,BIN,preflight,sha,utc,write_json,audit_spectrum

EXP=WORK/'experiments/endpoint_a75_joint'
DEST=ROOT/'runs/endpoint_a75_joint_M1L6_20261001_v1'
OLD=ROOT/'runs/thickness_ratio75_fullpath_M2L4_20260926/mcst_ratio0.32_M2L4'
SHIFT=ROOT/'runs/endpoint_a75_gamma_shift_20261001_v1'
FIELDS=ROOT/'runs/endpoint_a75_shift_fields_20261001_v1/completion.json'
REPORT=SHIFT/'diagnostic_report_v1/audit.json'
PREFS=ROOT/'runs/endpoint_a75_gamma_repeat_20261001_v1/prefs'
FILES=('SolveA75JointPath.java','SolveA75JointPath.class','run_a75_joint_path.py','run_joint_baseline.py')


def freeze():
    assert not (EXP/'inputs.json').exists() and not DEST.exists()
    assert json.loads(FIELDS.read_text())['physical_identity']=='PASS'
    assert json.loads((SHIFT/'completion.json').read_text())['status']=='PASS_SHIFT_REPEATABILITY'
    assert json.loads(REPORT.read_text())['status']=='PASS_RAW_SPECTRA_AND_PHYSICAL_FIELDS_RECOMPUTED'
    old=json.loads((OLD/'manifest.json').read_text());assert old['status']=='PASS_FULL_PATH_EXPORT'
    paths=[OLD/'manifest.json',OLD/'frequencies.csv',OLD/'model_Model.mph',OLD.parent/'sources/BuildTwoPhaseValidation.java',
        FIELDS,REPORT,SHIFT/'completion.json',SHIFT/'shift1_repeat1/model_A75Shift.mph',
        SHIFT/'shift1_repeat1/frequencies.csv',SHIFT/'shift1_repeat2/frequencies.csv',PREFS/'comsol.prefs']
    write_json(EXP/'inputs.json',dict(frozen_utc=utc(),protocol_sha256=sha(EXP/'protocol.md'),
        input_hashes={str(p):sha(p) for p in paths},source_hashes={n:sha(HERE/n) for n in FILES},settings=old['settings']))
    print('PASS_A75_JOINT_INPUT_FREEZE')


def reference(frozen):
    folder=DEST/'reference_M2L4_gamma_recomputed';folder.mkdir()
    with (OLD/'frequencies.csv').open(newline='') as stream:
        reader=csv.DictReader(stream);columns=reader.fieldnames;old=list(reader)
    merged=[]
    for p in range(13):
        if p not in (0,12):merged.extend(r for r in old if int(r['point_index'])==p)
        else:
            source=SHIFT/f"shift1_repeat{1 if p==0 else 2}/frequencies.csv"
            with source.open(newline='') as stream:
                for r in csv.DictReader(stream):merged.append(dict(r,point_index=str(p),kx_pi_over_a='0.0',ky_pi_over_a='0.0'))
    assert len(merged)==260
    path=folder/'frequencies_derived.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(merged)
    audit=audit_spectrum(path,.7);audit.update(input_hashes=frozen['input_hashes'],gamma_source='1MHz repeat1/repeat2',
        internal_points_source=str(OLD/'frequencies.csv'),original_status_unchanged=True)
    write_json(folder/'audit.json',audit);return audit


def run():
    frozen=json.loads((EXP/'inputs.json').read_text())
    def verify():
        assert sha(EXP/'protocol.md')==frozen['protocol_sha256']
        assert all(sha(Path(p))==h for p,h in frozen['input_hashes'].items())
        assert all(sha(HERE/n)==h for n,h in frozen['source_hashes'].items())
    verify();machine=preflight();DEST.mkdir(exist_ok=False);(DEST/'sources').mkdir();(DEST/'checkpoints').mkdir()
    for n in FILES:shutil.copy2(HERE/n,DEST/'sources'/n)
    rec=dict(status='PREPARING',controller_pid=os.getpid(),started_utc=utc(),machine=machine,source_hashes=frozen['source_hashes'],
        inputs_sha256=sha(EXP/'inputs.json'),protocol_sha256=frozen['protocol_sha256'],
        settings=dict(frozen['settings'],COMSOL_MESH_LEVEL='1',COMSOL_DRY_SWEEP_LAYERS='6'),
        gamma_solver=dict(shift_MHz=1,rtol=1e-9),internal_solver=dict(shift_MHz=1e-4,rtol=1e-6),
        polarization_threshold=.7,field_MAC='NOT_RUN_CROSS_MESH',time_limit_seconds=None)
    write_json(DEST/'manifest.json',rec);start=time.perf_counter()
    try:
        rec['baseline_audit']=reference(frozen)
        scratch=Path(tempfile.mkdtemp(prefix='mcst_a75_joint_'));assert scratch.is_dir()
        env={k:v for k,v in os.environ.items() if not k.startswith(('COMSOL_','MCST_JOINT_'))}
        env.update(MCST_JOINT_INPUT=str(SHIFT/'shift1_repeat1/model_A75Shift.mph'),MCST_JOINT_OUTPUT=str(DEST))
        cmd=[str(BIN),'-np','4','-prefsdir',str(require_prefs(PREFS)),'-tmpdir',str(scratch),'-recoverydir',str(scratch),'-autosave','off',
            '-inputfile',str(DEST/'sources/SolveA75JointPath.class'),'-outputfile',str(DEST/'model.mph'),'-batchlog',str(DEST/'batch.log')]
        rec.update(command=cmd,status='RUNNING');preflight();verify()
        with (DEST/'console.log').open('w') as stream:
            proc=subprocess.Popen(cmd,cwd=PROJECT/'comsol',env=env,stdout=stream,stderr=subprocess.STDOUT)
            rec['comsol_pid']=proc.pid;write_json(DEST/'manifest.json',rec);rec['exit_code']=proc.wait()
        rec.update(seconds=time.perf_counter()-start,finished_utc=utc());verify()
        assert rec['exit_code']==0 and (DEST/'model_JointA75.mph').is_file()
        logged=(DEST/'console.log').read_text(errors='replace')
        assert 'JOINT_MESH_READY mesh=1 layers=6' in logged and 'JOINT_PATH_COMPLETE' in logged
        matches=re.findall(r'POINT_SETTINGS point=(\d+) kx=([^ ]+) ky=([^ ]+) shift=([^ ]+) rtol=([^ ]+) neigs=(\d+)',logged)
        assert [int(m[0]) for m in matches]==list(range(13))
        for p,x,y,shift,tol,count in matches:
            p=int(p);expected=(p/4,0) if p<=4 else ((1,(p-4)/4) if p<=8 else ((12-p)/4,(12-p)/4))
            assert int(count)==20 and math.isclose(float(x),expected[0],abs_tol=1e-12) and math.isclose(float(y),expected[1],abs_tol=1e-12)
            assert math.isclose(float(shift.removesuffix('[MHz]')),1 if p in (0,12) else 1e-4,rel_tol=1e-12)
            assert math.isclose(float(tol),1e-9 if p in (0,12) else 1e-6,rel_tol=1e-12)
        with (DEST/'frequencies.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==260 and all(sum(int(r['point_index'])==p for r in rows)==20 for p in range(13))
        assert all(math.isfinite(float(v)) for r in rows for v in r.values())
        assert all((DEST/f'checkpoints/point_{p}.mph').is_file() for p in range(13))
        audit=audit_spectrum(DEST/'frequencies.csv',.7);write_json(DEST/'spectrum_audit.json',audit)
        rec.update(status='PASS_EXPORT_AND_PATH_SPECTRUM',raw_sha256=sha(DEST/'frequencies.csv'),spectrum_audit=audit,
            checkpoint_hashes={str(p):sha(DEST/f'checkpoints/point_{p}.mph') for p in range(13)},cross_mesh_pairing='NOT_RUN')
        write_json(DEST/'manifest.json',rec);write_json(DEST/'completion.json',rec);print('PASS_A75_PATH',rec['seconds'],flush=True)
    except Exception as exc:
        rec.update(status='FAIL',reason=repr(exc),finished_utc=utc(),seconds=time.perf_counter()-start)
        write_json(DEST/'manifest.json',rec);write_json(DEST/'failure.json',rec);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args();freeze() if args.freeze else run()
