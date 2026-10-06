"""Single endpoint full path after explicit Gamma reference repair; no retries."""
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
from run_joint_baseline import ROOT, OUT, HERE, WORK, PROJECT, BIN, preflight, sha, utc, write_json, audit_spectrum

EXP = WORK/'experiments/endpoint_a40_joint'
DEST = ROOT/'runs/endpoint_a40_joint_M1L6_20261001'
OLD = ROOT/'runs/ratio_scale_fullpath_M2L4_20260926/mcst_a40_M2L4'
SHIFT = OUT/'diagnostics/endpoint_gamma_shift_v1'
FIELDS = OUT/'diagnostics/endpoint_shift_fields_v1/completion.json'
FILES = ('SolveA40JointPath.java','SolveA40JointPath.class','run_a40_joint_path.py','run_joint_baseline.py')


def freeze():
    assert not (EXP/'inputs.json').exists()
    assert json.loads(FIELDS.read_text())['physical_identity'] == 'PASS'
    assert json.loads((SHIFT/'completion.json').read_text())['status'] == 'PASS_SHIFT_REPEATABILITY'
    old = json.loads((OLD/'manifest.json').read_text())
    assert old['status'] == 'PASS_FULL_PATH_EXPORT'
    paths = [OLD/'manifest.json',OLD/'frequencies.csv',FIELDS,SHIFT/'completion.json',
             SHIFT/'shift1_repeat1/model_GammaShift.mph', SHIFT/'shift1_repeat1/frequencies.csv',
             SHIFT/'shift1_repeat2/frequencies.csv']
    write_json(EXP/'inputs.json',dict(frozen_utc=utc(),protocol_sha256=sha(EXP/'protocol.md'),
        input_hashes={str(p):sha(p) for p in paths},source_hashes={n:sha(HERE/n) for n in FILES},
        settings=old['settings']))
    print('PASS_FROZEN_INPUTS')


def verify(frozen):
    assert sha(EXP/'protocol.md') == frozen['protocol_sha256']
    assert all(sha(Path(p)) == h for p,h in frozen['input_hashes'].items())
    assert all(sha(HERE/n) == h for n,h in frozen['source_hashes'].items())


def reference(frozen):
    folder = DEST/'reference_M2L4_gamma_recomputed'; folder.mkdir()
    with (OLD/'frequencies.csv').open(newline='') as stream:
        reader = csv.DictReader(stream); columns = reader.fieldnames; old = list(reader)
    merged = []
    for point in range(13):
        if point not in (0,12):
            merged.extend(r for r in old if int(r['point_index']) == point)
        else:
            source = SHIFT/f"shift1_repeat{1 if point==0 else 2}/frequencies.csv"
            with source.open(newline='') as stream:
                for row in csv.DictReader(stream):
                    merged.append(dict(row,point_index=str(point),kx_pi_over_a='0.0',ky_pi_over_a='0.0'))
    assert len(merged) == 260
    path = folder/'frequencies_derived.csv'
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns); writer.writeheader(); writer.writerows(merged)
    audit = audit_spectrum(path)
    audit.update(input_hashes=frozen['input_hashes'],gamma_source='1MHz repeat1 and repeat2',
                 internal_points_source=str(OLD/'frequencies.csv'),original_status_unchanged=True)
    write_json(folder/'audit.json',audit)
    return audit


def run():
    frozen = json.loads((EXP/'inputs.json').read_text()); verify(frozen)
    machine = preflight(); DEST.mkdir(exist_ok=False)
    (DEST/'sources').mkdir(); (DEST/'checkpoints').mkdir()
    for n in FILES: shutil.copy2(HERE/n,DEST/'sources'/n)
    baseline = reference(frozen)
    scratch = Path(tempfile.mkdtemp(prefix='mcst_a40_joint_')); assert scratch.is_dir()
    rec = dict(status='RUNNING',started_utc=utc(),controller_pid=os.getpid(),machine=machine,
        source_hashes=frozen['source_hashes'],inputs_sha256=sha(EXP/'inputs.json'),protocol_sha256=frozen['protocol_sha256'],
        baseline_audit=baseline,settings=dict(frozen['settings'],COMSOL_MESH_LEVEL='1',COMSOL_DRY_SWEEP_LAYERS='6'),
        gamma_solver=dict(shift_MHz=1,rtol=1e-9),internal_solver=dict(shift_MHz=6.25e-5,rtol=1e-6),
        field_MAC='NOT_RUN_CROSS_MESH',time_limit_seconds=None)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('COMSOL_','MCST_JOINT_'))}
    env.update(MCST_JOINT_INPUT=str(SHIFT/'shift1_repeat1/model_GammaShift.mph'),MCST_JOINT_OUTPUT=str(DEST))
    cmd = [str(BIN),'-np','4','-prefsdir',str(require_prefs(OUT/'prefs')),'-tmpdir',str(scratch),'-recoverydir',str(scratch),
           '-autosave','off','-inputfile',str(DEST/'sources/SolveA40JointPath.class'),'-outputfile',str(DEST/'model.mph'),
           '-batchlog',str(DEST/'batch.log')]
    rec['command'] = cmd; start = time.perf_counter()
    write_json(DEST/'manifest.json',rec)
    try:
        with (DEST/'console.log').open('w') as stream:
            proc = subprocess.Popen(cmd,cwd=PROJECT/'comsol',env=env,stdout=stream,stderr=subprocess.STDOUT)
            rec['comsol_pid'] = proc.pid; write_json(DEST/'manifest.json',rec); rec['exit_code'] = proc.wait()
        rec.update(seconds=time.perf_counter()-start,finished_utc=utc()); verify(frozen)
        assert rec['exit_code'] == 0 and (DEST/'model_JointA40.mph').is_file()
        logged = (DEST/'console.log').read_text(errors='replace')
        assert 'JOINT_MESH_READY mesh=1 layers=6' in logged and 'JOINT_PATH_COMPLETE' in logged
        matches = re.findall(r'POINT_SETTINGS point=(\d+) kx=([^ ]+) ky=([^ ]+) shift=([^ ]+) rtol=([^ ]+) neigs=(\d+)',logged)
        assert len(matches) == 13
        for p,x,y,shift,rtol,count in matches:
            p = int(p); assert int(count) == 20
            assert math.isclose(float(shift.removesuffix('[MHz]')),1 if p in (0,12) else 6.25e-5,rel_tol=1e-12)
            assert math.isclose(float(rtol),1e-9 if p in (0,12) else 1e-6,rel_tol=1e-12)
        with (DEST/'frequencies.csv').open(newline='') as stream: rows = list(csv.DictReader(stream))
        assert len(rows) == 260 and all(sum(int(r['point_index']) == p for r in rows)==20 for p in range(13))
        assert all((DEST/f'checkpoints/point_{p}.mph').is_file() for p in range(13))
        audit = audit_spectrum(DEST/'frequencies.csv'); write_json(DEST/'spectrum_audit.json',audit)
        rec.update(status='PASS_EXPORT_AND_PATH_SPECTRUM',raw_sha256=sha(DEST/'frequencies.csv'),
                   spectrum_audit=audit,cross_mesh_pairing='NOT_RUN')
        write_json(DEST/'manifest.json',rec); write_json(DEST/'completion.json',rec)
        print('PASS_A40_PATH',rec['seconds'],flush=True)
    except Exception as exc:
        rec.update(status='FAIL',reason=repr(exc),finished_utc=utc(),seconds=time.perf_counter()-start)
        write_json(DEST/'manifest.json',rec); write_json(DEST/'failure.json',rec); raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--freeze',action='store_true')
    args = ap.parse_args(); freeze() if args.freeze else run()
