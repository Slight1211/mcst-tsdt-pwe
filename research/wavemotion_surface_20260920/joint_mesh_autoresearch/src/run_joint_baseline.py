"""One serial joint-refinement run. Never overwrite an existing attempt."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
ROOT = WORK.parent
PROJECT = ROOT.parents[1]
OLD = ROOT / 'runs/section41_fullpath_M2L4_20260924'
OUT = ROOT / 'runs/section41_joint_M1L6_20260930'
sys.path.insert(0, str(ROOT))
from comsol_runtime import comsol_batch, preferences_file, require_file, require_prefs
BIN = Path(os.environ.get('COMSOL_BIN', '.')).expanduser().resolve() / 'comsolbatch.exe'
HOOK = '''    // Joint-mesh validation: preserve solutions without changing the solve.
    String checkpointDir = envString("COMSOL_CHECKPOINT_DIR", "");
    if (!checkpointDir.isEmpty()) {
      try {
        model.save(checkpointDir + "/point_" + point + ".mph");
      } catch (Exception exception) {
        throw new RuntimeException("Could not save path checkpoint " + point, exception);
      }
    }
'''


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(require_file(path).read_bytes()).hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)


def check_source():
    old = require_file(OLD / 'sources/BuildTwoPhaseValidation.java', 'historical builder source for the original source-difference check').read_text()
    new = (HERE / 'BuildTwoPhaseValidation.java').read_text()
    assert new.count(HOOK) == 1 and new.replace(HOOK, '') == old, 'Unexpected equation/source change'


def audit_spectrum(path, threshold=.8):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert sorted({int(r['point_index']) for r in rows}) == list(range(13)), 'Missing path point'
    spectrum, min_pol, max_imag = [], 1., 0.
    for p in range(13):
        expected = (p/4, 0) if p <= 4 else ((1, (p-4)/4) if p <= 8 else ((12-p)/4, (12-p)/4))
        hits = [r for r in rows if int(r['point_index']) == p]
        assert len(hits) >= 6
        assert len({int(r['mode_index']) for r in hits}) == len(hits)
        for r in hits:
            for key in ('frequency_mhz', 'imag_frequency_mhz', 'vertical_fraction'):
                assert math.isfinite(float(r[key])), (p, key)
            assert math.isclose(float(r['kx_pi_over_a']), expected[0], abs_tol=1e-12)
            assert math.isclose(float(r['ky_pi_over_a']), expected[1], abs_tol=1e-12)
        selected = sorted((r for r in hits if float(r['frequency_mhz']) > 1e-5
                           and float(r['vertical_fraction']) > threshold),
                          key=lambda r: float(r['frequency_mhz']))
        need = 5 if p in (0, 12) else 6
        assert len(selected) >= need, (p, len(selected), 'Not enough flexural modes')
        chosen = selected[:need]
        values = [float(r['frequency_mhz']) for r in chosen]
        if p in (0, 12):
            values.insert(0, 0.)  # analytical rigid branch, raw file unchanged
        spectrum.append(values)
        for r in chosen:
            ratio = abs(float(r['imag_frequency_mhz'])) / float(r['frequency_mhz'])
            assert ratio < 1e-7, (p, 'Complex eigenfrequency', ratio)
            max_imag = max(max_imag, ratio)
            min_pol = min(min_pol, float(r['vertical_fraction']))
    assert all(math.isclose(x, y, rel_tol=1e-8, abs_tol=1e-8)
               for x, y in zip(spectrum[0], spectrum[12])), 'Gamma closure'
    lp = max(range(12), key=lambda p: spectrum[p][3])
    up = min(range(12), key=lambda p: spectrum[p][4])
    return dict(status='PASS_PATH_SPECTRUM', point_count=13, polarization_threshold=threshold,
                min_vertical=min_pol, max_relative_imaginary=max_imag,
                L_MHz=spectrum[lp][3], U_MHz=spectrum[up][4], L_point=lp, U_point=up,
                spectra_MHz=spectrum, raw_sha256=sha(path))


def preflight():
    comsol_batch()
    if os.name != 'nt':
        raise RuntimeError('This archived joint controller requires Windows PowerShell preflight; port that check before running on another operating system.')
    command = "@{os=(Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory,FreeVirtualMemory,TotalVirtualMemorySize);cpu=(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors);processes=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^comsol(batch|mphserver)?\\.exe$' } | Select-Object ProcessId,Name,CommandLine)} | ConvertTo-Json -Depth 4 -Compress"
    result = subprocess.run(['powershell', '-NoProfile', '-Command', command],
                            check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)
    assert not info['processes'], 'A COMSOL process already exists; do not run in parallel'
    assert int(info['os']['FreeVirtualMemory']) > 20 * 1024**2, 'Insufficient commit headroom'
    assert shutil.disk_usage(tempfile.gettempdir()).free > 30 * 1024**3, 'Insufficient scratch space'
    assert shutil.disk_usage(ROOT).free > 10 * 1024**3, 'Insufficient output space'
    return info


def self_test():
    check_source()
    reports = {mesh: audit_spectrum(OLD / mesh / 'frequencies.csv')
               for mesh in ('mcst_M2_L4', 'mcst_M1_L4', 'mcst_M2_L6')}
    with tempfile.TemporaryDirectory(prefix='mcst_audit_test_') as td:
        original = (OLD / 'mcst_M2_L4/frequencies.csv').read_text().splitlines()
        bad = Path(td) / 'missing_point.csv'
        bad.write_text('\n'.join(line for line in original if not line.startswith('12,')))
        try:
            audit_spectrum(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('Missing-point negative control was not rejected')
    print(json.dumps(dict(status='PASS_SELF_TEST', source_delta='checkpoint_only',
                         reference_edges={k: [v['L_MHz'], v['U_MHz']] for k, v in reports.items()},
                         missing_point_rejected=True), indent=2))


def run():
    comsol_batch()
    profile = preferences_file(ROOT)
    check_source()
    machine = preflight()
    OUT.mkdir(exist_ok=True)
    folder = OUT / 'mcst_M1_L6'
    folder.mkdir()  # intentionally refuses to replace or resume an old attempt
    (folder / 'checkpoints').mkdir()
    (OUT / 'sources').mkdir(exist_ok=True)
    (OUT / 'prefs').mkdir(exist_ok=True)
    for name in ('BuildTwoPhaseValidation.java', 'BuildTwoPhaseValidation.class', 'run_joint_baseline.py'):
        shutil.copy2(require_file(HERE / name), OUT / 'sources' / name)
    shutil.copy2(profile, OUT / 'prefs/comsol.prefs')
    scratch = Path(tempfile.mkdtemp(prefix='mcst_joint_M1L6_'))
    assert scratch.is_dir()
    baseline = json.loads((OLD / 'mcst_M1_L4/manifest.json').read_text())
    assert baseline['status'] == 'PASS_EXPORT_AND_PATH_SPECTRUM'
    settings = dict(baseline['settings'])
    settings.update(COMSOL_DRY_SWEEP_LAYERS='6', COMSOL_RESULT_CSV=str(folder / 'frequencies.csv'),
                    COMSOL_TIMING_CSV=str(folder / 'timing.csv'),
                    COMSOL_CHECKPOINT_DIR=str(folder / 'checkpoints'))
    assert settings['COMSOL_MESH_LEVEL'] == '1' and settings['COMSOL_PATH_POINTS_PER_SEGMENT'] == '4'
    env = {k:v for k,v in os.environ.items() if not k.startswith(('COMSOL_', 'VALIDATION_'))}
    env.update(settings)
    cmd = [str(BIN), '-np', '4', '-prefsdir', str(OUT / 'prefs'), '-tmpdir', str(scratch),
           '-recoverydir', str(scratch), '-autosave', 'off', '-inputfile',
           str(OUT / 'sources/BuildTwoPhaseValidation.class'), '-outputfile', str(folder / 'model.mph'),
           '-batchlog', str(folder / 'batch.log')]
    rec = dict(status='RUNNING', started_utc=utc(), controller_pid=os.getpid(), settings=settings,
               command=cmd, machine=machine, python=sys.version, platform=platform.platform(),
               protocol_sha256=sha(WORK / 'protocol.md'), baseline_manifest_sha256=sha(OLD / 'mcst_M1_L4/manifest.json'),
               source_hashes={p.name: sha(p) for p in (OUT / 'sources').iterdir()},
               time_limit_seconds=None, field_MAC='NOT_RUN')
    mf = folder / 'manifest.json'
    write_json(mf, rec)
    started = time.perf_counter()
    print('START', utc(), folder, flush=True)
    try:
        with (folder / 'console.log').open('w') as log:
            proc = subprocess.Popen(cmd, cwd=PROJECT / 'comsol', env=env, stdout=log, stderr=subprocess.STDOUT)
            rec['comsol_pid'] = proc.pid
            write_json(mf, rec)
            rec['exit_code'] = proc.wait()  # no artificial timeout
        rec.update(seconds=time.perf_counter()-started, finished_utc=utc())
        assert rec['exit_code'] == 0 and (folder / 'model_Model.mph').is_file()
        audit = audit_spectrum(folder / 'frequencies.csv')
        assert all((folder / 'checkpoints' / f'point_{p}.mph').is_file() for p in range(13))
        write_json(folder / 'spectrum_audit.json', audit)
        rec.update(status='PASS_EXPORT_AND_PATH_SPECTRUM', point_count=13,
                   min_vertical=audit['min_vertical'], max_relative_imaginary=audit['max_relative_imaginary'])
        write_json(mf, rec)
        write_json(OUT / 'completion.json', dict(status='PASS_JOINT_BASELINE_PATH', finished_utc=utc(),
                   manifest_sha256=sha(mf), mesh_sensitivity='PENDING_REPORT', field_MAC='NOT_RUN'))
        print('COMPLETE', rec['seconds'], flush=True)
    except Exception as exc:
        rec.update(status='FAIL', reason=repr(exc), finished_utc=utc(), seconds=time.perf_counter()-started)
        write_json(mf, rec)
        write_json(OUT / 'failure.json', rec)
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    self_test() if args.self_test else run()
