"""Preserve interrupted runs; recover points11/12 and audit a three-source path."""
import argparse,csv,json,math,os,re,shutil,subprocess,tempfile,time
from pathlib import Path
from run_joint_baseline import require_prefs
from run_joint_baseline import ROOT,HERE,WORK,PROJECT,BIN,preflight,sha,utc,write_json,audit_spectrum
from run_a75_joint_n15 import read_csv,check_rows

EXP=WORK/'experiments/endpoint_a75_reboot_resume'
OLD=ROOT/'runs/endpoint_a75_joint_M1L6_20261001_v1'
PRIOR=ROOT/'runs/endpoint_a75_joint_M1L6_n15_20261001_v1'
DEST=ROOT/'runs/endpoint_a75_joint_M1L6_n15_reboot_20261002_v1'
PREFS=ROOT/'runs/endpoint_a75_gamma_repeat_20261001_v1/prefs'
FILES=('ResumeA75N15.java','ResumeA75N15.class','run_a75_reboot_resume.py',
       'run_joint_baseline.py','run_a75_joint_n15.py')

def retained():
    columns,prefix=read_csv(OLD/'frequencies.csv');check_rows(prefix,range(2),20)
    other,previous=read_csv(PRIOR/'frequencies_remaining.csv');assert other==columns
    check_rows(previous,range(2,11),15)
    logged=(PRIOR/'console.log').read_text(errors='replace')
    assert [int(p) for p in re.findall(r'JOINT_POINT_COMPLETE point=(\d+)',logged)]==list(range(2,11))
    assert not (PRIOR/'completion.json').exists()
    assert not (PRIOR/'checkpoints/point_11.mph').exists()
    return columns,prefix,previous

def freeze():
    assert not (EXP/'inputs.json').exists() and not DEST.exists()
    machine=preflight();retained()
    earlier=json.loads((WORK/'experiments/endpoint_a75_joint_n15/inputs.json').read_text())
    assert all(sha(Path(p))==h for p,h in earlier['input_hashes'].items())
    assert all(sha(PRIOR/'sources'/n)==h for n,h in earlier['source_hashes'].items())
    assert sha(WORK/'experiments/endpoint_a75_joint_n15/protocol.md')==earlier['protocol_sha256']
    checkpaths=[(OLD if p<2 else PRIOR)/f'checkpoints/point_{p}.mph' for p in range(11)]
    paths=checkpaths+[OLD/'frequencies.csv',PRIOR/'frequencies_remaining.csv',
      PRIOR/'manifest.json',PRIOR/'console.log',PRIOR/'batch.log',
      WORK/'experiments/endpoint_a75_joint_n15/inputs.json',
      WORK/'experiments/endpoint_a75_joint_n15/protocol.md',PREFS/'comsol.prefs']
    boot=subprocess.run(['powershell','-NoProfile','-Command',
      '(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString("o")'],
      check=True,capture_output=True,text=True).stdout.strip()
    write_json(EXP/'inputs.json',dict(frozen_utc=utc(),machine=machine,last_boot_local=boot,
      interruption='SYSTEM_REBOOT_20261002_1018_HK_PRIOR_RUNNING_MANIFEST_STALE',
      protocol_sha256=sha(EXP/'protocol.md'),input_hashes={str(p):sha(p) for p in paths},
      source_hashes={n:sha(HERE/n) for n in FILES},settings=earlier['settings'],
      retained_points=list(range(11)),new_points=[11,12]))
    print('PASS_FREEZE_11_COMPLETED_POINTS_AND_SOURCE_HASHES',flush=True)

def run():
    frozen=json.loads((EXP/'inputs.json').read_text())
    def verify():
        assert sha(EXP/'protocol.md')==frozen['protocol_sha256']
        assert all(sha(Path(p))==h for p,h in frozen['input_hashes'].items())
        assert all(sha(HERE/n)==h for n,h in frozen['source_hashes'].items())
    verify();machine=preflight();retained();DEST.mkdir(exist_ok=False)
    for name in ('sources','checkpoints'):(DEST/name).mkdir()
    for name in FILES:shutil.copy2(HERE/name,DEST/'sources'/name)
    rec=dict(status='PREPARING',controller_pid=os.getpid(),started_utc=utc(),machine=machine,
      protocol_sha256=frozen['protocol_sha256'],inputs_sha256=sha(EXP/'inputs.json'),
      source_hashes=frozen['source_hashes'],input_hashes=frozen['input_hashes'],
      settings=dict(frozen['settings'],COMSOL_EIGEN_COUNT='15'),
      prior_run=str(PRIOR),prior_interruption=frozen['interruption'],
      retained_points=list(range(11)),new_points=[11,12],
      eigenpairs_by_point={str(p):20 if p<2 else 15 for p in range(13)},
      polarization_threshold=.7,cross_mesh_pairing='NOT_RUN',time_limit_seconds=None)
    write_json(DEST/'manifest.json',rec);start=time.perf_counter()
    try:
        scratch=Path(tempfile.mkdtemp(prefix='mcst_a75_reboot_'));assert scratch.is_dir()
        env={k:v for k,v in os.environ.items() if not k.startswith(('COMSOL_','MCST_JOINT_'))}
        env.update(MCST_JOINT_INPUT=str(PRIOR/'checkpoints/point_10.mph'),MCST_JOINT_OUTPUT=str(DEST))
        cmd=[str(BIN),'-np','4','-prefsdir',str(require_prefs(PREFS)),'-tmpdir',str(scratch),
          '-recoverydir',str(scratch),'-autosave','off','-inputfile',str(DEST/'sources/ResumeA75N15.class'),
          '-outputfile',str(DEST/'model.mph'),'-batchlog',str(DEST/'batch.log')]
        preflight();verify();rec.update(status='RUNNING',command=cmd)
        with (DEST/'console.log').open('w') as stream:
            proc=subprocess.Popen(cmd,cwd=PROJECT/'comsol',env=env,stdout=stream,stderr=subprocess.STDOUT)
            rec['comsol_pid']=proc.pid;write_json(DEST/'manifest.json',rec);rec['exit_code']=proc.wait()
        verify();assert rec['exit_code']==0 and (DEST/'model_JointA75N15Resume.mph').is_file()
        logged=(DEST/'console.log').read_text(errors='replace')
        assert 'RESUME_MESH mesh=1 layers=6 neigs=15 start_point=11' in logged
        assert 'JOINT_RECOVERY_COMPLETE' in logged
        assert [int(p) for p in re.findall(r'JOINT_POINT_COMPLETE point=(\d+)',logged)]==[11,12]
        matches=re.findall(r'POINT_SETTINGS point=(\d+) kx=([^ ]+) ky=([^ ]+) shift=([^ ]+) rtol=([^ ]+) neigs=([^\s]+)',logged)
        assert [int(m[0]) for m in matches]==[11,12]
        for p,x,y,shift,tol,count in matches:
            p=int(p);k=(12-p)/4
            assert float(count)==15 and math.isclose(float(x),k,abs_tol=1e-12) and math.isclose(float(y),k,abs_tol=1e-12)
            assert math.isclose(float(shift.removesuffix('[MHz]')),1 if p==12 else 1e-4,rel_tol=1e-12)
            assert math.isclose(float(tol),1e-9 if p==12 else 1e-6,rel_tol=1e-12)
        columns,prefix,previous=retained()
        other,recovery=read_csv(DEST/'frequencies_recovery.csv');assert other==columns
        check_rows(recovery,range(11,13),15)
        merged=DEST/'frequencies_derived_fullpath.csv'
        allrows=prefix+previous+recovery;assert len(allrows)==205
        with merged.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(allrows)
        assert read_csv(merged)[1]==allrows
        def origin(p):return OLD if p<2 else (PRIOR if p<11 else DEST)
        def rawpath(p):return origin(p)/('frequencies.csv' if p<2 else ('frequencies_remaining.csv' if p<11 else 'frequencies_recovery.csv'))
        models={p:origin(p)/f'checkpoints/point_{p}.mph' for p in range(13)}
        assert all(p.is_file() for p in models.values())
        audit=audit_spectrum(merged,.7);write_json(DEST/'spectrum_audit.json',audit)
        with (DEST/'point_sources.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['point','eigenpairs','raw_source','raw_sha256','checkpoint','checkpoint_sha256'])
            writer.writeheader()
            for p in range(13):
                raw=rawpath(p);checkpoint=models[p]
                writer.writerow(dict(point=p,eigenpairs=20 if p<2 else 15,raw_source=str(raw),
                  raw_sha256=sha(raw),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)))
        perpoint={str(p):float(next(r['solve_seconds'] for r in allrows if int(r['point_index'])==p)) for p in range(13)}
        rec.update(status='PASS_DERIVED_PATH_MIXED_20_15_REBOOT_RECOVERY',finished_utc=utc(),
          recovery_wall_seconds=time.perf_counter()-start,completed_solve_seconds_by_point=perpoint,
          total_completed_solve_seconds=sum(perpoint.values()),spectrum_audit=audit,
          recovery_raw_sha256=sha(DEST/'frequencies_recovery.csv'),derived_sha256=sha(merged),
          source_map_sha256=sha(DEST/'point_sources.csv'),spectral_count_study='NOT_RUN_FULL_PATH_15_VS20')
        write_json(DEST/'manifest.json',rec);write_json(DEST/'completion.json',rec)
        print(rec['status'],flush=True)
    except Exception as exc:
        rec.update(status='FAIL',reason=repr(exc),finished_utc=utc(),recovery_wall_seconds=time.perf_counter()-start)
        write_json(DEST/'manifest.json',rec);write_json(DEST/'failure.json',rec);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args();freeze() if args.freeze else run()
