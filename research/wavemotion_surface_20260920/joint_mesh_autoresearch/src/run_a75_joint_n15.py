"""User-directed 15-pair continuation; original raw spectra remain untouched."""
import argparse,csv,json,math,os,re,shutil,subprocess,tempfile,time
from pathlib import Path
from run_joint_baseline import require_prefs
from run_joint_baseline import ROOT,HERE,WORK,PROJECT,BIN,preflight,sha,utc,write_json,audit_spectrum

EXP=WORK/'experiments/endpoint_a75_joint_n15'
OLD=ROOT/'runs/endpoint_a75_joint_M1L6_20261001_v1'
DEST=ROOT/'runs/endpoint_a75_joint_M1L6_n15_20261001_v1'
PREFS=ROOT/'runs/endpoint_a75_gamma_repeat_20261001_v1/prefs'
FILES=('ContinueA75N15.java','ContinueA75N15.class','run_a75_joint_n15.py','run_joint_baseline.py')

def read_csv(path):
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream);return reader.fieldnames,list(reader)

def check_rows(rows,points,count):
    assert sorted({int(r['point_index']) for r in rows})==list(points)
    assert len(rows)==len(points)*count
    for p in points:
        hits=[r for r in rows if int(r['point_index'])==p]
        assert sorted(int(r['mode_index']) for r in hits)==list(range(1,count+1))
        xy=(p/4,0) if p<=4 else ((1,(p-4)/4) if p<=8 else ((12-p)/4,(12-p)/4))
        for r in hits:
            assert all(math.isfinite(float(v)) for v in r.values())
            assert math.isclose(float(r['kx_pi_over_a']),xy[0],abs_tol=1e-12)
            assert math.isclose(float(r['ky_pi_over_a']),xy[1],abs_tol=1e-12)

def freeze():
    assert not (EXP/'inputs.json').exists() and not DEST.exists()
    preflight()
    _,prefix=read_csv(OLD/'frequencies.csv');check_rows(prefix,range(2),20)
    old=json.loads((OLD/'manifest.json').read_text());assert old['status']=='FAIL'
    paths=[OLD/'manifest.json',OLD/'failure.json',OLD/'frequencies.csv',OLD/'checkpoints/point_0.mph',OLD/'checkpoints/point_1.mph',
      OLD/'reference_M2L4_gamma_recomputed/frequencies_derived.csv',OLD/'reference_M2L4_gamma_recomputed/audit.json',
      WORK/'experiments/endpoint_a75_joint/inputs.json',PREFS/'comsol.prefs']
    _,reference=read_csv(paths[5]);coverage=[]
    for p in range(13):
        modes=[r for r in reference if int(r['point_index'])==p and int(r['mode_index'])<=15
          and float(r['frequency_mhz'])>1e-5 and float(r['vertical_fraction'])>.7]
        assert len(modes)>=(5 if p in (0,12) else 6),('reference coverage',p)
        coverage.append(dict(point=p,selected_in_first15=len(modes)))
    write_json(EXP/'inputs.json',dict(frozen_utc=utc(),protocol_sha256=sha(EXP/'protocol.md'),
      input_hashes={str(p):sha(p) for p in paths},source_hashes={n:sha(HERE/n) for n in FILES},
      settings=old['settings'],reference_first15_coverage=coverage,user_request='做到15就行',
      old_interruption='USER_REQUESTED_EIGENPAIR_COUNT_CHANGE_NOT_SOLVER_FAILURE'))
    print('PASS_FREEZE_PREFIX_AND_REFERENCE_COVERAGE',flush=True)

def run():
    frozen=json.loads((EXP/'inputs.json').read_text())
    def verify():
        assert sha(EXP/'protocol.md')==frozen['protocol_sha256']
        assert all(sha(Path(p))==h for p,h in frozen['input_hashes'].items())
        assert all(sha(HERE/n)==h for n,h in frozen['source_hashes'].items())
    verify();machine=preflight();DEST.mkdir(exist_ok=False)
    for name in ('sources','checkpoints'): (DEST/name).mkdir()
    for name in FILES:shutil.copy2(HERE/name,DEST/'sources'/name)
    rec=dict(status='PREPARING',controller_pid=os.getpid(),started_utc=utc(),machine=machine,
      protocol_sha256=frozen['protocol_sha256'],inputs_sha256=sha(EXP/'inputs.json'),source_hashes=frozen['source_hashes'],
      input_hashes=frozen['input_hashes'],settings=dict(frozen['settings'],COMSOL_EIGEN_COUNT='15'),
      raw_output=str(DEST/'frequencies_remaining.csv'),retained_points=[0,1],new_points=list(range(2,13)),
      eigenpairs_by_point={str(p):20 if p<2 else 15 for p in range(13)},polarization_threshold=.7,
      gamma_solver=dict(shift_MHz=1,rtol=1e-9),internal_solver=dict(shift_MHz=1e-4,rtol=1e-6),
      cross_mesh_pairing='NOT_RUN',time_limit_seconds=None,prior_run=str(OLD))
    write_json(DEST/'manifest.json',rec);start=time.perf_counter()
    try:
        scratch=Path(tempfile.mkdtemp(prefix='mcst_a75_n15_'));assert scratch.is_dir()
        env={k:v for k,v in os.environ.items() if not k.startswith(('COMSOL_','MCST_JOINT_'))}
        env.update(MCST_JOINT_INPUT=str(OLD/'checkpoints/point_1.mph'),MCST_JOINT_OUTPUT=str(DEST))
        cmd=[str(BIN),'-np','4','-prefsdir',str(require_prefs(PREFS)),'-tmpdir',str(scratch),'-recoverydir',str(scratch),'-autosave','off',
          '-inputfile',str(DEST/'sources/ContinueA75N15.class'),'-outputfile',str(DEST/'model.mph'),'-batchlog',str(DEST/'batch.log')]
        preflight();verify();rec.update(status='RUNNING',command=cmd)
        with (DEST/'console.log').open('w') as stream:
            proc=subprocess.Popen(cmd,cwd=PROJECT/'comsol',env=env,stdout=stream,stderr=subprocess.STDOUT)
            rec['comsol_pid']=proc.pid;write_json(DEST/'manifest.json',rec);rec['exit_code']=proc.wait()
        verify();assert rec['exit_code']==0 and (DEST/'model_JointA75N15.mph').is_file()
        logged=(DEST/'console.log').read_text(errors='replace')
        assert 'CONTINUE_MESH mesh=1 layers=6 neigs=15 start_point=2' in logged and 'JOINT_REMAINING_COMPLETE' in logged
        matches=re.findall(r'POINT_SETTINGS point=(\d+) kx=([^ ]+) ky=([^ ]+) shift=([^ ]+) rtol=([^ ]+) neigs=([^\s]+)',logged)
        assert [int(m[0]) for m in matches]==list(range(2,13))
        for p,x,y,shift,tol,count in matches:
            p=int(p);xy=(p/4,0) if p<=4 else ((1,(p-4)/4) if p<=8 else ((12-p)/4,(12-p)/4))
            assert float(count)==15 and math.isclose(float(x),xy[0],abs_tol=1e-12) and math.isclose(float(y),xy[1],abs_tol=1e-12)
            assert math.isclose(float(shift.removesuffix('[MHz]')),1 if p==12 else 1e-4,rel_tol=1e-12)
            assert math.isclose(float(tol),1e-9 if p==12 else 1e-6,rel_tol=1e-12)
        columns,prefix=read_csv(OLD/'frequencies.csv');check_rows(prefix,range(2),20)
        new_columns,remaining=read_csv(DEST/'frequencies_remaining.csv');assert columns==new_columns
        check_rows(remaining,range(2,13),15)
        merged=DEST/'frequencies_derived_fullpath.csv'
        with merged.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(prefix+remaining)
        assert read_csv(merged)[1]==prefix+remaining
        checkpoint_sources={str(p):(OLD if p<2 else DEST)/f'checkpoints/point_{p}.mph' for p in range(13)}
        assert all(p.is_file() for p in checkpoint_sources.values())
        audit=audit_spectrum(merged,.7);write_json(DEST/'spectrum_audit.json',audit)
        with (DEST/'point_sources.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['point','eigenpairs','raw_source','raw_sha256','checkpoint','checkpoint_sha256'])
            writer.writeheader()
            for p in range(13):
                raw=OLD/'frequencies.csv' if p<2 else DEST/'frequencies_remaining.csv';checkpoint=checkpoint_sources[str(p)]
                writer.writerow(dict(point=p,eigenpairs=20 if p<2 else 15,raw_source=str(raw),raw_sha256=sha(raw),
                  checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)))
        rec.update(status='PASS_DERIVED_PATH_MIXED_20_15',finished_utc=utc(),seconds=time.perf_counter()-start,
          spectrum_audit=audit,raw_sha256=sha(DEST/'frequencies_remaining.csv'),derived_sha256=sha(merged),
          source_map_sha256=sha(DEST/'point_sources.csv'),spectral_count_study='NOT_RUN_FULL_PATH_15_VS20')
        write_json(DEST/'manifest.json',rec);write_json(DEST/'completion.json',rec);print(rec['status'],flush=True)
    except Exception as exc:
        rec.update(status='FAIL',reason=repr(exc),finished_utc=utc(),seconds=time.perf_counter()-start)
        write_json(DEST/'manifest.json',rec);write_json(DEST/'failure.json',rec);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args();freeze() if args.freeze else run()
