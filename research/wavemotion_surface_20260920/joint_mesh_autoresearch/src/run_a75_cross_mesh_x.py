"""Preregistered single X solve followed by serial read-only physical field pairing."""
import argparse,json,math,os,shutil,subprocess,tempfile,time
from pathlib import Path
from run_joint_baseline import require_prefs
from run_joint_baseline import HERE,WORK,PROJECT,BIN,preflight,sha,utc,write_json
from report_a75_joint import RUN,OLD,PRIOR,rows

EXP=WORK/'experiments/a75_cross_mesh'
DEST=RUN/'cross_mesh_X_v1'
SOURCE=OLD/'model_Model.mph'
PREFS=OLD.parent.parent/'endpoint_a75_gamma_repeat_20261001_v1/prefs'
FILES=('SolveA75X.java','SolveA75X.class','run_a75_cross_mesh_x.py','run_a75_cross_mesh_fields.py',
 'run_joint_baseline.py','report_a75_joint.py','ExportJointFields.java','ExportJointFields.class','mode_overlap.py')

def selected(path,count):
    group=[r for r in rows(path) if int(r['point_index'])==4]
    assert sorted(int(r['mode_index']) for r in group)==list(range(1,count+1))
    assert all(all(math.isfinite(float(v)) for v in r.values()) for r in group)
    assert all(float(r['kx_pi_over_a'])==1 and float(r['ky_pi_over_a'])==0 for r in group)
    chosen=sorted([r for r in group if float(r['frequency_mhz'])>1e-5 and float(r['vertical_fraction'])>.7],key=lambda r:float(r['frequency_mhz']))[:6]
    assert len(chosen)==6 and all(abs(float(r['imag_frequency_mhz']))/float(r['frequency_mhz'])<1e-7 for r in chosen)
    return chosen

def freeze():
    assert not (EXP/'inputs.json').exists() and not DEST.exists();preflight()
    audit=json.loads((RUN/'report_v1/audit.json').read_text());assert audit['status']=='PASS_RAW_PATH_AND_PWE_RECOMPUTATION'
    assert all(a['L_point']==a['U_point']==4 for a in audit['audits'].values())
    selected(OLD/'frequencies.csv',20);selected(PRIOR/'frequencies_remaining.csv',15)
    paths=[SOURCE,OLD/'manifest.json',OLD/'frequencies.csv',RUN/'report_v1/audit.json',
      RUN/'completion.json',PRIOR/'checkpoints/point_4.mph',PRIOR/'frequencies_remaining.csv',PREFS/'comsol.prefs']
    write_json(EXP/'inputs.json',dict(frozen_utc=utc(),protocol_sha256=sha(EXP/'protocol.md'),
      input_hashes={str(p):sha(p) for p in paths},source_hashes={n:sha(HERE/n) for n in FILES}))
    print('PASS_FROZEN_A75_X_AND_FIELD_PROTOCOL',flush=True)

def verify(frozen):
    assert sha(EXP/'protocol.md')==frozen['protocol_sha256']
    assert all(sha(Path(p))==h for p,h in frozen['input_hashes'].items())
    assert all(sha(HERE/n)==h for n,h in frozen['source_hashes'].items())

def run():
    frozen=json.loads((EXP/'inputs.json').read_text());verify(frozen);machine=preflight()
    old=selected(OLD/'frequencies.csv',20);DEST.mkdir(exist_ok=False)
    for name in FILES:shutil.copy2(HERE/name,DEST/name)
    scratch=Path(tempfile.mkdtemp(prefix='mcst_a75_X_'));assert scratch.is_dir()
    env={k:v for k,v in os.environ.items() if not k.startswith(('COMSOL_','MCST_X_'))}
    env.update(MCST_X_MODEL=str(SOURCE),MCST_X_CSV=str(DEST/'frequencies.csv'))
    cmd=[str(BIN),'-np','4','-prefsdir',str(require_prefs(PREFS)),'-tmpdir',str(scratch),'-recoverydir',str(scratch),
      '-autosave','off','-inputfile',str(DEST/'SolveA75X.class'),'-outputfile',str(DEST/'model.mph'),'-batchlog',str(DEST/'batch.log')]
    rec=dict(status='RUNNING',started_utc=utc(),controller_pid=os.getpid(),machine=machine,
      inputs_sha256=sha(EXP/'inputs.json'),protocol_sha256=frozen['protocol_sha256'],source_hashes=frozen['source_hashes'],
      command=cmd,cross_mesh_MAC='NOT_RUN',time_limit_seconds=None)
    write_json(DEST/'manifest.json',rec);start=time.perf_counter()
    try:
        preflight();verify(frozen)
        with (DEST/'console.log').open('w') as stream:
            proc=subprocess.Popen(cmd,cwd=PROJECT/'comsol',env=env,stdout=stream,stderr=subprocess.STDOUT)
            rec['comsol_pid']=proc.pid;write_json(DEST/'manifest.json',rec);rec['exit_code']=proc.wait()
        verify(frozen);assert rec['exit_code']==0 and (DEST/'model_A75X.mph').is_file()
        log=(DEST/'console.log').read_text(errors='replace')
        assert 'A75X_SETTINGS a75 h24 M2L4' in log and 'A75X_SOLVE_COMPLETE' in log
        new=selected(DEST/'frequencies.csv',15)
        comparisons=[dict(branch=i+1,old_MHz=float(a['frequency_mhz']),new_MHz=float(b['frequency_mhz']),
          difference_Hz=1e6*abs(float(a['frequency_mhz'])-float(b['frequency_mhz'])),
          passed=math.isclose(float(a['frequency_mhz']),float(b['frequency_mhz']),rel_tol=1e-8,abs_tol=1e-8)) for i,(a,b) in enumerate(zip(old,new))]
        rec['comparisons']=comparisons;assert all(c['passed'] for c in comparisons),'X repeatability'
        rec.update(status='PASS_X_SPECTRUM_REPEAT',finished_utc=utc(),seconds=time.perf_counter()-start,
          raw_sha256=sha(DEST/'frequencies.csv'),model_sha256=sha(DEST/'model_A75X.mph'))
        write_json(DEST/'manifest.json',rec);write_json(DEST/'completion.json',rec)
        print('PASS_A75_X',rec['seconds'],flush=True)
    except Exception as exc:
        rec.update(status='FAIL_EXECUTION_OR_AUDIT',reason=repr(exc),finished_utc=utc(),seconds=time.perf_counter()-start)
        write_json(DEST/'manifest.json',rec);write_json(DEST/'failure.json',rec);raise
    from run_a75_cross_mesh_fields import main
    main()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');args=ap.parse_args()
    freeze() if args.freeze else run()
