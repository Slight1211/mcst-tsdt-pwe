"""Serial storage recovery; do not alter equations, mesh or requested eigenpairs."""
from pathlib import Path
import json,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_ell2_3d_v8 import OUT,run,dump
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=[OUT/f'{s}_M2L6_n24_GX/manifest.json' for s in ['initial','final']]
deadline=time.monotonic()+7200
while any(load(p)['status'] in {'RUNNING','COMPILING'} for p in old):
    assert time.monotonic()<deadline,'Existing thickness process exceeded recovery wait'
    time.sleep(5)
scratch_root=Path('E:/MCSTv8_scratch_20261031')
assert scratch_root.is_absolute() and str(scratch_root).isascii()
amendment=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    reason='One thickness X solve failed after a low-memory out-of-core switch and an out-of-core file I/O error.',
    original_case_status={p.parent.name:load(p)['status'] for p in old},
    changes=['Serial execution of incomplete thickness cases','ASCII temporary/recovery directory on E drive with ample free disk space','Child process TEMP/TMP only; no global preference changes'],
    unchanged=['constitutive energy','exact vertices','mesh and thickness layers','quadrature','eigenpair count','frequency shift','tolerance','diagnostic thresholds'],
    archived_initial_launcher_sha256=sha(OUT/'source/run_ell2_3d_v8_initial_launcher.py'),
    documentation='https://www.comsol.com/support/knowledgebase/1083',
    failed_attempts_retained=True,human_verified=False)
dump(OUT/'storage_recovery_amendment.json',amendment)
state=dict(status='RUNNING',jobs=[])
dump(OUT/'queue_recovery.json',state)
for p in old:
    m=load(p)
    if m['status']=='COMPLETED_RAW_EXPORT':continue
    assert m['status']=='FAILED_OR_INCOMPLETE'
    shape=m['shape'];tag=f'{shape}_M2L6_n24_GX_retry1'
    result=run((shape,2,6,24,'GX',dict(tag=tag,scratch=str(scratch_root/f'{shape}_retry1'))))
    state['jobs'].append(result);dump(OUT/'queue_recovery.json',state)
state['status']='COMPLETED';dump(OUT/'queue_recovery.json',state)
print(json.dumps(state,indent=2),flush=True)
