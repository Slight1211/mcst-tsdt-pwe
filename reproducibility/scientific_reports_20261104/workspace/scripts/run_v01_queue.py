"""Serial continuation of the requested local COMSOL validation cases."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_validation/V01_20261030'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
first=OUT/'initial_M2L4_n24_GXM/manifest.json'
while read(first)['status']=='RUNNING':time.sleep(10)
assert read(first)['status']=='COMPLETED_RAW_EXPORT','Initial refined case failed; inspect before continuing'
jobs=[
 ('final_M2L4_n24_GXM',['--shape','final','--mesh','2','--layers','4','--points','GXM']),
 ('final_M2L4_n32_X',['--shape','final','--mesh','2','--layers','4','--count','32','--points','X']),
 ('initial_M2L4_n32_X',['--shape','initial','--mesh','2','--layers','4','--count','32','--points','X']),
 ('final_M2L6_n24_X',['--shape','final','--mesh','2','--layers','6','--points','X'])]
state={'status':'RUNNING','jobs':[]}
for name,args in jobs:
    manifest=OUT/name/'manifest.json'
    if manifest.exists():
        assert read(manifest)['status']=='COMPLETED_RAW_EXPORT',name
        state['jobs'].append(dict(case=name,status='ALREADY_COMPLETED'));continue
    state['active']=name
    (OUT/'queue.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    print('START_CASE',name,flush=True)
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_v01.py')]+args,cwd=ROOT)
    state['jobs'].append(dict(case=name,exit_code=result.returncode,status=read(manifest)['status'] if manifest.exists() else 'NO_MANIFEST'))
    if result.returncode!=0:
        state['status']='STOPPED_ON_FAILURE';break
else:state['status']='COMPLETED'
(OUT/'queue.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
print(json.dumps(state,indent=2),flush=True)
