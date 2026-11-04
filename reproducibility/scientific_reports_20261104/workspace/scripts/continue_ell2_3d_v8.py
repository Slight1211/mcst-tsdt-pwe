"""Continue the same active local run after the baseline queue completes."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'05_validation/V01_ell2_20261031_v8'
p=OUT/'queue_base.json';deadline=time.monotonic()+7200
while True:
    state=json.loads(p.read_text(encoding='utf-8'))
    if state['status']!='RUNNING':break
    assert time.monotonic()<deadline,'Baseline did not finish within the continuation deadline'
    time.sleep(5)
assert state['status']=='COMPLETED',state
result=subprocess.run([sys.executable,str(ROOT/'scripts/run_ell2_3d_v8.py'),'--phase','checks'],cwd=ROOT)
assert result.returncode==0,'Refinement checks failed; retained raw logs require inspection'
