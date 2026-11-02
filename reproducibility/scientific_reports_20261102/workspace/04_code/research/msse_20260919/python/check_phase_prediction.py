"""Selected simple-mode T04 pilot; freeze prediction before perturbation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,datetime,hashlib,time
import numpy as np
from scipy.linalg import eigh
from two_phase import assemble,solve
ROOT=Path(__file__).resolve().parents[1]
def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'runs'/('t04_X4_'+stamp);out.mkdir();print('OUTPUT',out,flush=True)
    s=assemble(13,[np.pi,0]);d=solve(s);j=3;v=d['v'][:,j];lam=d['lam'][j];w=np.sqrt(lam)
    separation=min(abs(d['lam'][j+1]-lam),abs(lam-d['lam'][j-1]))/lam
    assert separation>1e-3,'Mode not safely isolated'
    steps=[.01,.001,.0001,.00001];cfg=dict(N=13,k=[float(np.pi),0],h=.1,ell_h=[.25,.25],mode_sorted=4,steps=steps)
    meta=dict(project_id='MSSE-PCP',task_id='T04_pilot',execution_status='running',validation_level='internal_consistency',created_at=stamp,config=cfg,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));pred=dict(created_before_perturbed_solve=True,omega=w,R_A=float(d['RA'][j]),R_B=float(d['RB'][j]),relative_lambda_separation=separation,predictions=[dict(phase=phase,log_step=sign*step,predicted_delta_log_omega=sign*step*float(d['RA'][j] if phase=='A' else d['RB'][j])) for phase in ['A','B'] for step in steps for sign in [-1,1]])
    freeze=out/'prediction_baseline.json';freeze.write_text(json.dumps(pred,indent=2));digest=hashlib.sha256(freeze.read_bytes()).hexdigest();rows=[]
    K=s['Kcl']+.25**2*(s['KA']+s['KB']);Mv=s['M']@v
    for phase in ['A','B']:
      for eps in steps:
        change=[];macs=[]
        for sign in [-1,1]:
            shifted=K+.25**2*np.expm1(2*sign*eps)*s['K'+phase]
            vals,vec=eigh(shifted,s['M'],subset_by_index=[0,7],driver='gvx')
            mac=abs(Mv.conj()@vec)**2;idx=int(np.argmax(mac));assert mac[idx]>.99
            change.append(float(.5*np.log(vals[idx]/lam)));macs.append(float(mac[idx]))
        fd=(change[1]-change[0])/(2*eps);exact=pred['R_'+phase]
        rows.append(dict(phase=phase,log_step=eps,derivative_fd=fd,predicted=exact,absolute_error=abs(fd-exact),min_MAC=min(macs),delta_log_omega=change,predicted_delta_log_omega=[-eps*exact,eps*exact],passed=abs(fd-exact)<=.002))
        (out/'sensitivity.json').write_text(json.dumps(rows,indent=2));print(phase,eps,fd,exact,abs(fd-exact),flush=True)
    assert hashlib.sha256(freeze.read_bytes()).hexdigest()==digest
    meta.update(execution_status='completed',all_tests_passed=all(x['passed'] for x in rows),prediction_sha256=digest,scope='Single isolated X mode4 only; not bandgap/geometry/holdout generalization')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
