"""T03 bounded pilot; preserves failed and unresolved convergence records."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,time,datetime,hashlib,sys,platform
import numpy as np
from two_phase import reciprocal,disk,rectangle,polygon,assemble,solve
ROOT=Path(__file__).resolve().parents[1]

def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'runs'/('t03_'+stamp);out.mkdir(parents=True)
    tests=[];cases=[];conv=[]
    def save():
        for n,x in [('tests',tests),('cases',cases),('convergence',conv)]: (out/(n+'.json')).write_text(json.dumps(x,indent=2))
    def check(n,v,t):tests.append(dict(name=n,value=float(v),threshold=t,passed=bool(np.isfinite(v) and v<=t)))
    cfg=dict(N=[3,5,7,9],k={'X':[np.pi,0],'M':[np.pi,np.pi]},h=.1,theory='TSDT',ell=[.25,.25],materials=[[10,.3,2],[1,.3,1]],fraction=.3)
    meta=dict(project_id='MSSE-PCP',task_id='T03',run_id=out.name,case_id='two_phase_circle_pilot',created_at=stamp,execution_status='running',validation_level='internal_consistency',model_version='full5_direct_complex_v1',config=cfg,config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),runtime_name='Python primary',runtime_version=sys.version,platform=platform.platform(),code_commit=None,working_tree_dirty=None,random_seed=20260919,toolboxes=[],unit_system='a_EB_rhoB',time_convention='exp(-i*omega*t)',fourier_convention='exp(i*(k+G).r)',dof_order=['u0','v0','w','theta_x','theta_y'],basis_normalization='no area normalization',geometry_source='analytic circle',material_parameter_status='hypothetical',source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));print('OUTPUT',out,flush=True)
    _,g=reciprocal(4);r=np.sqrt(.3/np.pi);center=np.array([.07,-.04])
    check('disk_area',abs(disk(np.zeros((1,2)),r)[0]-.3),1e-14)
    check('complex_fourier_conjugacy',np.max(abs(disk(-g,r,center)-disk(g,r,center).conj())),1e-12)
    check('disk_translation_phase',np.max(abs(disk(g,r,center)-disk(g,r)*np.exp(-1j*g@center))),1e-12)
    box=np.array([[-.2,-.15],[.2,-.15],[.2,.15],[-.2,.15]])
    check('rectangle_polygon',np.max(abs(rectangle(g,.4,.3)-polygon(g,box))),1e-12)
    errors=[]
    for n in [32,64,128,256]:
        theta=2*np.pi*np.arange(n)/n;p=r*np.column_stack([np.cos(theta),np.sin(theta)])
        error=np.max(abs(polygon(g,p)-disk(g,r)));errors.append(float(error))
        check(f'polygon_conjugacy_{n}',np.max(abs(polygon(-g,p)-polygon(g,p).conj())),1e-12)
    check('polygon_refinement_decreases',max(np.diff(errors)),0);check('polygon_256_absolute_error',errors[-1],1e-4)
    for name,p in [('clockwise',box[::-1]),('repeated',np.vstack([box,box[-1]])),('crossing',box[[0,2,1,3]]),('outside',box*4)]:
        rejected=False
        try:polygon(g,p)
        except ValueError:rejected=True
        check('reject_'+name,0 if rejected else 1,0)
    # Translation is a complex unitary congruence, not just a frequency test.
    s=assemble(2,[.71,1.21]);st=assemble(2,[.71,1.21],indicator=lambda g:disk(g,r,center))
    phase=np.repeat(np.exp(-1j*s['G']@center),5)
    for key in ['Kcl','KA','KB','M']:
        check('translation_'+key,np.linalg.norm(st[key]-phase[:,None]*s[key]*phase.conj()[None,:])/np.linalg.norm(s[key]),1e-10)
    check('translation_spectrum',np.max(abs(solve(s)['lam']-solve(st)['lam']))/max(solve(s)['lam']),1e-10)
    # Exact uniform Fourier limit of general assembly vs existing one-wave blocks.
    from msse_core import blocks
    su=assemble(1,[.71,1.21],materials=[(10,.3,2),(10,.3,2)])
    from scipy.linalg import block_diag
    small=[blocks('TSDT',q+[.71,1.21],.1,'analytic') for q in su['G']]
    check('uniform_Kcl',np.linalg.norm(su['Kcl']-block_diag(*[s[0] for s in small]))/np.linalg.norm(su['Kcl']),1e-10)
    check('uniform_M',np.linalg.norm(su['M']-block_diag(*[s[2] for s in small]))/np.linalg.norm(su['M']),1e-10)
    check('uniform_Kmc',np.linalg.norm(su['KA']+su['KB']-block_diag(*[s[1] for s in small]))/np.linalg.norm(su['KA']+su['KB']),1e-10)
    save()
    if not all(t['passed'] for t in tests):
        meta['execution_status']='failed';(out/'metadata.json').write_text(json.dumps(meta,indent=2));raise AssertionError([t for t in tests if not t['passed']])
    for point,k in cfg['k'].items():
      previous=None
      for N in cfg['N']:
        start=time.perf_counter();s=assemble(N,k);d=solve(s);elapsed=time.perf_counter()-start
        check(f'{point}_{N}_hermitian',d['hermitian'],1e-10);check(f'{point}_{N}_residual',d['residual'],1e-9);check(f'{point}_{N}_energy_closure',d['closure'],1e-9)
        check(f'{point}_{N}_participation',max(0,-min(d['RA']),-min(d['RB']),max(d['RA']+d['RB'])-1),1e-8)
        check(f'{point}_{N}_difference_range',abs(np.max(s['ij'][:,None,:]-s['ij'][None,:,:])-2*N),0)
        omega=np.sqrt(d['lam']);row=dict(point=point,N=N,ndof=len(s['M']),seconds=elapsed,omega=omega.tolist(),RA=d['RA'].tolist(),RB=d['RB'].tolist())
        cases.append(row)
        np.savez_compressed(out/f'{point}_N{N}.npz',**s,lambda_raw=d['lam'],modes=d['v'],RA=d['RA'],RB=d['RB'])
        if previous:
            freq=float(np.max(abs(omega[:12]/np.array(previous['omega'][:12])-1)))
            # Trace over first12 is rotation invariant, but not proof of each
            # mode/cluster convergence. Keep this distinction explicit.
            energy=float(max(abs(sum(d['RA'][:12])-sum(previous['RA'][:12])),abs(sum(d['RB'][:12])-sum(previous['RB'][:12]))))
            conv.append(dict(point=point,N_from=previous['N'],N_to=N,max_frequency_change_first12=freq,phase_participation_trace_change_first12=energy,frequency_gate=freq<=.005,per_mode_participation_gate='not_assessed_requires_cluster_matching'))
        previous=row;save();print(point,N,'seconds',round(elapsed,2),'omega1',omega[0],flush=True)
    meta['execution_status']='completed' if all(t['passed'] for t in tests) else 'failed'
    meta['convergence_status']='pilot_only_not_all_acceptance_gates_met';meta['polygon_coefficient_errors']=errors
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));save();print('TESTS',len(tests),'FAILED',[t for t in tests if not t['passed']],flush=True)
if __name__=='__main__':main()
