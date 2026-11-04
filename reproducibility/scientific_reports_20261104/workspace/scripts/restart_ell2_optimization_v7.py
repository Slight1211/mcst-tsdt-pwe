"""Continue the same circle-origin search after pre-recorded numerical stagnation."""
from pathlib import Path
import csv,hashlib,json,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_ell2_optimization_v7 as base
import numpy as np
from threadpoolctl import threadpool_limits
seed_path=ROOT/'05_validation/V07_ell2_reoptimization_20261030/restart_seed.json'
seed=json.loads(seed_path.read_text(encoding='utf-8'))
out=ROOT/'05_validation/V07_ell2_reoptimization_20261030/restart';out.mkdir(exist_ok=False)
orig_minimize=base.minimize
def restarted_minimize(*args,**kwargs):
    kwargs['options']['ftol']=1e-7
    return orig_minimize(*args,**kwargs)
base.minimize=restarted_minimize
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=dict(status='RUNNING',N=15,a_um=75,h_um=6,ell_epoxy_um=2,vertices=48,shape_variables=6,
    fixed_area=True,ftol=1e-7,feasibility_tolerance=1e-6,single_original_start='zero six-vector / equal-area regular48gon',
    solver_restart=True,restart_seed=seed,initial_search='run',predeclared_restart_tolerance=True,
    source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),Path(base.__file__),seed_path,
          ROOT/'05_validation/V07_ell2_reoptimization_20261030/source/node_shape_sensitivity_v7.py']},
    no_global_optimality_claim=True,no_continuous_BZ_claim=True,no_3d_FEM_claim=True)
base.save_json(out/'manifest.json',meta);t=time.perf_counter();history=[];rounds=[]
try:
 with threadpool_limits(2):
  y=np.array(seed['design']);ks=base.ibz(8);active={0,36,44}
  for rid in range(1,9):
   ids=sorted(active);ev=base.LoggedEvaluator(15,ks[ids],1/3,h_over_a=.08)
   local=base.optimize_round(ev,y,rid,out,history,t,100);y=np.array(local['design'])
   values=base.spectrum(y,ks,out,f'round{rid}_ibz45');full=base.sampled_gap(values)
   L,U=local['x'][-2:];violation=max(0,full['L']-L,U-full['U'])
   new=set(map(int,np.where((values[:,3]>L+1e-6)|(values[:,4]<U-1e-6))[0]))-active
   rounds.append(dict(optimizer=local,active_indices=ids,new_indices=sorted(new),full=full,full_constraint_violation=violation))
   meta.update(rounds=rounds,elapsed_seconds=time.perf_counter()-t);base.save_json(out/'manifest.json',meta)
   if local['success'] and local['constraint_violation']<1e-6 and violation<1e-6:break
   if not new:raise ArithmeticError('Restart did not converge; do not classify as a solved optimum.')
   active.update(new)
  else:raise ArithmeticError('Active-set rounds exceeded')
  initial=ROOT/'05_validation/V08_N15_20261030_v3/initial_ell2_N15_ibz45.csv'
  original=list(csv.DictReader(initial.open()))
  baseline=np.array([[float(r[f'Omega{i}']) for i in range(1,7)] for r in original])
  dense=base.spectrum(y,base.ibz(16),out,'final_ibz153')
  gate=base.gradient_gate(y,'restarted_final')
  np.savetxt(out/'final_vertices.csv',base.boundary(y),delimiter=',',header='x_over_a,y_over_a',comments='')
  meta.update(status='PASS_SAMPLED_N15_LOCAL_SQP',design=y.tolist(),initial_ibz45=base.sampled_gap(baseline),
      reused_initial_source=initial.relative_to(ROOT).as_posix(),reused_initial_sha256=sha(initial),
      final_ibz45=full,final_ibz153=base.sampled_gap(dense),gradient_gates=[gate],
      accepted_iterations=sum(r['optimizer']['iterations'] for r in rounds),
      area_fraction=float(base.signed_area(base.boundary(y))),elapsed_seconds=time.perf_counter()-t,
      gain_from_circle_pp=full['J']-base.sampled_gap(baseline)['J'],N21_check='PENDING',FEM_check='NOT_RUN')
  meta['output_sha256']={p.name:sha(p) for p in out.glob('*') if p.is_file() and p.name!='manifest.json'}
  base.save_json(out/'manifest.json',meta);print('RESTART_COMPLETE',json.dumps(meta),flush=True)
except BaseException as e:
 meta.update(status='FAIL',error=repr(e),traceback=traceback.format_exc());base.save_json(out/'manifest.json',meta);raise
