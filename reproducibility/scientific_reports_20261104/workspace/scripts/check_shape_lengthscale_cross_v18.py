"""Two frozen-geometry length-scale cross-evaluations at N15; no optimization.

The eigensolver and diagnostic gates follow check_ell0_order_cross.py. The
constitutive construction follows check_lengthscale_modal_energy.py. Both
historical programs and every physical implementation remain untouched.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,platform,subprocess,sys,time,traceback,shutil

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'04_code/research/wavemotion_surface_20260920'
CORE=CODE.parent/'msse_20260919/python'
sys.path[:0]=[str(CODE/'src'),str(CORE)]
import numpy as np
import scipy
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse.linalg import eigsh,LinearOperator
from threadpoolctl import threadpool_limits,threadpool_info
from node_shape_bending import AREA,boundary,fourier,signed_area
from inverse_circle_bending import prepare,classical_inverse
from real_circle_bending import assemble_real
from check_ell0_order_cross import read_grid_spectrum,gap

JOBS={
 'g0_at5':dict(shape='g0',own_ell_um=0.,evaluation_ell_um=5.,manifest='05_validation/ELL0_MULTISTART_20261101/lhs_05/manifest.json'),
 'g5_at0':dict(shape='g5',own_ell_um=5.,evaluation_ell_um=0.,manifest='05_validation/GAP_CLOSURE_SCAN_20261031/ell_5/warm/manifest.json')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def save(p,x):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');t.replace(p)
def sources():
 paths=[Path(__file__),ROOT/'scripts/check_ell0_order_cross.py',ROOT/'scripts/check_lengthscale_modal_energy.py',
        CODE/'src/node_shape_bending.py',CODE/'src/inverse_circle_bending.py',CODE/'src/real_circle_bending.py',CORE/'msse_core.py',CORE/'two_phase.py']
 return {rel(p):sha(p) for p in paths}

def load_case(job):
 p=ROOT/job['manifest'];m=json.loads(p.read_text(encoding='utf8'))
 assert m['status'].startswith('PASS')
 assert (m['N'],m['a_um'],m['h_um'],m['ell_epoxy_um'],m.get('ell_steel_um',0.))==(15,75,6,job['own_ell_um'],0.)
 y=np.asarray(m['design'],float);assert y.shape==(6,) and np.isfinite(y).all() and np.max(abs(y))<=.16+1e-10
 vertices=boundary(y);assert vertices.shape==(48,2) and abs(signed_area(vertices)-AREA)<1e-13
 vp=p.parent/'final_vertices.csv';raw=list(csv.DictReader(vp.open(encoding='utf-8-sig',newline='')))
 saved=np.array([[float(r['x_over_a']),float(r['y_over_a'])] for r in raw]);assert np.max(abs(saved-vertices))<1e-12
 sp=p.parent/'final_ibz153.csv';k,f=read_grid_spectrum(sp);g=gap(f,k)
 assert all(abs(g[v]-m['final_ibz153'][v])<1e-9 for v in ['L','U','J'])
 for q in [vp,sp]:
  assert sha(q)==m['output_sha256'][q.name]
 inputs={rel(q):sha(q) for q in [p,vp,sp]}
 return m,y,k,f,g,inputs

def solve(cache,F,k,ell):
 tick=time.perf_counter();parts=assemble_real(15,k,h=.08,theory='TSDT',indicator_matrix=F)
 M,KB=parts['M'],parts['KB'];del parts
 K=classical_inverse(cache,k)+(ell/6.)**2*KB;del KB
 factor=cho_factor(K+.001*M,lower=True,check_finite=False)
 inverse=LinearOperator(K.shape,matvec=lambda v:cho_solve(factor,v,check_finite=False),dtype=float)
 vals,vecs=eigsh(K,k=12,M=M,sigma=-.001,OPinv=inverse,tol=1e-10,ncv=60,v0=np.random.default_rng(20261030).normal(size=len(K)))
 order=np.argsort(vals);vals,vecs=vals[order],vecs[:,order]
 gamma=np.max(abs(k))<1e-14;first=1 if gamma else 0
 assert np.all(vals[first:]>0)
 if gamma:assert abs(vals[0])<1e-7
 Kv,Mv=K@vecs,M@vecs
 res=np.linalg.norm(Kv-Mv*vals,axis=0)/(np.linalg.norm(Kv,axis=0)+abs(vals)*np.linalg.norm(Mv,axis=0)+1e-30)
 maxres=float(np.max(res[first:6]));orth=float(np.max(abs(vecs.T@Mv-np.eye(12))))
 assert maxres<1e-7 and orth<1e-7,(k,ell,maxres,orth)
 return np.sqrt(np.maximum(vals[:6],0)),dict(max_first6_nonrigid_relative_residual=maxres,mass_orthogonality_max=orth,Gamma_rigid_lambda=float(vals[0]) if gamma else None,solve_seconds=time.perf_counter()-tick)

def worker(label,out):
 job=JOBS[label];m,y,k,diagonal,own_gap,inputs=load_case(job);before=sources()
 out=out.resolve();assert out.is_relative_to(ROOT);out.mkdir(parents=True,exist_ok=False)
 meta=dict(status='RUNNING_DIAGONAL_SPOTCHECKS',job=label,pid=os.getpid(),N=15,a_um=75.,h_um=6.,
  ell_steel_um=0.,own_ell_um=job['own_ell_um'],evaluation_ell_um=job['evaluation_ell_um'],
  shape_label=job['shape'],design=y.tolist(),shape_manifest=job['manifest'],source_sha256=before,input_sha256=inputs,
  own_diagonal=own_gap,grid=16,points=153,threads=2,eigen_count=12,shift=-.001,tolerance=1e-10,ncv=60,random_seed=20261030,
  construction='K=classical_inverse(prepare(N15,h/a=.08,indicatorF),k)+(ell_um/6)^2*assemble_real(...)[KB]; KB epoxy only; M=assemble_real(...)[M]',
  scope='Frozen geometry, selected endpoint lengths, same153 grid; no optimization, COMSOL, global optimality, continuous-zone or circularization-mechanism certification.',
  new_optimization=False,new_COMSOL_runs=False,human_verified=False,environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
 save(out/'manifest.json',meta);tick=time.perf_counter()
 try:
  with threadpool_limits(2):
   meta['threadpool']=threadpool_info();F=fourier(y,15);cache=prepare(15,.08,theory='TSDT',indicator_matrix=F)
   spots=[]
   for index in [0,27,136]:
    f,check=solve(cache,F,k[index],job['own_ell_um']);first=1 if index==0 else 0
    error=float(np.max(abs(f[first:]-diagonal[index,first:])/diagonal[index,first:]));assert error<1e-7
    spots.append(dict(point=index,k_a=k[index].tolist(),Omega_fresh=f.tolist(),Omega_saved=diagonal[index].tolist(),max_nonrigid_relative_frequency_error=error,**check))
    print('SPOT_PASS',label,index,error,flush=True)
   save(out/'diagonal_spotchecks.json',spots)
   meta.update(status='RUNNING_FROZEN_CROSS',diagonal_spotchecks='PASS');save(out/'manifest.json',meta)
   vals=[];checks=[];spectrum_name=label+'_ibz153.csv';check_name=label+'_checks.csv'
   with (out/spectrum_name).open('x',encoding='utf8',newline='') as f,(out/check_name).open('x',encoding='utf8',newline='') as a:
    writer=csv.writer(f);writer.writerow(['point','kx_a','ky_a']+[f'Omega{i}' for i in range(1,7)])
    auditor=csv.DictWriter(a,fieldnames=['point','kx_a','ky_a','max_first6_nonrigid_relative_residual','mass_orthogonality_max','Gamma_rigid_lambda','solve_seconds']);auditor.writeheader()
    for index,wave in enumerate(k):
     freq,check=solve(cache,F,wave,job['evaluation_ell_um']);check=dict(point=index,kx_a=float(wave[0]),ky_a=float(wave[1]),**check)
     vals.append(freq);checks.append(check);writer.writerow([index,*wave,*freq]);auditor.writerow(check);f.flush();a.flush()
     meta['progress']=dict(completed_points=index+1,total_points=153);save(out/'manifest.json',meta)
     if (index+1)%10==0 or index==152:print('CROSS_PROGRESS',label,index+1,'/153',check['max_first6_nonrigid_relative_residual'],flush=True)
   saved_k,saved_f=read_grid_spectrum(out/spectrum_name);assert np.max(abs(saved_k-k))<1e-14 and np.max(abs(saved_f-np.asarray(vals)))<1e-14
   result=gap(saved_f,k)
   assert before==sources() and inputs=={p:sha(ROOT/p) for p in inputs}
   meta.update(status='PASS_FROZEN_SHAPE_LENGTHSCALE_CROSS_N15',result=result,
    max_first6_nonrigid_relative_residual=max(x['max_first6_nonrigid_relative_residual'] for x in checks),max_mass_orthogonality_error=max(x['mass_orthogonality_max'] for x in checks),
    source_hash_check='PASS',input_hash_check='PASS',source_sha256_post=sources(),input_sha256_post={p:sha(ROOT/p) for p in inputs},
    elapsed_seconds=time.perf_counter()-tick,output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
   save(out/'manifest.json',meta);print('CROSS_PASS',label,json.dumps(result),flush=True)
 except BaseException:
  meta.update(status='FAILED_FROZEN_CROSS',error=traceback.format_exc(),elapsed_seconds=time.perf_counter()-tick);save(out/'manifest.json',meta);raise

def controller(out):
 out=out.resolve();assert out.is_relative_to(ROOT);out.mkdir(parents=True,exist_ok=False)
 meta=dict(status='RUNNING_TWO_FROZEN_CROSSES',jobs=[],new_optimization=False,new_COMSOL_runs=False,human_verified=False,source_sha256=sources())
 handles=[];processes=[]
 for label in JOBS:
  log=out/(label+'.console.log');h=log.open('x',encoding='utf8');handles.append(h)
  command=[sys.executable,'-u','-X','utf8',str(Path(__file__).resolve()),'--job',label,'--out',str(out/label)]
  process=subprocess.Popen(command,stdout=h,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
  processes.append((label,process));meta['jobs'].append(dict(label=label,pid=process.pid,command=command,folder=rel(out/label)))
 save(out/'manifest.json',meta)
 try:
  for label,p in processes:assert p.wait()==0,label
  cells=[]
  for label,_ in processes:
   m=json.loads((out/label/'manifest.json').read_text());assert m['status']=='PASS_FROZEN_SHAPE_LENGTHSCALE_CROSS_N15'
   shape=m['shape_label'];cells.extend([dict(shape=shape,evaluation_ell_um=m['own_ell_um'],provenance='SAVED_OWN_ELL_DIAGONAL',**m['own_diagonal']),dict(shape=shape,evaluation_ell_um=m['evaluation_ell_um'],provenance='FRESH_FROZEN_GEOMETRY_CROSS',**m['result'])])
   sp=ROOT/JOBS[label]['manifest'];shutil.copyfile(sp.parent/'final_ibz153.csv',out/(shape+'_saved_diagonal_ibz153.csv'))
  with (out/'cross_2x2_summary.csv').open('x',encoding='utf8',newline='') as h:
   w=csv.DictWriter(h,fieldnames=['shape','evaluation_ell_um','provenance','L','U','J'],extrasaction='ignore');w.writeheader();w.writerows(cells)
  meta.update(status='PASS_FROZEN_LENGTHSCALE_2X2_N15',cells=cells,fresh_cross_points=306,diagonal_fresh_spotchecks=6,
   no_circularization_mechanism_claim=True,source_hash_check='PASS' if meta['source_sha256']==sources() else 'FAIL',
   output_sha256={rel(p):sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json' and not p.name.endswith('.console.log')})
  assert meta['source_hash_check']=='PASS';save(out/'manifest.json',meta);print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
 except BaseException:
  meta.update(status='FAILED_TWO_CROSSES',error=traceback.format_exc());save(out/'manifest.json',meta);raise
 finally:
  for h in handles:h.close()

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',choices=list(JOBS));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.job:worker(a.job,a.out)
 else:controller(a.out)
