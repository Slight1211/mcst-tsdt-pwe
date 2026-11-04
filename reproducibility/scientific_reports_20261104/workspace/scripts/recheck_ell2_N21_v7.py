"""Higher-order evaluation of the newly reoptimized ell2 shape, without a new search."""
from pathlib import Path
import csv,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'04_code/research/wavemotion_surface_20260920'
sys.path[:0]=[str(CODE/'src'),str(CODE.parent/'msse_20260919/python')]
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse.linalg import eigsh,LinearOperator
from threadpoolctl import threadpool_limits
from node_shape_bending import fourier,boundary,signed_area,AREA
from inverse_circle_bending import prepare,classical_inverse
from real_circle_bending import assemble_real
opt=ROOT/'05_validation/V07_ell2_reoptimization_20261030/restart'
out=ROOT/'05_validation/V07_ell2_reoptimization_20261030/N21'
manifest=json.loads((opt/'manifest.json').read_text())
assert manifest['status']=='PASS_SAMPLED_N15_LOCAL_SQP'
out.mkdir(parents=True,exist_ok=False);y=np.array(manifest['design'])
initial=ROOT/'05_validation/V08_N21_ell2_20261030/initial_ell2_N21_ibz45.csv'
data=list(csv.DictReader(initial.open()));ks=np.array([[float(r['kx_a']),float(r['ky_a'])] for r in data])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=dict(status='RUNNING',N=21,ell_um=2,reoptimized=False,evaluation_of_new_ell2_optimum=True,
    reused_initial_spectrum=initial.relative_to(ROOT).as_posix(),reused_initial_sha256=sha(initial),
    source_optimizer_sha256=sha(opt/'manifest.json'),new_final_eigensolves=0,threads=2,
    source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),CODE/'src/inverse_circle_bending.py',CODE/'src/real_circle_bending.py']})
t=time.perf_counter();rows=[];columns=['point','kx_a','ky_a']+[f'Omega{i}' for i in range(1,7)]+['nonrigid_residual','mass_orthogonality']
with threadpool_limits(2):
 F=fourier(y,21);cache=prepare(21,.08,theory='TSDT',indicator_matrix=F)
 with (out/'final_ell2_N21_ibz45.csv').open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
  for i,k in enumerate(ks):
   p=assemble_real(21,k,h=.08,theory='TSDT',indicator_matrix=F);M=p['M'];K=classical_inverse(cache,k)+(1/3)**2*p['KB'];del p
   cf=cho_factor(K+.001*M,lower=True,check_finite=False);inv=LinearOperator(K.shape,matvec=lambda x:cho_solve(cf,x,check_finite=False),dtype=float)
   v,q=eigsh(K,k=12,M=M,sigma=-.001,OPinv=inv,ncv=60,tol=1e-10,v0=np.random.default_rng(20261030).normal(size=len(K)))
   ids=np.argsort(v);v=v[ids];q=q[:,ids];omega=np.sqrt(np.maximum(v,0))
   res=np.linalg.norm(K@q-M@q*v,axis=0)/(np.linalg.norm(K@q,axis=0)+abs(v)*np.linalg.norm(M@q,axis=0)+1e-30)
   nr=float(res[1:].max());orth=float(abs(q.T@M@q-np.eye(12)).max());assert nr<1e-7 and orth<1e-7
   r=dict(point=i,kx_a=k[0],ky_a=k[1],**{f'Omega{j+1}':omega[j] for j in range(6)},nonrigid_residual=nr,mass_orthogonality=orth)
   w.writerow(r);f.flush();rows.append(r);meta.update(new_final_eigensolves=i+1,elapsed_seconds=time.perf_counter()-t)
   (out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
   print('N21_REOPTIMIZED',i+1,'/45',flush=True)
   del K,M,cf,inv,q
def gap(rr):
 lo=max(rr,key=lambda r:float(r['Omega4']));up=min(rr,key=lambda r:float(r['Omega5']));L=float(lo['Omega4']);U=float(up['Omega5'])
 return dict(L=L,U=U,J=200*(U-L)/(U+L),kx_L=float(lo['kx_a']),ky_L=float(lo['ky_a']),kx_U=float(up['kx_a']),ky_U=float(up['ky_a']))
g=gap(rows);old=gap(data);meta.update(status='COMPLETED_N21_FROZEN_GEOMETRY',initial=old,final=g,gain_pp=g['J']-old['J'],area_error=abs(signed_area(boundary(y))-AREA),
 output_sha256={'final_ell2_N21_ibz45.csv':sha(out/'final_ell2_N21_ibz45.csv')})
(out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps(meta,indent=2))
