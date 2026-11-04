"""Compute saved-shape modal energies and one-sided length-scale responses."""
from pathlib import Path
import csv,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'04_code/research/wavemotion_surface_20260920'
NODE=ROOT/'02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1'
sys.path[:0]=[str(CODE/'src'),str(CODE.parent/'msse_20260919/python')]
import numpy as np
from scipy.linalg import cho_factor,cho_solve,eigh
from scipy.sparse.linalg import eigsh,LinearOperator
from threadpoolctl import threadpool_limits
from node_shape_bending import fourier
from inverse_circle_bending import prepare,classical_inverse
from real_circle_bending import assemble_real
out=ROOT/'05_validation/V08_edge_switch_20261030';out.mkdir(parents=True,exist_ok=False)
y=np.asarray(json.loads((NODE/'manifest.json').read_text())['design'])
records=[];t=time.perf_counter()
with threadpool_limits(2):
 for N,ells in [(15,[.5,1.,1.5,2.]),(21,[1.,1.5])]:
  F=fourier(y,N);cache=prepare(N,.08,theory='TSDT',indicator_matrix=F)
  for name,k in [('Gamma',np.array([0.,0.])),('X',np.array([np.pi,0.]))]:
   parts=assemble_real(N,k,h=.08,theory='TSDT',indicator_matrix=F);K0=classical_inverse(cache,k);M=parts['M'];KB=parts['KB'];del parts
   for ell in ells:
    Kmc=(ell/6)**2*KB;K=K0+Kmc
    cf=cho_factor(K+.001*M,lower=True,check_finite=False)
    inv=LinearOperator(K.shape,matvec=lambda x:cho_solve(cf,x,check_finite=False),dtype=float)
    val,q=eigsh(K,k=12,M=M,sigma=-.001,OPinv=inv,ncv=60,tol=1e-10,v0=np.random.default_rng(20261030).normal(size=len(K)))
    ids=np.argsort(val);val=val[ids];q=q[:,ids]
    dense_drift=None
    if N==15:
     dense=eigh(K,M,subset_by_index=(0,7),driver='gvx',eigvals_only=True,check_finite=False)
     dense_drift=float(np.max(abs(val[1:6]/dense[1:6]-1)));assert dense_drift<1e-7
    for band in [4,5]:
     v=q[:,band-1];lam=val[band-1];frequency=np.sqrt(lam)
     energy=float(v@K@v);R=float(v@Kmc@v/energy)
     residual=float(np.linalg.norm(K@v-lam*(M@v))/(np.linalg.norm(K@v)+lam*np.linalg.norm(M@v)))
     nearest=float(np.min(abs(lam-np.delete(val,band-1)))/lam)
     assert 0<=R<=1 and residual<1e-7
     records.append(dict(shape='ell1_terminal_frozen',N=N,ell_um=ell,point=name,band=band,Omega=frequency,R_mc=R,S_ell=R,relative_eigenvalue_spacing=nearest,relative_residual=residual,dense_first6_relative_drift=dense_drift))
    print('ENERGY',N,name,ell,[(x['band'],x['R_mc']) for x in records[-2:]],flush=True)
    del K,Kmc,cf,inv,q
   del K0,M,KB
  del F,cache
with (out/'modal_energy_sensitivity.csv').open('x',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
sources=[Path(__file__),CODE/'src/real_circle_bending.py',CODE/'src/inverse_circle_bending.py',NODE/'manifest.json']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',shape='same ell1 terminal 48-gon',new_eigensolves=True,
 records=len(records),elapsed_seconds=time.perf_counter()-t,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
 output_sha256={'modal_energy_sensitivity.csv':sha(out/'modal_energy_sensitivity.csv')},scope='local fixed-wavevector energy identities; no assumption that frequency decreases with length scale'),indent=2),encoding='utf-8')
