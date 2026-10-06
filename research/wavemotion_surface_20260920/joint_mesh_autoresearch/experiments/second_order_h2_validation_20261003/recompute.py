"""Two serial real PWE eigensolves; existing frozen model is never refitted."""
from __future__ import annotations
import argparse
import csv
import datetime
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, eigsh
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
ROOT=WORK.parent
DATA_ROOT=Path(os.environ.get('MCST_DATA_ROOT',str(ROOT))).expanduser().resolve()
OLD=ROOT.parent/'msse_20260919/python'
SNAPSHOT=ROOT/'runs/scale_addendum_20260927/sources'
ORIGINAL=DATA_ROOT/'runs/manuscript_revision_20260925'
COEFF=DATA_ROOT/'joint_mesh_autoresearch/experiments/second_order_local_prediction_20261003/coefficients_frozen.json'
BASE=ORIGINAL/'ibz_N15_tau0.08_ix4_iy0.npz'
RAW=ROOT/'runs/second_order_h2_validation_20261003'
FROZEN=HERE/'predictions_frozen.json'
SOURCE_VARIANT='public-source-port-20261006'
sys.path[:0]=[str(SNAPSHOT),str(OLD)]
from inverse_circle_bending import prepare, classical_inverse
from real_circle_bending import assemble_real

LOCKED={
 COEFF:'e633d4ef514b756f2c6e18bd82ef8e2c2fa6052f9c7d96ee070eef679376800a',
 BASE:'8ea7397a750bbc0f203d944dccb5c777566ae1e823990f9b4f2d8b2d813b4fa9',
 ORIGINAL/'edge_prediction_frozen.json':'804841bfdda9ebd02cb21f9a4e6b1062f2f052d6dfaeb389b7be2d1685bc5f4e',
 ORIGINAL/'prediction_checks.csv':'aec5f6a0f7f591543d13e7a08ffdd448837a471b51b1c90192e4292676df4176',
 SNAPSHOT/'inverse_circle_bending.py':'679a57197224239f2bd9e06603cb414ae2873fada1b5c47554d8f39d222745e2',
 SNAPSHOT/'real_circle_bending.py':'d49fd896b4bdbef91f2fec33a3cb633746b249c89f82669954c3634e70a484d4',
 OLD/'two_phase.py':'41ac4d0b0e0549c4e49cdfc38d8bc960f58df2dfa966eae4e0dc6018ef66cf51',
 OLD/'msse_core.py':'338e41d56d089b13288018eecbd87b8dcaaecf0994682bd106d14d9e1c345356',
}
PUBLIC_SOURCES=[ROOT/'run_revision_edge_prediction.py',ROOT/'run_verified_inverse_scans.py',
                HERE/'protocol.md',Path(__file__),HERE/'audit.py',HERE/'audit_v2.py',HERE/'verify_saved.py']
KPOINT=np.array([np.pi,0.])
FACTOR=np.sqrt(4.35e9/1180)/(75e-6*2*np.pi)/1e6

def require(path):
 path=Path(path)
 if not path.is_file():
  raise FileNotFoundError(f'Required input missing: {path}. This source-only release omits numerical results. Set MCST_DATA_ROOT to a data root containing runs/ and archived joint_mesh_autoresearch/experiments/ inputs; generated public outputs require freeze/compute first.')
 return path
def sha(path): return hashlib.sha256(require(path).read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,data,exclusive=True):
 with Path(path).open('x' if exclusive else 'w',encoding='utf-8') as f:
  json.dump(data,f,indent=2,allow_nan=False)
def csvout(path,rows):
 with Path(path).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def checks():
 for path,expected in LOCKED.items(): assert sha(path)==expected,('hash mismatch',str(path))
 for name,folder in [('inverse_circle_bending',SNAPSHOT),('real_circle_bending',SNAPSHOT),('two_phase',OLD),('msse_core',OLD)]:
  assert Path(importlib.import_module(name).__file__).resolve()==(folder/(name+'.py')).resolve(),name
def gap(freq): return 200*(freq[1]-freq[0])/(freq[1]+freq[0])
def matrix(h):
 cache=prepare(15,h/75,.3,'TSDT')
 s=assemble_real(15,KPOINT,h/75,.3,theory='TSDT')
 del s['Kcl']
 K=classical_inverse(cache,KPOINT)+(1/h)**2*s['KB']
 return K,s['M']
def metrics(K,M,lam,V):
 kv=K@V;mv=M@V
 residual=np.linalg.norm(kv-mv*lam,axis=0)/(np.linalg.norm(kv,axis=0)+abs(lam)*np.linalg.norm(mv,axis=0)+1e-30)
 orth=float(np.linalg.norm(V.T@M@V-np.eye(V.shape[1])))
 return residual,orth
def macmatrix(V0,M0,V):
 return abs(V0.T@M0@V)**2/(np.einsum('ij,ij->j',V0,M0@V0)[:,None]*np.einsum('ij,ij->j',V,M0@V)[None,:])

def freeze():
 checks();assert not RAW.exists(),'Existing raw directory; inspect instead of overwrite'
 assert not FROZEN.exists(),'Predictions already frozen'
 model=json.loads(COEFF.read_text());rows=[]
 for d in [-.02,.02]:
  x=float(np.log1p(d));f0=np.array(model['baseline_f4_f5_MHz'])
  S=np.array([c['S'] for c in model['coefficients']['h']]);T=np.array([c['T'] for c in model['coefficients']['h']])
  f1=f0*np.exp(S*x);f2=f0*np.exp(S*x+.5*T*x*x)
  rows.append(dict(fractional_change=d,h_um=6*(1+d),first_f4_f5_MHz=f1.tolist(),second_f4_f5_MHz=f2.tolist(),first_J_percent=float(gap(f1)),second_J_percent=float(gap(f2))))
 files={str(p):sha(p) for p in [*LOCKED,*PUBLIC_SOURCES]}
 dump(FROZEN,dict(status='FROZEN_BEFORE_TWO_NEW_EIGENSOLVES',intent='PUBLIC_REPRODUCTION_OF_HISTORICAL_SAME_MODEL_EVALUATION_NOT_EXTERNAL_VALIDATION',source_variant=SOURCE_VARIANT,created_utc=now(),protocol_sha256=sha(HERE/'protocol.md'),baseline_f4_f5_MHz=model['baseline_f4_f5_MHz'],baseline_J_percent=model['baseline_X_J45_percent'],cases=rows,sha256=files))
 print('FROZEN',str(FROZEN),flush=True)

def solve(K,M):
 fact=cho_factor(K+.001*M,lower=True,check_finite=False)
 inv=LinearOperator(K.shape,matvec=lambda x:cho_solve(fact,x,check_finite=False),dtype=float)
 for ncv in [40,80]:
  spectra=[];vectors=[]
  for seed in [20260921,20260922]:
   ev,vec=eigsh(K,k=16,ncv=ncv,M=M,sigma=-.001,OPinv=inv,tol=1e-11,v0=np.random.default_rng(seed).normal(size=len(K)))
   ix=np.argsort(ev);spectra.append(ev[ix]);vectors.append(vec[:,ix])
  agreement=float(np.max(abs(spectra[0][:8]/spectra[1][:8]-1)))
  if agreement<1e-7:break
 residual,orth=metrics(K,M,spectra[0],vectors[0])
 return spectra[0],vectors[0],residual,orth,agreement,np.array(spectra),np.array(vectors),ncv

def compute():
 checks();frozen=json.loads(require(FROZEN).read_text())
 assert frozen.get('source_variant')==SOURCE_VARIANT,'Re-freeze using this public source; historical source hashes do not authenticate the port'
 for path,expected in frozen['sha256'].items():assert sha(path)==expected,path
 assert frozen['protocol_sha256']==sha(HERE/'protocol.md'),'Public protocol changed after freeze'
 RAW.mkdir(exist_ok=False)
 manifest=dict(status='RUNNING',source_variant=SOURCE_VARIANT,pid=os.getpid(),started_utc=now(),command=sys.argv,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,threads=1,points_serial=True,time_cutoff=None,expected_points=2,completed=[],frozen_sha256=sha(FROZEN),source_sha256=frozen['sha256'],scope=frozen['intent'])
 dump(RAW/'manifest.json',manifest);start=time.perf_counter()
 try:
  K0,M0=matrix(6.)
  with np.load(BASE) as base: V0=base['vectors'][:,3:5].copy()
  baseline_f=np.sqrt(np.einsum('ij,ij->j',V0,K0@V0)/np.einsum('ij,ij->j',V0,M0@V0))*FACTOR
  baseline_error=float(np.max(abs(baseline_f/np.array(frozen['baseline_f4_f5_MHz'])-1)))
  assert baseline_error<1e-7,baseline_error
  manifest['baseline_reconstructed_relative_error']=baseline_error
  del K0;gc.collect();rows=[];evaluation=[]
  for point,case in enumerate(frozen['cases']):
   tick=time.perf_counter();print('SOLVING',case['h_um'],'um',flush=True)
   K,M=matrix(case['h_um'])
   lam,V,res,orth,agree,seed_lam,seed_V,ncv=solve(K,M)
   frequencies=np.sqrt(lam)*FACTOR;mac=macmatrix(V0,M0,V[:,:8]);matched=np.argmax(mac,axis=1)+1
   quality=bool(np.all(np.isfinite(lam)) and np.min(lam)>0 and max(res[:8])<1e-7 and orth<1e-7 and agree<1e-7 and np.array_equal(matched,[4,5]) and min(mac[0,3],mac[1,4])>=.99)
   path=RAW/f'point{point}_h{case["h_um"]:.2f}.npz'
   np.savez_compressed(path,frequency_MHz=frequencies,eigenvalues=lam,vectors=V,residual=res,mass_orthogonality=orth,seed_agreement=agree,seed_eigenvalues=seed_lam,seed_vectors=seed_V,mass_MAC=mac,k=KPOINT,N=15,a_um=75,h_um=case['h_um'],radius_over_a=.3,ell_epoxy_um=1.,ell_steel_um=0.,ncv=ncv)
   for j in range(16):rows.append(dict(point=point,fractional_change=case['fractional_change'],h_um=case['h_um'],band=j+1,eigenvalue=float(lam[j]),frequency_MHz=float(frequencies[j]),residual=float(res[j]),source_npz=str(path),source_sha256=sha(path)))
   actual=float(gap(frequencies[3:5]));delta=actual-frozen['baseline_J_percent']
   item=dict(fractional_change=case['fractional_change'],h_um=case['h_um'],actual_f4_MHz=float(frequencies[3]),actual_f5_MHz=float(frequencies[4]),actual_J_percent=actual,actual_delta_J_pp=delta,quality='PASS' if quality else 'FAIL',min_mass_MAC=float(min(mac[0,3],mac[1,4])),max_residual=float(max(res[:8])),seed_agreement=agree,mass_orthogonality=orth,seconds=time.perf_counter()-tick,source_npz=str(path),source_sha256=sha(path))
   for order in ['first','second']:
    jp=case[order+'_J_percent'];err=abs(jp-actual)
    item.update({order+'_predicted_J_percent':jp,order+'_absolute_error_pp':err,order+'_increment_relative_error_percent':100*err/abs(delta),order+'_direction_agrees':bool((jp-frozen['baseline_J_percent'])*delta>0)})
   evaluation.append(item);manifest['completed'].append(item)
   dump(RAW/'manifest.json',manifest,exclusive=False)
   print('POINT_COMPLETE',json.dumps(item),flush=True)
   assert quality,'Raw evidence saved; numerical quality failure requires diagnosis'
   del K,M,V,seed_V;gc.collect()
  csvout(RAW/'spectra.csv',rows);csvout(HERE/'evaluation.csv',evaluation)
  useful=all(r['second_absolute_error_pp']<r['first_absolute_error_pp'] and r['second_direction_agrees'] for r in evaluation) and max(r['second_increment_relative_error_percent'] for r in evaluation)<=10
  manifest.update(status='PASS_NUMERICAL_QUALITY',usefulness='PASS' if useful else 'FAIL',finished_utc=now(),seconds=time.perf_counter()-start,raw_spectra_sha256=sha(RAW/'spectra.csv'),evaluation_sha256=sha(HERE/'evaluation.csv'),NOT_RUN=['new COMSOL','full path/BZ extrema','higher-N perturbation validation','mixed derivative','external validation'])
  checks();dump(RAW/'manifest.json',manifest,exclusive=False)
  print('COMPLETE',json.dumps(manifest),flush=True)
 except BaseException as exc:
  manifest.update(status='FAIL',error=repr(exc),failed_utc=now());dump(RAW/'manifest.json',manifest,exclusive=False);raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['freeze','compute'],required=True);args=p.parse_args()
 with threadpool_limits(limits=1): (freeze if args.stage=='freeze' else compute)()
