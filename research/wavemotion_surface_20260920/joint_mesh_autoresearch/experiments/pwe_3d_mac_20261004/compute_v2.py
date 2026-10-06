"""Physical three-component PWE/3D overlap from immutable saved eigenvectors."""
from __future__ import annotations
import argparse
import csv
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
ROOT=WORK.parent
DATA_ROOT=Path(os.environ.get('MCST_DATA_ROOT',str(ROOT))).expanduser().resolve()
SOURCE_VARIANT='public-source-port-20261006'
OLD=ROOT.parent/'msse_20260919/python'
SNAP=ROOT/'runs/scale_addendum_20260927/sources'
sys.path[:0]=[str(SNAP),str(OLD),str(WORK/'src')]
from inverse_circle_bending import prepare,classical_inverse
from real_circle_bending import assemble_real
from msse_core import operators
from two_phase import reciprocal
from mode_overlap import read_fields,overlaps

CASES=[dict(name='baseline',a=500.,h=20.,N=21,modes=[1,2,3,4,5,6],
  field='section41_joint_M1L6_20260930/cross_mesh_fields_v1',
  pwe='large_scale_3d_20260921T145045670173Z',pattern='pwe_p4_{theory}_inverse_l1.npz',
  config='config.json',provenance='metadata.json'),
 dict(name='small',a=40.,h=3.2,N=15,modes=[1,2,3,5,8,9],
  field='endpoint_a40_joint_M1L6_20261001_v2/cross_mesh_fields_v1',
  pwe='fixed_ratio_scale_20260922T175850279921Z',pattern='pwe/N15_a40_p4_{theory}_l1.npz',
  config='config.json',provenance='provenance.json'),
 dict(name='thick',a=75.,h=24.,N=15,modes=[1,2,6,7,8,10],
  field='endpoint_a75_joint_M1L6_n15_reboot_20261002_v1/cross_mesh_fields_v1',
  pwe='thickness_ratio75_20260925/pwe',pattern='N15/N15_a75_h24_p4_{theory}_l1.npz',
  config='config.json',provenance='provenance.json')]
GRIDS=[(21,5),(31,7)]

def require(p):
 p=Path(p)
 if not p.is_file():
  raise FileNotFoundError(f'Required input missing: {p}. This source-only release omits saved spectra and 3D fields. Set MCST_DATA_ROOT to a data root containing the historical runs/ directories; freeze this public implementation before compute/verify.')
 return p
def sha(p):return hashlib.sha256(require(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,obj):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def write_csv(p,rows):
 with Path(p).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def paths(c,theory):
 return DATA_ROOT/'runs'/c['pwe']/c['pattern'].format(theory=theory)
def fieldpath(c,n,nz):return DATA_ROOT/'runs'/c['field']/f'M1L6_{n}x{n}x{nz}.csv'

def inputs():
 files=[HERE/'protocol.md',Path(__file__),SNAP/'inverse_circle_bending.py',SNAP/'real_circle_bending.py',
        OLD/'msse_core.py',OLD/'two_phase.py',WORK/'src/mode_overlap.py']
 for c in CASES:
  folder=DATA_ROOT/'runs'/c['field'];files.extend([folder/'completion.json',folder/'manifest.json',folder/'mac_31.json'])
  pwe=DATA_ROOT/'runs'/c['pwe'];files.extend([pwe/c['config'],pwe/c['provenance']])
  for theory in ['FSDT','TSDT']:files.append(paths(c,theory))
  for n,nz in GRIDS:
   p=fieldpath(c,n,nz);files.extend([p,Path(str(p)+'.modes.csv')])
 return {str(p):sha(p) for p in files}

def freeze():
 assert not (HERE/'inputs_frozen_v2.json').exists()
 hashes=inputs()
 dump(HERE/'inputs_frozen_v2.json',dict(status='FROZEN_BEFORE_MAC',created_utc=now(),cases=CASES,
      source_variant=SOURCE_VARIANT,grids=GRIDS,input_sha256=hashes,new_solve='NOT_RUN',
      scope='Public reproduction of existing fixed-X fields; no historical execution claim'))
 print('FROZEN',len(hashes),'files',flush=True)

def reconstruct(coords,c,V,ij,k,theory):
 """No amplitude fitting: all components in the same dimensionless length unit."""
 q=2*np.pi*ij+k
 unique_xy,inverse=np.unique(coords[:,:2]/(c['a']*1e-6),axis=0,return_inverse=True)
 phase=np.exp(1j*(unique_xy@q.T))
 coef=V.reshape(len(ij),3,-1).astype(complex)*np.array([1,1j,1j])[None,:,None]
 w=(phase@coef[:,0])[inverse]
 bx=(phase@coef[:,1])[inverse];by=(phase@coef[:,2])[inverse]
 wx=(phase@(1j*q[:,0,None]*coef[:,0]))[inverse]
 wy=(phase@(1j*q[:,1,None]*coef[:,0]))[inverse]
 z=coords[:,2]/(c['a']*1e-6);tau=c['h']/c['a']
 A=z if theory=='FSDT' else z-4*z**3/(3*tau*tau)
 B=np.zeros_like(z) if theory=='FSDT' else -4*z**3/(3*tau*tau)
 field=np.stack([A[:,None]*bx+B[:,None]*wx,A[:,None]*by+B[:,None]*wy,w],axis=1)
 return field.reshape(3*len(coords),-1)

def test_reconstruction():
 c=dict(a=75.,h=24.);ij=np.array([[0,0],[1,-1],[-1,2]])
 k=np.array([np.pi,.37]);V=np.random.default_rng(20261004).normal(size=(9,2))
 coords=np.array([[.1,.2,-.16],[-.2,.08,.16],[0,0,0]])*75e-6
 errs=[]
 for theory in ['FSDT','TSDT']:
  target=[]
  for xyz in coords/(75e-6):
   out=np.zeros((3,2),complex)
   for j,g in enumerate(ij):
    q=2*np.pi*g+k
    co=V[3*j:3*j+3]*np.array([1,1j,1j])[:,None]
    out+=np.exp(1j*(q@xyz[:2]))*operators(theory,q,xyz[2],24/75)[0][:,[2,3,4]]@co
   target.append(out)
  err=float(np.max(abs(reconstruct(coords,c,V,ij,k,theory)-np.array(target).reshape(9,2))))
  assert err<1e-12;errs.append(err)
 a=np.eye(9,dtype=complex)[:,:3];w=np.arange(1,10)
 assert np.allclose(overlaps(a,a*np.exp(1j*np.array([.1,.3,.9])),w)['MAC'],np.eye(3),atol=1e-14)
 assert np.allclose(overlaps(a,a[:,::-1],w)['MAC'],np.eye(3)[:,::-1],atol=1e-14)
 try:overlaps(a,a,np.zeros(9))
 except AssertionError:pass
 else:raise AssertionError('invalid weights accepted')
 return dict(closed_formula_vs_archived_operator_max_abs=errs,phase_permutation_invalid_weights='PASS')

def source_check():
 expected={'inverse_circle_bending.py':'679a57197224239f2bd9e06603cb414ae2873fada1b5c47554d8f39d222745e2',
           'real_circle_bending.py':'d49fd896b4bdbef91f2fec33a3cb633746b249c89f82669954c3634e70a484d4'}
 for name,h in expected.items():assert sha(SNAP/name)==h
 assert sha(OLD/'msse_core.py')=='338e41d56d089b13288018eecbd87b8dcaaecf0994682bd106d14d9e1c345356'
 for c in CASES:
  pwe=DATA_ROOT/'runs'/c['pwe'];cfg=json.loads((pwe/c['config']).read_text())
  assert cfg['r_over_a']==.3 and cfg['ell_host_um']==1 and cfg['ell_steel_um']==0
  assert cfg.get('N',cfg.get('N_path'))==c['N']
  if c['name']=='baseline':assert cfg['a_um']==c['a'] and cfg['h_um']==c['h']
  elif c['name']=='small':assert cfg['h_over_a']==c['h']/c['a'] and c['a'] in cfg['a_um']
  else:assert cfg['a_um']==c['a'] and c['h']/c['a'] in cfg['h_over_a']
  prov=json.loads((pwe/c['provenance']).read_text())['source_sha256']
  for name,h in expected.items():assert any(str(p).replace('\\','/').rsplit('/',1)[-1]==name and v==h for p,v in prov.items())
  folder=DATA_ROOT/'runs'/c['field'];done=json.loads((folder/'completion.json').read_text())
  assert done['status']=='PASS_EXPORT_AND_MAC_COMPUTATION'
  for n,nz in GRIDS:
   rec=next(r for r in done['exports'] if r['mesh']=='M1L6' and r['grid']==[n,n,nz])
   p=fieldpath(c,n,nz);assert sha(p)==rec['csv_sha256']
   assert sha(str(p)+'.modes.csv')==rec['metadata_sha256']
  meta=json.loads((folder/'mac_31.json').read_text());assert meta['right_modes']==c['modes']

def read_pwe(c,theory):
 with np.load(paths(c,theory)) as d:
  V=d['vectors'].copy();freq=d['frequency_MHz'][:8].copy();k=d['k'].copy()
  ij=d['ij'].copy() if 'ij' in d else reciprocal(c['N'])[0]
  assert V.shape==(3*(2*c['N']+1)**2,8) and len(freq)==8
  assert np.max(d['residual'][:8])<1e-5
  if 'a_um' in d:assert d['a_um']==c['a'] and d['h_um']==c['h'] and d['ell_um']==1
  saved_res=d['residual'][:8].copy()
 assert np.array_equal(ij,reciprocal(c['N'])[0])
 assert np.allclose(k,[np.pi,0],rtol=0,atol=1e-14) and np.isrealobj(V)
 assert np.all(np.isfinite(V)) and np.all(freq>0) and np.all(np.diff(freq)>0)
 tau=c['h']/c['a'];cache=prepare(c['N'],tau,.3,theory)
 s=assemble_real(c['N'],k,tau,.3,theory=theory);del s['Kcl']
 K=classical_inverse(cache,k)+(1/c['h'])**2*s['KB'];M=s['M']
 factor=np.sqrt(4.35e9/1180)/(c['a']*1e-6*2*np.pi)/1e6
 lam=(freq/factor)**2;kv=K@V;mv=M@V
 residual=np.linalg.norm(kv-mv*lam,axis=0)/(np.linalg.norm(kv,axis=0)+abs(lam)*np.linalg.norm(mv,axis=0))
 orth=float(np.linalg.norm(V.T@mv-np.eye(8)))
 rayleigh=np.sqrt(np.sum(V*kv,axis=0)/np.sum(V*mv,axis=0))*factor
 assert np.max(residual)<1e-7 and orth<1e-7
 assert np.max(abs(rayleigh/freq-1))<1e-8
 quality=dict(saved_residual=saved_res.tolist(),reassembled_residual=residual.tolist(),
              mass_orthogonality=orth,rayleigh_relative_difference=float(np.max(abs(rayleigh/freq-1))))
 del cache,s,K,M,kv,mv;gc.collect()
 return V[:,:6],freq[:6],ij,k,quality

def read_3d(c,n,nz):
 p=fieldpath(c,n,nz);coords,weights,V,k,f=read_fields(p,c['modes'])
 data=np.genfromtxt(p,delimiter=',',names=True);rho=data['rho_kg_m3']
 grid=np.array([[c['a']*1e-6*((ix+.5)/n-.5),c['a']*1e-6*((iy+.5)/n-.5),
                 c['h']*1e-6*((iz+.5)/nz-.5)] for iz in range(nz) for iy in range(n) for ix in range(n)])
 assert np.allclose(coords,grid,rtol=1e-12,atol=1e-18)
 density=np.where(np.sum(coords[:,:2]**2,axis=1)<(.3*c['a']*1e-6)**2,7780.,1180.)
 assert np.allclose(rho,density,rtol=1e-12,atol=0)
 volume=(c['a']*1e-6)**2*c['h']*1e-6/len(coords)
 assert np.allclose(weights,np.repeat(density*volume,3),rtol=1e-12,atol=0)
 assert np.allclose(k,[1,0],rtol=0,atol=1e-12)
 meta=np.atleast_1d(np.genfromtxt(str(p)+'.modes.csv',delimiter=',',names=True))
 assert all(abs(meta[meta['mode']==m]['imag_frequency_mhz'][0])/f[j]<1e-7 for j,m in enumerate(c['modes']))
 # Frequencies and solution numbers must agree with the already audited selection.
 prior=json.loads((p.parent/'mac_31.json').read_text())
 assert np.allclose(f,prior['right_frequencies_MHz'],rtol=0,atol=1e-12)
 return coords,weights,V,np.array(f)

def scalar_mac(a,b,w):
 out=np.empty((a.shape[1],b.shape[1]))
 for i in range(a.shape[1]):
  for j in range(b.shape[1]):
   norm_a=np.sum(w*abs(a[:,i])**2);norm_b=np.sum(w*abs(b[:,j])**2)
   out[i,j]=abs(np.sum(w*np.conj(a[:,i])*b[:,j]))**2/(norm_a*norm_b)
 return out

def compute():
 frozen=json.loads(require(HERE/'inputs_frozen_v2.json').read_text())
 assert frozen.get('source_variant')==SOURCE_VARIANT,'Re-freeze this public implementation; historical source hashes no longer apply'
 assert inputs()==frozen['input_sha256'],'Frozen input changed'
 assert not (HERE/'results_v2').exists(),'Never overwrite existing analysis'
 out=HERE/'results_v2';out.mkdir()
 start=time.perf_counter();tests=test_reconstruction();source_check()
 matrix_rows=[];edge_rows=[];comparisons=[]
 for c in CASES:
  for theory in ['FSDT','TSDT']:
   print('RECONSTRUCT',c['name'],theory,flush=True)
   V,f,ij,k,quality=read_pwe(c,theory);item=dict(case=c,theory=theory,pwe_quality=quality,grids=[])
   for n,nz in GRIDS:
    coords,w,fe,ff=read_3d(c,n,nz)
    pwe=reconstruct(coords,c,V,ij,k,theory)
    result=overlaps(pwe,fe,w);mac=np.array(result['MAC'])
    independent=scalar_mac(pwe,fe,w);err=float(np.max(abs(mac-independent)))
    assert err<1e-12
    pnorm=np.sum(w[:,None]*abs(pwe)**2,axis=0)
    fnorm=np.sum(w[:,None]*abs(fe)**2,axis=0)
    # Real standing-wave diagnostic; no conjugation/mirroring chosen to increase MAC.
    realness_p=abs(np.sum(w[:,None]*pwe**2,axis=0))/pnorm
    realness_fe=abs(np.sum(w[:,None]*fe**2,axis=0))/fnorm
    sub=dict(grid=[n,n,nz],independent_difference=err,
      pwe_frequencies_MHz=f.tolist(),fe_frequencies_MHz=ff.tolist(),
      pwe_sampled_mass_norm=pnorm.tolist(),fe_sampled_mass_norm=fnorm.tolist(),
      pwe_standing_wave_realness=realness_p.tolist(),fe_standing_wave_realness=realness_fe.tolist(),**result)
    item['grids'].append(sub)
    for i in range(6):
     for j in range(6):matrix_rows.append(dict(case=c['name'],a_um=c['a'],h_um=c['h'],N=c['N'],theory=theory,
        nxy=n,nz=nz,pwe_branch=i+1,fe_branch=j+1,fe_solution=c['modes'][j],MAC=float(mac[i,j])))
    for branch in [4,5]:
     i=branch-1;best=int(np.argmax(mac[i]))+1;reverse=int(np.argmax(mac[:,i]))+1
     row=dict(case=c['name'],a_um=c['a'],h_um=c['h'],N=c['N'],theory=theory,nxy=n,nz=nz,
       branch=branch,fe_solution=c['modes'][i],MAC=float(mac[i,i]),best_fe_branch=best,
       reverse_best_pwe_branch=reverse,largest_competing_MAC=float(np.max(np.delete(mac[i],i))),
       same_branch_screen=bool(best==reverse==branch and mac[i,i]>.95),
       pwe_frequency_MHz=float(f[i]),fe_frequency_MHz=float(ff[i]),
       signed_frequency_error_percent=float(100*(f[i]/ff[i]-1)))
     edge_rows.append(row)
   m0=np.array(item['grids'][0]['MAC']);m1=np.array(item['grids'][1]['MAC'])
   item['edge_sampling_changes']=[float(abs(m1[i,i]-m0[i,i])) for i in [3,4]]
   item['sampling_screen']=bool(max(item['edge_sampling_changes'])<=1e-3)
   comparisons.append(item)
   print('EDGES',c['name'],theory,'MAC',m1[3,3],m1[4,4],'sampling',item['edge_sampling_changes'],flush=True)
   del V;gc.collect()
 assert inputs()==frozen['input_sha256'],'Raw input changed during analysis'
 write_csv(out/'mac_matrices.csv',matrix_rows);write_csv(out/'edge_summary.csv',edge_rows)
 dump(out/'audit.json',dict(status='PASS_NUMERICAL_RECONSTRUCTION_AND_RAW_AUDIT',tests=tests,
     source_variant=SOURCE_VARIANT,frozen_sha256=sha(HERE/'inputs_frozen_v2.json'),command=sys.argv,
     implementation_sha256=sha(Path(__file__)),completed_utc=now(),seconds=time.perf_counter()-start,
     cases=comparisons,new_eigensolves='NOT_RUN',new_COMSOL='NOT_RUN',paper_edit='NOT_RUN',
     scope='Existing fixed-X fields, not full-path/BZ identity or continuum error'))
 print('PASS',out,flush=True)

def verify():
 frozen=json.loads(require(HERE/'inputs_frozen_v2.json').read_text());assert inputs()==frozen['input_sha256']
 audit=json.loads(require(HERE/'results_v2/audit.json').read_text())
 assert frozen.get('source_variant')==audit.get('source_variant')==SOURCE_VARIANT,'Verify new public freeze/compute outputs; historical source hashes do not authenticate this port'
 assert audit['implementation_sha256']==sha(Path(__file__))
 for item in audit['cases']:
  c=item['case'];V,f,ij,k,quality=read_pwe(c,item['theory'])
  for sub in item['grids']:
   n,_,nz=sub['grid'];coords,w,fe,ff=read_3d(c,n,nz)
   result=scalar_mac(reconstruct(coords,c,V,ij,k,item['theory']),fe,w)
   assert np.max(abs(result-np.array(sub['MAC'])))<1e-12
 print('PASS_READ_ONLY_RECOMPUTATION',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['freeze','compute','verify'],required=True)
 args=ap.parse_args()
 with threadpool_limits(limits=1):globals()[args.stage]()
