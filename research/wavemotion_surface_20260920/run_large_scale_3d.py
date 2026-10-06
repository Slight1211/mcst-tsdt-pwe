"""Weak-size-effect dry validation: a=500um,h=20um, physical ell=1um."""
from run_same_order_bands import *
from inverse_circle_bending import prepare,classical_inverse,strain,regression
from run_classical_comsol_baseline import BIN,PROJECT
from comsol_runtime import comsol_bin, preferences_file, require_file
from concurrent.futures import ThreadPoolExecutor
import subprocess,shutil

def inverse_checks():
 checks=[dict(theory='TSDT',**regression())]
 for theory in ['FSDT','TSDT']:
  for h in [.04,.08]:
   p=prepare(2,h,.3,theory);k=np.array([.73,.41]);B=strain(p['G']+k,theory);n=len(B);d=B.shape[1]
   A=np.zeros((d*n,3*n))
   for j in range(n):A[d*j:d*j+d,3*j:3*j+3]=B[j]
   S=(np.einsum('ij,ab->iajb',p['F'],np.linalg.inv(p['DA']))+np.einsum('ij,ab->iajb',np.eye(n)-p['F'],np.linalg.inv(p['DB']))).reshape(d*n,d*n)
   D=(np.einsum('ij,ab->iajb',p['F'],p['DA'])+np.einsum('ij,ab->iajb',np.eye(n)-p['F'],p['DB'])).reshape(d*n,d*n)
   Ki=classical_inverse(p,k);ref=A.T@np.linalg.solve(S,A);direct=assemble_real(2,k,h,.3,theory=theory)['Kcl']
   ei=float(np.linalg.norm(Ki-ref)/np.linalg.norm(ref));ed=float(np.linalg.norm(A.T@D@A-direct)/np.linalg.norm(direct));assert max(ei,ed)<1e-10
   checks.append(dict(theory=theory,h=h,full_inverse_equivalence=ei,direct_equivalence=ed))
 return checks

def main():
 BIN=comsol_bin();profile=preferences_file(ROOT)
 require_file(PROJECT/'comsol/BuildTwoPhaseValidation.class', 'compiled COMSOL builder')
 require_file(ROOT/'runs/original_parameters_20260921T013511952049Z/mcst_manifest.json', 'historical COMSOL settings')
 out=ROOT/'runs'/('large_scale_3d_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir();print('OUTPUT',out,flush=True)
 cfg=dict(a_um=500,h_um=20,r_over_a=.3,ell_host_um=1,ell_steel_um=0,N=21,segments=4,mesh_path=3,mesh_check_X=2,scope='dry bending sampled path; generalized classical compliance inverse, direct MCST curvature',length_calibration='numerical uncalibrated ell',threads_each=2)
 (out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'checks.json').write_text(json.dumps(inverse_checks(),indent=2));print('INVERSE_CHECKS PASS',flush=True)
 vertices=np.array([[0.,0.],[np.pi,0.],[np.pi,np.pi],[0.,0.]])
 path=np.array([(1-t)*vertices[j]+t*vertices[j+1] for j in range(3) for t in np.arange(4)/4]);np.savetxt(out/'path.csv',path,delimiter=',',header='kx_a,ky_a',comments='')
 start=time.perf_counter();factor=np.sqrt(4.35e9/1180)/(500e-6*2*np.pi)/1e6
 def pwe():
  rows=[]
  def save():
   with (out/'pwe.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
  with threadpool_limits(limits=2):
   for theory in ['FSDT','TSDT']:
    cache=prepare(21,.04,.3,theory)
    for point,k in enumerate(path):
     tick=time.perf_counter();s=assemble_real(21,k,.04,.3,theory=theory)
     methods=['direct','inverse'] if point in [4,8] else ['inverse']
     for method in methods:
      kc=s['Kcl'] if method=='direct' else classical_inverse(cache,k)
      for ell in [0.,1.]:
       K=kc+(ell/20)**2*s['KB'];lam,V,res,raw=solve(K,s['M'],k)
       np.savez_compressed(out/f'pwe_p{point}_{theory}_{method}_l{ell:g}.npz',frequency_MHz=np.sqrt(lam)*factor,vectors=V,raw=raw,residual=res,ij=s['ij'],k=k)
       for j,value in enumerate(lam):rows.append(dict(point=point,N=21,theory=theory,method=method,ell_um=ell,band=j+1,frequency_MHz=float(np.sqrt(value)*factor),residual=float(res[j])))
       del K,V
      del kc
     save();print('PWE',theory,point,round(time.perf_counter()-tick,1),flush=True);del s;gc.collect()
    del cache;gc.collect()
  return len(rows)
 def fem():
  for tag in ['prefs','temporary']:(out/tag).mkdir()
  shutil.copy2(profile,out/'prefs/comsol.prefs')
  template=json.loads((ROOT/'runs/original_parameters_20260921T013511952049Z/mcst_manifest.json').read_text())['settings'];records=[]
  for mesh,sweep in [(3,4),(2,0)]:
   for ell,label in [(0,'classic'),(1,'mcst')]:
    tag=f'{label}_M{mesh}';folder=out/tag;folder.mkdir();settings=template.copy()
    settings.update(COMSOL_LATTICE_UM='500',COMSOL_THICKNESS_UM='20',COMSOL_CORE_RADIUS_UM='75',COMSOL_RING_RADIUS_UM='150',COMSOL_ELECTRODE_SPLIT_UM='112.5',COMSOL_MCST_LENGTH_UM=str(ell),COMSOL_MESH_LEVEL=str(mesh),COMSOL_EIGEN_COUNT='16',COMSOL_EIGEN_SHIFT_MHZ='0.0001',COMSOL_PATH_POINTS_PER_SEGMENT=str(sweep),COMSOL_RESULT_CSV=str(folder/'frequencies.csv'),COMSOL_TIMING_CSV=str(folder/'timing.csv'))
    env={k:v for k,v in os.environ.items() if not k.startswith(('COMSOL_','VALIDATION_'))};env.update(settings)
    cmd=[str(BIN/'comsolbatch.exe'),'-np','2','-prefsdir',str(out/'prefs'),'-tmpdir',str(out/'temporary'),'-recoverydir',str(out/'temporary'),'-autosave','off','-inputfile',str(PROJECT/'comsol/BuildTwoPhaseValidation.class'),'-outputfile',str(folder/'model.mph'),'-batchlog',str(folder/'batch.log')]
    meta=dict(status='RUNNING',settings=settings,command=cmd);mf=folder/'manifest.json';mf.write_text(json.dumps(meta,indent=2));tick=time.perf_counter();print('FEM_START',tag,flush=True)
    with (folder/'console.log').open('w') as f:r=subprocess.run(cmd,cwd=PROJECT/'comsol',env=env,stdout=f,stderr=subprocess.STDOUT)
    meta.update(status='EXPORTED_NOT_VALIDATED' if r.returncode==0 and (folder/'frequencies.csv').exists() else 'FAIL',seconds=time.perf_counter()-tick,exit_code=r.returncode);mf.write_text(json.dumps(meta,indent=2));records.append(meta);(out/'fem_records.json').write_text(json.dumps(records,indent=2));print('FEM_DONE',tag,meta['status'],meta['seconds'],flush=True)
    if meta['status']=='FAIL':raise RuntimeError('3D FEM failed; retained logs')
  return records
 with ThreadPoolExecutor(max_workers=2) as pool:
  a=pool.submit(pwe);b=pool.submit(fem);count=a.result();records=b.result()
 sources=[Path(__file__),ROOT/'src/inverse_circle_bending.py',ROOT/'src/real_circle_bending.py',OLD/'msse_core.py',PROJECT/'comsol/BuildTwoPhaseValidation.java',PROJECT/'comsol/BuildTwoPhaseValidation.class']
 (out/'metadata.json').write_text(json.dumps(dict(status='COMPUTED_NEEDS_POLARIZATION_AND_MESH_CHECK',seconds=time.perf_counter()-start,pwe_frequency_count=count,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},command=sys.argv,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,config_sha256=hashlib.sha256((out/'config.json').read_bytes()).hexdigest()),indent=2));print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
