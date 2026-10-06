"""Fresh geometric-similarity scale scan, h/a=.08; physical host ell=1 um."""
from run_large_scale_3d import *
from run_verified_inverse_scans import verified_solve

SCALES=[10,15,25,40,75,100,200,500,1000,2000]

def main():
 out=(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'runs'/('fixed_ratio_scale_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
 out.mkdir(exist_ok=True);print('OUTPUT',out,flush=True)
 cfg=dict(a_um=SCALES,h_over_a=.08,r_over_a=.3,ell_host_um=1,ell_steel_um=0,N_controls=21,N_path=15,path_segments=4,controls=['X','M'],branches=6,mesh_inplane=2,sweep_layers=4,comsol_threads=4,pwe_threads=2,scope='dry bending; sorted first6 at X/M, not full-BZ error; fixed mesh reference')
 if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==cfg,'Config changed on resume'
 else:(out/'config.json').write_text(json.dumps(cfg,indent=2))
 for name in ['prefs','temporary','pwe','sources']:(out/name).mkdir(exist_ok=True)
 shutil.copy2(ROOT/'configs/comsol_batch_authorized.prefs',out/'prefs/comsol.prefs')
 sources=[Path(__file__),ROOT/'run_verified_inverse_scans.py',ROOT/'src/inverse_circle_bending.py',ROOT/'src/real_circle_bending.py',OLD/'msse_core.py',PROJECT/'comsol/BuildTwoPhaseValidation.java',PROJECT/'comsol/BuildTwoPhaseValidation.class']
 for p in sources:
  target=out/'sources'/p.name
  if target.exists():assert target.read_bytes()==p.read_bytes(),f'Source changed on resume: {p}'
  else:shutil.copy2(p,target)
 meta=dict(command=sys.argv,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},config_sha256=hashlib.sha256((out/'config.json').read_bytes()).hexdigest())
 if not (out/'provenance.json').exists():(out/'provenance.json').write_text(json.dumps(meta,indent=2))
 vertices=np.array([[0.,0.],[np.pi,0.],[np.pi,np.pi],[0.,0.]])
 path=np.array([(1-t)*vertices[j]+t*vertices[j+1] for j in range(3) for t in np.arange(4)/4])
 np.savetxt(out/'path.csv',path,delimiter=',',header='kx_a,ky_a',comments='')
 tick=time.perf_counter()
 def pwe():
  rows=[]
  with threadpool_limits(limits=2):
   for N,points in [(21,[4,8]),(15,range(12))]:
    for theory in ['FSDT','TSDT']:
     cache=None
     for point in points:
      k=path[point];s=None;kc=None
      for a in SCALES:
       h=.08*a
       for ell in [0.,1.]:
        fn=out/'pwe'/f'N{N}_a{a}_p{point}_{theory}_l{ell:g}.npz'
        if fn.exists():
         with np.load(fn) as z:
          freq=z['frequency_MHz'];res=z['residual'];agree=float(z['seed_agreement']);orth=float(z['mass_orthogonality'])
          assert float(z['a_um'])==a and abs(float(z['h_um'])-h)<1e-12
          start=1 if point==0 else 0
          assert len(freq)==8 and np.all(np.isfinite(freq)) and min(freq[start:])>0 and max(res[start:8])<1e-5 and agree<1e-7 and orth<1e-7
        else:
         if cache is None:cache=prepare(N,.08,.3,theory)
         if s is None:s=assemble_real(N,k,.08,.3,theory=theory);kc=classical_inverse(cache,k)
         K=kc+(ell/h)**2*s['KB'];lam,V,res,raw,agree,orth=verified_solve(K,s['M'],k,N)
         freq=np.sqrt(lam[:8])*np.sqrt(4.35e9/1180)/(a*1e-6*2*np.pi)/1e6
         np.savez_compressed(fn,frequency_MHz=freq,vectors=V[:,:8],raw=raw,residual=res,seed_agreement=agree,mass_orthogonality=orth,a_um=a,h_um=h,ell_um=ell,k=k)
         del K,V
        for j,f in enumerate(freq):rows.append(dict(N=N,a_um=a,h_um=h,theory=theory,ell_um=ell,point=point,band=j+1,frequency_MHz=float(f),residual=float(res[j]),seed_agreement=agree,mass_orthogonality=orth))
      with (out/'pwe.csv').open('w',newline='') as f:
       w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
      print('PWE',N,theory,point,'all10 scales',flush=True);del s,kc;gc.collect()
     del cache;gc.collect()
  assert len(rows)==4480
  return len(rows)
 def fem():
  template=json.loads((ROOT/'runs/wider_validation_20260921T232234189390Z/mcst_M2path/manifest.json').read_text())['settings'];records=[]
  for a in SCALES:
   h=.08*a;folder=out/f'mcst_a{a}';folder.mkdir(exist_ok=True);mf=folder/'manifest.json'
   if mf.exists() and json.loads(mf.read_text()).get('exit_code')==0 and (folder/'model_Model.mph').exists() and (folder/'frequencies.csv').stat().st_size>1000:
    records.append(json.loads(mf.read_text()));print('FEM_REUSED',a,flush=True);continue
   settings=template.copy();settings.update(COMSOL_LATTICE_UM=str(a),COMSOL_THICKNESS_UM=str(h),COMSOL_CORE_RADIUS_UM=str(.15*a),COMSOL_RING_RADIUS_UM=str(.3*a),COMSOL_ELECTRODE_SPLIT_UM=str(.225*a),COMSOL_MCST_LENGTH_UM='1',COMSOL_MESH_LEVEL='2',COMSOL_DRY_SWEEP_LAYERS='4',COMSOL_EIGEN_COUNT='20',COMSOL_EIGEN_SHIFT_MHZ=str(.0001*25/a),COMSOL_PATH_POINTS_PER_SEGMENT='0',COMSOL_IBZ_SUBDIVISIONS='1',COMSOL_CONTROL_XM_ONLY='1',COMSOL_RESULT_CSV=str(folder/'frequencies.csv'),COMSOL_TIMING_CSV=str(folder/'timing.csv'))
   env={k:v for k,v in os.environ.items() if not k.startswith(('COMSOL_','VALIDATION_'))};env.update(settings)
   cmd=[str(BIN/'comsolbatch.exe'),'-np','4','-prefsdir',str(out/'prefs'),'-tmpdir',str(out/'temporary'),'-recoverydir',str(out/'temporary'),'-autosave','off','-inputfile',str(PROJECT/'comsol/BuildTwoPhaseValidation.class'),'-outputfile',str(folder/'model.mph'),'-batchlog',str(folder/'batch.log')]
   record=dict(status='RUNNING',settings=settings,command=cmd);mf.write_text(json.dumps(record,indent=2));print('FEM_START',a,'h',h,flush=True);t=time.perf_counter()
   with (folder/'console.log').open('w') as f:r=subprocess.run(cmd,cwd=PROJECT/'comsol',env=env,stdout=f,stderr=subprocess.STDOUT)
   valid=r.returncode==0 and (folder/'model_Model.mph').exists() and (folder/'frequencies.csv').exists() and (folder/'frequencies.csv').stat().st_size>1000
   record.update(status='EXPORTED_NOT_VALIDATED' if valid else 'FAIL',exit_code=r.returncode,seconds=time.perf_counter()-t);mf.write_text(json.dumps(record,indent=2));records.append(record);print('FEM_DONE',a,record['status'],flush=True)
  return records
 with ThreadPoolExecutor(max_workers=2) as pool:
  p=pool.submit(pwe);f=pool.submit(fem);count=p.result();records=f.result()
 (out/'metadata.json').write_text(json.dumps(dict(status='COMPUTED_NEEDS_VALIDATION',pwe_rows=count,fem_exports=sum(r['status']=='EXPORTED_NOT_VALIDATED' for r in records),seconds=time.perf_counter()-tick),indent=2))
 print('COMPUTED_NEEDS_VALIDATION',out,flush=True)
if __name__=='__main__':main()
