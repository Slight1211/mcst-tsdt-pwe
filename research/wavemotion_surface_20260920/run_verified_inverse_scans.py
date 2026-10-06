"""Recompute both scans after detecting symmetry-induced ARPACK mode omission."""
from run_inverse_scan_bands import *
from concurrent.futures import ProcessPoolExecutor,as_completed

def verified_solve(K,M,k,N):
 if N==9:
  lam,V=eigh(K,M,subset_by_index=[0,15],check_finite=False);agreement=0.
 else:
  fact=cho_factor(K+.001*M,lower=True,check_finite=False)
  inv=LinearOperator(K.shape,matvec=lambda x:cho_solve(fact,x,check_finite=False),dtype=float)
  for ncv in [40,80]:
   spectra=[]
   for seed in [20260921,20260922]:
    ev,vec=eigsh(K,k=16,ncv=ncv,M=M,sigma=-.001,OPinv=inv,tol=1e-11,v0=np.random.default_rng(seed).normal(size=len(K)))
    ix=np.argsort(ev);ev,vec=ev[ix],vec[:,ix];spectra.append(ev)
    if len(spectra)==1:lam,V=ev,vec
   start=1 if np.linalg.norm(k)==0 else 0
   agreement=float(np.max(abs(spectra[0][start:8]/spectra[1][start:8]-1)))
   if agreement<1e-7:break
  assert agreement<1e-7, ('Independent-start mismatch',agreement,N,k)
 raw=lam.copy();kv=K@V;mv=M@V
 residual=np.linalg.norm(kv-mv*lam,axis=0)/(np.linalg.norm(kv,axis=0)+abs(lam)*np.linalg.norm(mv,axis=0)+1e-30)
 orth=float(np.linalg.norm(V.T@M@V-np.eye(16)))
 assert orth<1e-7, ('M orthogonality',orth)
 start=0
 if np.linalg.norm(k)==0:
  center=(len(K)//3)//2*3
  assert np.linalg.norm(K[:,center])<1e-10 and abs(lam[0])<1e-7
  lam[0]=0.;start=1
 assert min(lam[start:8])>0 and max(residual[start:8])<1e-5
 return lam,V,residual,raw,agreement,orth

def worker(job):
 N,a,h,outstr,segments=job;out=Path(outstr);rows=[];timings=[]
 vertices=np.array([[0.,0.],[np.pi,0.],[np.pi,np.pi],[0.,0.]])
 path=np.array([(1-t)*vertices[j]+t*vertices[j+1] for j in range(3) for t in np.arange(segments)/segments])
 with threadpool_limits(limits=1):
  for theory in ['FSDT','TSDT']:
   tick=time.perf_counter();cache=prepare(N,h/a,.3,theory)
   for point,k in enumerate(path):
    s=assemble_real(N,k,h/a,.3,theory=theory);del s['Kcl'];kc=classical_inverse(cache,k)
    for ell in [0.,1.]:
     saved=out/f'N{N}_a{a:g}_h{h:g}_p{point}_{theory}_l{ell:g}.npz'
     if saved.exists():
      try:
       with np.load(saved) as prior:
        pf=prior['frequency_MHz'];pr=prior['residual'];pa=float(prior['seed_agreement']);po=float(prior['mass_orthogonality'])
        assert len(pf)==8 and np.all(np.isfinite(pf)) and pa<1e-7 and po<1e-7
        for j in range(8):rows.append(dict(N=N,a_um=a,h_um=h,point=point,theory=theory,ell_um=ell,band=j+1,frequency_MHz=float(pf[j]),residual=float(pr[j]),seed_agreement=pa,mass_orthogonality=po))
       continue
      except (OSError,ValueError,EOFError):pass
     K=kc+(ell/h)**2*s['KB'];lam,V,res,raw,agreement,orth=verified_solve(K,s['M'],k,N)
     freq=np.sqrt(lam)*np.sqrt(4.35e9/1180)/(a*1e-6*2*np.pi)/1e6
     np.savez_compressed(out/f'N{N}_a{a:g}_h{h:g}_p{point}_{theory}_l{ell:g}.npz',frequency_MHz=freq[:8],expanded_frequency_MHz=freq,vectors=V[:,:8],raw=raw,residual=res,ij=s['ij'],k=k,seed_agreement=agreement,mass_orthogonality=orth)
     for j in range(8):rows.append(dict(N=N,a_um=a,h_um=h,point=point,theory=theory,ell_um=ell,band=j+1,frequency_MHz=float(freq[j]),residual=float(res[j]),seed_agreement=agreement,mass_orthogonality=orth))
     del K,V
    del s,kc;gc.collect()
   timings.append(dict(a_um=a,h_um=h,theory=theory,seconds=time.perf_counter()-tick));del cache;gc.collect()
 return N,a,h,rows,timings

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--resume-stamp');args=parser.parse_args()
 stamp=args.resume_stamp or datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');outs={};configs={};rows={9:[],15:[]};times={9:[],15:[]};jobs=[];start=time.perf_counter()
 checks=inverse_checks()
 for N,segments in [(9,1),(15,4)]:
  out=ROOT/'runs'/f'inverse_scan_verified_N{N}_{stamp}';out.mkdir(exist_ok=bool(args.resume_stamp));outs[N]=out
  source=ROOT/'runs/inverse_scan_bands_20260921T153727153752Z'
  cfg=json.loads((source/'config.json').read_text());cfg.update(N=N,segments=segments,workers=3,threads=1,extraction='N9 dense first16; N15 k16 ncv40, fallback80; two independent seeded starts; retain first8',resumed_verified_checkpoints_ncv80=bool(args.resume_stamp))
  configs[N]=cfg;(out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'checks.json').write_text(json.dumps(checks,indent=2))
  vertices=np.array([[0.,0.],[np.pi,0.],[np.pi,np.pi],[0.,0.]])
  path=np.array([(1-t)*vertices[j]+t*vertices[j+1] for j in range(3) for t in np.arange(segments)/segments]);np.savetxt(out/'path.csv',path,delimiter=',',header='kx_a,ky_a',comments='')
  for a,h in sorted(set([(float(a),2.) for a in cfg['a_um']]+[(25.,float(h)) for h in cfg['h_um']])):jobs.append((N,a,h,str(out),segments))
  print('OUTPUT',N,out,flush=True)
 with ProcessPoolExecutor(max_workers=3) as pool:
  futures=[pool.submit(worker,j) for j in jobs]
  for future in as_completed(futures):
   N,a,h,rr,tt=future.result();rows[N]+=rr;times[N]+=tt;out=outs[N]
   with (out/'spectra.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[N][0]);w.writeheader();w.writerows(rows[N])
   (out/'timings.json').write_text(json.dumps(times[N],indent=2));print('DONE',N,a,h,'rows',len(rows[N]),flush=True)
 for N,out in outs.items():
  sources=[Path(__file__),ROOT/'run_inverse_scan_bands.py',ROOT/'src/inverse_circle_bending.py',ROOT/'src/real_circle_bending.py',ROOT/'run_same_order_bands.py',OLD/'msse_core.py']
  meta=dict(status='COMPUTED_EXTRACTION_VERIFIED_NOT_N_CONVERGENCE_CERTIFIED',seconds=time.perf_counter()-start,timing_scope='resumed combined N9/N15 three-worker wall time; excludes prior checkpoint computation',frequency_count=len(rows[N]),command=sys.argv,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},config_sha256=hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),supersedes='initial fixed-ones k8 extraction; original files retained',NOT_RUN=['new FEM','full BZ','higher N full-path convergence'])
  (out/'metadata.json').write_text(json.dumps(meta,indent=2))
 print('COMPLETE',json.dumps({k:str(v) for k,v in outs.items()}),flush=True)
if __name__=='__main__':main()
