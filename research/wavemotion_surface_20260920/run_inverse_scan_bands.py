"""Real inverse PWE path spectra for both original parameter cuts; no FEM."""
from run_same_order_bands import *
from inverse_circle_bending import prepare,classical_inverse
from run_large_scale_3d import inverse_checks

def main():
 p=argparse.ArgumentParser();p.add_argument('--N',type=int,default=15);p.add_argument('--segments',type=int,default=4);args=p.parse_args()
 out=ROOT/'runs'/('inverse_scan_bands_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir();print('OUTPUT',out,flush=True)
 cfg=dict(N=args.N,segments=args.segments,a_um=[10,15,25,40,75,100,200],h_um=[.5,1,1.5,2,3,5,8],fixed_h_um=2,fixed_a_um=25,r_over_a=.3,ell_host_um=1,ell_steel_um=0,threads=2,method='classical generalized compliance inverse; direct MCST curvature and mass',scope='first eight bending-related path branches; not full-BZ gap certification')
 (out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'checks.json').write_text(json.dumps(inverse_checks(),indent=2));print('CHECKS PASS',flush=True)
 vertices=np.array([[0.,0.],[np.pi,0.],[np.pi,np.pi],[0.,0.]])
 path=np.array([(1-t)*vertices[j]+t*vertices[j+1] for j in range(3) for t in np.arange(args.segments)/args.segments]);np.savetxt(out/'path.csv',path,delimiter=',',header='kx_a,ky_a',comments='')
 geometries=sorted(set([(float(a),2.) for a in cfg['a_um']]+[(25.,float(h)) for h in cfg['h_um']]))
 rows=[];timings=[];start=time.perf_counter()
 def checkpoint():
  with (out/'spectra.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
  (out/'timings.json').write_text(json.dumps(timings,indent=2))
 with threadpool_limits(limits=2):
  for a,h in geometries:
   for theory in ['FSDT','TSDT']:
    tick=time.perf_counter();cache=prepare(args.N,h/a,.3,theory)
    for point,k in enumerate(path):
     s=assemble_real(args.N,k,h/a,.3,theory=theory);del s['Kcl'];kc=classical_inverse(cache,k)
     for ell in [0.,1.]:
      K=kc+(ell/h)**2*s['KB'];lam,V,res,raw=solve(K,s['M'],k)
      freq=np.sqrt(lam)*np.sqrt(4.35e9/1180)/(a*1e-6*2*np.pi)/1e6
      tag=f'N{args.N}_a{a:g}_h{h:g}_p{point}_{theory}_l{ell:g}'
      np.savez_compressed(out/(tag+'.npz'),frequency_MHz=freq,vectors=V,raw=raw,residual=res,ij=s['ij'],k=k)
      for j in range(8):rows.append(dict(N=args.N,a_um=a,h_um=h,point=point,theory=theory,ell_um=ell,band=j+1,frequency_MHz=float(freq[j]),residual=float(res[j])))
      del K,V
     checkpoint();del s,kc;gc.collect()
    elapsed=time.perf_counter()-tick;timings.append(dict(a_um=a,h_um=h,theory=theory,seconds=elapsed));checkpoint();print('DONE',a,h,theory,round(elapsed,1),'seconds',flush=True);del cache;gc.collect()
 sources=[Path(__file__),ROOT/'src/inverse_circle_bending.py',ROOT/'src/real_circle_bending.py',ROOT/'run_same_order_bands.py',OLD/'msse_core.py']
 (out/'metadata.json').write_text(json.dumps(dict(status='COMPUTED_NOT_CONVERGENCE_CERTIFIED',seconds=time.perf_counter()-start,frequency_count=len(rows),command=sys.argv,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},config_sha256=hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),NOT_RUN=['new FEM','full BZ','cross-parameter field MAC','inverse N convergence for every geometry']),indent=2));print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
