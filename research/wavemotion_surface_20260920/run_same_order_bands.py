from run_convergence import *
from real_circle_bending import assemble_real
from symmetric_bending import assemble_bending
from scipy.linalg import eigh,cho_factor,cho_solve
from scipy.sparse.linalg import eigsh,LinearOperator
from threadpoolctl import threadpool_limits
import argparse,gc
def solve(K,M,k):
    shift=-.001;factor=cho_factor(K-shift*M,lower=True,overwrite_a=True,check_finite=False)
    inv=LinearOperator(K.shape,matvec=lambda x:cho_solve(factor,x,check_finite=False),dtype=float)
    lam,v=eigsh(K,k=8,M=M,sigma=shift,OPinv=inv,tol=1e-10,v0=np.ones(len(K)))
    ix=np.argsort(lam);lam,v=lam[ix],v[:,ix];raw=lam.copy();kv=K@v;mv=M@v
    residual=np.linalg.norm(kv-mv*lam,axis=0)/(np.linalg.norm(kv,axis=0)+abs(lam)*np.linalg.norm(mv,axis=0)+1e-30)
    if np.linalg.norm(k)==0:
        center=(len(K)//3)//2*3
        assert np.linalg.norm(K[:,center])<1e-10 and abs(lam[0])<1e-7
        lam[0]=0 # known exact rigid translation only; raw eigenvalue retained
        assert min(lam[1:])>0 and max(residual[1:])<1e-5
    else:assert min(lam)>0 and max(residual)<1e-5
    return lam,v,residual,raw
def main():
 p=argparse.ArgumentParser();p.add_argument('--N',type=int,default=27);p.add_argument('--segments',type=int,default=4);a=p.parse_args()
 out=ROOT/'runs'/('same_order_bands_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir();print('OUTPUT',out,flush=True)
 cfg=dict(N=a.N,segments=a.segments,theory='TSDT',ell_matrix_over_h=[0,.1],ell_steel=0,h_over_a=.1,a_m=.05,area_fraction=.3,surface=False,fluid=False,scope='Bending branches on Gamma-X-M-Gamma only; not complete 2D/all-polarization gap certification',length_calibration='UNAVAILABLE numerical parameter',threads=2)
 (out/'config.json').write_text(json.dumps(cfg,indent=2));checks=[];rows=[]
 with threadpool_limits(limits=2):
    for k in [[.71,.39],[np.pi,0]]:
        s=assemble_real(3,k);old=assemble_bending(3,k,materials=[(210.6/4.35,.3,7780/1180),(1.,4.35/(2*1.59)-1,1.)],surface_constants={'A':(0.,0.),'B':(0.,0.)});d=np.tile([1,1j,1j],len(s['ij']))
        for name in ['Kcl','M','KB']:
            target=d.conj()[:,None]*old[name]*d[None,:];err=np.linalg.norm(s[name]-target)/np.linalg.norm(target);assert err<1e-11;checks.append(dict(k=k,matrix=name,error=err))
        K=s['Kcl']+.01*s['KB'];lam,_,_,_=solve(K,s['M'],k);ref=eigh(K,s['M'],eigvals_only=True,subset_by_index=[0,7]);assert max(abs(lam/ref-1))<1e-7
    (out/'regression.json').write_text(json.dumps(checks,indent=2));print('REAL_REGRESSION PASS',flush=True);del s,old,K;gc.collect()
    vertices=np.array([[0,0],[np.pi,0],[np.pi,np.pi],[0,0]])
    path=np.array([(1-t)*vertices[j]+t*vertices[j+1] for j in range(3) for t in np.arange(a.segments)/a.segments]+[[0,0]])
    np.savetxt(out/'path.csv',path,delimiter=',',header='kx_a,ky_a',comments='')
    order=[a.segments]+[i for i in range(len(path)-1) if i!=a.segments]
    for point in order:
        t=time.perf_counter();k=path[point];s=assemble_real(a.N,k);K=s['Kcl'];M=s['M']
        for ellh in [0.,.1]:
            if ellh:K+=ellh**2*s['KB']
            lam,v,res,raw=solve(K,M,k)
            np.savez_compressed(out/f'point{point}_ellh{ellh}.npz',Omega=np.sqrt(lam),eigenvalues_raw=raw,vectors=v,ij=s['ij'],k=k,residual=res,complex_column_phase=np.array([1,1j,1j]))
            for j,x in enumerate(lam):rows.append(dict(point=point,kx=float(k[0]),ky=float(k[1]),ellh=ellh,band=j+1,Omega=float(np.sqrt(x)),residual=float(res[j])))
            with (out/'bands.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            print('SOLVED',point,ellh,time.perf_counter()-t,flush=True)
        del s,K,M,v;gc.collect()
    meta=dict(status='PASS',scope=cfg['scope'],actually_executed=True,closing_Gamma_reused=True,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'src/real_circle_bending.py',OLD/'msse_core.py',OLD/'two_phase.py']},config_sha256=hashlib.sha256((out/'config.json').read_bytes()).hexdigest())
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
