"""Frozen-geometry length-scale scan using archived inverse MCST-TSDT operators.

No optimisation. Every CSV row is a fresh eigensolve. Source and geometry hashes
are recorded before computation. New outputs never overwrite archived results.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, shutil, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'04_code/research/wavemotion_surface_20260920'
NODE=ROOT/'02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1'
sys.path[:0]=[str(CODE/'src'),str(CODE.parent/'msse_20260919/python')]
import numpy as np
import scipy
from scipy.linalg import cho_factor,cho_solve,eigh
from scipy.sparse.linalg import eigsh,LinearOperator
from threadpoolctl import threadpool_limits,threadpool_info
from node_shape_bending import fourier,boundary,signed_area,AREA
from inverse_circle_bending import prepare,classical_inverse
from real_circle_bending import assemble_real

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def gap(rows):
    a=max(rows,key=lambda r:r['Omega4']);b=min(rows,key=lambda r:r['Omega5'])
    L,U=a['Omega4'],b['Omega5']
    return dict(L=L,U=U,J=200*(U-L)/(U+L),kx_L=a['kx_a'],ky_L=a['ky_a'],kx_U=b['kx_a'],ky_U=b['ky_a'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--N',type=int,default=15);p.add_argument('--points',choices=['ibz45','GX'],default='ibz45');p.add_argument('--threads',type=int,default=2);p.add_argument('--ells',type=float,nargs='+',default=[.5,1.,2.]);args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    clock=time.perf_counter();manifest=dict(status='RUNNING',pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),N=args.N,a_um=75,h_um=6,ell_epoxy_um=[0.5,1,2],ell_steel_um=0,points=args.points,threads=args.threads,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),eigen_count=12,shift=-.001,tolerance=1e-10,random_seed=20261030,new_eigensolves=True,reoptimised=False,human_verified=False,dense_symmetric_eigensolver_check='Gamma, X, M at N15; all length scales and both shapes')
    sources=[Path(__file__)]+list((CODE/'src').glob('*.py'))+list((CODE.parent/'msse_20260919/python').glob('*.py'))
    manifest['source_sha256']={str(x.relative_to(ROOT)):sha(x) for x in sources}
    manifest['input_sha256']={str((NODE/x).relative_to(ROOT)):sha(NODE/x) for x in ['manifest.json','initial_vertices.csv','final_vertices.csv','initial_ibz45.csv','round1_ibz45.csv']}
    manifest['ell_epoxy_um']=args.ells
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for i,x in enumerate(sources):shutil.copy2(x,snapshot/f'{i:03d}_{x.name}')
    dump(out/'manifest.json',manifest)
    old_manifest=json.loads((NODE/'manifest.json').read_text(encoding='utf-8'))
    shapes={'initial':np.zeros(6),'final':np.asarray(old_manifest['design'])}
    all_rows={};summary=[];comparisons=[];done=0
    ks=[(np.pi*i/8,np.pi*j/8) for i in range(9) for j in range(i+1)] if args.points=='ibz45' else [(0.,0.),(np.pi,0.)]
    try:
      with threadpool_limits(limits=args.threads):
       manifest['threadpool']=threadpool_info()
       for shape,y in shapes.items():
        vertices=boundary(y);archived=np.genfromtxt(NODE/f'{shape}_vertices.csv',delimiter=',',names=True)
        xy=np.column_stack([archived['x_over_a'],archived['y_over_a']])
        assert np.max(abs(vertices-xy))<1e-14
        assert abs(signed_area(vertices)-AREA)<1e-13
        F=fourier(y,args.N);cache=prepare(args.N,.08,theory='TSDT',indicator_matrix=F)
        handles={};writers={}
        for ell in args.ells:
            stem=f'{shape}_ell{ell:g}_N{args.N}_{args.points}'
            h=(out/(stem+'.csv')).open('x',newline='',encoding='utf-8');handles[ell]=h
            w=csv.DictWriter(h,fieldnames=['point','kx_a','ky_a']+[f'Omega{i}' for i in range(1,7)]+['max_nonrigid_relative_residual','mass_orthogonality_max','eigen_count','solve_seconds']);w.writeheader();writers[ell]=w;all_rows[(shape,ell)]=[]
        old=np.genfromtxt(NODE/('initial_ibz45.csv' if shape=='initial' else 'round1_ibz45.csv'),delimiter=',',names=True)
        for i,k in enumerate(ks):
            parts=assemble_real(args.N,np.asarray(k),h=.08,theory='TSDT',indicator_matrix=F)
            base=classical_inverse(cache,np.asarray(k));M=parts['M'];KB=parts['KB']
            del parts
            for ell in args.ells:
                start=time.perf_counter();K=base+(ell/6.)**2*KB
                factor=cho_factor(K+.001*M,lower=True,check_finite=False)
                inverse=LinearOperator(K.shape,matvec=lambda v:cho_solve(factor,v,check_finite=False),dtype=float)
                val,vec=eigsh(K,k=12,M=M,sigma=-.001,OPinv=inverse,tol=1e-10,v0=np.random.default_rng(20261030).normal(size=len(K)),ncv=60)
                order=np.argsort(val);val=val[order];vec=vec[:,order]
                assert np.all(val[1:]>0),(shape,ell,i,val.tolist())
                omega=np.sqrt(np.maximum(val,0))
                Kv=K@vec;Mv=M@vec
                residual=np.linalg.norm(Kv-Mv*val,axis=0)/(np.linalg.norm(Kv,axis=0)+abs(val)*np.linalg.norm(Mv,axis=0)+1e-30)
                nr=float(np.max(residual[1:6]));orth=float(np.max(abs(vec.T@Mv-np.eye(12))))
                assert nr<1e-7 and orth<1e-7,(nr,orth)
                dense_drift=None
                if args.N==15 and (np.allclose(k,(0,0)) or np.allclose(k,(np.pi,0)) or np.allclose(k,(np.pi,np.pi))):
                    dense=np.sqrt(np.maximum(eigh(K,M,subset_by_index=(0,7),driver='gvx',eigvals_only=True,check_finite=False),0))
                    dense_drift=float(np.max(abs(omega[1:6]/dense[1:6]-1)))
                    assert dense_drift<1e-7,dense_drift
                    manifest.setdefault('dense_checks',[]).append(dict(shape=shape,ell_um=ell,point=i,max_nonrigid_first6_relative_drift=dense_drift))
                row=dict(point=i,kx_a=float(k[0]),ky_a=float(k[1]),**{f'Omega{b+1}':float(omega[b]) for b in range(6)},max_nonrigid_relative_residual=nr,mass_orthogonality_max=orth,eigen_count=12,solve_seconds=time.perf_counter()-start)
                writers[ell].writerow(row);handles[ell].flush();all_rows[(shape,ell)].append(row)
                if ell==1. and args.N==15:
                    idx=int(np.argmin((old['kx_a']-k[0])**2+(old['ky_a']-k[1])**2))
                    reference=np.array([old[f'Omega{b}'][idx] for b in range(2,7)])
                    drifts=abs(omega[1:6]/reference-1)
                    drift=float(np.max(drifts[:4]))
                    comparisons.append(dict(shape=shape,point=i,max_relative_drift_bands2_to5=drift,max_relative_drift_gap_bands4_5=float(np.max(drifts[2:4])),relative_drift_band6=float(drifts[4])));assert drift<1e-5,drift
                done+=1;manifest.update(completed_eigensolves=done,elapsed_seconds=time.perf_counter()-clock,last_completed=dict(shape=shape,ell_um=ell,point=i));dump(out/'manifest.json',manifest)
                print(shape,f'ell={ell:g}',i+1,'/',len(ks),'J_partial',gap(all_rows[(shape,ell)])['J'],'residual',nr,flush=True)
                del K,factor,vec,Kv,Mv,inverse
            del base,M,KB
        for ell,h in handles.items():
            h.close();summary.append(dict(shape=shape,ell_um=ell,N=args.N,points=len(ks),**gap(all_rows[(shape,ell)])))
        del cache,F
       for row in summary:
           ell=row['ell_um'];other=next(x for x in summary if x['ell_um']==ell and x['shape']=='initial')
           row['gain_from_initial_pp']=row['J']-other['J']
       with (out/'V08_summary.csv').open('x',encoding='utf-8',newline='') as h:
           w=csv.DictWriter(h,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
       manifest.update(status='COMPLETED_SAMPLED_SCAN',completed_eigensolves=done,summary=summary,ell1_archived_consistency=comparisons,elapsed_seconds=time.perf_counter()-clock,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
       manifest['output_sha256']={p.name:sha(p) for p in out.glob('*.csv')};dump(out/'manifest.json',manifest)
       print('V08_COMPLETED',json.dumps(summary),flush=True)
    except BaseException as error:
       manifest.update(status='FAILED',error=repr(error),traceback=traceback.format_exc(),completed_eigensolves=done,elapsed_seconds=time.perf_counter()-clock);dump(out/'manifest.json',manifest);raise

if __name__=='__main__':main()
