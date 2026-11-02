"""New N15 prescribed-length-scale spectra for three frozen geometries.

No optimization or COMSOL run. Input geometry and code are hash-bound before
computation. New output directories are exclusive; archived evidence is unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'04_code/research/wavemotion_surface_20260920'
NODE=ROOT/'02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1'
ELL2=ROOT/'05_validation/V07_ell2_reoptimization_20261030/restart'
sys.path[:0]=[str(CODE/'src'),str(CODE.parent/'msse_20260919/python')]
import numpy as np
import scipy
from scipy.linalg import cho_factor,cho_solve,eigh
from scipy.sparse.linalg import eigsh,LinearOperator
from threadpoolctl import threadpool_limits,threadpool_info
from node_shape_bending import boundary,fourier,signed_area,AREA
from inverse_circle_bending import prepare,classical_inverse
from real_circle_bending import assemble_real


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)
def summary(rows):
    lower=max(rows,key=lambda r:r['Omega4']);upper=min(rows,key=lambda r:r['Omega5'])
    L,U=lower['Omega4'],upper['Omega5']
    return dict(L=L,U=U,J=200*(U-L)/(U+L),L_MHz=lower['f4_MHz'],U_MHz=upper['f5_MHz'],
                lower_point=lower['point'],upper_point=upper['point'],
                kx_L=lower['kx_a'],ky_L=lower['ky_a'],kx_U=upper['kx_a'],ky_U=upper['ky_a'])
def point_name(k):
    for name,q in [('Gamma',(0.,0.)),('X',(np.pi,0.)),('M',(np.pi,np.pi))]:
        if np.allclose(k,q,atol=1e-14,rtol=0):return name
    return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ell',type=float,default=10.)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if not np.isfinite(args.ell) or args.ell<0:raise ValueError('ell must be finite and nonnegative, in micrometres')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    node_manifest=json.loads((NODE/'manifest.json').read_text(encoding='utf-8'))
    ell2_manifest=json.loads((ELL2/'manifest.json').read_text(encoding='utf-8'))
    shapes={
        'initial':(np.zeros(6),NODE/'initial_vertices.csv'),
        'ell1_design':(np.asarray(node_manifest['design']),NODE/'final_vertices.csv'),
        'ell2_design':(np.asarray(ell2_manifest['design']),ELL2/'final_vertices.csv')}
    sources=[Path(__file__)]+sorted((CODE/'src').glob('*.py'))+sorted((CODE.parent/'msse_20260919/python').glob('*.py'))
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for i,p in enumerate(sources):shutil.copy2(p,snapshot/f'{i:03d}_{p.name}')
    normalization=float(np.sqrt(4.35e9/1180)/(75e-6*2*np.pi*1e6))
    meta=dict(status='RUNNING',pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        N=15,a_um=75.,h_um=6.,ell_epoxy_um=args.ell,ell_steel_um=0.,ell_over_h=args.ell/6.,
        ell_squared_over_h_squared=(args.ell/6.)**2,h_over_a=.08,points='ibz45',grid=8,
        eigen_count=12,shift=-.001,tolerance=1e-10,ncv=60,random_seed=20261030,threads=args.threads,
        new_eigensolves=True,reoptimized=False,no_global_optimality_claim=True,no_full_zone_or_3D_gap_claim=True,
        source_sha256=source_hashes,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [NODE/'manifest.json',ELL2/'manifest.json']+[p for _,p in shapes.values()]},
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        normalization=dict(f_MHz='Omega*sqrt(E_ref/rho_ref)/(2*pi*a*1e6)',E_ref_Pa=4.35e9,rho_ref_kg_m3=1180.,a_m=75e-6,MHz_per_Omega=normalization),
        shape_designs={name:y.tolist() for name,(y,_) in shapes.items()},human_verified=False)
    dump(out/'manifest.json',meta)
    ks=[(np.pi*i/8,np.pi*j/8) for i in range(9) for j in range(i+1)]
    total=0;summary_rows=[];dense_rows=[]
    try:
        with threadpool_limits(limits=args.threads):
            meta['threadpool']=threadpool_info()
            for name,(design,vertices_file) in shapes.items():
                vertices=boundary(design)
                archived=np.genfromtxt(vertices_file,delimiter=',',names=True)
                saved=np.column_stack([archived['x_over_a'],archived['y_over_a']])
                assert np.max(abs(vertices-saved))<1e-14,(name,'geometry does not match archive')
                assert abs(signed_area(vertices)-AREA)<1e-13,(name,'area')
                np.savetxt(out/f'{name}_vertices.csv',vertices,delimiter=',',header='x_over_a,y_over_a',comments='')
                F=fourier(design,15);cache=prepare(15,.08,theory='TSDT',indicator_matrix=F)
                records=[]
                fields=['point','kx_a','ky_a']+[f'Omega{b}' for b in range(1,7)]+[f'f{b}_MHz' for b in range(1,7)]+['max_nonrigid_relative_residual','mass_orthogonality_max','dense_first6_relative_drift','solve_seconds']
                with (out/f'{name}_ell{args.ell:g}_N15_ibz45.csv').open('x',encoding='utf-8',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
                    for index,k in enumerate(ks):
                        tick=time.perf_counter();wave=np.asarray(k)
                        parts=assemble_real(15,wave,h=.08,theory='TSDT',indicator_matrix=F)
                        M=parts['M'];K=classical_inverse(cache,wave)+(args.ell/6.)**2*parts['KB'];del parts
                        factor=cho_factor(K+.001*M,lower=True,check_finite=False)
                        inverse=LinearOperator(K.shape,matvec=lambda v:cho_solve(factor,v,check_finite=False),dtype=float)
                        values,vectors=eigsh(K,k=12,M=M,sigma=-.001,OPinv=inverse,tol=1e-10,ncv=60,v0=np.random.default_rng(20261030).normal(size=len(K)))
                        ordering=np.argsort(values);values=values[ordering];vectors=vectors[:,ordering]
                        gamma=point_name(k)=='Gamma'
                        assert np.all(values[1 if gamma else 0:]>0),(name,index,'nonpositive nonrigid eigenvalues',values.tolist())
                        if gamma:assert abs(values[0])<1e-7,(name,'Gamma first mode not rigid',values[0])
                        omega=np.sqrt(np.maximum(values,0));Kv=K@vectors;Mv=M@vectors
                        residual=np.linalg.norm(Kv-Mv*values,axis=0)/(np.linalg.norm(Kv,axis=0)+abs(values)*np.linalg.norm(Mv,axis=0)+1e-30)
                        nr=float(np.max(residual[1 if gamma else 0:6]));orth=float(np.max(abs(vectors.T@Mv-np.eye(12))))
                        assert nr<1e-7 and orth<1e-7,(name,index,'numerical gate',nr,orth)
                        drift=None
                        if point_name(k):
                            dense=np.sqrt(np.maximum(eigh(K,M,subset_by_index=(0,7),driver='gvx',eigvals_only=True,check_finite=False),0))
                            lower=1 if gamma else 0
                            drift=float(np.max(abs(omega[lower:6]/dense[lower:6]-1)))
                            assert drift<1e-7,(name,index,'dense discrepancy',drift)
                            dense_rows.extend(dict(shape=name,point=point_name(k),band=b+1,Omega_sparse=float(omega[b]),Omega_dense=float(dense[b]),
                                relative_drift=None if gamma and b==0 else float(omega[b]/dense[b]-1)) for b in range(6))
                        row=dict(point=index,kx_a=float(k[0]),ky_a=float(k[1]),
                            **{f'Omega{b}':float(omega[b-1]) for b in range(1,7)},
                            **{f'f{b}_MHz':float(omega[b-1]*normalization) for b in range(1,7)},
                            max_nonrigid_relative_residual=nr,mass_orthogonality_max=orth,
                            dense_first6_relative_drift=drift,solve_seconds=time.perf_counter()-tick)
                        records.append(row);writer.writerow(row);handle.flush();total+=1
                        meta.update(completed_eigensolves=total,last_completed=dict(shape=name,index=index,point=point_name(k)),elapsed_seconds=time.perf_counter()-start)
                        dump(out/'manifest.json',meta)
                        print(name,index+1,'/45','J_partial',summary(records)['J'],'max_nonrigid_residual',nr,flush=True)
                        del K,M,factor,inverse,Kv,Mv,vectors
                item=dict(shape=name,ell_um=args.ell,N=15,points=len(records),**summary(records))
                summary_rows.append(item);meta['completed_shape_summaries']=summary_rows
                dump(out/'manifest.json',meta)
                print('SHAPE_COMPLETED',json.dumps(item),flush=True)
                del cache,F
        baseline=summary_rows[0]['J']
        for row in summary_rows:row['gain_from_initial_pp']=row['J']-baseline
        for filename,rows in [('summary.csv',summary_rows),('dense_checks.csv',dense_rows)]:
            with (out/filename).open('x',encoding='utf-8',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        assert all(sha(ROOT/p)==digest for p,digest in source_hashes.items()),'source changed during run'
        meta.update(status='COMPLETED_SAMPLED_FROZEN_SCAN',summary=summary_rows,completed_eigensolves=total,
            dense_check_count=len(dense_rows),elapsed_seconds=time.perf_counter()-start,
            finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_hash_check='PASS',
            output_sha256={p.name:sha(p) for p in out.glob('*.csv')})
        dump(out/'manifest.json',meta)
        print('FROZEN_SCAN_COMPLETED',json.dumps(summary_rows),flush=True)
    except BaseException as exc:
        meta.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),completed_eigensolves=total,elapsed_seconds=time.perf_counter()-start)
        dump(out/'manifest.json',meta);raise


if __name__=='__main__':main()
