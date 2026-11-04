"""Independent N15 sparse/dense checks of a saved local-search geometry.

This is a bounded three-wavevector eigenfrequency check, not an optimization,
3D validation, or higher-order study. Outputs stay outside the search directory.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'04_code/research/wavemotion_surface_20260920'
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
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--round',type=int)
    args=parser.parse_args();run=args.run_dir.resolve();out=args.out.resolve()
    if out==run or run in out.parents:raise ValueError('Checker outputs must stay outside the optimization directory')
    out.mkdir(parents=True,exist_ok=False)
    run_meta=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    roundid=args.round or max(int(p.stem.removeprefix('round').removesuffix('_optimizer')) for p in run.glob('round*_optimizer.json'))
    optfile=run/f'round{roundid}_optimizer.json';opt=json.loads(optfile.read_text(encoding='utf-8'))
    referencefile=run/f'round{roundid}_ibz45.csv';reference=rows(referencefile);assert len(reference)==45
    N=int(run_meta['N']);a_um=float(run_meta['a_um']);h_um=float(run_meta['h_um']);ell_um=float(run_meta['ell_epoxy_um'])
    assert (N,a_um,h_um)==(15,75.,6.)
    design=np.asarray(opt['design']);vertices=boundary(design)
    assert np.all(np.abs(design)<=.16+1e-12)
    assert abs(signed_area(vertices)-AREA)<1e-13
    sources=[Path(__file__)]+sorted((CODE/'src').glob('*.py'))+sorted((CODE.parent/'msse_20260919/python').glob('*.py'))
    sources_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [optfile,referencefile]}
    meta=dict(status='RUNNING',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),
        N=N,a_um=a_um,h_um=h_um,ell_epoxy_um=ell_um,ell_steel_um=0.,h_over_a=h_um/a_um,ell_over_h=ell_um/h_um,
        construction='K=classical_inverse(prepare(N,h/a,indicatorF),k)+(ell_um/h_um)**2*KB; M=assemble_real(...)[M]',
        source_run=str(run.relative_to(ROOT)),source_round=roundid,source_run_status_at_start=run_meta['status'],
        source_run_manifest_is_live=True,input_sha256=inputs,source_sha256=sources_hashes,design=design.tolist(),
        geometry=dict(vertices=48,shape_variables=6,area_fraction=float(signed_area(vertices)),target_area_fraction=AREA,
            area_error=float(signed_area(vertices)-AREA),shape_bounds=[-.16,.16],
            max_abs_vertex_over_a=float(np.max(abs(vertices))),within_cell=bool(np.max(abs(vertices))<.5)),
        points=['Gamma','X','M'],sparse_eigen_count=12,dense_eigen_count=8,checked_first_frequencies=6,
        sparse_shift=-.001,sparse_tolerance=1e-10,ncv=60,seed=20261030,threads=2,
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        new_spectrum_checks=True,new_optimization=False,new_COMSOL_runs=False,no_continuous_BZ_claim=True,human_verified=False)
    dump(out/'run_parameters_snapshot.json',run_meta);dump(out/'manifest.json',meta)
    np.savetxt(out/'vertices.csv',vertices,delimiter=',',header='x_over_a,y_over_a',comments='')
    clock=time.perf_counter();records=[];point_checks=[]
    try:
        with threadpool_limits(limits=2):
            meta['threadpool']=threadpool_info()
            F=fourier(design,N);cache=prepare(N,h_um/a_um,theory='TSDT',indicator_matrix=F)
            norm=float(np.sqrt(4.35e9/1180)/(a_um*1e-6*2*np.pi*1e6))
            for name,k in [('Gamma',(0.,0.)),('X',(np.pi,0.)),('M',(np.pi,np.pi))]:
                tick=time.perf_counter();wave=np.asarray(k)
                parts=assemble_real(N,wave,h=h_um/a_um,theory='TSDT',indicator_matrix=F)
                M=parts['M'];K=classical_inverse(cache,wave)+(ell_um/h_um)**2*parts['KB'];del parts
                factor=cho_factor(K+.001*M,lower=True,check_finite=False)
                inv=LinearOperator(K.shape,matvec=lambda v:cho_solve(factor,v,check_finite=False),dtype=float)
                values,vectors=eigsh(K,k=12,M=M,sigma=-.001,OPinv=inv,tol=1e-10,ncv=60,v0=np.random.default_rng(20261030).normal(size=len(K)))
                order=np.argsort(values);values=values[order];vectors=vectors[:,order]
                lower=1 if name=='Gamma' else 0
                assert np.all(values[lower:]>0)
                if name=='Gamma':assert abs(values[0])<1e-7
                sparse=np.sqrt(np.maximum(values,0))
                Kv=K@vectors;Mv=M@vectors
                res=np.linalg.norm(Kv-Mv*values,axis=0)/(np.linalg.norm(Kv,axis=0)+abs(values)*np.linalg.norm(Mv,axis=0)+1e-30)
                maxres=float(np.max(res[lower:6]));orth=float(np.max(abs(vectors.T@Mv-np.eye(12))))
                dense=np.sqrt(np.maximum(eigh(K,M,subset_by_index=(0,7),driver='gvx',eigvals_only=True,check_finite=False),0))
                dense_drift=float(np.max(abs(sparse[lower:6]/dense[lower:6]-1)))
                found=[r for r in reference if abs(float(r['kx_a'])-k[0])<1e-12 and abs(float(r['ky_a'])-k[1])<1e-12]
                assert len(found)==1
                saved=np.array([float(found[0][f'Omega{b}']) for b in range(1,7)])
                saved_drift=float(np.max(abs(sparse[lower:6]/saved[lower:6]-1)))
                assert maxres<1e-7 and orth<1e-7 and dense_drift<1e-7 and saved_drift<1e-7,(name,maxres,orth,dense_drift,saved_drift)
                point_checks.append(dict(point=name,max_nonrigid_first6_relative_residual=maxres,
                    mass_orthogonality_max=orth,max_nonrigid_first6_dense_relative_drift=dense_drift,
                    max_nonrigid_first6_saved_relative_drift=saved_drift,solve_seconds=time.perf_counter()-tick))
                for band in range(6):
                    rigid=name=='Gamma' and band==0
                    records.append(dict(point=name,kx_a=k[0],ky_a=k[1],band=band+1,
                        Omega_sparse=float(sparse[band]),Omega_dense=float(dense[band]),Omega_saved=float(saved[band]),
                        f_sparse_MHz=float(sparse[band]*norm),
                        relative_dense_difference=None if rigid else float(sparse[band]/dense[band]-1),
                        relative_saved_difference=None if rigid else float(sparse[band]/saved[band]-1),
                        relative_residual=None if rigid else float(res[band]),rigid_translation=rigid))
                meta.update(point_checks=point_checks,elapsed_seconds=time.perf_counter()-clock);dump(out/'manifest.json',meta)
                print('CHECK',name,json.dumps(point_checks[-1]),flush=True)
                del M,K,factor,inv,Kv,Mv,vectors
        with (out/'frequency_checks.csv').open('x',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        assert all(sha(ROOT/p)==digest for p,digest in inputs.items()),'saved round inputs changed during check'
        assert all(sha(ROOT/p)==digest for p,digest in sources_hashes.items()),'construction source changed during check'
        meta.update(status='PASS_INDEPENDENT_SPARSE_DENSE_GXM',elapsed_seconds=time.perf_counter()-clock,
            max_nonrigid_first6_relative_residual=max(p['max_nonrigid_first6_relative_residual'] for p in point_checks),
            max_nonrigid_first6_dense_relative_drift=max(p['max_nonrigid_first6_dense_relative_drift'] for p in point_checks),
            max_nonrigid_first6_saved_relative_drift=max(p['max_nonrigid_first6_saved_relative_drift'] for p in point_checks),
            max_mass_orthogonality_error=max(p['mass_orthogonality_max'] for p in point_checks),
            normalization_MHz_per_Omega=norm,source_hash_check='PASS',input_hash_check='PASS',
            output_sha256={p.name:sha(p) for p in out.glob('*.csv')})
        dump(out/'manifest.json',meta);print('CHECK_COMPLETED',json.dumps(meta),flush=True)
    except BaseException as exc:
        meta.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-clock)
        dump(out/'manifest.json',meta);raise

if __name__=='__main__':main()
