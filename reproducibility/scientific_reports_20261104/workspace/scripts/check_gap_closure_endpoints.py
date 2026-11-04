"""Independent matrix checks of the gap-closure scan's vetted endpoint witnesses.

The chosen geometry is case_summary.best_vetted_candidate, which can be a seed
or intermediate geometry rather than a stationary terminal. A negative local
search result is never interpreted as a proof over all admissible shapes.
Run only after endpoint selection; --plan-only performs no eigensolves.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
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

DEFAULT_BASE=ROOT/'05_validation/GAP_CLOSURE_SCAN_20261031'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
locate=lambda p:Path(p).relative_to(ROOT).as_posix()

def save(path,data):
    temp=Path(str(path)+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)

def source_path(value):
    path=Path(value)
    if not path.is_absolute():path=ROOT/path
    path=path.resolve()
    assert path.is_relative_to(ROOT.resolve()),('Input escapes project',path)
    return path

def read_spectrum(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        records=list(csv.DictReader(handle))
    assert len(records)==153,(path,'Expected153point grid')
    k=np.array([[float(r['kx_a']),float(r['ky_a'])] for r in records])
    frequencies=np.array([[float(r[f'Omega{b}']) for b in range(1,7)] for r in records])
    expected=np.array([(i,j) for i in range(17) for j in range(i+1)])
    assert np.max(abs(k*16/np.pi-expected))<1e-10,(path,'Grid/order mismatch')
    assert np.isfinite(frequencies).all() and np.all(np.diff(frequencies,axis=1)>=-1e-9)
    assert np.all(frequencies[:,1:]>0) and np.all(frequencies[1:,0]>0)
    lo=int(np.argmax(frequencies[:,3]));up=int(np.argmin(frequencies[:,4]))
    L=float(frequencies[lo,3]);U=float(frequencies[up,4])
    lower=list(map(int,np.flatnonzero(abs(frequencies[:,3]-L)<=1e-10)))
    upper=list(map(int,np.flatnonzero(abs(frequencies[:,4]-U)<=1e-10)))
    gap=dict(L=L,U=U,J=200*(U-L)/(U+L),active_L=lo,active_U=up,
        lower_k_a=k[lo].tolist(),upper_k_a=k[up].tolist(),
        lower_tied_indices=lower,upper_tied_indices=upper)
    return records,k,frequencies,gap

def validate_geometry(design):
    y=np.asarray(design,float)
    assert y.shape==(6,) and np.isfinite(y).all() and max(abs(y))<=.16+1e-10
    vertices=boundary(y)
    area=.5*float(np.sum(vertices[:,0]*np.roll(vertices[:,1],-1)-vertices[:,1]*np.roll(vertices[:,0],-1)))
    assert vertices.shape==(48,2) and abs(area-AREA)<1e-13
    assert np.max(abs(vertices))<.49
    assert np.min(np.linalg.norm(np.roll(vertices,-1,axis=0)-vertices,axis=1))>=1e-5
    assert abs(area-float(signed_area(vertices)))<1e-14
    return y,vertices,dict(vertices=48,variables=6,area_fraction=area,target_area_fraction=AREA,
        area_error=area-AREA,shape_bounds=[-.16,.16],max_abs_vertex_over_a=float(np.max(abs(vertices))))

def resolve_candidate(base,controller,ell):
    """Read immutable witness data, never substitute the selected terminal."""
    matches=[c for c in controller.get('cases',[]) if abs(float(c['ell_um'])-ell)<1e-12]
    inputs={}
    if matches:
        assert len(matches)==1
        summary_path=base/f'ell_{ell:g}/case_summary.json'
        summary=load(summary_path);assert summary==matches[0]
        candidate=summary['best_vetted_candidate'];assert candidate is not None
        spectrum=source_path(candidate['spectrum'])
        assert spectrum.is_relative_to((base/f'ell_{ell:g}').resolve())
        assert sha(spectrum)==candidate['spectrum_sha256']
        worker_path=spectrum.parent/'manifest.json';worker=load(worker_path)
        assert candidate['run']==spectrum.parent.name
        assert all(worker[k]==v for k,v in [('N',15),('a_um',75.),('h_um',6.),('ell_steel_um',0.),('optimization_points',153)])
        assert abs(float(worker['ell_epoxy_um'])-ell)<1e-12
        assert abs(float(worker['ell_over_h'])-ell/6)<1e-14
        matching=[c for c in worker['full153_candidates']
            if c['round']==candidate['round'] and c['spectrum']==spectrum.name and c['design']==candidate['design']]
        assert len(matching)==1
        assert matching[0]['optimizer_success']==candidate['local_optimizer_success']
        assert all(abs(matching[0][key]-candidate[key])<1e-9 for key in ['L','U','J'])
        if worker['status'].startswith('PASS'):
            assert sha(spectrum)==worker['output_sha256'][spectrum.name]
        for name,digest in worker['source_sha256'].items():
            assert sha(source_path(name))==digest,('Worker construction source changed',name)
        inputs.update({locate(p):sha(p) for p in [summary_path,worker_path,spectrum]})
        if candidate['round']>0:
            optimizer_path=spectrum.parent/f"round{candidate['round']}_optimizer.json"
            assert load(optimizer_path)['design']==candidate['design']
            inputs[locate(optimizer_path)]=sha(optimizer_path)
        return candidate,summary['classification'],worker['status'],inputs,summary
    state=controller['initial_state']
    assert abs(float(state['ell_um'])-ell)<1e-12,('Unknown selected ell',ell)
    seed_path=source_path(state['seed_manifest']);assert sha(seed_path)==state['seed_sha256']
    worker=load(seed_path);assert worker['status'].startswith('PASS')
    assert all(worker[k]==v for k,v in [('N',15),('a_um',75.),('h_um',6.),('ell_steel_um',0.)])
    assert abs(float(worker['ell_epoxy_um'])-ell)<1e-12
    spectrum=seed_path.parent/'final_ibz153.csv'
    assert sha(spectrum)==worker['output_sha256'][spectrum.name]
    candidate=dict(run='initial_anchor',round=None,design=worker['design'],
        spectrum=locate(spectrum),spectrum_sha256=sha(spectrum),
        local_optimizer_success=True,**{k:worker['final_ibz153'][k] for k in ['L','U','J']})
    assert abs(candidate['J']-state['J153'])<1e-9
    inputs.update({locate(p):sha(p) for p in [seed_path,spectrum]})
    return candidate,'POSITIVE_SAMPLED_GAP_INITIAL_ANCHOR',worker['status'],inputs,None

def point_plan(k,gap):
    points={}
    def add(index,role,label):
        key=tuple(np.round(k[index],14))
        if key not in points:points[key]=dict(index=index,label=label,k_a=k[index].tolist(),roles=[])
        points[key]['roles'].append(role)
    for label,target in [('Gamma',(0.,0.)),('X',(np.pi,0.)),('M',(np.pi,np.pi))]:
        indices=np.flatnonzero(np.max(abs(k-np.asarray(target)),axis=1)<1e-13)
        assert len(indices)==1
        add(int(indices[0]),'symmetry_point',label)
    for role,indices in [('lower_controller',gap['lower_tied_indices']),('upper_controller',gap['upper_tied_indices'])]:
        for index in indices:add(index,role,f'{role}_{index}')
    return list(points.values())

def check_one(base,controller,ell,role,plan_only):
    candidate,classification,worker_status,inputs,summary=resolve_candidate(base,controller,ell)
    path=source_path(candidate['spectrum'])
    records,k,saved,gap=read_spectrum(path)
    assert all(abs(gap[key]-candidate[key])<1e-9 for key in ['L','U','J'])
    for key in ['lower_k_a','upper_k_a']:
        if key in candidate:assert np.max(abs(np.asarray(candidate[key])-np.asarray(gap[key])))<1e-13
    margin=float(controller['gap_sign_buffer_pp'])
    if classification.startswith('POSITIVE'):assert gap['J']>margin
    elif classification=='NO_POSITIVE_SAMPLED_GAP_FOUND':
        assert gap['J']<-margin and summary['all_required_starts_converged']
        assert all(r['status'].startswith('PASS') for r in summary['runs'])
    elif classification=='NEAR_ZERO_UNRESOLVED':assert abs(gap['J'])<=margin
    else:raise AssertionError(('Unexpected classification',classification))
    y,vertices,geometry=validate_geometry(candidate['design'])
    points=point_plan(k,gap)
    if plan_only:
        return dict(status='PLAN_ONLY_NO_EIGENSOLVES',ell_um=ell,role=role,classification=classification,
            witness_spectrum=locate(path),candidate_run=candidate['run'],candidate_round=candidate['round'],
            actual_J153=gap['J'],point_plan=points,geometry=geometry)
    out=base/'independent_endpoint_checks'/f'ell_{ell:g}'
    out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__)]+sorted((CODE/'src').glob('*.py'))+sorted((CODE.parent/'msse_20260919/python').glob('*.py'))
    source_hashes={locate(p):sha(p) for p in sources}
    snap=out/'source_snapshot';snap.mkdir()
    for i,p in enumerate(sources):shutil.copy2(p,snap/f'{i:03d}_{p.name}')
    # The controller may still poll during an explicitly requested case check.
    # Its snapshot is retained; immutable case/worker/witness inputs are guarded.
    controller_path=base/'manifest.json'
    controller_live=controller['status']=='RUNNING'
    controller_hash=sha(controller_path)
    if not controller_live:inputs[locate(controller_path)]=controller_hash
    norm=float(np.sqrt(4.35e9/1180)/(75e-6*2*np.pi*1e6))
    meta=dict(status='RUNNING',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),
        N=15,a_um=75.,h_um=6.,ell_epoxy_um=ell,ell_steel_um=0.,h_over_a=.08,ell_over_h=ell/6,
        endpoint_role=role,controller_classification=classification,source_worker_status=worker_status,
        gap_sign_buffer_pp=margin,buffer_semantics=controller['buffer_semantics'],
        witness_selection='case_summary.best_vetted_candidate; initial anchor only if no case exists atell3',
        candidate_stationarity_not_asserted=True,candidate=candidate,full153_gap=gap,
        point_plan=points,geometry=geometry,source_sha256=source_hashes,input_sha256=inputs,
        controller_manifest_observation=dict(path=locate(controller_path),sha256=controller_hash,live=controller_live),
        sparse_eigen_count=12,dense_eigen_count=8,first_frequencies_checked=6,
        sparse_shift=-.001,sparse_tolerance=1e-10,ncv=60,seed=20261030,threads=2,
        construction='K=classical_inverse(prepare(N,h/a,indicatorF),k)+(ell_um/h_um)**2*KB; M=assemble_real(...)[M]',
        normalization_MHz_per_Omega=norm,no_global_absence_proof=True,no_continuous_zone_certification=True,
        no_independent_physical_surface_or_interface_validation=True,new_COMSOL_runs=False,human_verified=False,
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
    save(out/'controller_snapshot.json',controller)
    save(out/'manifest.json',meta)
    np.savetxt(out/'witness_vertices.csv',vertices,delimiter=',',header='x_over_a,y_over_a',comments='')
    clock=time.perf_counter();point_checks=[]
    fields=['point_label','roles','source153_index','kx_a','ky_a','band','Omega_sparse','Omega_dense','Omega_saved',
            'f_sparse_MHz','relative_dense_difference','relative_saved_difference','relative_residual','rigid_translation']
    try:
        with threadpool_limits(2):
            meta['threadpool']=threadpool_info()
            F=fourier(y,15);cache=prepare(15,.08,theory='TSDT',indicator_matrix=F)
            with (out/'frequency_checks.csv').open('x',encoding='utf-8',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
                for point in points:
                    tick=time.perf_counter();wave=np.asarray(point['k_a']);index=point['index']
                    parts=assemble_real(15,wave,h=.08,theory='TSDT',indicator_matrix=F)
                    M=parts['M'];K=classical_inverse(cache,wave)+(ell/6)**2*parts['KB'];del parts
                    factor=cho_factor(K+.001*M,lower=True,check_finite=False)
                    inv=LinearOperator(K.shape,matvec=lambda v:cho_solve(factor,v,check_finite=False),dtype=float)
                    values,vectors=eigsh(K,k=12,M=M,sigma=-.001,OPinv=inv,tol=1e-10,ncv=60,
                        v0=np.random.default_rng(20261030).normal(size=len(K)))
                    order=np.argsort(values);values=values[order];vectors=vectors[:,order]
                    gamma=np.max(abs(wave))<1e-14;lower=1 if gamma else 0
                    assert np.all(values[lower:]>0)
                    if gamma:assert abs(values[0])<1e-7
                    sparse=np.sqrt(np.maximum(values,0))
                    Kv=K@vectors;Mv=M@vectors
                    residual=np.linalg.norm(Kv-Mv*values,axis=0)/(np.linalg.norm(Kv,axis=0)+abs(values)*np.linalg.norm(Mv,axis=0)+1e-30)
                    dense=np.sqrt(np.maximum(eigh(K,M,subset_by_index=(0,7),driver='gvx',eigvals_only=True,check_finite=False),0))
                    maxres=float(np.max(residual[lower:6]));orth=float(np.max(abs(vectors.T@Mv-np.eye(12))))
                    dense_drift=float(np.max(abs(sparse[lower:6]/dense[lower:6]-1)))
                    saved_drift=float(np.max(abs(sparse[lower:6]/saved[index,lower:6]-1)))
                    check=dict(**point,max_nonrigid_first6_relative_residual=maxres,mass_orthogonality_max=orth,
                        max_nonrigid_first6_dense_relative_drift=dense_drift,max_nonrigid_first6_saved_relative_drift=saved_drift,
                        solve_seconds=time.perf_counter()-tick)
                    point_checks.append(check);meta['point_checks']=point_checks
                    save(out/'manifest.json',meta)
                    for b in range(6):
                        rigid=gamma and b==0
                        writer.writerow(dict(point_label=point['label'],roles=';'.join(point['roles']),source153_index=index,
                            kx_a=float(wave[0]),ky_a=float(wave[1]),band=b+1,Omega_sparse=float(sparse[b]),
                            Omega_dense=float(dense[b]),Omega_saved=float(saved[index,b]),f_sparse_MHz=float(sparse[b]*norm),
                            relative_dense_difference=None if rigid else float(sparse[b]/dense[b]-1),
                            relative_saved_difference=None if rigid else float(sparse[b]/saved[index,b]-1),
                            relative_residual=None if rigid else float(residual[b]),rigid_translation=rigid))
                    handle.flush()
                    assert maxres<1e-7 and orth<1e-7 and dense_drift<1e-7 and saved_drift<1e-7,check
                    print('CHECK',ell,point['label'],json.dumps(check),flush=True)
                    del K,M,Kv,Mv,factor,inv,vectors
        assert all(sha(source_path(p))==h for p,h in source_hashes.items()),'Construction source changed'
        assert all(sha(source_path(p))==h for p,h in inputs.items()),'Immutable witness input changed'
        meta.update(status='PASS_INDEPENDENT_VETTED_ENDPOINT_MATRIX_CHECKS',source_hash_check='PASS',input_hash_check='PASS',
            point_count=len(points),max_nonrigid_first6_relative_residual=max(p['max_nonrigid_first6_relative_residual'] for p in point_checks),
            max_nonrigid_first6_dense_relative_drift=max(p['max_nonrigid_first6_dense_relative_drift'] for p in point_checks),
            max_nonrigid_first6_saved_relative_drift=max(p['max_nonrigid_first6_saved_relative_drift'] for p in point_checks),
            max_mass_orthogonality_error=max(p['mass_orthogonality_max'] for p in point_checks),elapsed_seconds=time.perf_counter()-clock,
            finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            output_sha256={p.name:sha(p) for p in out.glob('*.csv')})
        save(out/'manifest.json',meta)
        return dict(status=meta['status'],ell_um=ell,actual_ell_um=ell,role=role,classification=classification,
            candidate=candidate,input_sha256=inputs,source_sha256=source_hashes,
            J153=gap['J'],full153_gap=gap,point_count=len(points),checked_points=points,
            max_nonrigid_relative_residual=meta['max_nonrigid_first6_relative_residual'],
            max_dense_frequency_relative_delta=meta['max_nonrigid_first6_dense_relative_drift'],
            max_raw_frequency_relative_delta=meta['max_nonrigid_first6_saved_relative_drift'],
            max_mass_orthogonality_error=meta['max_mass_orthogonality_error'],
            manifest=locate(out/'manifest.json'),manifest_sha256=sha(out/'manifest.json'))
    except BaseException as exc:
        meta.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-clock)
        save(out/'manifest.json',meta);raise

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--ells',type=float,nargs='+')
    parser.add_argument('--plan-only',action='store_true',help='Resolve witnesses and checking points; no eigensolves or output folders')
    args=parser.parse_args();base=args.base.resolve();controller=load(base/'manifest.json')
    assert all(controller[k]==v for k,v in [('N',15),('a_um',75),('h_um',6),('ell_steel_um',0)])
    if args.ells:ells=args.ells
    else:
        assert controller['status']=='COMPLETED_FIRST_TESTED_NO_POSITIVE_GAP','Endpoint scan has not completed'
        ells=[float(controller['last_positive_tested_ell_um']),float(controller['first_no_positive_gap_tested_ell_um'])]
    assert len(ells)==len(set(ells)) and all(math.isfinite(e) and e>0 for e in ells)
    results=[]
    for ell in ells:
        role='explicit_case'
        if abs(ell-float(controller.get('last_positive_tested_ell_um',float('inf'))))<1e-12:role='last_positive_tested'
        if abs(ell-float(controller.get('first_no_positive_gap_tested_ell_um',float('inf'))))<1e-12:role='first_no_positive_gap_tested'
        results.append(check_one(base,controller,float(ell),role,args.plan_only))
    report=dict(status='PLAN_ONLY_NO_EIGENSOLVES' if args.plan_only else 'PASS_REQUESTED_ENDPOINT_CHECKS',endpoints=results,
        witness_is_not_necessarily_stationary=True,no_global_absence_proof=True,no_continuous_zone_certification=True,human_verified=False)
    if not args.plan_only:
        folder=base/'independent_endpoint_checks'
        with (folder/'manifest.json').open('x',encoding='utf-8') as h:h.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
