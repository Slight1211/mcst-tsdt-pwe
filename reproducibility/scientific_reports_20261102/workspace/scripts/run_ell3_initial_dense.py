"""Fresh153-point initial48gon baseline for same-grid ell3 gain comparisons.

Uses the existing corrected N15 frequency operator without changing any source
or search. All physical lengths are prescribed in micrometres; no calibration
or global/full-zone/three-dimensional certification is implied.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import traceback

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'scripts'))
import run_ell3_optimization as base
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info

base.ELL_UM=3.0
base.ELL_OVER_H=3.0/6.0
OUT=PROJECT/'05_validation/ELL3_exploration_20261031/initial_dense'
FROZEN=PROJECT/'05_validation/ELL3_exploration_20261031/frozen'
CODE=PROJECT/'04_code/research/wavemotion_surface_20260920'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))

def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))

def main():
    assert load(FROZEN/'manifest.json')['status']=='COMPLETED_SAMPLED_FROZEN_SCAN'
    sources=[Path(__file__),Path(base.__file__),
        PROJECT/'05_validation/V07_ell2_reoptimization_20261030/source/node_shape_sensitivity_v7.py',
        CODE/'run_node_shape_pilot.py']
    sources+=sorted((CODE/'src').glob('*.py'))
    sources+=sorted((CODE.parent/'msse_20260919/python').glob('*.py'))
    sources=list(dict.fromkeys(sources))
    inputs=[FROZEN/'initial_ell3_N15_ibz45.csv',FROZEN/'initial_vertices.csv',FROZEN/'manifest.json']
    source_hashes={p.relative_to(PROJECT).as_posix():sha(p) for p in sources}
    input_hashes={p.relative_to(PROJECT).as_posix():sha(p) for p in inputs}
    OUT.mkdir(parents=True,exist_ok=False)
    snap=OUT/'source_snapshot';snap.mkdir()
    for i,p in enumerate(sources):shutil.copy2(p,snap/f'{i:03d}_{p.name}')
    clock=time.perf_counter()
    meta=dict(status='RUNNING',pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        N=15,a_um=75.,h_um=6.,ell_epoxy_um=3.,ell_steel_um=0.,ell_over_h=.5,
        ell_squared_over_h_squared=.25,h_over_a=.08,points=153,grid=16,threads=2,
        shape='Equal-area regular48gon',design=[0.]*6,new_eigensolves=True,new_optimization=False,
        new_COMSOL_runs=False,human_verified=False,source_sha256=source_hashes,input_sha256=input_hashes,
        no_global_optimality_claim=True,no_continuous_BZ_claim=True,
        baseline_role='Same153-point set as frozen local-search terminal checks',
        frequency_normalization_MHz_per_Omega=float(np.sqrt(4.35e9/1180)/(75e-6*2*np.pi*1e6)))
    base.save_json(OUT/'manifest.json',meta)
    try:
        with threadpool_limits(2):
            meta['threadpool']=threadpool_info();base.save_json(OUT/'manifest.json',meta)
            ks=base.ibz(16)
            values=base.spectrum(np.zeros(6),ks,OUT,'initial_ibz153',N=15)
        assert values.shape==(153,6) and np.isfinite(values).all()
        assert np.all(values[:,1:]>0) and np.all(np.diff(values,axis=1)>=-1e-10)
        summary=base.sampled_gap(values)
        for side in ['L','U']:
            index=summary['active_'+side]
            summary['kx_'+side]=float(ks[index,0]);summary['ky_'+side]=float(ks[index,1])
        reference=rows(FROZEN/'initial_ell3_N15_ibz45.csv')
        dense=rows(OUT/'initial_ibz153.csv')
        key=lambda r:tuple(round(float(r[n]),12) for n in ['kx_a','ky_a'])
        lookup={key(r):r for r in dense};assert len(lookup)==153
        drifts=[]
        for r in reference:
            d=lookup[key(r)]
            start=2 if key(r)==(0.,0.) else 1
            drifts.extend(abs(float(d[f'Omega{i}'])/float(r[f'Omega{i}'])-1) for i in range(start,7))
        drift=max(drifts);assert drift<1e-7
        values45=np.array([[float(r[f'Omega{i}']) for i in range(1,7)] for r in reference])
        summary45=base.sampled_gap(values45)
        assert summary['J']<=summary45['J']+1e-8
        vertices=base.boundary(np.zeros(6))
        np.savetxt(OUT/'initial_vertices.csv',vertices,delimiter=',',header='x_over_a,y_over_a',comments='')
        assert all(sha(PROJECT/p)==h for p,h in source_hashes.items())
        assert all(sha(PROJECT/p)==h for p,h in input_hashes.items())
        meta.update(status='PASS_N15_DENSE_INITIAL_BASELINE',initial_ibz153=summary,initial_ibz45=summary45,
            dense_gap_drift_pp=summary['J']-summary45['J'],nested45_153_max_nonrigid_relative_drift=drift,
            nested45_points_checked=45,area_fraction=float(base.signed_area(vertices)),
            source_hash_check='PASS',input_hash_check='PASS',elapsed_seconds=time.perf_counter()-clock,
            finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            output_sha256={p.name:sha(p) for p in OUT.glob('*.csv')})
        base.save_json(OUT/'manifest.json',meta)
        print('INITIAL_DENSE_COMPLETED',json.dumps(dict(initial_ibz153=summary,initial_ibz45=summary45,
            nested_drift=drift,dense_gap_drift_pp=meta['dense_gap_drift_pp'])),flush=True)
    except BaseException as exc:
        meta.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-clock)
        base.save_json(OUT/'manifest.json',meta);raise

if __name__=='__main__':main()
