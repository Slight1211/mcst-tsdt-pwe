"""N15 nonnegative-length-scale local search on the full153point set.

Each output is exclusive. Every fully evaluated geometry is retained, including
positive witnesses from a failed or intermediate optimizer round. No global
absence of a gap is inferred from a local search.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,platform,shutil,sys,time,traceback
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
import run_lengthscale_optimization as base

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'04_code/research/wavemotion_surface_20260920'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
save=base.save_json

def export45(values,k,out,rid):
    chosen=[i for i,q in enumerate(k) if np.max(abs(q*8/np.pi-np.rint(q*8/np.pi)))<1e-12]
    assert len(chosen)==45
    with (out/f'round{rid}_ibz45.csv').open('x',encoding='utf-8',newline='') as h:
        w=csv.writer(h);w.writerow(['point','kx_a','ky_a']+[f'Omega{i}' for i in range(1,7)])
        for j,i in enumerate(chosen):w.writerow([j,*k[i],*values[i]])
    return base.sampled_gap(values[chosen])

def main():
    p=argparse.ArgumentParser();p.add_argument('--ell-um',type=float,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed-manifest',type=Path)
    p.add_argument('--maxiter',type=int,default=100);p.add_argument('--max-rounds',type=int,default=16)
    args=p.parse_args()
    assert np.isfinite(args.ell_um) and args.ell_um>=0
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    base.ELL_UM=float(args.ell_um);base.ELL_OVER_H=base.ELL_UM/6.
    seed_path=args.seed_manifest.resolve() if args.seed_manifest else None
    seed_meta=json.loads(seed_path.read_text(encoding='utf-8')) if seed_path else None
    y0=np.asarray(seed_meta['design'] if seed_meta else np.zeros(6),dtype=float)
    assert y0.shape==(6,) and np.isfinite(y0).all() and max(abs(y0))<=.16+1e-12
    seed_vertices=base.boundary(y0)
    assert abs(base.signed_area(seed_vertices)-base.AREA)<1e-13
    k=base.ibz(16)
    active={i for i,q in enumerate(k) if any(np.max(abs(q-np.asarray(t)))<1e-14 for t in [(0.,0.),(np.pi,0.),(np.pi,np.pi)])}
    assert len(k)==153 and active=={0,136,152}
    sources=[Path(__file__),ROOT/'scripts/run_gap_closure_case.py',Path(base.__file__),
        ROOT/'05_validation/V07_ell2_reoptimization_20261030/source/node_shape_sensitivity_v7.py',
        CODE/'run_node_shape_pilot.py']+list((CODE/'src').glob('*.py'))+list((CODE.parent/'msse_20260919/python').glob('*.py'))
    hashes={str(x.relative_to(ROOT)):sha(x) for x in sources}
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for i,x in enumerate(sources):shutil.copy2(x,snapshot/f'{i:03d}_{x.name}')
    meta=dict(status='RUNNING',pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        N=15,a_um=75.,h_um=6.,ell_epoxy_um=base.ELL_UM,ell_steel_um=0.,ell_over_h=base.ELL_OVER_H,
        vertices=48,shape_variables=6,shape_bounds=[-.16,.16],fixed_area_fraction=base.AREA,
        optimization_points=153,optimization_grid=16,active_start_indices=sorted(active),
        ftol=1e-7,feasibility_tolerance=1e-6,projected_lagrangian_tolerance=1e-3,
        complementarity_tolerance=1e-4,max_same_tolerance_resets=2,
        maxiter=args.maxiter,max_rounds=args.max_rounds,threads=2,source_sha256=hashes,
        seed_source=seed_path.relative_to(ROOT).as_posix() if seed_path else 'regular48gon',
        seed_source_sha256=sha(seed_path) if seed_path else None,seed_design=y0.tolist(),
        seed_semantics='Geometry only; all frequencies at the new ell are freshly solved.',
        full153_candidates=[],environment=dict(python=sys.version,platform=platform.platform()),
        no_global_optimality_claim=True,no_continuous_BZ_certification=True,new_COMSOL_runs=False,human_verified=False)
    save(out/'manifest.json',meta)
    np.savetxt(out/'seed_vertices.csv',seed_vertices,delimiter=',',header='x_over_a,y_over_a',comments='')
    clock=time.perf_counter();history=[];rounds=[];start=y0.copy();resets=0
    try:
        with threadpool_limits(limits=2):
            meta['threadpool']=threadpool_info();meta['gradient_gates']=[base.gradient_gate(start,'seed')]
            save(out/'manifest.json',meta)
            for rid in range(1,args.max_rounds+1):
                idx=sorted(active);ev=base.LoggedEvaluator(15,k[idx],base.ELL_OVER_H,h_over_a=.08)
                local=base.optimize_round(ev,start,rid,out,history,clock,args.maxiter)
                y=np.asarray(local['design']);vertices=base.boundary(y)
                assert max(abs(y))<=.16+1e-10 and abs(base.signed_area(vertices)-base.AREA)<1e-13
                values=base.spectrum(y,k,out,f'round{rid}_ibz153')
                full=base.sampled_gap(values);coarse=export45(values,k,out,rid)
                L,U=local['x'][-2:]
                violation=float(max(0.,full['L']-L,U-full['U']))
                all_violating=set(map(int,np.where((values[:,3]>L+1e-6)|(values[:,4]<U-1e-6))[0]))
                extrema=set()
                if full['L']>L+1e-6:
                    extrema.update(map(int,np.flatnonzero(values[:,3]>=full['L']-1e-10)))
                if full['U']<U-1e-6:
                    extrema.update(map(int,np.flatnonzero(values[:,4]<=full['U']+1e-10)))
                new=extrema-active
                rec=dict(optimizer=local,active_indices=idx,new_indices=sorted(new),full=full,
                    full_constraint_violation=violation,subset45=coarse,
                    all_violating_indices=sorted(all_violating),cut_rule='Mostviolated lower/upper constraints including numericalties; existing activeconstraints retained.')
                rounds.append(rec)
                candidate=dict(round=rid,design=y.tolist(),J=full['J'],L=full['L'],U=full['U'],
                    area_fraction=float(base.signed_area(vertices)),bounds_valid=True,
                    spectrum=f'round{rid}_ibz153.csv',optimizer_success=local['success'])
                meta['full153_candidates'].append(candidate)
                meta.update(rounds=rounds,elapsed_seconds=time.perf_counter()-clock)
                save(out/'manifest.json',meta)
                print('FULL153',rid,'J',full['J'],'violation',violation,'newcuts',sorted(new),flush=True)
                kkt=(local['projected_lagrangian_gradient_inf']<1e-3 and local['complementarity_inf']<1e-4)
                if local['success'] and local['constraint_violation']<1e-6 and violation<1e-6 and kkt:
                    break
                if new:
                    active.update(new);start=y;continue
                if resets<2:
                    resets+=1;start=y
                    meta.setdefault('resets',[]).append(dict(after_round=rid,same_ftol=1e-7,
                        reason='Reset SLSQP Hessian and auxiliary edges at saved geometry; no unseen full-grid cut.'))
                    save(out/'manifest.json',meta);continue
                raise ArithmeticError('Local optimizer did not satisfy full-grid feasibility/KKT checks; do not classify as no gap.')
            else:raise ArithmeticError('Maximum active-set rounds exceeded; not a no-gap result.')
            # If the selected terminal is near/non-positive, explicitly evaluate the
            # incoming seed at this ell so a positive witness cannot be discarded.
            if full['J']<=.001:
                seed_values=base.spectrum(y0,k,out,'seed_ibz153')
                seed_gap=base.sampled_gap(seed_values)
                meta['seed_ibz153']=seed_gap
                meta['full153_candidates'].append(dict(round=0,design=y0.tolist(),J=seed_gap['J'],L=seed_gap['L'],U=seed_gap['U'],
                    area_fraction=float(base.signed_area(seed_vertices)),bounds_valid=True,spectrum='seed_ibz153.csv',optimizer_success=None))
            meta['gradient_gates'].append(base.gradient_gate(y,'terminal'))
            np.savetxt(out/'final_vertices.csv',vertices,delimiter=',',header='x_over_a,y_over_a',comments='')
            shutil.copy2(out/f'round{rid}_ibz153.csv',out/'final_ibz153.csv')
            shutil.copy2(out/f'round{rid}_ibz45.csv',out/'final_ibz45.csv')
            best=max(meta['full153_candidates'],key=lambda c:c['J'])
            assert all(sha(ROOT/path)==digest for path,digest in hashes.items()),'source changed during run'
            assert seed_path is None or sha(seed_path)==meta['seed_source_sha256'],'seed changed during run'
            meta.update(status='PASS_SAMPLED_N15_GRID16_LOCAL_SQP',design=y.tolist(),final_ibz153=full,
                final_ibz45=coarse,subset45_semantics='Nested subset exported from the153point solves; not the optimization grid.',
                area_fraction=float(base.signed_area(vertices)),accepted_iterations=sum(r['optimizer']['iterations'] for r in rounds),
                best_evaluated153=best,elapsed_seconds=time.perf_counter()-clock,source_hash_check='PASS',
                finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
            save(out/'manifest.json',meta)
            print('CASE_COMPLETE',base.ELL_UM,'J153',full['J'],'best_evaluated',best['J'],flush=True)
    except BaseException as exc:
        meta.update(status='FAILED_OPTIMIZATION',error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-clock)
        save(out/'manifest.json',meta);raise

if __name__=='__main__':main()
