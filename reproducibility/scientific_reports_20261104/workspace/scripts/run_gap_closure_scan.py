"""Increase prescribed epoxy ell; optimize each state before classifying its gap.

The reported endpoint means no positive sampled interval was found by the listed
local searches. It is not a proof that no admissible shape has a band gap.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,os,subprocess,sys,time,traceback
from audit_ell3_trial_outputs import spectrum as read_spectrum,gap,reconstructed_polygon,polygon_area

ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'scripts/run_gap_closure_case.py'
BASE=ROOT/'05_validation/GAP_CLOSURE_SCAN_20261031'
REVIEW=ROOT/'03_review/gap_closure_scan_20261031'
SHA=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
MARGIN=.001

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):
    q=Path(str(p)+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def locate(p):return p.relative_to(ROOT).as_posix()

def vet_run(folder,expected_ell):
    m=load(folder/'manifest.json')
    assert (m['N'],m['a_um'],m['h_um'],m['ell_steel_um'],m['optimization_points'])==(15,75.,6.,0.,153)
    assert abs(m['ell_epoxy_um']-expected_ell)<1e-12 and abs(m['ell_over_h']-expected_ell/6)<1e-14
    assert all(SHA(ROOT/p)==digest for p,digest in m['source_sha256'].items())
    if m['seed_source']!='regular48gon':assert SHA(ROOT/m['seed_source'])==m['seed_source_sha256']
    checked=[]
    for c in m.get('full153_candidates',[]):
        y=c['design'];assert len(y)==6 and all(math.isfinite(v) and abs(v)<=.16+1e-10 for v in y)
        polygon=reconstructed_polygon(y)
        assert abs(polygon_area(polygon)-math.pi*.09)<1e-12
        assert max(abs(x) for point in polygon for x in point)<.49
        path=(folder/c['spectrum']).resolve();assert path.is_relative_to(folder.resolve())
        if m['status'].startswith('PASS'):assert SHA(path)==m['output_sha256'][path.name]
        actual=gap(read_spectrum(path,16))
        assert all(abs(actual[k]-c[k])<1e-9 for k in ['J','L','U'])
        checked.append(dict(run=folder.name,round=c['round'],design=y,J=actual['J'],L=actual['L'],U=actual['U'],
            lower_k_a=actual['lower_k_a'],upper_k_a=actual['upper_k_a'],
            spectrum=locate(path),spectrum_sha256=SHA(path),local_optimizer_success=c['optimizer_success']))
    if m['status'].startswith('PASS'):
        assert checked
        final=gap(read_spectrum(folder/'final_ibz153.csv',16))
        assert abs(final['J']-m['final_ibz153']['J'])<1e-10
        assert m['rounds'][-1]['optimizer']['success'] and m['rounds'][-1]['full_constraint_violation']<1e-6
    return m,checked

def launch_batch(case,ell,specs,meta):
    processes=[];folders=[]
    try:
        for label,seed in specs:
            folder=case/label;folders.append(folder)
            if (folder/'manifest.json').exists():
                existing=load(folder/'manifest.json')
                if existing['status'].startswith('PASS'):
                    assert abs(existing['ell_epoxy_um']-ell)<1e-12
                    if seed is None:assert existing['seed_source']=='regular48gon'
                    else:assert existing['seed_source']==locate(seed) and existing['seed_source_sha256']==SHA(seed)
                    continue
                raise RuntimeError(f'Existing unfinished/failed result needs explicit review, not overwrite: {folder}')
            log=(case/(label+'.console.log')).open('x',encoding='utf-8')
            cmd=[sys.executable,'-u','-X','utf8',str(WORKER),'--ell-um',str(ell),'--out',str(folder)]
            if seed is not None:cmd+=['--seed-manifest',str(seed)]
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'PYTHONUNBUFFERED':'1'})
            processes.append((process,log,label))
            meta.setdefault('launches',[]).append(dict(ell_um=ell,label=label,pid=process.pid,command=cmd))
            save(BASE/'manifest.json',meta)
            print('LAUNCHED',ell,label,'PID',process.pid,flush=True)
        while any(p.poll() is None for p,_,_ in processes):
            meta.update(current_ell_um=ell,last_poll_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            save(BASE/'manifest.json',meta);time.sleep(5)
        for p,log,label in processes:
            log.close();print('WORKER_EXIT',ell,label,p.returncode,flush=True)
    finally:
        for p,log,_ in processes:
            if p.poll() is None:p.terminate();p.wait(timeout=30)
            if not log.closed:log.close()
    return folders

def inspect(folders,ell):
    records=[];witnesses=[]
    for folder in folders:
        m,candidates=vet_run(folder,ell);witnesses+=candidates
        records.append(dict(run=folder.name,folder=locate(folder),status=m['status'],
            final_J153=m.get('final_ibz153',{}).get('J'),
            iterations=m.get('accepted_iterations'),best_vetted_J153=max((c['J'] for c in candidates),default=None)))
    return records,witnesses

def export_summary(meta):
    rows=[]
    start=meta['initial_state']
    rows.append(dict(ell_um=start['ell_um'],best_converged_J153=start['J153'],best_vetted_J153=start['J153'],
        classification='POSITIVE_SAMPLED_GAP',selected_run=start['seed_manifest']))
    for c in meta.get('cases',[]):
        rows.append(dict(ell_um=c['ell_um'],best_converged_J153=c.get('best_converged_J153'),
            best_vetted_J153=c.get('best_vetted_J153'),classification=c['classification'],selected_run=c.get('selected_run')))
    with (BASE/'scan_summary.csv').open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--start-ell',type=float,default=4.)
    p.add_argument('--step',type=float,default=1.);p.add_argument('--max-ell',type=float,default=12.)
    p.add_argument('--seed-manifest',type=Path)
    args=p.parse_args();assert args.step>0 and args.start_ell>0 and args.max_ell>=args.start_ell
    BASE.mkdir(parents=True,exist_ok=True);REVIEW.mkdir(parents=True,exist_ok=True)
    if (BASE/'manifest.json').exists():raise FileExistsError('A scan manifest already exists; inspect/resume explicitly, never overwrite.')
    if args.seed_manifest:
        seed=args.seed_manifest.resolve();source=load(seed);assert source['status'].startswith('PASS')
    else:
        root3=ROOT/'05_validation/ELL3_exploration_20261031'
        while True:
            paths=[root3/name/'manifest.json' for name in ['optimization','warmstart']]
            if all(p.exists() and load(p)['status'].startswith('PASS') for p in paths):break
            print('WAITING_FOR_COMPLETED_ELL3_STARTS',flush=True);time.sleep(15)
        seed=max(paths,key=lambda p:load(p)['final_ibz153']['J']);source=load(seed)
    assert source['N']==15 and source['a_um']==75 and source['h_um']==6 and source['ell_steel_um']==0
    assert source['final_ibz153']['J']>MARGIN and source['ell_epoxy_um']<args.start_ell
    meta=dict(status='RUNNING',pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        N=15,a_um=75,h_um=6,ell_steel_um=0,vertices=48,variables=6,shape_bounds=[-.16,.16],optimization_points=153,
        start_ell_um=args.start_ell,step_um=args.step,max_ell_um=args.max_ell,gap_sign_buffer_pp=MARGIN,
        buffer_semantics='Practical reporting buffer, not a certified numerical error bound.',
        initial_state=dict(ell_um=source['ell_epoxy_um'],J153=source['final_ibz153']['J'],seed_manifest=locate(seed),seed_sha256=SHA(seed)),
        case_worker_sha256=SHA(WORKER),controller_sha256=SHA(Path(__file__)),cases=[],
        no_global_absence_proof=True,no_continuous_zone_certification=True,no_monotonicity_assumption=True,
        stopping_rule='First testedell with no positive vetted full153candidate, all required starts converged, bestJ<-0.001pp; extraell1/ell2starts at proposedendpoint.',
        optimizer_failures_never_count_as_no_gap=True,new_COMSOL_runs=False,human_verified=False)
    save(BASE/'manifest.json',meta);export_summary(meta)
    last_positive=float(source['ell_epoxy_um'])
    extras=[('ell1_seed',ROOT/'02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1/manifest.json'),
            ('ell2_seed',ROOT/'05_validation/V07_ell2_reoptimization_20261030/restart/manifest.json')]
    try:
        ell=float(args.start_ell)
        while ell<=args.max_ell+1e-12:
            assert SHA(WORKER)==meta['case_worker_sha256'] and SHA(Path(__file__))==meta['controller_sha256']
            case=BASE/f'ell_{ell:g}';case.mkdir(exist_ok=True)
            meta['current_ell_um']=ell;save(BASE/'manifest.json',meta)
            folders=launch_batch(case,ell,[('regular',None),('warm',seed)],meta)
            records,witnesses=inspect(folders,ell)
            best_witness=max(witnesses,key=lambda c:c['J']) if witnesses else None
            # Before a proposed endpoint, try the two previously useful shapes too.
            if best_witness is None or best_witness['J']<=MARGIN:
                extra_folders=launch_batch(case,ell,extras,meta);folders+=extra_folders
                records,witnesses=inspect(folders,ell)
                best_witness=max(witnesses,key=lambda c:c['J']) if witnesses else None
            converged=[r for r in records if r['status'].startswith('PASS')]
            if not converged:raise ArithmeticError(f'No converged local search atell{ell}; cannot classify as no gap.')
            best=max(converged,key=lambda r:r['final_J153'])
            # A verified positive intermediate geometry cannot be erased just
            # because a local solver ended at a poorer design.
            if best_witness and best_witness['J']>best['final_J153']+MARGIN:
                witness_seed=case/'witness_seed.json'
                save(witness_seed,dict(status='VERIFIED_FULL153_GEOMETRY',ell_epoxy_um=ell,
                    design=best_witness['design'],spectrum=best_witness['spectrum'],spectrum_sha256=best_witness['spectrum_sha256']))
                folders+=launch_batch(case,ell,[('witness_refinement',witness_seed)],meta)
                records,witnesses=inspect(folders,ell);best_witness=max(witnesses,key=lambda c:c['J'])
                converged=[r for r in records if r['status'].startswith('PASS')]
                best=max(converged,key=lambda r:r['final_J153'])
            has_positive=best_witness is not None and best_witness['J']>MARGIN
            all_pass=all(r['status'].startswith('PASS') for r in records)
            if has_positive:classification='POSITIVE_SAMPLED_GAP_FOUND'
            elif all_pass and best_witness and best_witness['J']<-MARGIN:classification='NO_POSITIVE_SAMPLED_GAP_FOUND'
            elif all_pass:classification='NEAR_ZERO_UNRESOLVED'
            else:raise ArithmeticError(f'Failedrequiredstart atell{ell}; a no-gap endpoint is not justified.')
            rec=dict(ell_um=ell,classification=classification,runs=records,
                best_converged_J153=best['final_J153'],best_vetted_J153=best_witness['J'] if best_witness else None,
                best_vetted_candidate=best_witness,selected_run=best['folder'],all_required_starts_converged=all_pass)
            meta['cases'].append(rec);save(case/'case_summary.json',rec)
            save(BASE/'manifest.json',meta);export_summary(meta)
            print('ELL_RESULT',ell,classification,'J153',best['final_J153'],'best_vetted',rec['best_vetted_J153'],flush=True)
            if classification=='NO_POSITIVE_SAMPLED_GAP_FOUND':
                meta.update(status='COMPLETED_FIRST_TESTED_NO_POSITIVE_GAP',last_positive_tested_ell_um=last_positive,
                    first_no_positive_gap_tested_ell_um=ell,
                    bracket_semantics='Lasttested positive state and firsttested no-positive state; intervening unresolved states are retained. Not a certified criticalvalue or proof over all shapes.',
                    finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
                save(BASE/'manifest.json',meta);export_summary(meta);print('SCAN_STOPPED_AT_VERIFIED_LOCAL_SEARCH_ENDPOINT',ell,flush=True);return
            if has_positive:last_positive=ell
            seed=case/'continuation_seed.json'
            save(seed,dict(status='PASS_VERIFIED_FULL153_GEOMETRY',N=15,a_um=75,h_um=6,ell_steel_um=0,ell_epoxy_um=ell,
                design=best_witness['design'],final_ibz153=dict(J=best_witness['J'],L=best_witness['L'],U=best_witness['U']),
                witness_spectrum=best_witness['spectrum'],witness_spectrum_sha256=best_witness['spectrum_sha256'],
                geometry_selection='Largest independentlyvetted full153interval among all evaluatedgeometries, not necessarily a stationary/globaloptimum.',
                source_candidate_run=best_witness['run'],source_candidate_round=best_witness['round']))
            ell=round(ell+args.step,12)
        meta['status']='MAX_RANGE_REACHED_REQUIRES_EXTENSION';save(BASE/'manifest.json',meta)
        print('RANGE_END_WITHOUT_CLOSURE_EXTEND_SCAN',flush=True)
    except BaseException as exc:
        meta.update(status='NEEDS_REVIEW_NOT_NO_GAP',error=repr(exc),traceback=traceback.format_exc())
        save(BASE/'manifest.json',meta);raise

if __name__=='__main__':main()
