"""Saved-file audit of ell0 multistart; no eigensolver imports or calls.

The saved-run checker below is copied from audit_gap_closure_case.py with only
its expected initial active set parameterized. Original archived audit code is
unchanged. Completed but unregistered full-grid candidates are recovered too.
"""
from pathlib import Path
import argparse, csv, datetime, hashlib, json, math, re, time, traceback
from audit_gap_closure_case import (ROOT, TARGET_AREA, load, sha, rows, area,
    polygon, validate_geometry, spectrum, compare_gap)

BASE=ROOT/'05_validation/ELL0_MULTISTART_20261101'
REVIEW=ROOT/'03_review/ell0_multistart_20261101'
INITIAL=(0,27,136,152)

def check_run(folder,ell,expected_active=(0,27,136,152)):
    m=load(folder/'manifest.json')
    assert (m['N'],m['a_um'],m['h_um'],m['ell_steel_um'],m['optimization_points'],m['optimization_grid'])==(15,75.,6.,0.,153,16)
    assert m['vertices']==48 and m['shape_variables']==6 and m['shape_bounds']==[-.16,.16] and m['threads']==2
    assert abs(m['fixed_area_fraction']-TARGET_AREA)<1e-13
    assert abs(m['ell_epoxy_um']-ell)<1e-12 and abs(m['ell_over_h']-ell/6)<1e-14
    assert m['active_start_indices']==list(expected_active) and m['ftol']==1e-7
    assert m['feasibility_tolerance']==1e-6 and m['projected_lagrangian_tolerance']==1e-3 and m['complementarity_tolerance']==1e-4
    assert all(sha(ROOT/name)==digest for name,digest in m['source_sha256'].items())
    for i,(name,digest) in enumerate(m['source_sha256'].items()):
        assert sha(folder/'source_snapshot'/f'{i:03d}_{Path(name).name}')==digest
    assert all(sha(folder/name)==digest for name,digest in m.get('output_sha256',{}).items())
    if m['seed_source']=='regular48gon':assert max(abs(v) for v in m['seed_design'])==0
    else:
        assert sha(ROOT/m['seed_source'])==m['seed_source_sha256']
        assert m['seed_design']==load(ROOT/m['seed_source'])['design']
    validate_geometry(m['seed_design'])
    observed={};prior=None
    for rid,rec in enumerate(m.get('rounds',[]),1):
        actual=spectrum(folder/f'round{rid}_ibz153.csv');observed[rid]=actual
        compare_gap(actual,rec['full'])
        assert actual['active_L']==rec['full']['active_L'] and actual['active_U']==rec['full']['active_U']
        opt=load(folder/f'round{rid}_optimizer.json');assert opt==rec['optimizer']
        validate_geometry(opt['design'])
        active=set(rec['active_indices'])
        assert active==(set(expected_active) if prior is None else set(prior['active_indices'])|set(prior['new_indices']))
        L,U=opt['x'][-2:]
        vio=max(0.,actual['L']-L,U-actual['U'])
        assert abs(vio-rec['full_constraint_violation'])<1e-11
        allbad={i for i,f in enumerate(actual['f']) if f[3]>L+1e-6 or f[4]<U-1e-6}
        assert allbad==set(rec['all_violating_indices'])
        extrema=set()
        if actual['L']>L+1e-6:extrema|={i for i,f in enumerate(actual['f']) if f[3]>=actual['L']-1e-10}
        if actual['U']<U-1e-6:extrema|={i for i,f in enumerate(actual['f']) if f[4]<=actual['U']+1e-10}
        assert set(rec['new_indices'])==extrema-active
        prior=rec
    candidates=[]
    for c in m['full153_candidates']:
        validate_geometry(c['design']);actual=spectrum(folder/c['spectrum']);compare_gap(actual,c)
        assert abs(c['area_fraction']-TARGET_AREA)<1e-13 and c['bounds_valid']
        if c['round']>0:
            assert c['design']==m['rounds'][c['round']-1]['optimizer']['design']
            assert c['optimizer_success']==m['rounds'][c['round']-1]['optimizer']['success']
        else:assert c['design']==m['seed_design'] and c['optimizer_success'] is None
        candidates.append(dict(run=folder.name,round=c['round'],J=actual['J'],L=actual['L'],U=actual['U'],
            design=c['design'],spectrum=c['spectrum'],spectrum_sha256=sha(folder/c['spectrum'])))
    if not m['status'].startswith('PASS'):
        final=None
        if (folder/'final_ibz153.csv').exists() and 'final_ibz153' in m:
            final=spectrum(folder/'final_ibz153.csv');compare_gap(final,m['final_ibz153'])
        return dict(status='PASS_AVAILABLE_CANDIDATE_ARITHMETIC_ONLY',run=folder.name,run_status=m['status'],
            J153=final['J'] if final else None,rounds=len(m.get('rounds',[])),candidates=candidates,
            best_J153=max((c['J'] for c in candidates),default=None),source_hashes=len(m['source_sha256']),
            output_hashes=len(m.get('output_sha256',{})),manifest_sha256=sha(folder/'manifest.json'))
    last=m['rounds'][-1];local=last['optimizer']
    assert local['success'] and local['constraint_violation']<1e-6 and last['full_constraint_violation']<1e-6
    assert local['projected_lagrangian_gradient_inf']<1e-3 and local['complementarity_inf']<1e-4
    assert m['design']==local['design']
    final=spectrum(folder/'final_ibz153.csv');compare_gap(final,m['final_ibz153'])
    assert final['f']==observed[len(m['rounds'])]['f'] and final['k']==observed[len(m['rounds'])]['k']
    coarse=spectrum(folder/'final_ibz45.csv',8);compare_gap(coarse,m['final_ibz45'])
    lookup={tuple(round(t,12) for t in q):f for q,f in zip(final['k'],final['f'])}
    assert all(f==lookup[tuple(round(t,12) for t in q)] for q,f in zip(coarse['k'],coarse['f']))
    p=validate_geometry(m['design']);saved=rows(folder/'final_vertices.csv')
    assert len(saved)==48
    drift=max(abs(a-float(r[key])) for q,r in zip(p,saved) for a,key in zip(q,['x_over_a','y_over_a']))
    assert drift<1e-13 and abs(area(p)-m['area_fraction'])<1e-13
    gates=m['gradient_gates'];assert len(gates)==2
    for gate in gates:
        assert gate['status']=='PASS'
        assert len(gate['direction'])==6 and abs(sum(v*v for v in gate['direction'])-1)<1e-12
        for item in gate['records']:
            norm=math.sqrt(sum((a-b)**2 for a,b in zip(item['analytic'],item['central'])))/max(math.sqrt(sum(x*x for x in item['central'])),1e-12)
            assert abs(norm-item['relative_error'])<1e-12 and norm<1e-4
    best=max(candidates,key=lambda c:c['J']);compare_gap(best,m['best_evaluated153'])
    assert m['best_evaluated153']==max(m['full153_candidates'],key=lambda c:c['J'])
    assert m['accepted_iterations']==sum(r['optimizer']['iterations'] for r in m['rounds'])
    return dict(status='PASS_INDEPENDENT_SAVED_RUN_AUDIT',run=folder.name,run_status=m['status'],J153=final['J'],L=final['L'],U=final['U'],
        rounds=len(m['rounds']),candidates=candidates,best_J153=best['J'],source_hashes=len(m['source_sha256']),
        output_hashes=len(m['output_sha256']),geometry_max_coordinate_drift=drift,
        KKT=dict(projected_lagrangian_gradient_inf=local['projected_lagrangian_gradient_inf'],complementarity_inf=local['complementarity_inf']),
        manifest_sha256=sha(folder/'manifest.json'))


def inspect_run(folder, expected_active, controller_record=None):
    result=check_run(folder,0.,expected_active)
    m=load(folder/'manifest.json')
    assert m['status']!='RUNNING'
    if m['status'].startswith('PASS'):
        assert m['source_hash_check']=='PASS'
    recorded={c['spectrum'] for c in result['candidates']}
    orphan=[];incomplete=[]
    for f in sorted(folder.glob('*ibz153.csv')):
        if f.name=='final_ibz153.csv' or f.name in recorded:continue
        if f.name=='seed_ibz153.csv':
            rid=0;design=m['seed_design'];optimizer_success=None
        else:
            match=re.fullmatch(r'round(\d+)_ibz153\.csv',f.name)
            if not match:continue
            rid=int(match.group(1));op=folder/f'round{rid}_optimizer.json'
            if not op.exists():
                incomplete.append(dict(spectrum=f.name,reason='No saved optimizer geometry; no physical candidate assigned.'));continue
            q=load(op);design=q['design'];optimizer_success=q['success']
        count=len(rows(f))
        if count!=153:
            incomplete.append(dict(spectrum=f.name,rows=count,reason='Incomplete grid; excluded from candidate comparisons.'));continue
        validate_geometry(design);actual=spectrum(f)
        candidate=dict(run=folder.name,round=rid,J=actual['J'],L=actual['L'],U=actual['U'],
            design=design,spectrum=f.name,spectrum_sha256=sha(f),
            recovered_unregistered_complete_candidate=True,optimizer_success=optimizer_success)
        result['candidates'].append(candidate);orphan.append(candidate)
    for candidate in result['candidates']:
        candidate['folder']=folder.relative_to(ROOT).as_posix()
        candidate['local_optimizer_success']=(m['rounds'][candidate['round']-1]['optimizer']['success']
            if candidate['round']>0 and candidate['round']<=len(m.get('rounds',[])) else candidate.get('optimizer_success'))
    result['best_J153']=max((c['J'] for c in result['candidates']),default=None)
    result.update(folder=folder.relative_to(ROOT).as_posix(),ell_um=0.,
        expected_initial_active_indices=list(expected_active),recovered_complete_candidates=orphan,
        incomplete_artifacts=incomplete,manifest_sha256=sha(folder/'manifest.json'))
    if controller_record:
        assert controller_record['status']==m['status']
        assert controller_record['manifest_sha256']==result['manifest_sha256']
        assert controller_record.get('seed_sha256')==m['seed_source_sha256']
        assert controller_record['exit_code']==0 or not m['status'].startswith('PASS'),'Nonzero exit with PASS manifest requires review.'
        if m['status'].startswith('PASS'):
            assert abs(controller_record['J153']-result['J153'])<1e-10
            assert abs(controller_record['best_evaluated_J153']-m['best_evaluated153']['J'])<1e-10
        result['exit_code']=controller_record['exit_code']
    if m['status'].startswith('PASS'):
        result.update(final_vertices=(folder/'final_vertices.csv').relative_to(ROOT).as_posix(),
            final_vertices_sha256=sha(folder/'final_vertices.csv'),design=m['design'])
        final=spectrum(folder/'final_ibz153.csv')
        result.update(terminal_lower_k_a=final['k'][final['active_L']],terminal_upper_k_a=final['k'][final['active_U']],
            gradient_gate_max_relative_error=max(q['relative_error'] for g in m['gradient_gates'] for q in g['records']))
    return result


def audit_once(refresh=False,quiet=False):
    REVIEW.mkdir(parents=True,exist_ok=True);(REVIEW/'runs').mkdir(exist_ok=True)
    meta=load(BASE/'manifest.json')
    assert (meta['N'],meta['a_um'],meta['h_um'],meta['ell_epoxy_um'],meta['ell_steel_um'])==(15,75,6,0,0)
    assert meta['shape_bounds']==[-.16,.16] and meta['active_start_indices']==list(INITIAL)
    assert meta['new_starts']==16 and meta['previous_starts']==2 and len(meta['seeds'])==16 and len(meta['baseline'])==2
    assert meta['optimization_points']==153 and meta['vertices']==48 and meta['shape_variables']==6
    assert abs(meta['fixed_area_fraction']-TARGET_AREA)<1e-13
    assert meta['no_global_optimality_claim'] and not meta['new_COMSOL_runs']
    assert sha(ROOT/'scripts/run_ell0_multistart.py')==meta['controller_sha256']
    assert sha(ROOT/'scripts/run_ell0_multistart_case.py')==meta['worker_sha256']
    assert all(sha(ROOT/p)==h for p,h in meta['input_sha256'].items())
    seedmap={s['label']:s for s in meta['seeds']};assert len(seedmap)==16
    for s in seedmap.values():
        assert sha(ROOT/s['path'])==s['sha256'];validate_geometry(s['design'])
    planned=[dict(kind='baseline',label=b['label'],folder=(ROOT/b['manifest']).parent,record=b) for b in meta['baseline']]
    for r in meta['runs']:
        assert r['label'] in seedmap
        assert ROOT/r['out']==BASE/r['label']
        if r['status']=='RUNNING':continue
        planned.append(dict(kind='new_start',label=r['label'],folder=ROOT/r['out'],record=r))
    assert len({r['label'] for r in meta['runs']})==len(meta['runs'])
    audits=[];newly=[];issues=[]
    for item in planned:
        folder=item['folder'];path=folder/'manifest.json';r=item['record'];label=item['label']
        cache=REVIEW/'runs'/f'{label}.json'
        try:
            assert path.exists(),'Completed record has no manifest.'
            if item['kind']=='baseline':assert sha(path)==r['manifest_sha256']
            else:
                w=load(path);s=seedmap[label]
                assert w['seed_source']==s['path'] and w['seed_source_sha256']==s['sha256'] and w['seed_design']==s['design']
            cached=load(cache) if cache.exists() else {}
            reuse=(not refresh and cached.get('manifest_sha256')==sha(path)
                and cached.get('audit_script_sha256')==sha(Path(__file__)))
            if reuse:
                result=load(cache)
                # Verify immutable bindings even when the full arithmetic audit is cached.
                w=load(path)
                assert all(sha(ROOT/n)==h for n,h in w['source_sha256'].items())
                assert all(sha(folder/n)==h for n,h in w.get('output_sha256',{}).items())
            else:
                result=inspect_run(folder,(0,136,152) if item['kind']=='baseline' else INITIAL,
                    r if item['kind']=='new_start' else None)
                result.update(kind=item['kind'],label=label,audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    audit_script_sha256=sha(Path(__file__)))
                cache.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');newly.append(label)
            if item['kind']=='baseline':assert abs(result['J153']-r['J153'])<1e-10
            audits.append(result)
        except Exception as exc:
            issue=dict(label=label,folder=folder.relative_to(ROOT).as_posix(),error=repr(exc),traceback=traceback.format_exc())
            issues.append(issue)
            (REVIEW/'runs'/f'{label}_AUDIT_ERROR.json').write_text(json.dumps(issue,indent=2)+'\n',encoding='utf-8')
    candidates=[dict(c,label=a['label'],kind=a['kind']) for a in audits for c in a['candidates']]
    terminals=[a for a in audits if a['run_status'].startswith('PASS') and a.get('exit_code',0)==0]
    best_terminal=max(terminals,key=lambda a:a['J153']) if terminals else None
    best_candidate=max(candidates,key=lambda c:c['J']) if candidates else None
    difference=(best_candidate['J']-best_terminal['J153']) if best_candidate and best_terminal else None
    complete=len(meta['runs'])==16 and all(r['status']!='RUNNING' for r in meta['runs'])
    if meta['status'].startswith('COMPLETED'):
        assert complete,'Controller declares completion before all16 starts exit.'
    finished=complete and meta['status'].startswith('COMPLETED')
    failed=[a['label'] for a in audits if not a['run_status'].startswith('PASS')]
    status=('NEEDS_AUDIT_REVIEW' if issues else
        'COMPLETE_SAVED_DATA_AUDIT_WITH_OPTIMIZER_FAILURES' if finished and failed else
        'PASS_COMPLETED_SAVED_DATA_AUDIT' if finished else 'RUNNING_COMPLETED_RUNS_AUDITED')
    result=dict(status=status,study_status=meta['status'],study_manifest_sha256=sha(BASE/'manifest.json'),
        audit_script_sha256=sha(Path(__file__)),copied_checker_source='scripts/audit_gap_closure_case.py',
        copied_checker_source_sha256=sha(ROOT/'scripts/audit_gap_closure_case.py'),
        audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),newly_audited_labels=newly,
        planned_new_starts=16,previous_baseline_starts=2,launched_new_starts=len(meta['runs']),
        completed_new_starts=sum(r['status']!='RUNNING' for r in meta['runs']),audited_runs=len(audits),
        optimization_failed_starts=failed,
        all_required_starts_converged=finished and all(r['status'].startswith('PASS') and r.get('exit_code')==0 for r in meta['runs']),
        best_PASS_terminal=best_terminal,best_evaluated_candidate=best_candidate,best_evaluated_advantage_pp=difference,
        refinement_required=difference is not None and difference>.001,
        source_hash_checks=sum(a['source_hashes'] for a in audits),output_hash_checks=sum(a['output_hashes'] for a in audits),
        complete_full153_candidates=len(candidates),runs=audits,issues=issues,
        new_solver_runs=False,no_global_optimality_claim=True,no_continuous_zone_certification=True,
        scope='Fixed N15,153-point,48-nodeC4v,6-variable[-.16,.16] fixed-area local searches. Failed starts retain all valid complete-grid candidates; missing and partial grids are not conclusions.',
        limitations='Arithmetic,geometry,source/output bindings and recorded feasibility/KKT/gradient gates. No eigenvectors,derivatives or residuals independently recomputed here.')
    (REVIEW/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if not quiet or newly or issues or finished:
        print(json.dumps(dict(status=status,newly_audited_labels=newly,completed_new_starts=result['completed_new_starts'],
            best_PASS_terminal_J153=best_terminal['J153'] if best_terminal else None,
            best_evaluated_J153=best_candidate['J'] if best_candidate else None,
            best_evaluated_advantage_pp=difference,refinement_required=result['refinement_required'],issues=issues),indent=2),flush=True)
    if issues:raise SystemExit(2)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true',help='Recompute all completed reports once, e.g. final audit.')
    parser.add_argument('--quiet-unchanged',action='store_true')
    parser.add_argument('--watch',action='store_true',help='Audit new completions every15s; no solver launches.')
    args=parser.parse_args()
    while True:
        result=audit_once(args.refresh,args.quiet_unchanged or args.watch)
        if not args.watch:return
        if result['study_status'].startswith('COMPLETED'):
            audit_once(True,False)
            return
        time.sleep(15)


if __name__=='__main__':main()
