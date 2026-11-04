"""Independent saved-file arithmetic/control audit; never run an eigensolver."""
from pathlib import Path
import argparse,csv,hashlib,json,math

ROOT=Path(__file__).resolve().parents[1]
TARGET_AREA=.09*math.pi
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
def area(p):return .5*sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(p,p[1:]+p[:1]))
def polygon(y):
    sector=[]
    for node in range(7):
        z=0.
        for c in range(6):
            i=c+1;h=(1. if node<i else -i if node==i else 0.)/math.sqrt(i*(i+1))
            z+=h*y[c]
        sector.append(math.exp(z))
    raw=[]
    for j in range(48):
        radius=sector[min(j%12,12-j%12)];theta=2*math.pi*j/48
        raw.append([radius*math.cos(theta),radius*math.sin(theta)])
    scale=math.sqrt(TARGET_AREA/area(raw))
    return [[x*scale,y*scale] for x,y in raw]
def validate_geometry(y):
    assert len(y)==6 and all(math.isfinite(v) and abs(v)<=.16+1e-10 for v in y)
    p=polygon(y)
    assert abs(area(p)-TARGET_AREA)<1e-12
    assert max(abs(x) for q in p for x in q)<.49
    assert min(math.hypot(a[0]-b[0],a[1]-b[1]) for a,b in zip(p,p[1:]+p[:1]))>=1e-5
    return p
def spectrum(p,grid=16):
    rr=rows(p);assert len(rr)==(grid+1)*(grid+2)//2
    coords=[];frequency=[]
    for r in rr:
        q=(float(r['kx_a']),float(r['ky_a']));coords.append(q)
        assert all(abs(t*grid/math.pi-round(t*grid/math.pi))<1e-10 for t in q)
        f=[float(r[f'Omega{b}']) for b in range(1,7)];frequency.append(f)
        assert all(math.isfinite(t) for t in f)
        assert all(a<=b+1e-9 for a,b in zip(f,f[1:]))
        assert all(t>0 for t in (f[1:] if max(abs(t) for t in q)<1e-14 else f))
    lattice=[tuple(round(t*grid/math.pi) for t in q) for q in coords]
    expected=[(i,j) for i in range(grid+1) for j in range(i+1)]
    assert lattice==expected
    low=max(range(len(rr)),key=lambda i:frequency[i][3]);up=min(range(len(rr)),key=lambda i:frequency[i][4])
    L,U=frequency[low][3],frequency[up][4]
    return dict(rows=rr,k=coords,f=frequency,L=L,U=U,J=200*(U-L)/(U+L),active_L=low,active_U=up)
def compare_gap(actual,stored,tolerance=1e-9):
    assert all(abs(actual[k]-stored[k])<tolerance for k in ['L','U','J'])

def check_run(folder,ell):
    m=load(folder/'manifest.json')
    assert (m['N'],m['a_um'],m['h_um'],m['ell_steel_um'],m['optimization_points'],m['optimization_grid'])==(15,75.,6.,0.,153,16)
    assert m['vertices']==48 and m['shape_variables']==6 and m['shape_bounds']==[-.16,.16] and m['threads']==2
    assert abs(m['fixed_area_fraction']-TARGET_AREA)<1e-13
    assert abs(m['ell_epoxy_um']-ell)<1e-12 and abs(m['ell_over_h']-ell/6)<1e-14
    assert m['active_start_indices']==[0,136,152] and m['ftol']==1e-7
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
        assert active==({0,136,152} if prior is None else set(prior['active_indices'])|set(prior['new_indices']))
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

def main():
    p=argparse.ArgumentParser();p.add_argument('--ell',type=float,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    base=ROOT/'05_validation/GAP_CLOSURE_SCAN_20261031';case=base/f'ell_{args.ell:g}'
    summary=load(case/'case_summary.json');scan=load(base/'manifest.json')
    assert abs(summary['ell_um']-args.ell)<1e-12 and scan['N']==15
    initial=scan['initial_state'];seed_path=ROOT/initial['seed_manifest']
    assert sha(seed_path)==initial['seed_sha256']
    seed=load(seed_path);seed_spectrum=spectrum(seed_path.parent/'final_ibz153.csv')
    assert seed['ell_epoxy_um']==initial['ell_um'] and seed['ell_epoxy_um']<args.ell
    compare_gap(seed_spectrum,seed['final_ibz153'])
    assert abs(seed_spectrum['J']-initial['J153'])<1e-10 and seed_spectrum['J']>.001
    assert sha(ROOT/'scripts/run_gap_closure_case.py')==scan['case_worker_sha256']
    assert sha(ROOT/'scripts/run_gap_closure_scan.py')==scan['controller_sha256']
    audited=[check_run(ROOT/run['folder'],args.ell) for run in summary['runs']]
    byname={r['run']:r for r in audited};candidates=[]
    for stored in summary['runs']:
        audit=byname[stored['run']]
        assert stored['status']==audit['run_status']
        assert (stored['final_J153'] is None and audit['J153'] is None) or abs(stored['final_J153']-audit['J153'])<1e-10
        assert (stored['best_vetted_J153'] is None and audit['best_J153'] is None) or abs(stored['best_vetted_J153']-audit['best_J153'])<1e-10
        candidates+=audit['candidates']
    best=max(candidates,key=lambda c:c['J']);converged=[r for r in audited if r['run_status'].startswith('PASS')]
    assert converged
    best_final=max(converged,key=lambda r:r['J153'])
    assert abs(best['J']-summary['best_vetted_J153'])<1e-10
    assert abs(best_final['J153']-summary['best_converged_J153'])<1e-10
    compare_gap(best,summary['best_vetted_candidate'])
    assert best['design']==summary['best_vetted_candidate']['design']
    assert best['spectrum_sha256']==summary['best_vetted_candidate']['spectrum_sha256']
    assert summary['selected_run']==f'05_validation/GAP_CLOSURE_SCAN_20261031/ell_{args.ell:g}/{best_final["run"]}'
    margin=scan['gap_sign_buffer_pp'];assert margin==.001
    all_pass=all(r['run_status'].startswith('PASS') for r in audited)
    expected='POSITIVE_SAMPLED_GAP_FOUND' if best['J']>margin else 'NO_POSITIVE_SAMPLED_GAP_FOUND' if all_pass and best['J']<-margin else 'NEAR_ZERO_UNRESOLVED' if all_pass else 'FAILED_NOT_NO_GAP'
    assert expected!='FAILED_NOT_NO_GAP' and summary['classification']==expected and summary['all_required_starts_converged']==all_pass
    if expected=='NO_POSITIVE_SAMPLED_GAP_FOUND':assert {'ell1_seed','ell2_seed'}<=set(byname)
    stored=next(c for c in scan['cases'] if abs(c['ell_um']-args.ell)<1e-12);assert stored==summary
    row=next(r for r in rows(base/'scan_summary.csv') if abs(float(r['ell_um'])-args.ell)<1e-12)
    assert row['classification']==expected and abs(float(row['best_converged_J153'])-best_final['J153'])<1e-10 and abs(float(row['best_vetted_J153'])-best['J'])<1e-10
    if expected!='NO_POSITIVE_SAMPLED_GAP_FOUND':
        continuation=load(case/'continuation_seed.json')
        assert continuation['design']==best['design']
        assert continuation['witness_spectrum_sha256']==best['spectrum_sha256']
        assert abs(continuation['final_ibz153']['J']-best['J'])<1e-10
    report=dict(status='PASS_INDEPENDENT_CASE_AUDIT',ell_um=args.ell,classification=expected,runs=audited,
        all_required_starts_converged=all_pass,
        best_J153=best['J'],best_converged_J153=best_final['J153'],source_control_summary_sha256=sha(case/'case_summary.json'),
        limitations='Saved-file arithmetic/control audit, not a new eigensolve. KKT and residual gates are checked as recorded and source-bound; derivatives/eigenvectors are not independently recomputed here. No global or continuous-zone absence is inferred.',new_solver_runs=False,human_verified=False)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],ell_um=args.ell,best_J153=best['J'],classification=expected),indent=2))

if __name__=='__main__':main()
