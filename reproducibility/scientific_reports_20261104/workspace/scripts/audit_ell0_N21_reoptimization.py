"""Independent saved-data audit of genuine N21 ell0 optimization.

No eigensolver is imported or called. The existing saved-run checker is copied
with its Fourier order parameterized. Only N21 rows compete for N21 best;
the frozen g15/N21 witness is kept distinct from an N21 optimized terminal.
"""
from pathlib import Path
import argparse,csv,datetime,json,math,re,time,traceback
from audit_gap_closure_case import (ROOT,TARGET_AREA,load,sha,rows,area,
    polygon,validate_geometry,spectrum,compare_gap)
BASE=ROOT/'05_validation/ELL0_N21_REOPT_20261101'
REVIEW=ROOT/'03_review/ell0_N21_reoptimization_20261101'
FROZEN=BASE/'frozen_g15_N21/manifest.json'
CROSS=BASE/'order_cross_comparison/manifest.json'
INITIAL=(0,27,136,152)


def save(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def check_run(folder,ell,expected_active=(0,27,136,152),N=21):
    m=load(folder/'manifest.json')
    assert (m['N'],m['a_um'],m['h_um'],m['ell_steel_um'],m['optimization_points'],m['optimization_grid'])==(N,75.,6.,0.,153,16)
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


def strict_checks(path):
    data=rows(path)
    assert len(data)==153
    assert [int(r['point']) for r in data]==list(range(153))
    res=[float(r['nonrigid_first6_residual']) for r in data]
    orth=[float(r['mass_orthogonality']) for r in data]
    assert all(math.isfinite(x) and 0<=x<1e-7 for x in res+orth)
    return dict(spectrum=path.name,max_nonrigid_first6_residual=max(res),max_mass_orthogonality=max(orth))


def geometry_metrics(design,vertices_path=None):
    v=validate_geometry(design)
    if vertices_path:
        saved=rows(vertices_path);assert len(saved)==48
        assert max(abs(x-float(q[k])) for p,q in zip(v,saved) for x,k in zip(p,['x_over_a','y_over_a']))<1e-13
    turns=[]
    for i in range(48):
        a,b,c=v[i-1],v[i],v[(i+1)%48]
        turns.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
    depth=v[1][0]-v[0][0]
    return dict(design=design,y0=design[0],axis_notch_y0_threshold=math.log(math.cos(math.pi/24))/math.sqrt(2),
        axis_notch_depth_over_a=depth,axis_notch_depth_um=75*depth,axis_notch_present=depth>0,
        negative_turn_count=sum(t<-1e-12 for t in turns),minimum_signed_turn_over_a2=min(turns),
        at_upper_bound_indices=[i for i,y in enumerate(design) if abs(y-.16)<1e-8],
        at_lower_bound_indices=[i for i,y in enumerate(design) if abs(y+.16)<1e-8],area_fraction=area(v))


def audit_worker(folder,record):
    a=inspect_run(folder,INITIAL,record);m=load(folder/'manifest.json')
    assert m['N']==21
    assert all(g['N']==21 for g in m.get('gradient_gates',[]))
    bound={Path(p).as_posix() for p in m['source_sha256']}
    assert {'scripts/ell0_resolution_base.py','scripts/ell0_resolution_sensitivity.py',
        'scripts/run_ell0_resolution_case.py','scripts/run_lengthscale_optimization.py',
        '04_code/research/wavemotion_surface_20260920/run_node_shape_pilot.py'}<=bound
    complete={c['spectrum'] for c in a['candidates']}
    if m['status'].startswith('PASS'):complete.add('final_ibz153.csv')
    a['strict_spectrum_gates']=[strict_checks(folder/n) for n in sorted(complete)]
    for candidate in a['candidates']:candidate['N']=21
    a['N']=21
    if m['status'].startswith('PASS'):
        a['geometry']=geometry_metrics(m['design'],folder/'final_vertices.csv')
        s=spectrum(folder/'final_ibz153.csv')
        a['lower_ties_1e8']=[i for i,f in enumerate(s['f']) if abs(f[3]-s['L'])<=1e-8]
        a['upper_ties_1e8']=[i for i,f in enumerate(s['f']) if abs(f[4]-s['U'])<=1e-8]
    if m['status']=='RUNNING':
        a['run_status']='EXITED_WITH_STALE_RUNNING_MANIFEST'
        a['incomplete_manifest_after_process_exit']=True
    return a


def audit_frozen(path,baseline):
    if not path.exists():return dict(status='PENDING_FROZEN_G15_AT_N21',manifest=str(path.relative_to(ROOT)))
    m=load(path)
    if m['status']=='RUNNING':return dict(status='PENDING_FROZEN_G15_AT_N21',progress=m.get('progress'),manifest=str(path.relative_to(ROOT)))
    assert m['status']=='PASS_FROZEN_ELL0_G15_N21_SINGLE',m.get('error',m['status'])
    assert m['mode']=='single_g15_order21' and m['N_values']==[21]
    assert (m['a_um'],m['h_um'],m['ell_epoxy_um'],m['ell_steel_um'],m['optimization_points'],m['optimization_grid'],m['threads'])==(75,6,0,0,153,16,2)
    assert not m['new_optimization'] and not m['new_dense_reference'] and not m['new_COMSOL_runs']
    assert m['source_hash_check']==m['input_hash_check']=='PASS'
    assert m['source_sha256']==m['source_sha256_post'] and m['input_sha256']==m['input_sha256_post']
    assert all(sha(ROOT/p)==h for p,h in m['source_sha256'].items())
    assert all(sha(ROOT/p)==h for p,h in m['input_sha256'].items())
    assert all(sha(path.parent/p)==h for p,h in m['output_sha256'].items())
    assert len(m['shapes'])==len(m['cells'])==1
    shape,cell=m['shapes'][0],m['cells'][0]
    assert shape['shape_label']=='g15' and shape['optimization_N']==15 and shape['design']==baseline['design']
    assert shape['source_manifest']==baseline['manifest'] and shape['source_manifest_sha256']==baseline['manifest_sha256']
    assert shape['own_order_spectrum']==baseline['spectrum'] and shape['own_order_spectrum_sha256']==baseline['spectrum_sha256']
    compare_gap(shape['own_order_gap'],baseline['gap'])
    assert (cell['shape_label'],cell['optimization_N'],cell['evaluation_N'],cell['point_count'])==('g15',15,21,153)
    assert cell['provenance']=='FRESH_FROZEN_GEOMETRY_CROSS_EVALUATION'
    s=spectrum(path.parent/cell['spectrum']);compare_gap(s,cell)
    lower=[i for i,f in enumerate(s['f']) if abs(f[3]-s['L'])<=1e-8]
    upper=[i for i,f in enumerate(s['f']) if abs(f[4]-s['U'])<=1e-8]
    assert lower==cell['lower_tied_indices'] and upper==cell['upper_tied_indices']
    assert [list(s['k'][i]) for i in lower]==cell['lower_tied_k_a']
    assert [list(s['k'][i]) for i in upper]==cell['upper_tied_k_a']
    checks=rows(path.parent/'g15_N21_checks.csv');assert len(checks)==153
    assert [int(q['point']) for q in checks]==list(range(153))
    assert all(max(abs(float(r[k])-v) for k,v in zip(['kx_a','ky_a'],point))<1e-14 for r,point in zip(checks,s['k']))
    residual=[float(q['max_first6_nonrigid_relative_residual']) for q in checks]
    orth=[float(q['mass_orthogonality_max']) for q in checks]
    assert all(math.isfinite(x) and 0<=x<1e-7 for x in residual+orth)
    assert abs(float(checks[0]['Gamma_rigid_lambda']))<1e-7 and all(q['Gamma_rigid_lambda']=='' for q in checks[1:])
    assert max(residual)==cell['max_first6_nonrigid_relative_residual'] and max(orth)==cell['max_mass_orthogonality_error']
    candidate=dict(N=21,kind='FROZEN_G15_EVALUATED_AT_N21_NOT_OPTIMIZED',label='frozen_g15_at_N21',
        design=baseline['design'],J=s['J'],L=s['L'],U=s['U'],spectrum=cell['spectrum'],
        folder=path.parent.relative_to(ROOT).as_posix(),spectrum_sha256=sha(path.parent/cell['spectrum']),
        local_optimizer_success=None,round=None)
    return dict(status='PASS_INDEPENDENT_FROZEN_G15_N21_AUDIT',manifest=path.relative_to(ROOT).as_posix(),
        manifest_sha256=sha(path),candidate=candidate,lower_ties_1e8=lower,upper_ties_1e8=upper,
        max_nonrigid_first6_residual=max(residual),max_mass_orthogonality=max(orth),
        source_hashes=len(m['source_sha256']),input_hashes=len(m['input_sha256']),output_hashes=len(m['output_sha256']),
        N15_own_order_gap_not_used_as_N21_candidate=baseline['gap'])


def audit_full_cross(path,baseline,best,frozen):
    if not path.exists():return dict(status='PENDING_FULL_2X2_CROSS')
    m=load(path)
    if m['status']=='RUNNING':return dict(status='PENDING_FULL_2X2_CROSS',progress=m.get('progress'))
    assert best is not None and frozen['status'].startswith('PASS')
    assert m['status']=='PASS_FROZEN_ELL0_2X2_ORDER_CROSS',m.get('error',m['status'])
    assert m['mode']=='full2x2' and m['N_values']==[15,21]
    assert (m['a_um'],m['h_um'],m['ell_epoxy_um'],m['ell_steel_um'],m['optimization_points'],m['optimization_grid'],m['threads'])==(75,6,0,0,153,16,2)
    assert not m['new_optimization'] and not m['new_dense_reference'] and not m['new_COMSOL_runs']
    assert m['source_hash_check']==m['input_hash_check']=='PASS'
    assert m['source_sha256']==m['source_sha256_post'] and m['input_sha256']==m['input_sha256_post']
    assert all(sha(ROOT/p)==h for p,h in m['source_sha256'].items())
    assert all(sha(ROOT/p)==h for p,h in m['input_sha256'].items())
    for p,h in m['output_sha256'].items():
        artifact=(path.parent/p).resolve();assert artifact.parent==path.parent.resolve() and sha(artifact)==h
    assert m['fresh_point_count']==153 and m['reused_diagonal_point_count']==306 and m['reused_frozen_cross_point_count']==153
    shapes={g['shape_label']:g for g in m['shapes']};assert set(shapes)=={'g15','g21'} and len(m['shapes'])==2
    expected={'g15':(15,baseline['design'],baseline['manifest'],baseline['manifest_sha256']),
              'g21':(21,best['design'],best['folder']+'/manifest.json',best['manifest_sha256'])}
    for key,(N,design,mp,digest) in expected.items():
        g=shapes[key]
        assert g['optimization_N']==N and g['design']==design
        assert g['source_manifest']==mp and g['source_manifest_sha256']==digest==sha(ROOT/mp)
        assert g['own_order_spectrum']==str(Path(mp).parent/'final_ibz153.csv').replace('\\','/')
        assert sha(ROOT/g['own_order_spectrum'])==g['own_order_spectrum_sha256']
        validate_geometry(design)
    cells={};cross_gate_summary={}
    for cell in m['cells']:
        key=(cell['evaluation_N'],cell['shape_label']);assert key not in cells
        assert key[0] in [15,21] and key[1] in shapes and cell['optimization_N']==shapes[key[1]]['optimization_N'] and cell['point_count']==153
        sp=path.parent/cell['spectrum'];assert sp.parent==path.parent
        s=spectrum(sp);compare_gap(s,cell)
        assert [int(r['point']) for r in s['rows']]==list(range(153))
        lower=[i for i,f in enumerate(s['f']) if abs(f[3]-s['L'])<=1e-8]
        upper=[i for i,f in enumerate(s['f']) if abs(f[4]-s['U'])<=1e-8]
        assert lower==cell['lower_tied_indices'] and upper==cell['upper_tied_indices']
        assert [list(s['k'][i]) for i in lower]==cell['lower_tied_k_a']
        assert [list(s['k'][i]) for i in upper]==cell['upper_tied_k_a']
        if key[0]==cell['optimization_N']:
            assert cell['provenance']=='HASH_BOUND_SAVED_OWN_ORDER_DIAGONAL'
            assert sha(sp)==shapes[key[1]]['own_order_spectrum_sha256']
            compare_gap(s,shapes[key[1]]['own_order_gap'])
        else:
            cp=path.parent/f"{key[1]}_N{key[0]}_checks.csv";data=rows(cp)
            assert len(data)==153 and [int(q['point']) for q in data]==list(range(153))
            assert all(max(abs(float(q[k])-v) for k,v in zip(['kx_a','ky_a'],point))<1e-14 for q,point in zip(data,s['k']))
            res=[float(q['max_first6_nonrigid_relative_residual']) for q in data]
            orth=[float(q['mass_orthogonality_max']) for q in data]
            assert all(math.isfinite(x) and 0<=x<1e-7 for x in res+orth)
            assert abs(float(data[0]['Gamma_rigid_lambda']))<1e-7 and all(q['Gamma_rigid_lambda']=='' for q in data[1:])
            assert max(res)==cell['max_first6_nonrigid_relative_residual'] and max(orth)==cell['max_mass_orthogonality_error']
            cross_gate_summary[f'{key[1]}_N{key[0]}']=dict(max_NRfirst6_residual=max(res),max_mass_orthogonality=max(orth))
            if key==(21,'g15'):
                assert cell['provenance']=='HASH_BOUND_REUSED_FROZEN_CROSS_EVALUATION'
                assert cell['reused_manifest']==frozen['manifest']
                assert sha(sp)==frozen['candidate']['spectrum_sha256']
                compare_gap(s,frozen['candidate'])
                assert sha(cp)==sha((ROOT/frozen['manifest']).parent/'g15_N21_checks.csv')
            else:
                assert key==(15,'g21') and cell['provenance']=='FRESH_FROZEN_GEOMETRY_CROSS_EVALUATION'
        cells[key]=dict(N=key[0],shape=key[1],L=s['L'],U=s['U'],J=s['J'],
            spectrum=sp.relative_to(ROOT).as_posix(),spectrum_sha256=sha(sp),
            lower_tied_indices=lower,upper_tied_indices=upper,provenance=cell['provenance'])
    assert set(cells)=={(N,g) for N in [15,21] for g in ['g15','g21']}
    assert load(path.parent/'order_cross_summary.json')['cells']==m['cells']
    table=rows(path.parent/'order_cross_summary.csv');assert len(table)==4
    for row in table:
        c=cells[int(row['evaluation_N']),row['shape_label']]
        assert all(abs(float(row[k])-c[k])<1e-12 for k in ['L','U','J'])
    A,B,C,D=(cells[N,g]['J'] for N,g in [(15,'g15'),(21,'g15'),(15,'g21'),(21,'g21')])
    assert abs(A-baseline['gap']['J'])<1e-12 and abs(D-best['J153'])<1e-12
    assert abs(B-frozen['candidate']['J'])<1e-12
    return dict(status='PASS_INDEPENDENT_FULL_2X2_ORDER_AUDIT',manifest=path.relative_to(ROOT).as_posix(),manifest_sha256=sha(path),
        source_hashes=len(m['source_sha256']),input_hashes=len(m['input_sha256']),output_hashes=len(m['output_sha256']),
        cells=[cells[N,g] for N,g in [(15,'g15'),(21,'g15'),(15,'g21'),(21,'g21')]],
        J153_matrix={'N15':{'g15':A,'g21':C},'N21':{'g15':B,'g21':D}},
        fixed_g15_order_difference_pp=B-A,N21_geometry_gain_pp=D-B,
        N15_geometry_difference_pp=C-A,fixed_g21_order_difference_pp=D-C,
        fresh_point_count=153,reused_diagonal_point_count=306,reused_frozen_B_point_count=153,
        cross_recorded_numerical_gates=cross_gate_summary,
        scope='One fresh frozen g21/N15 evaluation; A,D are hash-bound optimized diagonals and B is hash-bound frozen reuse. No additional optimization or asymptotic convergence claim.')


def audit_once(refresh=False,quiet=False):
    REVIEW.mkdir(parents=True,exist_ok=True);(REVIEW/'runs').mkdir(exist_ok=True)
    meta=load(BASE/'manifest.json')
    assert (meta['N'],meta['a_um'],meta['h_um'],meta['ell_epoxy_um'],meta['ell_steel_um'])==(21,75,6,0,0)
    assert meta['vertices']==48 and meta['shape_variables']==6 and meta['shape_bounds']==[-.16,.16]
    assert meta['optimization_points']==153 and meta['optimization_grid']==16 and meta['initial_active_indices']==list(INITIAL)
    assert meta['starts']==4 and len(meta['seeds'])==4 and meta['threads_per_worker']==2
    assert abs(meta['fixed_area_fraction']-TARGET_AREA)<1e-13
    assert all(sha(ROOT/p)==h for p,h in meta['source_sha256'].items())
    assert all(sha(ROOT/p)==h for p,h in meta['input_sha256'].items())
    assert meta['no_global_optimality_claim'] and meta['no_continuous_zone_certification'] and not meta['new_COMSOL_runs']
    seeds={s['label']:s for s in meta['seeds']};assert len(seeds)==4
    for s in seeds.values():assert sha(ROOT/s['path'])==s['sha256'];validate_geometry(s['design'])
    bp=ROOT/meta['n15_baseline_manifest'];bm=load(bp);assert bm['N']==15 and bm['ell_epoxy_um']==0 and bm['status'].startswith('PASS')
    bs=spectrum(bp.parent/'final_ibz153.csv');compare_gap(bs,bm['final_ibz153'])
    assert bs['J']==meta['n15_baseline_J153']
    baseline=dict(N=15,manifest=bp.relative_to(ROOT).as_posix(),manifest_sha256=sha(bp),design=bm['design'],
        spectrum=(bp.parent/'final_ibz153.csv').relative_to(ROOT).as_posix(),spectrum_sha256=sha(bp.parent/'final_ibz153.csv'),
        gap={k:bs[k] for k in ['L','U','J']},geometry=geometry_metrics(bm['design'],bp.parent/'final_vertices.csv'))
    previous=load(REVIEW/'independent_audit.json') if (REVIEW/'independent_audit.json').exists() else {}
    audits=[];new=[];issues=[]
    assert len({r['label'] for r in meta['runs']})==len(meta['runs'])
    for record in meta['runs']:
        assert record['label'] in seeds and ROOT/record['out']==BASE/record['label']
        if record.get('exit_code') is None:continue
        folder=ROOT/record['out'];label=record['label'];cache=REVIEW/'runs'/f'{label}.json'
        try:
            mp=folder/'manifest.json';assert mp.exists(),'Exited process has no manifest.'
            m=load(mp);s=seeds[label]
            assert m['seed_source']==s['path'] and m['seed_source_sha256']==s['sha256'] and m['seed_design']==s['design']
            saved=load(cache) if cache.exists() else {}
            use_cache=not refresh and saved.get('manifest_sha256')==sha(mp) and saved.get('audit_script_sha256')==sha(Path(__file__))
            if use_cache:
                a=saved
                assert all(sha(ROOT/p)==h for p,h in m['source_sha256'].items())
                assert all(sha(folder/p)==h for p,h in m.get('output_sha256',{}).items())
                assert record['manifest_sha256']==a['manifest_sha256'] and record['exit_code']==a['exit_code']
            else:
                a=audit_worker(folder,record);a.update(label=label,audit_script_sha256=sha(Path(__file__)),
                    audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                save(cache,a);new.append(label)
            if a['run_status']=='EXITED_WITH_STALE_RUNNING_MANIFEST':
                issues.append(dict(label=label,error='Process exited but left a RUNNING worker manifest; available complete candidates audited, no convergence claim.'))
            audits.append(a)
        except Exception as exc:
            issue=dict(label=label,error=repr(exc),traceback=traceback.format_exc());issues.append(issue)
            save(REVIEW/'runs'/f'{label}_AUDIT_ERROR.json',issue)
    try:frozen=audit_frozen(FROZEN,baseline)
    except Exception as exc:
        frozen=dict(status='NEEDS_FROZEN_REVIEW',error=repr(exc),traceback=traceback.format_exc());issues.append(frozen)
    candidates=[dict(c,label=a['label'],kind='N21_OPTIMIZATION_CANDIDATE') for a in audits for c in a['candidates']]
    if frozen['status'].startswith('PASS'):candidates.append(frozen['candidate'])
    assert all(c['N']==21 for c in candidates)
    terminals=[a for a in audits if a['run_status'].startswith('PASS') and a.get('exit_code')==0]
    best_terminal=max(terminals,key=lambda a:a['J153']) if terminals else None
    best_candidate=max(candidates,key=lambda c:c['J']) if candidates else None
    try:cross=audit_full_cross(CROSS,baseline,best_terminal,frozen)
    except Exception as exc:
        cross=dict(status='NEEDS_FULL_2X2_REVIEW',error=repr(exc),traceback=traceback.format_exc());issues.append(cross)
    difference=best_candidate['J']-best_terminal['J153'] if best_candidate and best_terminal else None
    exited=len(meta['runs'])==4 and all(r.get('exit_code') is not None for r in meta['runs'])
    finished=exited and meta['status'].startswith('COMPLETED')
    if meta['status'].startswith('COMPLETED'):assert exited
    ready=finished and frozen['status'].startswith('PASS')
    failed=[a['label'] for a in audits if not a['run_status'].startswith('PASS')]
    status=('NEEDS_AUDIT_REVIEW' if issues else 'COMPLETE_N21_DATA_AUDIT_WITH_OPTIMIZER_FAILURES' if ready and failed else
        'PASS_N21_REOPT_DATA_WITH_FROZEN_WITNESS' if ready else
        'OPTIMIZATION_AUDITED_AWAITING_FROZEN_N21_G15' if finished else 'RUNNING_COMPLETED_N21_RUNS_AUDITED')
    result=dict(status=status,study_status=meta['status'],study_manifest_sha256=sha(BASE/'manifest.json'),
        audit_script_sha256=sha(Path(__file__)),copied_checker_source='scripts/audit_ell0_multistart.py',
        copied_checker_source_sha256=sha(ROOT/'scripts/audit_ell0_multistart.py'),
        audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),newly_audited_labels=new,
        planned_N21_starts=4,launched_N21_starts=len(meta['runs']),completed_N21_starts=sum(r.get('exit_code') is not None for r in meta['runs']),
        N21_audited_runs=len(audits),all_required_N21_starts_converged=finished and len(audits)==4 and not failed and not issues,
        optimization_failed_starts=failed,best_PASS_terminal=best_terminal,best_vetted_N21_candidate=best_candidate,
        best_vetted_advantage_pp=difference,refinement_required=difference is not None and difference>.001,
        complete_N21_full153_candidates=len(candidates),n15_baseline_not_a_N21_candidate=baseline,
        frozen_g15_at_N21=frozen,runs=audits,issues=issues,
        source_hash_checks=sum(a['source_hashes'] for a in audits)+frozen.get('source_hashes',0)+cross.get('source_hashes',0),
        output_hash_checks=sum(a['output_hashes'] for a in audits)+frozen.get('output_hashes',0)+cross.get('output_hashes',0),
        no_new_solver_runs=True,no_global_optimality_claim=True,no_continuous_zone_certification=True,
        comparison_scope='Only N21 geometry candidates compete with N21 J. A=N15(g15) is descriptive baseline only. Frozen B=N21(g15) is a witness, never mislabeled optimized N21.',
        two_by_two_g21_at_N15_pending=not cross['status'].startswith('PASS'),full_2x2_cross_audit=cross,
        limitations='Saved arithmetic/geometry/source/output and recorded numerical gates; eigenvectors/derivatives not recomputed. Two Fourier orders do not establish an asymptotic error bound.')
    save(REVIEW/'independent_audit.json',result)
    changed_frozen=frozen['status']!=previous.get('frozen_g15_at_N21',{}).get('status')
    if not quiet or new or issues or ready or changed_frozen:
        print(json.dumps(dict(status=status,newly_audited_labels=new,completed_N21_starts=result['completed_N21_starts'],
            frozen_status=frozen['status'],frozen_g15_N21_J=frozen.get('candidate',{}).get('J'),
            best_PASS_terminal_J153=best_terminal['J153'] if best_terminal else None,
            best_vetted_N21_J153=best_candidate['J'] if best_candidate else None,
            best_vetted_advantage_pp=difference,refinement_required=result['refinement_required'],issues=issues),indent=2),flush=True)
    if issues:raise SystemExit(2)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true')
    parser.add_argument('--refresh',action='store_true');parser.add_argument('--quiet-unchanged',action='store_true');args=parser.parse_args()
    while True:
        result=audit_once(args.refresh,args.watch or args.quiet_unchanged)
        if not args.watch:return
        if result['status'] in ['PASS_N21_REOPT_DATA_WITH_FROZEN_WITNESS','COMPLETE_N21_DATA_AUDIT_WITH_OPTIMIZER_FAILURES']:
            audit_once(True,False);return
        time.sleep(15)


if __name__=='__main__':main()
