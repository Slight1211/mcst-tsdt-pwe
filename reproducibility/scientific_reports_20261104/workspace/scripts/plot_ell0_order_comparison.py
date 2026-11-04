"""Saved-data plots for an N15-to-N21 reoptimization and frozen-shape check.

Geometry and spectra are separate objects: changing the solver truncation of a
fixed geometry is never labelled as reoptimization. No eigensolver is imported.
"""
from __future__ import annotations
from pathlib import Path
import argparse,csv,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_gap_closure_scan import Sources,polygon,spectrum,source_path,sha,locate

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'05_validation/ELL0_N21_REOPT_20261101'
DEFAULT_AUDIT=ROOT/'03_review/ell0_N21_reoptimization_20261101/independent_audit.json'
DEFAULT_CROSS=BASE/'order_cross_comparison/manifest.json'
DEFAULT_OUT=ROOT/'output/ell0_N15_N21'
AREA=.09*math.pi
BUFFER=.001
STYLE={'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
       'axes.spines.right':False,'svg.fonttype':'none'}
SHAPE_COLORS={'g15':'#D55E00','g21':'#0072B2'}
SHAPE_LABELS={'g15':'Optimized at $N=15$','g21':'Reoptimized at $N=21$'}


def close(v):return np.vstack([v,v[0]])


def vertex_distance(a,b):return float(np.max(np.linalg.norm(a-b,axis=1)))


def verify_root_hashes(values,sources):
    for value,digest in values.items():
        p=source_path(value);sources.read(p);assert sha(p)==digest,('Source/input hash changed',p)


def verify_outputs(folder,values,sources):
    for name,digest in values.items():
        p=(folder/name).resolve();assert p.parent==folder.resolve()
        sources.read(p);assert sha(p)==digest,('Saved output hash changed',p)


def read_full_spectrum(path,sources):
    sp=spectrum(path,sources);data=sources.rows(path)
    assert [int(r['point']) for r in data]==list(range(153))
    k=np.array([[float(r['kx_a']),float(r['ky_a'])] for r in data])
    expected=np.array([(i,j) for i in range(17) for j in range(i+1)])
    assert np.max(abs(k*16/math.pi-expected))<1e-10
    f=np.array([[float(r[f'Omega{i}']) for i in range(1,7)] for r in data]);g=sp['gap']
    for name,band,bound in [('lower',3,'L'),('upper',4,'U')]:
        indices=list(map(int,np.flatnonzero(abs(f[:,band]-g[bound])<=1e-8)))
        g[name+'_tied_indices']=indices;g[name+'_tied_k_a']=k[indices].tolist()
    sp['full_k']=k;sp['full_f']=f
    if 'nonrigid_first6_residual' in data[0]:
        res=max(float(r['nonrigid_first6_residual']) for r in data)
        orth=max(float(r['mass_orthogonality']) for r in data)
        assert res<1e-7 and orth<1e-7
        sp['recorded_numerical_checks']=dict(max_nonrigid_first6_residual=res,max_mass_orthogonality=orth)
    else:sp['recorded_numerical_checks']=None
    return sp


def geometry_metrics(v,design):
    """Actual polygon turns and signed right-axis distance from neighbour chord."""
    assert v.shape==(48,2) and np.isfinite(v).all()
    assert np.max(abs(v-polygon(design)))<1e-12
    area=.5*np.sum(v[:,0]*np.roll(v[:,1],-1)-v[:,1]*np.roll(v[:,0],-1))
    assert abs(area-AREA)<1e-12
    incoming=v-np.roll(v,1,axis=0);outgoing=np.roll(v,-1,axis=0)-v
    turns=incoming[:,0]*outgoing[:,1]-incoming[:,1]*outgoing[:,0]
    # C4v and fixed angular indexing imply a vertical chord at the right-axis tip.
    assert abs(v[1,0]-v[-1,0])<1e-12 and abs(v[1,1]+v[-1,1])<1e-12 and abs(v[0,1])<1e-12
    radii=np.linalg.norm(v,axis=1);ratio=radii[0]/radii[1]
    assert abs(ratio-math.exp(math.sqrt(2)*design[0]))<1e-12
    signed_inset=float(.5*(v[1,0]+v[-1,0])-v[0,0])
    return dict(y0=float(design[0]),axis_radius_ratio=float(ratio),
        analytical_axis_notch_y0_threshold=math.log(math.cos(math.pi/24))/math.sqrt(2),
        signed_axis_chord_inset_over_a=signed_inset,signed_axis_chord_inset_um=75*signed_inset,
        concave_vertices=int(np.count_nonzero(turns < -1e-8)),
        all_four_axis_tips_concave=bool(np.all(turns[[0,12,24,36]] < -1e-8)),
        minimum_signed_turn=float(np.min(turns)),area_fraction=float(area))


def read_optimized_geometry(folder,expected_N,key,sources):
    m=sources.json(folder/'manifest.json')
    assert m['status'].startswith('PASS')
    assert (m['N'],m['a_um'],m['h_um'],m['ell_epoxy_um'],m['ell_steel_um'])==(expected_N,75,6,0,0)
    assert m['optimization_points']==153 and m['shape_bounds']==[-.16,.16]
    assert m['source_hash_check']=='PASS'
    verify_root_hashes(m['source_sha256'],sources);verify_outputs(folder,m['output_sha256'],sources)
    final_round=m['rounds'][-1];opt=final_round['optimizer']
    assert opt['success'] and opt['constraint_violation']<1e-6 and final_round['full_constraint_violation']<1e-6
    assert opt['projected_lagrangian_gradient_inf']<1e-3 and opt['complementarity_inf']<1e-4
    assert len(m['gradient_gates'])==2 and all(g['status']=='PASS' for g in m['gradient_gates'])
    vp=folder/'final_vertices.csv';assert sha(vp)==m['output_sha256'][vp.name]
    data=sources.rows(vp);v=np.array([[float(r['x_over_a']),float(r['y_over_a'])] for r in data])
    metrics=geometry_metrics(v,m['design'])
    return dict(key=key,optimized_N=expected_N,run=locate(folder),vertices_path=locate(vp),
        design=m['design'],vertices=v,metrics=metrics,manifest=m)


def on_boundary_path(k):
    x,y=k
    return abs(y)<1e-12 or abs(x-math.pi)<1e-12 or abs(x-y)<1e-12


def cross_evaluation_rows(evaluations):
    rows=[]
    for N in [15,21]:
        for key in ['g15','g21']:
            d=evaluations[N,key];g=d['spectrum']['gap'];opt_N=15 if key=='g15' else 21
            rows.append(dict(solver_N=N,geometry=key,geometry_optimized_at_N=opt_N,
                evaluation_type='optimization result' if N==opt_N else 'frozen evaluation',
                J153_percent=g['J'],L_MHz=g['L_MHz'],U_MHz=g['U_MHz'],
                lower_kx_a=g['lower_k_a'][0],lower_ky_a=g['lower_k_a'][1],
                upper_kx_a=g['upper_k_a'][0],upper_ky_a=g['upper_k_a'][1],
                lower_tied_indices=g['lower_tied_indices'],upper_tied_indices=g['upper_tied_indices'],
                lower_tied_k_a=g['lower_tied_k_a'],upper_tied_k_a=g['upper_tied_k_a'],
                lower_bound_on_plotted_path=on_boundary_path(g['lower_k_a']),
                upper_bound_on_plotted_path=on_boundary_path(g['upper_k_a']),
                spectrum=d['spectrum']['spectrum'],points=153,boundary_path_points=49))
    return rows


def save_figure(fig,out,stem,outputs):
    for ext in ['png','svg']:
        name=stem+'.'+ext;fig.savefig(out/name,dpi=230,facecolor='white');outputs.append(name)
    plt.close(fig)


def draw_shapes(shapes,out,outputs):
    fig,axes=plt.subplots(1,2,figsize=(11.4,6),layout='constrained')
    d15=shapes['g15']['metrics']['signed_axis_chord_inset_um'];d21=shapes['g21']['metrics']['signed_axis_chord_inset_um']
    relative=100*(d21-d15)/d15
    fig.suptitle(r'$\ell=0\,\mu$m: $N=15$ optimization and $N=21$ reoptimization'+'\n'+
        f'Signed axis-tip inset: d15={d15:.5f} µm; d21={d21:.5f} µm; change {relative:+.2f}%',fontsize=13)
    initial=close(polygon(np.zeros(6)))
    axes[0].plot(initial[:,0],initial[:,1],color='#777777',ls='--',lw=1)
    maximum=max(float(np.max(abs(g['vertices']))) for g in shapes.values())
    limit=max(.4,math.ceil((maximum+.02)/.05)*.05)
    samples=np.vstack([g['vertices'][[47,0,1]] for g in shapes.values()])
    centre=.5*(float(np.min(samples[:,0]))+float(np.max(samples[:,0])))
    half=max(.025,.65*(float(np.max(samples[:,0]))-float(np.min(samples[:,0]))),1.3*float(np.max(abs(samples[:,1]))))
    for key,g in shapes.items():
        v=g['vertices'];c=close(v);color=SHAPE_COLORS[key];ls='-' if key=='g15' else '--'
        for a in axes:
            a.plot(c[:,0],c[:,1],color=color,ls=ls,lw=1.7)
            a.scatter(v[:,0],v[:,1],color=color,s=9,marker='o' if key=='g15' else 's')
        axes[1].plot([v[-1,0],v[1,0]],[v[-1,1],v[1,1]],color=color,ls=':',lw=1.1)
    axes[0].set(xlim=(-limit,limit),ylim=(-limit,limit),title='Exact saved 48-edge polygons')
    axes[1].set(xlim=(centre-half,centre+half),ylim=(-half,half),title='Right-axis tip; neighbour chords dotted')
    for a in axes:a.set(xlabel='$x/a$',ylabel='$y/a$');a.set_aspect('equal');a.grid(alpha=.15)
    handles=[Line2D([],[],color='#777777',ls='--',lw=1,label='Initial equal-area 48-gon')]
    handles.extend(Line2D([],[],color=SHAPE_COLORS[k],ls='-' if k=='g15' else '--',lw=1.7,label=SHAPE_LABELS[k]) for k in ['g15','g21'])
    fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=9,frameon=False)
    save_figure(fig,out,'ell0_N15_N21_shapes_and_tip_zoom',outputs)
    return dict(full_axis_limits=[-limit,limit],zoom_x_limits=[centre-half,centre+half],zoom_y_limits=[-half,half])


def draw_band_matrix(evaluations,out,outputs):
    values=np.r_[np.concatenate([d['spectrum']['f'][:,3:5].ravel() for d in evaluations.values()]),
        [d['spectrum']['gap'][bound] for d in evaluations.values() for bound in ['L_MHz','U_MHz']]]
    margin=max(.04,.06*float(np.ptp(values)));ylimits=(float(np.min(values))-margin,float(np.max(values))+margin)
    fig,axes=plt.subplots(2,2,figsize=(11.8,9.2),sharex=True,sharey=True,layout='constrained')
    fig.suptitle(r'$\ell=0\,\mu$m: ordered branches 4 and 5 across geometries and truncations'+'\n'+
        'Curves join the 49 boundary samples; bounds use all 153 samples.',fontsize=13)
    for row,N in enumerate([15,21]):
        for col,key in enumerate(['g15','g21']):
            a=axes[row,col];d=evaluations[N,key];sp=d['spectrum'];g=sp['gap'];opt_N=15 if key=='g15' else 21
            for band,color,ls in [(3,'#0072B2','-'),(4,'#D55E00','--')]:
                a.plot(sp['x'],sp['f'][:,band],color=color,ls=ls,lw=1.6)
            L,U=g['L_MHz'],g['U_MHz']
            if L!=U:a.axhspan(min(L,U),max(L,U),color='#009E73' if U>L else '#CC6677',alpha=.12)
            a.axhline(L,color='#0072B2',lw=.8,ls=':');a.axhline(U,color='#D55E00',lw=.8,ls=':')
            controls=g.get('lower_tied_k_a',[g['lower_k_a']])+g.get('upper_tied_k_a',[g['upper_k_a']])
            off_path=any(not on_boundary_path(k) for k in controls)
            role='Optimization result' if N==opt_N else 'Frozen evaluation'
            a.set_title(f'Solver N={N}; geometry optimized at N={opt_N}\n{role}; J153={g["J"]:.6f}%'+
                ('\nFull-grid extrema include off-path samples' if off_path else ''),fontsize=10)
            a.set(xticks=[0,1,2,2+math.sqrt(2)],xticklabels=[r'$\Gamma$','$X$','$M$',r'$\Gamma$'],
                xlim=(0,2+math.sqrt(2)),ylim=ylimits,xlabel='Bloch path',ylabel='Frequency (MHz)')
            for x in [1,2]:a.axvline(x,color='#999999',lw=.6,alpha=.4)
            a.grid(alpha=.15)
    handles=[Line2D([],[],color='#0072B2',lw=1.6,label='Branch 4'),Line2D([],[],color='#D55E00',lw=1.6,ls='--',label='Branch 5'),
        Line2D([],[],color='#777777',ls=':',lw=.8,label='Bounds over all 153 samples')]
    if any(d['spectrum']['gap']['U']>d['spectrum']['gap']['L'] for d in evaluations.values()):
        handles.append(Patch(facecolor='#009E73',alpha=.12,label='Sampled gap'))
    if any(d['spectrum']['gap']['U']<d['spectrum']['gap']['L'] for d in evaluations.values()):
        handles.append(Patch(facecolor='#CC6677',alpha=.12,label='Frequency overlap'))
    fig.legend(handles=handles,loc='outside lower center',ncol=4 if len(handles)==4 else 3,frameon=False,fontsize=9)
    save_figure(fig,out,'ell0_N15_N21_band45_cross_evaluation',outputs)
    return dict(common_frequency_limits_MHz=list(ylimits),path_segment_lengths=[1,1,math.sqrt(2)],
        all_bounds_use_153_samples=True,curves_use_49_boundary_samples=True,
        off_path_control_note='Off-path controlling bounds are stated in panel titles; full coordinates are in the CSV.')


def inspect_n21_candidates(folder,m,sources):
    registered={c['spectrum']:c for c in m['full153_candidates']};records=[]
    for p in sorted(folder.glob('*_ibz153.csv')):
        if p.name=='final_ibz153.csv':continue
        c=registered.get(p.name)
        if c is None:
            if p.name=='seed_ibz153.csv':design=m['seed_design'];rid=0
            else:
                stem=p.name.removesuffix('_ibz153.csv');assert stem.startswith('round')
                rid=int(stem[5:]);design=sources.json(folder/(stem+'_optimizer.json'))['design']
        else:design=c['design'];rid=c['round']
        geometry_metrics(polygon(design),design);sp=read_full_spectrum(p,sources);g=sp['gap']
        if c:assert all(abs(g[k]-c[k])<1e-9 for k in ['L','U','J'])
        records.append(dict(run=locate(folder),round=rid,design=design,J153=g['J'],spectrum=locate(p)))
    assert records
    return records


def read_cross_checks(folder,cell,sp,sources):
    p=folder/f"{cell['shape_label']}_N{cell['evaluation_N']}_checks.csv"
    data=sources.rows(p);assert len(data)==153 and [int(r['point']) for r in data]==list(range(153))
    k=np.array([[float(r['kx_a']),float(r['ky_a'])] for r in data]);assert np.max(abs(k-sp['full_k']))<1e-14
    res=max(float(r['max_first6_nonrigid_relative_residual']) for r in data)
    orth=max(float(r['mass_orthogonality_max']) for r in data)
    assert res<1e-7 and orth<1e-7 and abs(float(data[0]['Gamma_rigid_lambda']))<1e-7
    assert all(r['Gamma_rigid_lambda']=='' for r in data[1:])
    assert abs(res-cell['max_first6_nonrigid_relative_residual'])<1e-14
    assert abs(orth-cell['max_mass_orthogonality_error'])<1e-14
    return dict(max_nonrigid_first6_residual=res,max_mass_orthogonality=orth)


def load_completed_inputs(study_path,audit_path,cross_path,sources,selected_override=None):
    study=sources.json(study_path)
    assert study['status']=='COMPLETED_ALL_N21_SEARCHES','N21 optimization study is not complete'
    assert (study['N'],study['starts'],study['optimization_points'])==(21,4,153)
    assert (study['a_um'],study['h_um'],study['ell_epoxy_um'],study['ell_steel_um'])==(75,6,0,0)
    assert study['vertices']==48 and study['shape_variables']==6 and study['shape_bounds']==[-.16,.16]
    assert abs(study['fixed_area_fraction']-AREA)<1e-13
    assert len(study['runs'])==4 and not study.get('pending_labels')
    assert all(r['status'].startswith('PASS') and r['exit_code']==0 for r in study['runs'])
    verify_root_hashes(study['source_sha256'],sources);verify_root_hashes(study['input_sha256'],sources)
    for seed in study['seeds']:
        p=source_path(seed['path']);sources.read(p);assert sha(p)==seed['sha256']
    audit=sources.json(audit_path)
    assert audit['status']=='PASS_N21_REOPT_DATA_WITH_FROZEN_WITNESS','Independent N21 audit is not a completed PASS'
    assert audit['all_required_N21_starts_converged'] and audit['completed_N21_starts']==audit['N21_audited_runs']==4
    assert not audit['refinement_required'] and not audit['issues'] and not audit['optimization_failed_starts']
    assert audit['study_manifest_sha256']==sha(study_path)
    final_cross_audit=audit['full_2x2_cross_audit']
    assert final_cross_audit['status']=='PASS_INDEPENDENT_FULL_2X2_ORDER_AUDIT'
    assert final_cross_audit['manifest']==locate(cross_path) and final_cross_audit['manifest_sha256']==sha(cross_path)
    verify_root_hashes({'scripts/audit_ell0_N21_reoptimization.py':audit['audit_script_sha256'],
        audit['copied_checker_source']:audit['copied_checker_source_sha256']},sources)
    assert audit['frozen_g15_at_N21']['status']=='PASS_INDEPENDENT_FROZEN_G15_N21_AUDIT'
    selected=source_path(audit['best_PASS_terminal']['folder'])
    if selected_override:
        override=source_path(selected_override)
        if override.name=='manifest.json':override=override.parent
        assert override==selected,'Selection must be the audited best PASS N21 terminal, not a frozen witness'
    n15_manifest=source_path(study['n15_baseline_manifest'])
    shapes={'g15':read_optimized_geometry(n15_manifest.parent,15,'g15',sources)}
    assert abs(shapes['g15']['manifest']['final_ibz153']['J']-study['n15_baseline_J153'])<1e-9
    candidates=[];starts=[]
    for r in study['runs']:
        folder=source_path(r['out']);mp=folder/'manifest.json'
        assert sha(mp)==r['manifest_sha256']
        shape=read_optimized_geometry(folder,21,'g21',sources)
        sp=read_full_spectrum(folder/'final_ibz153.csv',sources)
        assert all(abs(sp['gap'][k]-shape['manifest']['final_ibz153'][k])<1e-9 for k in ['L','U','J'])
        assert abs(sp['gap']['J']-r['J153'])<1e-9
        candidates.extend(inspect_n21_candidates(folder,shape['manifest'],sources))
        starts.append(dict(label=r['label'],run=r['out'],status=r['status'],J153=sp['gap']['J']))
        if folder==selected:shapes['g21']=shape
    assert 'g21' in shapes
    assert sha(selected/'final_vertices.csv')==audit['best_PASS_terminal']['final_vertices_sha256']
    assert abs(shapes['g21']['manifest']['final_ibz153']['J']-audit['best_PASS_terminal']['J153'])<1e-9
    assert max(s['J153'] for s in starts)-shapes['g21']['manifest']['final_ibz153']['J']<1e-8
    cross=sources.json(cross_path);folder=cross_path.parent
    assert cross['status']=='PASS_FROZEN_ELL0_2X2_ORDER_CROSS' and cross['mode']=='full2x2'
    assert cross['N_values']==[15,21] and (cross['a_um'],cross['h_um'],cross['ell_epoxy_um'],cross['ell_steel_um'])==(75,6,0,0)
    assert cross['optimization_points']==153 and cross['optimization_grid']==16
    assert cross['source_hash_check']==cross['input_hash_check']=='PASS'
    assert cross['source_sha256']==cross['source_sha256_post'] and cross['input_sha256']==cross['input_sha256_post']
    verify_root_hashes(cross['source_sha256'],sources);verify_root_hashes(cross['input_sha256'],sources)
    verify_outputs(folder,cross['output_sha256'],sources)
    assert len(cross['shapes'])==2 and {g['shape_label'] for g in cross['shapes']}=={'g15','g21'}
    for g in cross['shapes']:
        shape=shapes[g['shape_label']]
        assert g['source_manifest']==shape['run']+'/manifest.json'
        assert g['source_manifest_sha256']==sha(source_path(g['source_manifest']))
        assert g['optimization_N']==shape['optimized_N'] and g['design']==shape['design']
    evaluations={}
    for cell in cross['cells']:
        N,key=cell['evaluation_N'],cell['shape_label'];assert (N,key) not in evaluations
        assert N in [15,21] and key in ['g15','g21'] and cell['optimization_N']==shapes[key]['optimized_N']
        assert cell['point_count']==153
        p=folder/cell['spectrum'];assert p.parent==folder
        sp=read_full_spectrum(p,sources);g=sp['gap']
        assert all(abs(g[k]-cell[k])<1e-9 for k in ['L','U','J'])
        for side in ['lower','upper']:
            assert g[side+'_tied_indices']==cell[side+'_tied_indices']
            assert np.max(abs(np.asarray(g[side+'_tied_k_a'])-np.asarray(cell[side+'_tied_k_a'])))<1e-14
        if N==shapes[key]['optimized_N']:
            assert cell['provenance']=='HASH_BOUND_SAVED_OWN_ORDER_DIAGONAL'
            assert sha(p)==sha(source_path(shapes[key]['run'])/'final_ibz153.csv')
            checks=sp['recorded_numerical_checks']
        else:
            assert cell['provenance'] in ['FRESH_FROZEN_GEOMETRY_CROSS_EVALUATION','HASH_BOUND_REUSED_FROZEN_CROSS_EVALUATION']
            checks=read_cross_checks(folder,cell,sp,sources)
        evaluations[N,key]=dict(spectrum=sp,cell=cell,numerical_checks=checks)
    assert set(evaluations)=={(N,key) for N in [15,21] for key in ['g15','g21']}
    bestJ=shapes['g21']['manifest']['final_ibz153']['J']
    observed=max([c['J153'] for c in candidates]+[evaluations[21,'g15']['spectrum']['gap']['J']])
    assert observed-bestJ<=BUFFER,'Better evaluated N21 geometry needs refinement before final plots'
    # The final independent four-cell audit and actual cross PASS close the
    # frozen-evaluation stage; no frozen cell is reclassified as optimization.
    return dict(study=study,audit=audit,cross=cross,shapes=shapes,evaluations=evaluations,starts=starts,candidates=candidates)


def save_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def comparison_summary(data):
    ev=data['evaluations'];sh=data['shapes'];j={(N,key):d['spectrum']['gap']['J'] for (N,key),d in ev.items()}
    d15=sh['g15']['metrics']['signed_axis_chord_inset_over_a'];d21=sh['g21']['metrics']['signed_axis_chord_inset_over_a']
    assert abs(d15)>1e-12,'Relative inset change is undefined for a zero-inset baseline'
    ratio=d21/d15;order_ratio=sh['g15']['optimized_N']/sh['g21']['optimized_N']
    notch_scale=dict(g15_optimization_N=sh['g15']['optimized_N'],g21_optimization_N=sh['g21']['optimized_N'],
        relative_signed_depth_change_percent=100*(ratio-1),signed_depth_ratio_g21_over_g15=ratio,
        optimization_order_ratio_g15_over_g21=order_ratio,
        relative_difference_of_depth_ratio_from_order_ratio_percent=100*(ratio/order_ratio-1),
        g15_optimization_N_times_signed_inset_over_a=sh['g15']['optimized_N']*d15,
        g21_optimization_N_times_signed_inset_over_a=sh['g21']['optimized_N']*d21,
        N_is_geometry_optimization_order_not_frozen_evaluation_order=True,
        two_point_descriptive_arithmetic_only=True,no_scaling_fit_performed=True,
        inverse_N_law_not_established=True,physical_or_numerical_origin_not_determined=True)
    delta=sh['g21']['vertices']-sh['g15']['vertices']
    max_displacement=vertex_distance(sh['g15']['vertices'],sh['g21']['vertices'])
    rms_displacement=float(np.sqrt(np.mean(np.sum(delta*delta,axis=1))))
    return dict(selected_g15_run=sh['g15']['run'],selected_g21_run=sh['g21']['run'],
        J153_matrix_rows_solverN_columns_geometry={'N15':{'g15':j[15,'g15'],'g21':j[15,'g21']},'N21':{'g15':j[21,'g15'],'g21':j[21,'g21']}},
        fixed_g15_N21_minus_N15_pp=j[21,'g15']-j[15,'g15'],fixed_g21_N21_minus_N15_pp=j[21,'g21']-j[15,'g21'],
        geometry_gain_at_N21_pp=j[21,'g21']-j[21,'g15'],geometry_gain_at_N15_pp=j[15,'g21']-j[15,'g15'],
        max_corresponding_vertex_displacement_over_a=max_displacement,max_corresponding_vertex_displacement_um=75*max_displacement,
        rms_corresponding_vertex_displacement_over_a=rms_displacement,rms_corresponding_vertex_displacement_um=75*rms_displacement,
        rms_displacement_definition='Square root of the mean squared Euclidean displacement across the 48 corresponding saved vertices.',
        notch_scale_comparison=notch_scale,
        g15_geometry=sh['g15']['metrics'],g21_geometry=sh['g21']['metrics'],N21_start_count=4,
        numerical_equivalence_reporting_buffer_pp=BUFFER,no_global_optimality_claim=True,no_full_order_convergence_certificate=True,
        no_continuous_zone_certification=True,no_parameterization_independence_claim=True,no_new_3D_validation=True)


def write_reports(out,summary,rows):
    m=summary['J153_matrix_rows_solverN_columns_geometry']
    table=['| Solver truncation | Geometry optimized at N15 | Geometry reoptimized at N21 |',
        '|---|---:|---:|',f"| N15 | {m['N15']['g15']:.8f}% (optimization) | {m['N15']['g21']:.8f}% (frozen) |",
        f"| N21 | {m['N21']['g15']:.8f}% (frozen) | {m['N21']['g21']:.8f}% (reoptimization) |",'']
    (out/'cross_evaluation_2x2.md').write_text('\n'.join(table),encoding='utf-8')
    a,b=summary['g15_geometry'],summary['g21_geometry'];scale=summary['notch_scale_comparison']
    existence_zh='两阶选定形状均保留四轴凹口。' if a['all_four_axis_tips_concave'] and b['all_four_axis_tips_concave'] else '两阶四轴凹口判据分别为 '+str(a['all_four_axis_tips_concave'])+' / '+str(b['all_four_axis_tips_concave'])+'。'
    existence_en='The selected geometries at both orders retain notches at all four axis tips.' if a['all_four_axis_tips_concave'] and b['all_four_axis_tips_concave'] else 'The four-axis-notch criteria at the two orders are '+str(a['all_four_axis_tips_concave'])+' / '+str(b['all_four_axis_tips_concave'])+'.'
    lines=['ell=0 µm：N15/N21 重优化与固定形状交叉评估',
        f"g15：{summary['selected_g15_run']}",f"g21：{summary['selected_g21_run']}",
        f"y0：N15 {a['y0']:.10g}；N21 {b['y0']:.10g}。带符号轴向弦线凹深：{a['signed_axis_chord_inset_um']:.8g} / {b['signed_axis_chord_inset_um']:.8g} µm（正为凹入，负为凸出）。",
        existence_zh+f"凹深相对变化为 {scale['relative_signed_depth_change_percent']:+.6g}%；存在性与幅度的截断敏感性需要分别判断，不能因两阶都有凹口就排除数值起源或声称幅度已收敛。",
        f"仅作两点描述：d21/d15={scale['signed_depth_ratio_g21_over_g15']:.9g}，15/21={scale['optimization_order_ratio_g15_over_g21']:.9g}；各自 optimization_N·d/a={scale['g15_optimization_N_times_signed_inset_over_a']:.9g} / {scale['g21_optimization_N_times_signed_inset_over_a']:.9g}。这不是1/N拟合或定律证据；冻结同一形状重新求谱不会改变其凹深。",
        f"最大对应顶点位移：{summary['max_corresponding_vertex_displacement_over_a']:.8g}a（{summary['max_corresponding_vertex_displacement_um']:.8g} µm）；顶点欧氏位移的RMS为 {summary['rms_corresponding_vertex_displacement_um']:.8g} µm。",
        f"固定 g15 后，N21−N15 的 J153 变化：{summary['fixed_g15_N21_minus_N15_pp']:+.8g} 个百分点；固定 g21 后为 {summary['fixed_g21_N21_minus_N15_pp']:+.8g} 个百分点。",
        f"同为 N21 时，重优化形状相对 g15 的目标值变化：{summary['geometry_gain_at_N21_pp']:+.8g} 个百分点；同为 N15 时为 {summary['geometry_gain_at_N15_pp']:+.8g} 个百分点。",
        '对角项来自相应 N 下通过门槛的局部优化；非对角项仅固定形状重新求谱，不能称为额外优化。0.001 个百分点仅为报告数值等价的描述性容差。',
        '只比较 N15→N21 这一步；不能据此认证完整截断收敛、连续布里渊区极值、全局最优或凹口的物理必然性，也未新增三维验证。两种形状均为同面积的48直边、6变量、相同参数边界。','',
        'ell=0 µm: N15/N21 reoptimization and frozen-geometry cross evaluation',
        f"g15: {summary['selected_g15_run']}",f"g21: {summary['selected_g21_run']}",
        f"y0: {a['y0']:.10g} / {b['y0']:.10g}. Signed axis-tip inset from the neighbour chord: {a['signed_axis_chord_inset_um']:.8g} / {b['signed_axis_chord_inset_um']:.8g} µm (positive inward, negative outward).",
        existence_en+f" The relative inset change is {scale['relative_signed_depth_change_percent']:+.6g}%. Existence and amplitude sensitivity to truncation must be assessed separately; retaining a notch at both orders does not exclude a numerical origin or establish amplitude convergence.",
        f"Descriptive two-point arithmetic only: d21/d15={scale['signed_depth_ratio_g21_over_g15']:.9g}, 15/21={scale['optimization_order_ratio_g15_over_g21']:.9g}; the respective optimization_N*d/a values are {scale['g15_optimization_N_times_signed_inset_over_a']:.9g} / {scale['g21_optimization_N_times_signed_inset_over_a']:.9g}. This is neither a 1/N fit nor evidence of a scaling law. Reevaluating a frozen geometry does not change its inset.",
        f"Maximum corresponding-vertex displacement: {summary['max_corresponding_vertex_displacement_over_a']:.8g}a ({summary['max_corresponding_vertex_displacement_um']:.8g} µm); RMS Euclidean vertex displacement: {summary['rms_corresponding_vertex_displacement_um']:.8g} µm.",
        f"At fixed g15, the N21-minus-N15 change in J153 is {summary['fixed_g15_N21_minus_N15_pp']:+.8g} percentage points; at fixed g21 it is {summary['fixed_g21_N21_minus_N15_pp']:+.8g} percentage points.",
        f"At fixed N21, changing g15 to the reoptimized geometry changes J153 by {summary['geometry_gain_at_N21_pp']:+.8g} percentage points; at fixed N15 the change is {summary['geometry_gain_at_N15_pp']:+.8g} percentage points.",
        'Diagonal cells reuse gated local optimizations at their respective N. Off-diagonal cells only reevaluate fixed geometries; they are not additional optimizations. The 0.001-percentage-point equivalence buffer is descriptive.',
        'This compares only N15 to N21; it does not certify asymptotic order convergence, continuous-zone extrema, global optimality, or physical necessity of a notch, and adds no 3D validation. Both shapes retain the same area, 48 straight edges, six variables and parameter bounds.','']
    (out/'RESULTS_ZH_EN.txt').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-manifest',type=Path,default=BASE/'manifest.json')
    parser.add_argument('--audit-report',type=Path,default=DEFAULT_AUDIT)
    parser.add_argument('--cross-manifest',type=Path,default=DEFAULT_CROSS)
    parser.add_argument('--selected-n21-run',help='Optional explicit selection; must equal the audited best PASS terminal.')
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT);parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--inspect-only',action='store_true',help='Verify completed inputs and print the comparison; do not create files.')
    args=parser.parse_args();sources=Sources()
    data=load_completed_inputs(args.study_manifest.resolve(),args.audit_report.resolve(),args.cross_manifest.resolve(),sources,args.selected_n21_run)
    summary=comparison_summary(data);rows=cross_evaluation_rows(data['evaluations'])
    if args.inspect_only:sources.unchanged();print(json.dumps(summary,indent=2));return
    out=args.out.resolve();assert out.is_relative_to(ROOT/'output');out.mkdir(parents=True,exist_ok=True)
    if not args.overwrite:assert not any(out.iterdir()),'Use --overwrite only for these new derived comparison assets'
    save_csv(out/'cross_evaluation_2x2.csv',rows)
    base_depth=data['shapes']['g15']['metrics']['signed_axis_chord_inset_um']
    geometry_rows=[dict(geometry=key,optimized_N=g['optimized_N'],run=g['run'],**g['metrics'],
        signed_inset_change_from_g15_um=g['metrics']['signed_axis_chord_inset_um']-base_depth,
        relative_signed_inset_change_from_g15_percent=100*(g['metrics']['signed_axis_chord_inset_um']/base_depth-1),
        optimization_N_times_signed_inset_over_a=g['optimized_N']*g['metrics']['signed_axis_chord_inset_over_a'],
        max_vertex_displacement_from_g15_over_a=vertex_distance(g['vertices'],data['shapes']['g15']['vertices']),
        max_vertex_displacement_from_g15_um=75*vertex_distance(g['vertices'],data['shapes']['g15']['vertices'])) for key,g in data['shapes'].items()]
    save_csv(out/'geometry_metrics.csv',geometry_rows);save_csv(out/'N21_start_results.csv',data['starts'])
    (out/'RESULT_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');write_reports(out,summary,rows)
    outputs=['cross_evaluation_2x2.csv','geometry_metrics.csv','N21_start_results.csv','RESULT_SUMMARY.json','cross_evaluation_2x2.md','RESULTS_ZH_EN.txt']
    with plt.rc_context(STYLE):
        geometry_plot=draw_shapes(data['shapes'],out,outputs);band_plot=draw_band_matrix(data['evaluations'],out,outputs)
    sources.unchanged()
    record=dict(artifact='ELL0_N15_N21_ORDER_COMPARISON',status='DERIVED_FROM_COMPLETED_REOPTIMIZATION_AND_FROZEN_CROSS',
        selected_g15_run=summary['selected_g15_run'],selected_g21_run=summary['selected_g21_run'],
        independent_full_cross_audit_status=data['audit']['full_2x2_cross_audit']['status'],
        study=locate(args.study_manifest.resolve()),audit=locate(args.audit_report.resolve()),cross=locate(args.cross_manifest.resolve()),
        source_sha256=sources.hashes,plot_script_sha256=sha(Path(__file__)),helper_script_sha256=sha(ROOT/'scripts/plot_gap_closure_scan.py'),
        geometry_plot=geometry_plot,band_plot=band_plot,extrema_tie_tolerance_absolute_Omega=1e-8,
        initial_regular_geometry='Reconstructed from the zero Helmert six-vector and exact fixed-area normalization; no missing-result substitution.',
        evaluation_numerical_checks={f'N{N}_{key}':d['numerical_checks'] for (N,key),d in data['evaluations'].items()},
        signed_inset_semantics='Positive: axis tip inside neighbouring chord; negative: axis tip outside chord. No clamping to zero.',
        notch_scale_semantics='Only two-point descriptive arithmetic; scaled inset uses each geometry optimization_N, never a frozen evaluation_N. No inverse-N law or physical/numerical origin is established.',
        no_new_eigensolves=True,no_new_COMSOL=True,no_full_order_convergence_certificate=True,
        no_global_optimality_claim=True,no_continuous_zone_certification=True,no_new_3D_validation=True,
        output_sha256={name:sha(out/name) for name in outputs},visual_review='PENDING_ACTUAL_OUTPUT_INSPECTION',human_verified=False)
    (out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(out=str(out),selected_g21=summary['selected_g21_run'],summary=summary),indent=2))


if __name__=='__main__':main()
