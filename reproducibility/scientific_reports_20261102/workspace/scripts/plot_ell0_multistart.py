"""Inspect completed ell=0 searches and plot an explicitly reviewed selection.

No solver is imported or executed. Failed starts remain failed even if they have
complete intermediate spectra. All such saved spectra are independently ranked.
"""
from __future__ import annotations
from pathlib import Path
import argparse,csv,json,math,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_gap_closure_scan import Sources,polygon,spectrum,source_path,sha,locate

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'05_validation/ELL0_MULTISTART_20261101'
DEFAULT_OUT=ROOT/'output/ell0_multistart'
GROUPING_PROTOCOL=ROOT/'03_review/ell0_multistart_20261101/shape_grouping_protocol.json'
COMPLETED={'COMPLETED_ALL_STARTS','COMPLETED_WITH_FAILED_STARTS_REQUIRES_REVIEW'}
BUFFER=.001
GEOMETRY_TOL=1e-3
SENSITIVITY_GEOMETRY_TOL=1e-4
TURN_TOL=1e-8
STYLE={'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
       'axes.spines.right':False,'svg.fonttype':'none'}


def tracked_hash(path,sources):
    sources.read(path)
    return sources.hashes[locate(path)]


def saved_vertices(path,design,sources):
    data=sources.rows(path)
    v=np.array([[float(r['x_over_a']),float(r['y_over_a'])] for r in data])
    expected=polygon(design)
    assert v.shape==(48,2) and np.max(abs(v-expected))<1e-12
    return v


def morphology(v):
    incoming=v-np.roll(v,1,axis=0);outgoing=np.roll(v,-1,axis=0)-v
    turns=incoming[:,0]*outgoing[:,1]-incoming[:,1]*outgoing[:,0]
    radii=np.linalg.norm(v,axis=1)
    return dict(concave_vertices=int(np.count_nonzero(turns < -TURN_TOL)),
        minimum_signed_turn=float(np.min(turns)),
        axis_notch=bool(np.all(turns[[0,12,24,36]] < -TURN_TOL)),
        axis_chord_notch_depth_over_a=float(max(0.,.5*(radii[1]+radii[-1])*math.cos(math.pi/24)-radii[0])),
        axis_radius_ratio=float(radii[0]/radii[1]))


def check_pass_gates(m):
    assert m.get('source_hash_check')=='PASS'
    assert len(m['gradient_gates'])>=2 and all(g['status']=='PASS' for g in m['gradient_gates'])
    r=m['rounds'][-1];o=r['optimizer']
    assert o['success'] and o['constraint_violation']<1e-6 and r['full_constraint_violation']<1e-6
    assert o['projected_lagrangian_gradient_inf']<1e-3 and o['complementarity_inf']<1e-4


def read_search(folder,label,origin,sources,expected_hash=None,record=None):
    mp=folder/'manifest.json'
    if not mp.exists():
        assert record and record['status']=='MISSING_MANIFEST'
        return dict(label=label,origin=origin,run=locate(folder),status='MISSING_MANIFEST',
            exit_code=record.get('exit_code'),converged=False,candidates=[],terminal=None,error='No worker manifest')
    if expected_hash:assert sha(mp)==expected_hash
    m=sources.json(mp);assert m['status']!='RUNNING','Do not consume a running worker'
    assert (m['N'],m['a_um'],m['h_um'],m['ell_epoxy_um'],m['ell_steel_um'],m['optimization_points'])==(15,75,6,0,0,153)
    assert m['shape_bounds']==[-.16,.16] and m['vertices']==48 and m['shape_variables']==6
    assert abs(m['fixed_area_fraction']-.09*math.pi)<1e-13
    for value,digest in m['source_sha256'].items():
        p=source_path(value);assert tracked_hash(p,sources)==digest
    if m.get('seed_source_sha256'):
        assert tracked_hash(source_path(m['seed_source']),sources)==m['seed_source_sha256']
    if record:
        assert record['status']==m['status']
    candidates=[]
    for c in m.get('full153_candidates',[]):
        p=(folder/c['spectrum']).resolve();assert p.is_relative_to(folder.resolve())
        if p.name in m.get('output_sha256',{}):assert sha(p)==m['output_sha256'][p.name]
        g=spectrum(p,sources)['gap']
        assert all(abs(g[k]-c[k])<1e-9 for k in ['L','U','J'])
        v=polygon(c['design'])
        candidates.append(dict(label=label,origin=origin,run=locate(folder),worker_status=m['status'],
            round=c['round'],J153_percent=g['J'],L_MHz=g['L_MHz'],U_MHz=g['U_MHz'],
            spectrum=locate(p),spectrum_sha256=sha(p),design=c['design'],
            optimizer_success=c.get('optimizer_success'),recorded_in_worker_manifest=True,**morphology(v)))
    # A failed process can leave a complete spectrum between saving the CSV and
    # appending it to its manifest. Recover it from the saved optimizer geometry;
    # never silently omit such a candidate or count an incomplete CSV as a result.
    registered={Path(c['spectrum']).name for c in candidates};partial=[]
    for p in sorted(folder.glob('*_ibz153.csv')):
        if p.name in registered or p.name=='final_ibz153.csv':continue
        data=sources.rows(p)
        if len(data)!=153:
            partial.append(dict(spectrum=locate(p),saved_rows=len(data),reason='Incomplete spectrum; not ranked'))
            continue
        match=re.fullmatch(r'round(\d+)_ibz153\.csv',p.name)
        if match:
            op=folder/f'round{int(match.group(1))}_optimizer.json';o=sources.json(op)
            design=o['design'];rid=int(match.group(1));success=o['success']
        else:
            assert p.name=='seed_ibz153.csv',('Unrecognized unregistered complete spectrum',p)
            design=m['seed_design'];rid=0;success=None
        g=spectrum(p,sources)['gap'];v=polygon(design)
        candidates.append(dict(label=label,origin=origin,run=locate(folder),worker_status=m['status'],
            round=rid,J153_percent=g['J'],L_MHz=g['L_MHz'],U_MHz=g['U_MHz'],
            spectrum=locate(p),spectrum_sha256=sha(p),design=design,
            optimizer_success=success,recorded_in_worker_manifest=False,**morphology(v)))
    candidates.sort(key=lambda c:c['round'])
    terminal=None;passed=m['status'].startswith('PASS')
    if passed:
        if record:assert record.get('exit_code')==0
        check_pass_gates(m)
        p=folder/'final_ibz153.csv';assert sha(p)==m['output_sha256'][p.name]
        g=spectrum(p,sources)['gap']
        assert all(abs(g[k]-m['final_ibz153'][k])<1e-9 for k in ['L','U','J'])
        vp=folder/'final_vertices.csv';assert sha(vp)==m['output_sha256'][vp.name]
        v=saved_vertices(vp,m['design'],sources)
        assert candidates and abs(max(c['J153_percent'] for c in candidates)-m['best_evaluated153']['J'])<1e-9
        terminal=dict(gap=g,v=v,design=m['design'],vertices=locate(vp),spectrum=locate(p),**morphology(v))
        if record:assert abs(record['J153']-g['J'])<1e-9
    return dict(label=label,origin=origin,run=locate(folder),status=m['status'],
        exit_code=record.get('exit_code') if record else 0 if passed else None,
        converged=passed,candidates=candidates,partial_spectra=partial,terminal=terminal,error=m.get('error'))


def load_completed(meta,sources,extra_runs):
    assert meta['status'] in COMPLETED,'Formal outputs require the controller to finish all starts'
    assert (meta['N'],meta['a_um'],meta['h_um'],meta['ell_epoxy_um'],meta['ell_steel_um'])==(15,75,6,0,0)
    assert meta['new_starts']==16 and meta['previous_starts']==2 and len(meta['runs'])==16
    assert not meta.get('pending_labels')
    assert meta['optimization_points']==153 and meta['shape_bounds']==[-.16,.16]
    for name,key in [('run_ell0_multistart.py','controller_sha256'),('run_ell0_multistart_case.py','worker_sha256')]:
        assert tracked_hash(ROOT/'scripts'/name,sources)==meta[key]
    for value,digest in meta['input_sha256'].items():assert tracked_hash(source_path(value),sources)==digest
    for seed in meta['seeds']:
        assert tracked_hash(source_path(seed['path']),sources)==seed['sha256']
    rows=[]
    for b in meta['baseline']:
        rows.append(read_search(source_path(b['manifest']).parent,b['label'],'previous',sources,b['manifest_sha256']))
        assert rows[-1]['converged'] and abs(rows[-1]['terminal']['gap']['J']-b['J153'])<1e-9
    for r in meta['runs']:
        assert r['status'] not in {'RUNNING','EXITED'}
        seed=next(s for s in meta['seeds'] if s['label']==r['label'])
        assert r['seed_sha256']==seed['sha256']
        rows.append(read_search(source_path(r['out']),r['label'],'new',sources,r['manifest_sha256'],r))
    for index,path in enumerate(extra_runs,1):
        folder=path.resolve() if path.is_absolute() else source_path(str(path))
        if folder.name=='manifest.json':folder=folder.parent
        assert locate(folder) not in {r['run'] for r in rows}
        rows.append(read_search(folder,f'refinement_{index}','refinement',sources))
    assert len(rows)==18+len(extra_runs)
    return rows


def vertex_distance(a,b):
    return float(np.max(np.linalg.norm(a-b,axis=1)))


def group_terminals(rows,tolerance=GEOMETRY_TOL,key='geometry_group'):
    """Predeclared deterministic complete-link grouping in fixed lattice axes."""
    passing=sorted((r for r in rows if r['converged']),key=lambda r:(-r['terminal']['gap']['J'],r['label']))
    groups=[]
    for row in passing:
        for group in groups:
            if all(vertex_distance(row['terminal']['v'],r['terminal']['v'])<=tolerance
                and abs(row['terminal']['gap']['J']-r['terminal']['gap']['J'])<=BUFFER for r in group):
                group.append(row);break
        else:groups.append([row])
    result=[]
    for index,group in enumerate(groups,1):
        name=f'G{index}'
        for r in group:r[key]=name
        result.append(dict(group=name,count=len(group),labels=sorted(r['label'] for r in group),
            J153_min=min(r['terminal']['gap']['J'] for r in group),J153_max=max(r['terminal']['gap']['J'] for r in group),
            maximum_within_group_vertex_displacement_over_a=max(vertex_distance(a['terminal']['v'],b['terminal']['v']) for a in group for b in group)))
    return result


def save_table(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def start_table(rows,best):
    result=[]
    for r in rows:
        t=r['terminal'];cs=r['candidates'];last=cs[-1] if cs else None
        result.append(dict(label=r['label'],origin=r['origin'],status=r['status'],exit_code=r['exit_code'],
            converged=r['converged'],terminal_J153_percent=t['gap']['J'] if t else None,
            terminal_L_MHz=t['gap']['L_MHz'] if t else None,terminal_U_MHz=t['gap']['U_MHz'] if t else None,
            best_complete_candidate_J153_percent=max(c['J153_percent'] for c in cs) if cs else None,
            last_complete_candidate_J153_percent=last['J153_percent'] if last else None,
            complete_candidate_count=len(cs),geometry_group=r.get('geometry_group'),
            incomplete_spectrum_count=len(r.get('partial_spectra',[])),
            geometry_group_at_1e_minus4=r.get('geometry_group_at_1e_minus4'),
            terminal_max_vertex_displacement_from_best_over_a=vertex_distance(t['v'],best['terminal']['v']) if t else None,
            terminal_J153_difference_from_best_pp=t['gap']['J']-best['terminal']['gap']['J'] if t else None,
            terminal_concave_vertices=t['concave_vertices'] if t else None,
            terminal_axis_notch=t['axis_notch'] if t else None,
            terminal_axis_chord_notch_depth_over_a=t['axis_chord_notch_depth_over_a'] if t else None,
            run=r['run'],error=r['error']))
    return result


def closed(v):return np.vstack([v,v[0]])


def format_label(label):
    names={'previous_regular':'Previous: regular', 'previous_ell1_seed':'Previous: ℓ=1 shape',
        'legacy_ell0p5':'Seed: ℓ=0.5 shape','legacy_ell2':'Seed: ℓ=2 shape','legacy_ell5':'Seed: ℓ=5 shape',
        'opposite_old':'Reversed contrast','rotated45_reduced_contrast':'45° reduced contrast',
        'smooth_cos4_positive':'Smooth fourfold (+)','smooth_cos4_negative':'Smooth fourfold (−)',
        'alternating_bounds':'Alternating bounds'}
    if label.startswith('lhs_'):return 'Latin hypercube '+label[-2:]
    return names.get(label,label.replace('_',' '))


def save_figure(fig,out,stem,outputs):
    for ext in ['png','svg']:
        name=stem+'.'+ext;fig.savefig(out/name,dpi=220,facecolor='white');outputs.append(name)
    plt.close(fig)


def shapes_comparison(previous,selected,limit,out,outputs):
    fig=plt.figure(figsize=(12.9,5.3));width=.23;height=width*12.9/5.3
    axes=[fig.add_axes([x,.23,width,height]) for x in [.065,.385,.705]]
    fig.suptitle(r'$\ell=0\,\mu$m: previous best and best converged result after additional starts',fontsize=14,y=.985)
    fig.text(.5,.91,r'$N=15$; $a=75\,\mu$m, $h=6\,\mu$m; identical area and scale; 153 sampled wavevectors',ha='center',fontsize=10)
    initial=closed(polygon(np.zeros(6)));colors=['#D55E00','#0072B2']
    for a in axes:
        a.plot(initial[:,0],initial[:,1],color='#777777',ls='--',lw=1,label='Initial equal-area 48-gon')
        a.set(xlim=(-limit,limit),ylim=(-limit,limit),xlabel='$x/a$',ylabel='$y/a$');a.set_aspect('equal');a.grid(alpha=.15)
    for ax,row,color,title in zip(axes[:2],[previous,selected],colors,['Previous best','Best after additional starts']):
        v=row['terminal']['v'];c=closed(v)
        ax.fill(c[:,0],c[:,1],color=color,alpha=.07);ax.plot(c[:,0],c[:,1],color=color,lw=1.8)
        ax.scatter(v[:,0],v[:,1],color=color,s=8)
        ax.set_title(f"{title}\n$J_{{153}}={row['terminal']['gap']['J']:.6f}\\%$",fontsize=11)
    for row,color,label in zip([previous,selected],colors,['Previous best','Best after additional starts']):
        c=closed(row['terminal']['v']);axes[2].plot(c[:,0],c[:,1],color=color,lw=1.5,
            ls='-' if label=='Previous best' else '--',label=label)
    axes[2].set_title('Overlay at the same scale',fontsize=11)
    handles=[Line2D([],[],color='#777777',ls='--',lw=1,label='Initial equal-area 48-gon'),
        Line2D([],[],color=colors[0],lw=1.5,label='Previous best'),
        Line2D([],[],color=colors[1],ls='--',lw=1.5,label='Best after additional starts')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),fontsize=9,frameon=False,ncol=3)
    fig.canvas.draw();sizes=[(a.get_position().width*12.9,a.get_position().height*5.3) for a in axes]
    assert max(x for x,y in sizes)-min(x for x,y in sizes)<1e-10
    assert all(abs(x-y)<1e-10 for x,y in sizes)
    save_figure(fig,out,'ell0_best_shape_comparison',outputs)


def start_comparison(rows,previous,out,outputs):
    fig,a=plt.subplots(figsize=(12,max(6.5,len(rows)*.34)),layout='constrained')
    for i,r in enumerate(rows):
        if r['converged']:
            a.plot(r['terminal']['gap']['J'],i,'o',color='#0072B2',ms=6)
        elif r['candidates']:
            a.plot(max(c['J153_percent'] for c in r['candidates']),i,'x',color='#CC6677',ms=7,mew=1.5)
        else:
            a.text(.99,i,'Failed: no complete spectrum',transform=a.get_yaxis_transform(),ha='right',va='center',color='#AA4455',fontsize=9)
        if r['candidates']:
            best=max(c['J153_percent'] for c in r['candidates'])
            if r['converged'] and abs(best-r['terminal']['gap']['J'])>BUFFER:
                a.plot(best,i,'D',mfc='none',mec='#CC6677',ms=6)
    a.axvline(previous['terminal']['gap']['J'],color='#777777',ls='--',lw=1)
    values=[r['terminal']['gap']['J'] for r in rows if r['converged']]
    values.extend(c['J153_percent'] for r in rows for c in r['candidates'])
    # Include zero and a broad absolute range; do not amplify round-off-sized
    # differences between numerically equivalent terminal objectives.
    a.set_xlim(min(0.,5*math.floor((min(values)-1)/5)),max(40.,5*math.ceil((max(values)+1)/5)))
    labels=[format_label(r['label'])+('  [not converged]' if not r['converged'] else '') for r in rows]
    a.set(yticks=range(len(rows)),yticklabels=labels,xlabel='Relative sampled gap $J_{153}$ (%)',
        title=r'$\ell=0\,\mu$m: outcomes from all retained starts ($N=15$)')
    a.invert_yaxis();a.grid(axis='x',alpha=.2)
    handles=[Line2D([],[],marker='o',color='#0072B2',ls='',label='Converged terminal')]
    if any(not r['converged'] and r['candidates'] for r in rows):
        handles.append(Line2D([],[],marker='x',color='#CC6677',ls='',label='Failed start: best complete evaluated shape'))
    if any(r['converged'] and max(c['J153_percent'] for c in r['candidates'])-r['terminal']['gap']['J']>BUFFER for r in rows):
        handles.append(Line2D([],[],marker='D',mfc='none',color='#CC6677',ls='',label='Better intermediate shape (>0.001 pp)'))
    handles.append(Line2D([],[],color='#777777',ls='--',label='Previous best'))
    a.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.09),ncol=2,fontsize=9,frameon=False)
    save_figure(fig,out,'ell0_start_outcomes',outputs)


def gallery(rows,limit,out,outputs):
    ncols=6;nrows=math.ceil(len(rows)/ncols);figheight=3.05*nrows+.95
    fig=plt.figure(figsize=(16.8,figheight));width=.134;height=width*16.8/figheight
    initial=closed(polygon(np.zeros(6)));layout=[]
    for i,r in enumerate(rows):
        col=i%ncols;row=i//ncols
        y=.09+(nrows-1-row)*2.8/figheight
        a=fig.add_axes([.045+.16*col,y,width,height]);t=r['terminal']
        a.plot(initial[:,0],initial[:,1],color='#888888',ls='--',lw=.8)
        if t:
            v=t['v'];color='#0072B2';subtitle=f"J={t['gap']['J']:.4f}% | {r['geometry_group']}"
        elif r['candidates']:
            c=r['candidates'][-1];v=polygon(c['design']);color='#CC6677';subtitle=f"NOT CONVERGED | last J={c['J153_percent']:.4f}%"
        else:v=None;color='#CC6677';subtitle='NOT CONVERGED | no complete spectrum'
        if v is not None:
            c=closed(v);a.plot(c[:,0],c[:,1],color=color,lw=1.3);a.scatter(v[:,0],v[:,1],color=color,s=3)
        else:a.text(.5,.5,'No complete\nevaluated shape',transform=a.transAxes,ha='center',va='center',color=color,fontsize=8)
        a.set(xlim=(-limit,limit),ylim=(-limit,limit),xticks=[-limit,0,limit],yticks=[-limit,0,limit])
        a.set_aspect('equal');a.tick_params(labelsize=7);a.grid(alpha=.12)
        a.set_title(format_label(r['label'])+'\n'+subtitle,fontsize=8.5,color='#222222' if t else color)
        position=a.get_position();layout.append([position.width*16.8,position.height*figheight])
    assert max(q[0] for q in layout)-min(q[0] for q in layout)<1e-10
    assert all(abs(q[0]-q[1])<1e-10 for q in layout)
    fig.suptitle(r'$\ell=0\,\mu$m: shapes from all retained starts ($N=15$)',fontsize=17,y=.99)
    fig.text(.5,.945,r'$a=75\,\mu$m, $h=6\,\mu$m; 153 sampled wavevectors; fixed area fraction $0.09\pi$',ha='center',fontsize=9.5)
    fig.text(.5,.055,r'All displayed spectra use $\ell=0$; seed labels identify starting geometries.',ha='center',fontsize=9)
    fig.text(.5,.032,'Blue: converged terminal. Red: last complete evaluated shape from a failed start. Gray: initial equal-area polygon.',ha='center',fontsize=10)
    fig.text(.5,.012,'Coordinates x/a and y/a; identical scales and areas; 48 straight edges. G labels describe geometric groups, not a complete basin map.',ha='center',fontsize=9)
    save_figure(fig,out,'ell0_all_start_shapes',outputs)
    return layout


def bilingual_report(rows,groups,sensitivity_groups,previous,selected,all_candidates,scope):
    passing=[r for r in rows if r['converged']];failed=[r for r in rows if not r['converged']]
    concave=sum(r['terminal']['concave_vertices']>0 for r in passing)
    notched=sum(r['terminal']['axis_notch'] for r in passing)
    improvement=selected['terminal']['gap']['J']-previous['terminal']['gap']['J']
    maximum_displacement=max(vertex_distance(r['terminal']['v'],selected['terminal']['v']) for r in passing)
    gain_zh='变化未超过预先记录的 0.001 个百分点容差，未观察到可分辨的目标值增益。' if abs(improvement)<=BUFFER else '目标值变化超过预先记录的 0.001 个百分点容差。'
    gain_en='The change does not exceed the predeclared 0.001-percentage-point buffer; no resolved objective improvement was observed.' if abs(improvement)<=BUFFER else 'The objective change exceeds the predeclared 0.001-percentage-point buffer.'
    if len(groups)==1:
        zh='在这些通过门槛的起点中只观察到一个形状组；这不能证明解唯一或全局最优。'
        en='Only one geometric group was observed among the converged starts; this does not prove uniqueness or global optimality.'
    else:
        zh='观察到多个通过相同门槛的终态形状组，说明终态对初值敏感；这支持存在不同局部收敛结果，但不是对全部吸引域的完备划分，也不排除近等价的终态族。'
        en='Several terminal geometric groups passed the same gates, showing start-dependent converged shapes. This is evidence of distinct local outcomes, not an exhaustive basin map or a proof excluding near-degenerate families.'
    if notched==len(passing):
        notch_zh='四个轴向凹口在本次所有收敛终态中均复现；这仅说明本设计空间和这些起点中的重复性，不能证明凹口具有物理必然性。'
        notch_en='Axis notches recurred in every converged terminal tested here. This establishes recurrence for these starts in this design space, not their physical necessity.'
    else:
        notch_zh='轴向凹口并非所有收敛起点的共同结果；存在不含四个轴向凹口且通过相同门槛的终态。'
        notch_en='Axis notches were not universal across converged starts: some terminals passed the same gates without notches at all four axis tips.'
    lines=[
        'ell = 0 µm 多起点形状检查',
        f'保留 {len(rows)} 个搜索结果：{len(passing)} 个收敛，{len(failed)} 个未收敛。失败搜索不计为局部最优解。',
        f'旧最佳 J153 = {previous["terminal"]["gap"]["J"]:.10f}%；选定最佳 J153 = {selected["terminal"]["gap"]["J"]:.10f}%；变化 {improvement:+.4g} 个百分点。',
        gain_zh,
        f'选定运行：{selected["run"]}',
        f'收敛终态中，{concave}/{len(passing)} 个含凹角，{notched}/{len(passing)} 个在四个坐标轴端点均含凹口。凹角以有向边转角叉积 < -{TURN_TOL:g} 判定。',
        f'选定最佳终态：'+('含四个轴向凹口。' if selected['terminal']['axis_notch'] else '不含四个轴向凹口。'),notch_zh,
        f'按固定晶格坐标中的最大对应顶点欧氏位移 ≤ {GEOMETRY_TOL:g}a、J153 差 ≤ {BUFFER:g} 个百分点作完整链接分组，共 {len(groups)} 组；位移阈值改为 {SENSITIVITY_GEOMETRY_TOL:g}a 时为 {len(sensitivity_groups)} 组。',
        '两个阈值在查看本次最终结果前预先记录，仅作描述，不是统计检验或误差认证；CSV 同时给出相对最佳终态的实际最大顶点位移。',zh,
        f'收敛终态相对选定最佳的最大对应顶点位移为 {maximum_displacement:.6g}a。',
        '轴向凹口按保存多边形的转角判定。该参数化中 r(0)/r(7.5°)=exp(sqrt(2)*y0)，凹入邻点弦线对应 y0 < log(cos(7.5°))/sqrt(2)。其他坐标达到边界不能单独证明凹口的成因。',
        '所有保存的完整 153 点候选均保留并重算，包括失败和中间轮次；候选优于最佳收敛终态超过 0.001 个百分点时，正式出图会停止并要求精化。',
        scope,'',
        'ell = 0 µm multistart shape check',
        f'{len(rows)} retained searches: {len(passing)} converged and {len(failed)} not converged. Failed searches are not counted as local optima.',
        f'Previous best J153 = {previous["terminal"]["gap"]["J"]:.10f}%; selected best J153 = {selected["terminal"]["gap"]["J"]:.10f}%; change {improvement:+.4g} percentage points.',
        gain_en,
        f'Selected run: {selected["run"]}',
        f'Among converged terminals, {concave}/{len(passing)} contain concave vertices and {notched}/{len(passing)} contain notches at all four coordinate-axis tips. A concave vertex has a signed edge-turn cross product < -{TURN_TOL:g}.',
        'The selected best terminal '+('contains notches at all four axis tips.' if selected['terminal']['axis_notch'] else 'does not contain notches at all four axis tips.'),notch_en,
        f'Complete-link geometric groups use a maximum corresponding-vertex Euclidean displacement ≤ {GEOMETRY_TOL:g}a in fixed lattice axes and a J153 difference ≤ {BUFFER:g} percentage points. Group count: {len(groups)}; at {SENSITIVITY_GEOMETRY_TOL:g}a the count is {len(sensitivity_groups)}.',
        'Both thresholds were recorded before inspecting the new final results. They are descriptive, not statistical tests or error certifications. The CSV also gives the actual maximum vertex displacement from the best terminal.',en,
        f'The maximum corresponding-vertex displacement from the selected best terminal is {maximum_displacement:.6g}a.',
        'Axis notches are checked using the saved polygon turns. Here r(0)/r(7.5°)=exp(sqrt(2)*y0), and an axis tip lies inside its neighbouring chord when y0 < log(cos(7.5°))/sqrt(2). Bound-active coordinates alone do not establish the cause of a notch.',
        f'All {len(all_candidates)} saved complete 153-point candidates were retained and recomputed, including intermediate and failed-search candidates. Formal plotting stops if one exceeds the best converged terminal by >0.001 percentage points.',
        'Scope: N=15; a=75 µm; h=6 µm; epoxy ell=0 and steel ell=0; area fraction 0.09π; 48 vertices and six log-radius contrast variables bounded by [-0.16,0.16]; 153 sampled wavevectors. No global-optimality, continuous-zone, parameterization-independence, or new 3D-validation claim. Other length scales and manuscripts were not changed.',
        '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=BASE/'manifest.json');p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--inspect-only',action='store_true',help='Only rank completed results and export tables; no formal figures or selection claim.')
    p.add_argument('--selected-run',help='Explicit root/auditor-reviewed best terminal directory, project-relative or absolute.')
    p.add_argument('--audit-report',type=Path,help='Bind a completed independent PASS audit for the original18 retained starts.')
    p.add_argument('--extra-run',type=Path,action='append',default=[],help='Additional completed refinement; original starts stay retained.')
    p.add_argument('--overwrite',action='store_true');args=p.parse_args()
    if not args.inspect_only and not args.selected_run:p.error('Formal figures require an explicitly reviewed --selected-run.')
    sources=Sources();mp=args.manifest.resolve();meta=sources.json(mp)
    audit=None
    if args.audit_report:
        assert not args.extra_run,'The original18-start audit does not certify later refinements'
        audit_path=args.audit_report.resolve();audit=sources.json(audit_path)
        assert audit['status']=='PASS_COMPLETED_SAVED_DATA_AUDIT' and not audit['refinement_required'] and not audit['issues']
        assert audit['audited_runs']==18 and audit['completed_new_starts']==16 and audit['all_required_starts_converged']
        assert audit['study_manifest_sha256']==sha(mp)
        assert tracked_hash(ROOT/'scripts/audit_ell0_multistart.py',sources)==audit['audit_script_sha256']
        assert tracked_hash(source_path(audit['copied_checker_source']),sources)==audit['copied_checker_source_sha256']
    protocol=sources.json(GROUPING_PROTOCOL)
    assert protocol['primary_max_vertex_euclidean_displacement_over_a']==GEOMETRY_TOL
    assert protocol['sensitivity_max_vertex_euclidean_displacement_over_a']==SENSITIVITY_GEOMETRY_TOL
    assert protocol['J153_difference_tolerance_pp']==BUFFER
    assert protocol['concavity_signed_turn_threshold']==-TURN_TOL
    rows=load_completed(meta,sources,args.extra_run);groups=group_terminals(rows)
    sensitivity_groups=group_terminals(rows,SENSITIVITY_GEOMETRY_TOL,'geometry_group_at_1e_minus4')
    passing=[r for r in rows if r['converged']];assert passing
    best=max(passing,key=lambda r:r['terminal']['gap']['J'])
    if audit:assert abs(audit['best_PASS_terminal']['J153']-best['terminal']['gap']['J'])<1e-9
    previous=max((r for r in passing if r['origin']=='previous'),key=lambda r:r['terminal']['gap']['J'])
    candidates=[c for r in rows for c in r['candidates']]
    higher=[c for c in candidates if c['J153_percent']>best['terminal']['gap']['J']+BUFFER]
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if not args.overwrite:assert not any(out.iterdir()),'Use --overwrite for derived outputs only'
    table=start_table(rows,best);save_table(out/'start_results.csv',table);save_table(out/'all_full153_candidates.csv',candidates)
    inspection=dict(status='REQUIRES_REFINEMENT_OF_BETTER_EVALUATED_CANDIDATE' if higher else 'PASS_COMPLETED_RESULTS_RANKING',
        controller_status=meta['status'],starts=len(rows),converged=len(passing),failed=len(rows)-len(passing),
        best_converged_run=best['run'],best_converged_J153=best['terminal']['gap']['J'],
        previous_best_run=previous['run'],previous_best_J153=previous['terminal']['gap']['J'],
        all_complete153_candidate_count=len(candidates),higher_candidates=higher,groups=groups,sensitivity_groups_at_1e_minus4=sensitivity_groups,
        recovered_unregistered_complete_candidate_count=sum(not c['recorded_in_worker_manifest'] for c in candidates),
        incomplete_spectra_not_ranked=[dict(run=r['run'],**item) for r in rows for item in r.get('partial_spectra',[])],
        grouping_vertex_tolerance_over_a=GEOMETRY_TOL,grouping_J_tolerance_pp=BUFFER,
        grouping_protocol=locate(GROUPING_PROTOCOL),grouping_semantics='Predeclared deterministic complete-link groups in fixed lattice axes; descriptive only.',
        independent_audit_report=locate(audit_path) if audit else None,independent_audit_status=audit['status'] if audit else None,
        source_sha256=sources.hashes,plot_script_sha256=sha(Path(__file__)),
        no_global_optimality_claim=True,no_new_eigensolves=True,no_new_COMSOL=True)
    sources.unchanged();(out/'ranking.json').write_text(json.dumps(inspection,indent=2)+'\n',encoding='utf-8')
    if args.inspect_only or higher:
        print(json.dumps({k:inspection[k] for k in ['status','converged','failed','best_converged_run','best_converged_J153','higher_candidates','groups']},indent=2))
        if higher and not args.inspect_only:raise SystemExit('Better complete evaluated candidate requires review/refinement; no formal figures produced.')
        return
    selected_path=source_path(args.selected_run)
    if selected_path.name=='manifest.json':selected_path=selected_path.parent
    selected=next(r for r in passing if r['run']==locate(selected_path))
    assert best['terminal']['gap']['J']-selected['terminal']['gap']['J']<1e-8,'Reviewed selection is not the best converged terminal'
    if audit:assert selected['run']==audit['best_PASS_terminal']['folder']
    maximum=max(float(np.max(abs(polygon(c['design'])))) for c in candidates)
    limit=max(.4,math.ceil((maximum+.02)/.05)*.05)
    outputs=['start_results.csv','all_full153_candidates.csv','ranking.json'];layout=[]
    with plt.rc_context(STYLE):
        shapes_comparison(previous,selected,limit,out,outputs)
        start_comparison(rows,previous,out,outputs)
        layout=gallery(rows,limit,out,outputs)
    scope='范围：N=15，a=75 µm，h=6 µm；环氧及钢的 ell=0；面积分数 0.09π；48 个顶点、6 个对数半径对比参数的取值范围 [-0.16,0.16]、153 个波矢采样。未证明全局最优、连续布里渊区极值或参数化无关性，也未新增三维验证。其他长度尺度与稿件未改。'
    report=bilingual_report(rows,groups,sensitivity_groups,previous,selected,candidates,scope)
    (out/'RESULTS_ZH_EN.txt').write_text(report,encoding='utf-8');outputs.append('RESULTS_ZH_EN.txt')
    sources.unchanged()
    provenance=dict(**inspection,selected_run=selected['run'],selected_J153=selected['terminal']['gap']['J'],
        selection_method='Explicit parent/auditor-reviewed run, independently checked against all converged terminals; no better saved complete candidate above the reporting buffer.',
        previous_best=previous['run'],common_axis_limits=[-limit,limit],gallery_axes_inches=layout,
        same_scale_and_area=True,saved_48_straight_edges=True,no_smoothing=True,
        failed_start_semantics='Not converged; gallery shows only the last complete evaluated shape if available, never labels it a local optimum.',
        helper_script_sha256=sha(ROOT/'scripts/plot_gap_closure_scan.py'),
        output_sha256={name:sha(out/name) for name in outputs},visual_review='PENDING_ACTUAL_OUTPUT_INSPECTION',human_verified=False)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='DERIVED_FIGURES_REQUIRE_VISUAL_INSPECTION',out=str(out),selected_run=selected['run'],
        selected_J153=selected['terminal']['gap']['J'],groups=groups,common_axis_limits=[-limit,limit]),indent=2))


if __name__=='__main__':main()
