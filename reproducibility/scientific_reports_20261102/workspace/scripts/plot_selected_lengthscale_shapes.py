"""Compare completed local designs at ell=0,0.5,1,2,5 micrometres.

The five axes have identical physical dimensions and common coordinate limits.
Saved 48-node polygons are joined by straight edges, without smoothing or individual
rescaling. The new low-length-scale study must finish successfully before plotting.
"""
from __future__ import annotations
from pathlib import Path
import argparse,csv,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_gap_closure_scan import Sources,polygon,spectrum,source_path,sha,locate

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_STUDY=ROOT/'05_validation/LOW_ELL_SHAPES_20261101/manifest.json'
DEFAULT_OUT=ROOT/'output/ell_0_0p5_1_2_5_shapes'
LEGACY={
    1.:ROOT/'02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1',
    2.:ROOT/'05_validation/V07_ell2_reoptimization_20261030/restart',
}
AREA=math.pi*.09
BUFFER=.001


def saved_vertices(path,sources,design):
    data=sources.rows(path)
    v=np.array([[float(r['x_over_a']),float(r['y_over_a'])] for r in data])
    assert v.shape==(48,2) and np.isfinite(v).all()
    assert np.max(abs(v))<.49
    area=.5*np.sum(v[:,0]*np.roll(v[:,1],-1)-v[:,1]*np.roll(v[:,0],-1))
    assert abs(area-AREA)<1e-12
    assert np.max(abs(v-polygon(design)))<1e-12,'Saved vertices differ from recorded design'
    return v


def read_run(folder,ell,sources,legacy_binding=None):
    mp=folder/'manifest.json';m=sources.json(mp)
    assert m['status'].startswith('PASS'),('Incomplete or failed local design',folder,m['status'])
    assert (m['N'],m['a_um'],m['h_um'],m['ell_epoxy_um'])==(15,75,6,ell)
    assert m.get('ell_steel_um',0)==0
    sp=folder/'final_ibz153.csv';vp=folder/'final_vertices.csv'
    for p in [sp,vp]:
        expected=m.get('output_sha256',{}).get(p.name)
        if expected is not None:assert sha(p)==expected
    if legacy_binding:
        for p in [mp,vp]:assert sha(p)==legacy_binding[locate(p)],('Legacy source changed',p)
    actual=spectrum(sp,sources)['gap']
    assert all(abs(actual[k]-m['final_ibz153'][k])<1e-9 for k in ['L','U','J'])
    v=saved_vertices(vp,sources,m['design'])
    result=dict(ell_um=float(ell),run=locate(folder),status=m['status'],vertices_path=locate(vp),
        spectrum_path=locate(sp),v=v,gap=actual,manifest=m,
        optimization_points=m.get('optimization_points',m.get('ibz_search_points',45)))
    # Only complete 153-point candidates can be compared with the terminal result.
    candidates=[]
    for c in m.get('full153_candidates',[]):
        p=(folder/c['spectrum']).resolve();assert p.is_relative_to(folder.resolve())
        assert sha(p)==m['output_sha256'][p.name]
        g=spectrum(p,sources)['gap']
        assert all(abs(g[k]-c[k])<1e-9 for k in ['L','U','J'])
        candidates.append(dict(J153=g['J'],spectrum=locate(p),design=c['design'],round=c['round']))
    if candidates:
        best=max(candidates,key=lambda c:c['J153'])
        assert abs(best['J153']-m['best_evaluated153']['J'])<1e-9
    else:
        best=dict(J153=actual['J'],spectrum=locate(sp),design=m['design'],round=None)
    result['best_evaluated']=best
    result['evaluated_candidate_scope']='All saved full 153-point candidates' if candidates else 'Only the saved terminal 153-point geometry is available in this legacy record'
    return result


def low_length_scale_cases(study,sources):
    assert study['status']=='COMPLETED_ALL_SEARCHES','Do not plot unfinished low-length-scale searches'
    assert (study['N'],study['a_um'],study['h_um'],study['ell_steel_um'],study['optimization_points'])==(15,75,6,0,153)
    assert study['length_scales_um']==[0,.5]
    assert sha(ROOT/'scripts/run_shape_comparison_case.py')==study['worker_sha256']
    assert sha(ROOT/'scripts/run_low_ell_shape_study.py')==study['controller_sha256']
    selected=[]
    for ell in [0.,.5]:
        records=[r for r in study['runs'] if r['ell_um']==ell]
        assert len(records)==2 and all(r['status'].startswith('PASS') and r['exit_code']==0 for r in records)
        runs=[]
        for r in records:
            folder=source_path(r['out']+'/manifest.json').parent
            assert sha(folder/'manifest.json')==r['manifest_sha256']
            item=read_run(folder,ell,sources)
            assert item['optimization_points']==153
            assert abs(item['gap']['J']-r['J153'])<1e-9
            for name,digest in item['manifest']['source_sha256'].items():assert sha(source_path(name))==digest
            runs.append(item)
        chosen=max(runs,key=lambda r:r['gap']['J'])
        case=next(c for c in study['cases'] if c['ell_um']==ell)
        assert case['selected_run']==chosen['run'] and abs(case['J153']-chosen['gap']['J'])<1e-9
        evaluated=max((r['best_evaluated'] for r in runs),key=lambda r:r['J153'])
        assert abs(evaluated['J153']-case['best_evaluated_J153'])<1e-9
        chosen['best_evaluated']=evaluated
        chosen['compared_terminal_runs']=[dict(run=r['run'],J153=r['gap']['J']) for r in runs]
        chosen['evaluated_candidate_scope']='All saved full 153-point candidates from both completed starts'
        selected.append(chosen)
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-manifest',type=Path,default=DEFAULT_STUDY)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args();sources=Sources();study_path=args.study_manifest.resolve()
    study=sources.json(study_path);items=low_length_scale_cases(study,sources)
    old_plot_path=ROOT/'output/ell1_to_5_shapes/provenance.json'
    legacy_binding=sources.json(old_plot_path)['source_sha256']
    for ell,folder in LEGACY.items():items.append(read_run(folder,ell,sources,legacy_binding))
    case5_path=ROOT/'05_validation/GAP_CLOSURE_SCAN_20261031/ell_5/case_summary.json'
    case5=sources.json(case5_path)
    assert case5['all_required_starts_converged']
    five=read_run(source_path(case5['selected_run']+'/manifest.json').parent,5.,sources,legacy_binding)
    assert abs(five['gap']['J']-case5['best_converged_J153'])<1e-9
    five['best_evaluated']=dict(J153=case5['best_vetted_J153'],spectrum=case5['best_vetted_candidate']['spectrum'],
        design=case5['best_vetted_candidate']['design'],round=case5['best_vetted_candidate']['round'])
    assert abs(spectrum(source_path(five['best_evaluated']['spectrum']),sources)['gap']['J']-five['best_evaluated']['J153'])<1e-9
    five['evaluated_candidate_scope']='Best full 153-point evaluated geometry recorded by the completed ell=5 case'
    items.append(five);items.sort(key=lambda d:d['ell_um'])
    assert [d['ell_um'] for d in items]==[0.,.5,1.,2.,5.]

    initial_path=LEGACY[1.]/'initial_vertices.csv'
    initial=saved_vertices(initial_path,sources,np.zeros(6));closed_initial=np.vstack([initial,initial[0]])
    maximum=max(float(np.max(abs(d['v']))) for d in items)
    limit=max(.4,math.ceil((maximum+.02)/.05)*.05)
    ticks=np.arange(-math.floor(limit/.2)*.2,math.floor(limit/.2)*.2+.01,.2)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    stem='ell_0_0p5_1_2_5_optimized_shapes';outputs=[stem+'.png',stem+'.svg','J153_summary.csv']
    if not args.overwrite:assert not any((out/name).exists() for name in outputs),'Use --overwrite to regenerate derived assets explicitly'
    style={'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'}
    layout=[]
    with plt.rc_context(style):
        fig=plt.figure(figsize=(12.3,9.2));width=.245;height=width*12.3/9.2
        slots=[[x,y,width,height] for x,y in [(.075,.57),(.39,.57),(.705,.57),(.2325,.15),(.5475,.15)]]
        axes=[]
        for item,slot in zip(items,slots):
            v=item['v'];closed=np.vstack([v,v[0]]);ax=fig.add_axes(slot);axes.append(ax)
            ax.fill(closed[:,0],closed[:,1],color='#D55E00',alpha=.09)
            ax.plot(closed_initial[:,0],closed_initial[:,1],color='#666666',ls='--',lw=1.2,zorder=2)
            ax.plot(closed[:,0],closed[:,1],color='#D55E00',lw=2,zorder=3)
            ax.scatter(v[:,0],v[:,1],color='#D55E00',s=8,zorder=4)
            ax.set(xlim=(-limit,limit),ylim=(-limit,limit),xlabel='$x/a$',ylabel='$y/a$',xticks=ticks,yticks=ticks,
                title=fr'$\ell = {item["ell_um"]:g}\,\mu\mathrm{{m}}$')
            ax.set_aspect('equal');ax.grid(alpha=.15)
        fig.suptitle('Locally optimized shapes at selected length scales ($N=15$)',fontsize=18)
        handles=[Line2D([],[],color='#D55E00',lw=2,marker='o',ms=3,label='Locally optimized 48-gon'),
            Line2D([],[],color='#666666',lw=1.2,ls='--',label='Initial equal-area 48-gon')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.042),ncol=2,frameon=False,fontsize=12)
        fig.text(.5,.021,r'Common scale and area; $a=75\,\mu$m, $h=6\,\mu$m. Saved vertices joined by straight edges.',
            ha='center',va='bottom',fontsize=10)
        fig.canvas.draw()
        for ax in axes:
            p=ax.get_position();layout.append(dict(rectangle=list(p.bounds),width_inches=p.width*12.3,height_inches=p.height*9.2))
        assert max(d['width_inches'] for d in layout)-min(d['width_inches'] for d in layout)<1e-10
        assert max(d['height_inches'] for d in layout)-min(d['height_inches'] for d in layout)<1e-10
        assert all(abs(d['width_inches']-d['height_inches'])<1e-10 for d in layout)
        for extension in ['png','svg']:fig.savefig(out/f'{stem}.{extension}',dpi=240,facecolor='white')
        plt.close(fig)

    table=[];reported=[]
    for d in items:
        difference=d['best_evaluated']['J153']-d['gap']['J']
        row=dict(ell_um=d['ell_um'],J153_terminal=d['gap']['J'],J153_best_evaluated=d['best_evaluated']['J153'],
            evaluated_minus_terminal_pp=difference,evaluated_exceeds_terminal_by_buffer=difference>BUFFER,
            L_MHz=d['gap']['L_MHz'],U_MHz=d['gap']['U_MHz'],optimization_points=d['optimization_points'],
            verification_points=153,selected_run=d['run'])
        table.append(row)
        if difference>BUFFER:reported.append(dict(ell_um=d['ell_um'],difference_pp=difference,best_evaluated=d['best_evaluated']))
    with (out/'J153_summary.csv').open('w',encoding='utf-8',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    sources.unchanged()
    record=dict(status='DERIVED_FROM_COMPLETED_LOCAL_DESIGNS',length_scales_um=[0,.5,1,2,5],N=15,a_um=75,h_um=6,ell_steel_um=0,
        common_area_fraction=AREA,common_axis_limits=[-limit,limit],maximum_saved_vertex_magnitude=maximum,
        axes_physical_layout=layout,all_axes_identical_physical_size=True,
        selections=[dict(**r,vertex_count=48,vertices=d['vertices_path'],spectrum=d['spectrum_path'],
            evaluated_candidate_scope=d['evaluated_candidate_scope'],compared_terminal_runs=d.get('compared_terminal_runs')) for r,d in zip(table,items)],
        evaluated_terminal_reporting_buffer_pp=BUFFER,evaluated_terminal_differences_to_report=reported,
        source_sha256=sources.hashes,plot_script_sha256=sha(Path(__file__)),helper_script_sha256=sha(ROOT/'scripts/plot_gap_closure_scan.py'),
        transformations=['Close each saved 48-node polygon by repeating the first vertex; no smoothing or individual geometric rescaling.',
            'Orange curves show selected converged terminal designs; gray dashed curves show the same initial equal-area polygon.',
            'Identical square axes and common limits give the same physical coordinate-to-pixel scale for all five panels.',
            'J153 values are recomputed from the complete 153-point spectra; search-grid sizes are explicitly retained in the CSV.',
            'New ell=0/0.5 designs are selected only after both starts at both length scales finish successfully.'],
        no_global_optimality_claim=True,new_eigensolves=False,new_COMSOL_runs=False,human_verified=False,
        output_sha256={name:sha(out/name) for name in outputs},visual_review='PENDING_ACTUAL_OUTPUT_INSPECTION')
    (out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(out/f'{stem}.png'),summary=table,evaluated_terminal_differences_to_report=reported,
        common_axis_limits=record['common_axis_limits']),indent=2))


if __name__=='__main__':main()
