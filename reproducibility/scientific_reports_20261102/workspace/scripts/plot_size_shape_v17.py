"""Create the N15 size/shape manuscript figure from saved scientific data only."""
from pathlib import Path
import csv,json,math,shutil,platform
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from pypdf import PdfReader
from plot_gap_closure_scan import Sources,polygon,spectrum,source_path,sha,locate,FACTOR

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'03_review/size_shape_v17'
OUT=ROOT/'01_manuscript/figures'
STEM='figure6_size_shape_v17'
DATA=ROOT/'output/data/Supplementary_Data_6_size_shape_N15.csv'
ENERGY=ROOT/'03_review/lengthscale_mechanism_20261101'
WIDTH_MM,HEIGHT_MM=164.,120.
CASES=[(0.,153,'05_validation/ELL0_MULTISTART_20261101/lhs_05'),
 (.5,153,'05_validation/LOW_ELL_SHAPES_20261101/ell_0p5/regular'),
 (1.,45,'02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1'),
 (2.,45,'05_validation/V07_ell2_reoptimization_20261030/restart'),
 (5.,153,'05_validation/GAP_CLOSURE_SCAN_20261031/ell_5/warm')]


def check_mapping(mapping,sources):
    for name,digest in mapping.items():
        p=source_path(name);sources.read(p);assert sha(p)==digest,(name,'Hash mismatch')


def load_case(ell,search,relative,sources):
    folder=ROOT/relative;mp=folder/'manifest.json';m=sources.json(mp)
    assert m['status'].startswith('PASS') and (m['N'],m['a_um'],m['h_um'],m['ell_epoxy_um'])==(15,75,6,ell)
    assert m.get('ell_steel_um',0)==0
    assert m.get('optimization_points',m.get('ibz_search_points',search))==search
    if ell==2:
        p=ROOT/'scripts/restart_ell2_optimization_v7.py';b=sources.read(p)
        assert sha(p)==m['source_sha256'][locate(p)] and b'base.ibz(8)' in b
    vp=folder/'final_vertices.csv';sp=folder/'final_ibz153.csv'
    for p in [vp,sp]:
        if p.name in m.get('output_sha256',{}):assert sha(p)==m['output_sha256'][p.name]
    vertices=np.array([[float(r['x_over_a']),float(r['y_over_a'])] for r in sources.rows(vp)])
    assert vertices.shape==(48,2) and np.max(abs(vertices-polygon(m['design'])))<1e-12
    spec=spectrum(sp,sources);raw=sources.rows(sp)
    assert [int(r['point']) for r in raw]==list(range(153))
    k=np.array([[float(r['kx_a']),float(r['ky_a'])] for r in raw]);expected=np.array([(i,j) for i in range(17) for j in range(i+1)])
    assert np.max(abs(k*16/math.pi-expected))<1e-10
    g=spec['gap'];assert all(abs(g[key]-m['final_ibz153'][key])<1e-9 for key in ['L','U','J'])
    f=np.array([[float(r[f'Omega{b}']) for b in range(1,7)] for r in raw])
    lower=list(map(int,np.flatnonzero(abs(f[:,3]-g['L'])<=1e-8)));upper=list(map(int,np.flatnonzero(abs(f[:,4]-g['U'])<=1e-8)))
    support={}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix in ['.json','.csv']:
            sources.read(p);support[locate(p)]=sha(p)
    unresolved=[]
    for name,digest in m.get('source_sha256',{}).items():
        p=Path(name);p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve()
        if p.is_relative_to(ROOT) and p.is_file():
            sources.read(p);assert sha(p)==digest;support[locate(p)]=digest
        else:unresolved.append(dict(original_path=name,sha256=digest))
    common=dict(ell_um=ell,N=15,shape_search_points=search,shape_evaluation_points=153,area_fraction=.09*math.pi,
        J153=g['J'],L_Omega=g['L'],U_Omega=g['U'],L_MHz=g['L_MHz'],U_MHz=g['U_MHz'],
        lower_tied_indices=lower,upper_tied_indices=upper,
        source_manifest=locate(mp),source_manifest_sha256=sha(mp),source_vertices=locate(vp),source_vertices_sha256=sha(vp),
        source_spectrum=locate(sp),source_spectrum_sha256=sha(sp),
        gap_notation='J153 log field equals manuscript J_{45}^{K_153}; 45 denotes bands 4 and 5, not sample count')
    return dict(ell=ell,search=search,folder=relative,m=m,v=vertices,raw=raw,g=g,common=common,
        supporting_files=support,archived_source_references=unresolved)


def load_energy(cases,sources):
    m=sources.json(ENERGY/'manifest.json')
    assert m['status']=='PASS_ACTIVE_EDGE_MODAL_ENERGY_N15' and (m['N'],m['a_um'],m['h_um'],m['ell_steel_um'])==(15,75,6,0)
    assert m['source_hash_check']==m['input_hash_check']=='PASS'
    assert m['source_sha256']==m['source_sha256_post'] and m['input_sha256']==m['input_sha256_post']
    check_mapping(m['source_sha256'],sources);check_mapping(m['input_sha256'],sources)
    for name,digest in m['output_sha256'].items():assert sha(ENERGY/name)==digest
    data=sources.rows(ENERGY/'active_modal_energy.csv');saved=sources.json(ENERGY/'active_modal_energy.json')
    assert len(data)==len(saved['modes'])==13
    chosen=[r for r in data if float(r['ell_um']) in [2.,5.]];assert len(chosen)==4
    for r in chosen:
        ell=float(r['ell_um']);case=next(c for c in cases if c['ell']==ell)
        origin=next(c for c in m['cases'] if c['ell_um']==ell)
        assert origin['worker_manifest']==case['common']['source_manifest'] and origin['design']==case['m']['design']
        index=int(r['source153_index']);band=int(r['band']);actual=float(case['raw'][index][f'Omega{band}'])
        assert abs(actual-float(r['Omega_saved']))<1e-10
        assert index in case['common']['lower_tied_indices'] if r['edge_role']=='lower_edge' else index in case['common']['upper_tied_indices']
        Ecl,Ecs,Etot=[float(r[key]) for key in ['classical_elastic_energy','MCST_elastic_energy','total_elastic_energy']]
        assert abs((Ecl+Ecs)/Etot-1)<1e-10 and abs(Ecs/Etot-float(r['R_cs']))<1e-10
        assert float(r['relative_residual'])<1e-7 and abs(float(r['mass_norm'])-1)<1e-10
        assert r['near_degeneracy']=='False' and r['single_mode_energy_interpretation']=='True'
    return m,chosen


def source_data(cases,energy):
    result=[]
    for c in cases:
        base=c['common']
        result.append(dict(record_type='gap_summary',figure_panel='b',**base,
            supporting_source_metadata=dict(files=c['supporting_files'],archived_source_references=c['archived_source_references'],design=c['m']['design'])))
        for i,v in enumerate(c['v']):result.append(dict(record_type='shape_vertex',figure_panel='a',**base,vertex_index=i,x_over_a=float(v[0]),y_over_a=float(v[1])))
        for r in c['raw']:
            i=int(r['point']);result.append(dict(record_type='spectrum_point',figure_panel='b',**base,point_index=i,
                kx_a=float(r['kx_a']),ky_a=float(r['ky_a']),**{f'Omega{b}':float(r[f'Omega{b}']) for b in range(1,7)},
                active_lower_1e8=i in base['lower_tied_indices'],active_upper_1e8=i in base['upper_tied_indices']))
    for r in energy:
        c=next(c for c in cases if c['ell']==float(r['ell_um']))
        result.append(dict(record_type='edge_modal_energy',figure_panel='c',**c['common'],point_index=int(r['source153_index']),
            edge_role=r['edge_role'],band=int(r['band']),point_label=r['point_label'],kx_a=float(r['kx_a']),ky_a=float(r['ky_a']),
            Omega_mode=float(r['Omega_fresh']),R_mc=float(r['R_cs']),source_energy_fraction_field='R_cs',classical_elastic_energy=float(r['classical_elastic_energy']),
            MCST_elastic_energy=float(r['MCST_elastic_energy']),total_elastic_energy=float(r['total_elastic_energy']),
            relative_residual=float(r['relative_residual']),mass_norm=float(r['mass_norm']),
            source_energy_manifest=locate(ENERGY/'manifest.json'),source_energy_manifest_sha256=sha(ENERGY/'manifest.json'),
            source_energy_csv=locate(ENERGY/'active_modal_energy.csv'),source_energy_csv_sha256=sha(ENERGY/'active_modal_energy.csv')))
    fields=list(dict.fromkeys(k for r in result for k in r))
    with DATA.open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
        w.writerows({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in result)
    assert Counter(r['record_type'] for r in result)==dict(gap_summary=5,shape_vertex=240,spectrum_point=765,edge_modal_energy=4)
    return result


def main():
    for p in [OUT,ROOT/'latex/figures',DATA.parent,REVIEW]:p.mkdir(parents=True,exist_ok=True)
    sources=Sources();cases=[load_case(*item,sources) for item in CASES];energy_manifest,energy=load_energy(cases,sources)
    rows=source_data(cases,energy)
    style={'font.family':'cmr10','font.size':7.5,'mathtext.fontset':'cm','axes.labelsize':8.,
        'xtick.labelsize':7.,'ytick.labelsize':7.,'legend.fontsize':7.,'axes.formatter.use_mathtext':True,
        'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'xtick.major.width':.65,'ytick.major.width':.65,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.45,'figure.facecolor':'white','axes.facecolor':'white'}
    physical=[]
    with plt.rc_context(style):
        fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4));w=.153;h=w*WIDTH_MM/HEIGHT_MM
        extent=max(.43,math.ceil((max(float(np.max(abs(c['v']))) for c in cases)+.01)/.05)*.05)
        initial=polygon(np.zeros(6));initial=np.vstack([initial,initial[0]])
        for i,c in enumerate(cases):
            a=fig.add_axes([.07+.18*i,.66,w,h]);v=c['v'];closed=np.vstack([v,v[0]])
            a.plot(initial[:,0],initial[:,1],color='#777777',ls='--',lw=.65)
            a.plot(closed[:,0],closed[:,1],color='#D55E00',lw=.9,marker='o',ms=1.35)
            a.set(xlim=(-extent,extent),ylim=(-extent,extent),xticks=[-.4,0,.4],yticks=[-.4,0,.4],
                xlabel='$x/a$',title=fr'$\ell={c["ell"]:g}\,\mu\mathrm{{m}}$')
            a.set_aspect('equal');a.tick_params(pad=2,length=2.5)
            if i:a.tick_params(labelleft=False)
            else:a.set_ylabel('$y/a$',labelpad=1.5)
            r=a.get_position();physical.append(dict(width_mm=r.width*WIDTH_MM,height_mm=r.height*HEIGHT_MM))
        assert max(r['width_mm'] for r in physical)-min(r['width_mm'] for r in physical)<1e-10
        assert all(abs(r['width_mm']-r['height_mm'])<1e-10 for r in physical)
        fig.text(.5,.95,r'Optimized shapes ($N=15$)',ha='center',fontsize=8.5)
        fig.legend(handles=[Line2D([],[],color='#D55E00',lw=.9,marker='o',ms=2,label='Optimized shape'),
            Line2D([],[],color='#777777',ls='--',lw=.65,label='Initial equal-area 48-gon')],
            loc='lower center',bbox_to_anchor=(.5,.548),ncol=2,frameon=False,handlelength=2.,columnspacing=2.)
        b=fig.add_axes([.11,.13,.36,.30]);b.plot([c['ell'] for c in cases],[c['g']['J'] for c in cases],
            color='#0072B2',lw=.9,marker='o',ms=3.5)
        b.set(xlabel=r'Epoxy length scale $\ell$ ($\mu$m)',ylabel='Relative gap (%)',xlim=(-.18,5.28),ylim=(0,40),
            xticks=[0,.5,1,2,5],xticklabels=['0','0.5','1','2','5'],yticks=[0,10,20,30,40])
        cax=fig.add_axes([.66,.13,.29,.30]);x=np.array([0.,1.]);offset=.16
        for role,color,shift,label in [('lower_edge','#0072B2',-offset,r'Band 4 at $X$ ($L$)'),
                                      ('upper_edge','#D55E00',offset,r'Band 5 at $\Gamma$ ($U$)')]:
            vals=[float(next(r for r in energy if float(r['ell_um'])==ell and r['edge_role']==role)['R_cs']) for ell in [2.,5.]]
            cax.bar(x+shift,vals,width=.27,color=color,label=label,zorder=3)
        cax.set(xticks=x,xticklabels=['2','5'],xlabel=r'Epoxy length scale $\ell$ ($\mu$m)',
            ylabel='Couple-stress energy\n'+r'fraction $R_{\mathrm{mc}}$',ylim=(0,.65),yticks=[0,.2,.4,.6],xlim=(-.6,1.6))
        cax.legend(loc='lower center',bbox_to_anchor=(.5,1.025),frameon=False,ncol=1,labelspacing=.25,handlelength=1.4)
        for letter,xpos,ypos in [('a',.02,.948),('b',.02,.50),('c',.55,.50)]:
            fig.text(xpos,ypos,letter,fontfamily='cmb10',fontsize=10.,va='bottom')
        fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
        for t in fig.findobj(match=matplotlib.text.Text):
            if t.get_visible() and t.get_text().strip():
                z=t.get_window_extent(renderer)
                if z.x0<fig.bbox.x0-1 or z.y0<fig.bbox.y0-1 or z.x1>fig.bbox.x1+1 or z.y1>fig.bbox.y1+1:outside.append(t.get_text())
        assert not outside,('Text outside fixed canvas',outside)
        fig.savefig(OUT/(STEM+'.png'),dpi=450,facecolor='white',transparent=False)
        fig.savefig(OUT/(STEM+'.pdf'),facecolor='white',transparent=False)
        plt.close(fig)
    target=ROOT/'latex/figures'/(STEM+'.pdf');shutil.copy2(OUT/(STEM+'.pdf'),target);assert sha(target)==sha(OUT/(STEM+'.pdf'))
    page=PdfReader(OUT/(STEM+'.pdf')).pages[0]
    dims=dict(width_mm=float(page.mediabox.width)*25.4/72,height_mm=float(page.mediabox.height)*25.4/72)
    assert abs(dims['width_mm']-WIDTH_MM)<1e-8 and abs(dims['height_mm']-HEIGHT_MM)<1e-8
    fonts=[dict(base_font=str(v.get_object().get('/BaseFont')),subtype=str(v.get_object().get('/Subtype'))) for v in page['/Resources']['/Font'].get_object().values()]
    assert all(v['subtype']!='/Type3' for v in fonts)
    with Image.open(OUT/(STEM+'.png')) as im:
        assert im.mode!='RGBA' or im.getchannel('A').getextrema()==(255,255)
        raster=dict(size_px=list(im.size),mode=im.mode,dpi=list(im.info.get('dpi',[])),fully_opaque=True)
    sources.unchanged();outputs=[OUT/(STEM+'.png'),OUT/(STEM+'.pdf'),target,DATA]
    meta=dict(figure='Figure 6',artifact='SIZE_SHAPE_N15_V17',N=15,a_um=75,h_um=6,ell_steel_um=0,
        ell_values_um=[c['ell'] for c in cases],all_shape_areas=.09*math.pi,common_shape_axis_limits=[-extent,extent],shape_axes_physical=physical,
        point_count_all_gap_values=153,search_points={str(c['ell']):c['search'] for c in cases},
        gap_notation='CSV J153 equals manuscript J_{45}^{K_153}; subscript45 denotes branches4/5; K153 is the153-point triangular set.',
        source_data_record_counts=dict(Counter(r['record_type'] for r in rows)),source_sha256=sources.hashes,
        selected_run_directories=[c['folder'] for c in cases],supporting_scientific_files={c['folder']:c['supporting_files'] for c in cases},
        legacy_unresolved_archived_code_references={c['folder']:c['archived_source_references'] for c in cases if c['archived_source_references']},
        plotting_script_sha256=sha(Path(__file__)),helper_script_sha256=sha(ROOT/'scripts/plot_gap_closure_scan.py'),
        output_sha256={locate(p):sha(p) for p in outputs},raster=raster,pdf_page=dims,pdf_fonts=fonts,
        energy_scope='Manuscript R_mc is MCST/total elastic energy of the active single mode at its own fixed geometry/k; not a derivative of the reoptimized gap/shape sequence.',
        energy_symbol_mapping='Input energy CSV field R_cs equals manuscript R_mc and new source-data field R_mc; values are unchanged.',
        scope='Five sampled local N15 designs with the same area,48 straight edges and six bounded variables; no global optimum,continuous-zone or interface-smoothing mechanism claim.',
        gap_line_semantics='Segments only guide the eye between the five evaluated states; no derivative or continuous optimization curve claimed.',
        initial_regular_geometry='Exact area-normalized polygon reconstructed from the zero six-vector; no smoothing.',
        artifact_operation_marker='Parent reported successful shared edit/4/pdf operation marker before first figure-PDF authoring; no duplicate child marker.',
        software=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),
        new_physical_solver_runs=0,N21_not_in_primary_figure=True,visual_review='PENDING_ACTUAL_PNG_AND_PDF_INSPECTION',human_verified=False)
    (REVIEW/'figure_provenance.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='GENERATED_REQUIRES_VISUAL_QA',outputs=[locate(p) for p in outputs],J153=[c['g']['J'] for c in cases],source_rows=len(rows),pdf_page=dims,fonts=fonts),indent=2))


if __name__=='__main__':main()
