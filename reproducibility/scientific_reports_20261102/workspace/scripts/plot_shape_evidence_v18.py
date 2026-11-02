"""Create the v18 controlled shape/length-scale figure from complete saved spectra.

The old v17 program/assets and Data6 are read-only. Full-PASS off-diagonal
records are required before this program creates any new manuscript artifact.
"""
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
REVIEW=ROOT/'03_review/shape_mechanism_v18'
OUT=ROOT/'01_manuscript/figures'
STEM='figure6_shape_evidence_v18'
DATA6=ROOT/'output/data/Supplementary_Data_6_size_shape_N15.csv'
DATA=ROOT/'output/data/Supplementary_Data_8_shape_lengthscale_cross.csv'
CROSS=ROOT/'05_validation/SHAPE_LENGTHSCALE_CROSS_V18_20261101'
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



def load_data6_cases(sources):
    """Use the published source-data rows for a/b and cross-check their originals."""
    rows=sources.rows(DATA6)
    assert Counter(r['record_type'] for r in rows)==dict(gap_summary=5,shape_vertex=240,spectrum_point=765,edge_modal_energy=4)
    cases=[load_case(*item,sources) for item in CASES]
    for c in cases:
        chosen=[r for r in rows if float(r['ell_um'])==c['ell']]
        g=next(r for r in chosen if r['record_type']=='gap_summary')
        assert (int(g['N']),int(g['shape_search_points']),int(g['shape_evaluation_points']))==(15,c['search'],153)
        vrows=sorted((r for r in chosen if r['record_type']=='shape_vertex'),key=lambda r:int(r['vertex_index']))
        v=np.array([[float(r['x_over_a']),float(r['y_over_a'])] for r in vrows])
        assert len(vrows)==48 and np.max(abs(v-c['v']))<1e-13
        srows=sorted((r for r in chosen if r['record_type']=='spectrum_point'),key=lambda r:int(r['point_index']))
        assert len(srows)==153
        f=np.array([[float(r[f'Omega{b}']) for b in range(1,7)] for r in srows])
        raw=np.array([[float(r[f'Omega{b}']) for b in range(1,7)] for r in c['raw']])
        assert np.max(abs(f-raw))<1e-13
        assert abs(float(g['J153'])-c['g']['J'])<1e-12
        assert g['source_spectrum_sha256']==sha(ROOT/g['source_spectrum'])
        assert g['source_vertices_sha256']==sha(ROOT/g['source_vertices'])
        # The plotted arrays and scalars below are explicitly from Data6.
        c['v']=v;c['g']['J']=float(g['J153']);c['data6_gap_row']=g
    return cases


def load_cross(cases,sources):
    p=CROSS/'manifest.json';m=sources.json(p)
    assert m['status']=='PASS_FROZEN_LENGTHSCALE_2X2_N15',(m['status'],'Incomplete cross cannot be plotted')
    assert m['source_hash_check']=='PASS' and m['fresh_cross_points']==306 and m['diagonal_fresh_spotchecks']==6
    assert not m['new_optimization'] and not m['new_COMSOL_runs']
    check_mapping(m['source_sha256'],sources);check_mapping(m['output_sha256'],sources)
    rows=[]
    for shape,job in [('g0','g0_at5'),('g5','g5_at0')]:
        c=next(c for c in cases if c['ell']==(0 if shape=='g0' else 5))
        wp=CROSS/job/'manifest.json';w=sources.json(wp)
        assert w['status']=='PASS_FROZEN_SHAPE_LENGTHSCALE_CROSS_N15'
        assert (w['N'],w['a_um'],w['h_um'],w['ell_steel_um'],w['shape_label'])==(15,75,6,0,shape)
        assert w['source_hash_check']==w['input_hash_check']=='PASS'
        assert w['source_sha256']==w['source_sha256_post'] and w['input_sha256']==w['input_sha256_post']
        assert w['progress']==dict(completed_points=153,total_points=153)
        check_mapping(w['source_sha256'],sources);check_mapping(w['input_sha256'],sources)
        for name,digest in w['output_sha256'].items():
            q=CROSS/job/name;sources.read(q);assert sha(q)==digest
        for evaluation,kind,sp,reference in [
            (w['own_ell_um'],'SAVED_OWN_ELL_DIAGONAL',CROSS/f'{shape}_saved_diagonal_ibz153.csv',w['own_diagonal']),
            (w['evaluation_ell_um'],'FRESH_FROZEN_GEOMETRY_CROSS',CROSS/job/f'{job}_ibz153.csv',w['result'])]:
            result=spectrum(sp,sources)['gap']
            assert all(abs(result[key]-reference[key])<1e-10 for key in ['L','U','J'])
            if kind=='SAVED_OWN_ELL_DIAGONAL':
                assert sha(sp)==c['common']['source_spectrum_sha256']
            row=dict(shape_label=shape,design_origin_ell_um=c['ell'],evaluation_ell_um=evaluation,
                provenance=kind,N=15,a_um=75,h_um=6,ell_steel_um=0,shape_search_points=c['search'],
                evaluation_points=153,L_Omega=result['L'],U_Omega=result['U'],J45_K153_percent=result['J'],
                L_MHz=result['L_MHz'],U_MHz=result['U_MHz'],
                source_spectrum=locate(sp),source_spectrum_sha256=sha(sp),
                source_cross_manifest=locate(p),source_cross_manifest_sha256=sha(p),
                source_worker_manifest=locate(wp),source_worker_manifest_sha256=sha(wp),
                source_shape_manifest=c['common']['source_manifest'],source_shape_manifest_sha256=c['common']['source_manifest_sha256'],
                source_vertices=c['common']['source_vertices'],source_vertices_sha256=c['common']['source_vertices_sha256'],
                fixed_geometry=True,new_optimization=False,new_COMSOL_runs=False)
            rows.append(row)
    assert {(r['shape_label'],r['evaluation_ell_um']) for r in rows}=={('g0',0),('g0',5),('g5',0),('g5',5)}
    rows.sort(key=lambda r:(r['evaluation_ell_um'],r['shape_label']))
    with DATA.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return m,rows

def main():
    for p in [OUT,ROOT/'latex/figures',DATA.parent,REVIEW]:p.mkdir(parents=True,exist_ok=True)
    immutable_v17=[ROOT/'scripts/plot_size_shape_v17.py',DATA6,OUT/'figure6_size_shape_v17.png',OUT/'figure6_size_shape_v17.pdf',ROOT/'latex/figures/figure6_size_shape_v17.pdf']
    preserved_v17={locate(p):sha(p) for p in immutable_v17}
    sources=Sources();cases=load_data6_cases(sources);cross_manifest,rows=load_cross(cases,sources)
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
        for shape,color,hatch,shift,label in [
                ('g0','#0072B2','//',-offset,r'Design from $\ell=0$ ($g_0$)'),
                ('g5','#D55E00','',offset,r'Design from $\ell=5\,\mu$m ($g_5$)')]:
            vals=[next(r['J45_K153_percent'] for r in rows if r['shape_label']==shape and r['evaluation_ell_um']==ell) for ell in [0.,5.]]
            bars=cax.bar(x+shift,vals,width=.27,color=color,hatch=hatch,edgecolor='white',linewidth=.4,label=label,zorder=3)
            cax.bar_label(bars,fmt='%.2f',padding=2,fontsize=6.5)
        allvalues=[r['J45_K153_percent'] for r in rows]
        ymin=min(0.,math.floor(min(allvalues)/10)*10.);ymax=max(0.,math.ceil(max(allvalues)/10)*10.)
        # Signed data retain zero and all negative observations on the linear axis.
        if ymin==ymax:ymin-=1;ymax+=1
        cax.axhline(0.,color='black',lw=.65,zorder=4)
        cax.set(xticks=x,xticklabels=['0','5'],xlabel=r'Evaluation length scale $\ell$ ($\mu$m)',
            ylabel='Signed relative separation (%)',ylim=(ymin-1.5,ymax+1.5),xlim=(-.6,1.6))
        cax.set_yticks(np.arange(ymin,ymax+.01,10.))
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
    sources.unchanged();assert preserved_v17=={locate(p):sha(p) for p in immutable_v17}
    outputs=[OUT/(STEM+'.png'),OUT/(STEM+'.pdf'),target,DATA]
    meta=dict(figure='Figure 6',artifact='SHAPE_LENGTHSCALE_CONTROLLED_EVIDENCE_V18',N=15,a_um=75,h_um=6,ell_steel_um=0,
        target_journal='Scientific Reports research manuscript, author-review revision; no compliance certification',
        intended_width_mm=WIDTH_MM,deterministic_estimates='Saved numerical spectra; no statistical replicates or invented error bars; solver residual gates remain in raw manifests.',
        ell_values_um=[c['ell'] for c in cases],all_shape_areas=.09*math.pi,
        common_shape_axis_limits=[-extent,extent],shape_axes_physical=physical,
        point_count_all_gap_values=153,search_points={str(c['ell']):c['search'] for c in cases},
        source_Data6=locate(DATA6),source_Data6_sha256=sha(DATA6),
        source_Data6_read_only=True,old_v17_script_sha256=sha(ROOT/'scripts/plot_size_shape_v17.py'),
        v17_preserved_sha256=preserved_v17,v17_source_and_figure_unchanged=True,
        primary_panel_data={'a':'Data6 actual 48 saved polygon vertices, unsmoothed','b':'Data6 five common153 final gap summaries','c':'Data8 four signed gaps independently recomputed from complete actual153 raw spectra'},
        gap_notation='J45_K153_percent equals manuscript J_{45}^{K_153}; 45 denotes branches4/5; K153 is the153-point set.',
        cross_results=rows,cross_controller_manifest=locate(CROSS/'manifest.json'),cross_controller_status=cross_manifest['status'],
        cross_fixed_geometry=True,cross_new_optimization=False,cross_fresh_spectral_points=306,cross_diagonal_spotchecks=6,
        cross_panel_legend='g0 and g5 identify design origin; x identifies material length scale during evaluation; color plus hatch is redundant.',
        cross_y_limits=[ymin-1.5,ymax+1.5],cross_zero_line=True,cross_negative_values_retained=True,
        source_sha256=sources.hashes,selected_run_directories=[c['folder'] for c in cases],
        supporting_scientific_files={c['folder']:c['supporting_files'] for c in cases},
        legacy_unresolved_archived_code_references={c['folder']:c['archived_source_references'] for c in cases if c['archived_source_references']},
        plotting_script_sha256=sha(Path(__file__)),helper_script_sha256=sha(ROOT/'scripts/plot_gap_closure_scan.py'),
        output_sha256={locate(p):sha(p) for p in outputs},raster=raster,pdf_page=dims,pdf_fonts=fonts,
        scope='Endpoint cross-evaluation tests shape/parameter performance matching only; no circularization mechanism,global optimum,continuous-zone or parameterization-independent-shape claim.',
        gap_line_semantics='Panelb segments only guide the eye between five actual evaluated states; no continuous envelope asserted.',
        alt_text='Five true48-vertex optimized interfaces at fixed steel area; common153-point relative widths decrease across the five computed material lengths. The endpoint-design panel holds material length fixed for each group and compares design origins g0 and g5 on a shared signed linear gap scale.',
        removed_energy_panel='The v17 fixed-shape R_mc bars are not used as an explanation of optimizer geometry; original Data6/energy records remain untouched.',
        initial_regular_geometry='Exact area-normalized polygon reconstructed from the zero six-vector; no smoothing.',
        artifact_operation_marker='Parent reported successful v18 edit/4/pdf marker before first figure authoring; no duplicate marker.',
        software=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),
        new_physical_solver_runs_by_plotting_program=0,N21_not_in_primary_figure=True,
        visual_review='PENDING_ACTUAL_PNG_AND_PDF_INSPECTION',human_verified=False)
    (REVIEW/'figure_provenance.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='GENERATED_REQUIRES_VISUAL_QA',outputs=[locate(p) for p in outputs],J153=[c['g']['J'] for c in cases],cross_rows=rows,pdf_page=dims,fonts=fonts),indent=2))


if __name__=='__main__':main()
