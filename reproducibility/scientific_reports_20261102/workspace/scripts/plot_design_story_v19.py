"""Plot saved N15 design, transfer, solid-comparison and fixed-shape energies.

No spectral solve or optimization is run. Historical sources and assets are
checked before and after authoring; only the new v19 products are written.
"""
from pathlib import Path
import csv, hashlib, json, platform, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'03_review/narrative_v19'
OUT=ROOT/'01_manuscript/figures'
STEM='figure6_design_story_v19'
DATA=ROOT/'output/data/Figure6_design_story_v19.csv'
OLD=ROOT/'03_review/new_figure_provenance_v16.json'
REVIEW=ROOT/'03_review/order_unification_v16'
SCAN=[ROOT/'05_validation/V08_N15_20261030_v3/V08_summary.csv',
      ROOT/'05_validation/V08_N15_ell1p5_20261030/V08_summary.csv']
COMPARISON=REVIEW/'N15_vs_3D_point_comparison.csv'
OPT=ROOT/'05_validation/V07_ell2_reoptimization_20261030/restart'
ENERGY=ROOT/'05_validation/V08_edge_switch_20261030'
WIDTH_MM,HEIGHT_MM=164.,84.
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rel=lambda p:p.relative_to(ROOT).as_posix()
load=lambda p:json.loads(p.read_text(encoding='utf8'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    for folder in [V,OUT,DATA.parent,ROOT/'latex/figures']:folder.mkdir(parents=True,exist_ok=True)
    old=load(OLD);sources={}
    for name,digest in old['source_sha256'].items():
        p=ROOT/name;assert sha(p)==digest,(name,'Historical source changed');sources[name]=digest
    em=load(ENERGY/'manifest.json')
    assert em['status']=='COMPLETED' and em['shape']=='same ell1 terminal 48-gon'
    assert sha(ENERGY/'modal_energy_sensitivity.csv')==em['output_sha256']['modal_energy_sensitivity.csv']
    for name,digest in em['source_sha256'].items():
        assert sha(ROOT/name)==digest;sources[name]=digest
    for p in [OLD,ENERGY/'manifest.json',ENERGY/'modal_energy_sensitivity.csv',COMPARISON,OPT/'manifest.json']+SCAN:
        sources[rel(p)]=sha(p)
    protected=[ROOT/'scripts/plot_primary_order_v16.py',ROOT/'scripts/plot_shape_evidence_v18.py',
      ROOT/'01_manuscript/figures/figure6_primary_N15_v16.pdf',
      ROOT/'latex/figures/figure6_primary_N15_v16.pdf',
      ROOT/'01_manuscript/figures/figure6_shape_evidence_v18.pdf',
      ROOT/'latex/figures/figure6_shape_evidence_v18.pdf',
      ROOT/'output/data/Supplementary_Data_5_primary_order_comparison.csv',
      ROOT/'output/data/Supplementary_Data_6_size_shape_N15.csv',
      ROOT/'output/data/Supplementary_Data_8_shape_lengthscale_cross.csv']
    preserved={rel(p):sha(p) for p in protected}
    scan=[dict(r,source=rel(p),source_sha256=sha(p)) for p in SCAN for r in rows(p)]
    assert len(scan)==8
    for shape in ['initial','final']:
        selected=[r for r in scan if r['shape']==shape]
        assert sorted(float(r['ell_um']) for r in selected)==[.5,1.,1.5,2.]
        assert all(int(r['N'])==15 and int(r['points'])==45 for r in selected)
        for r in selected:assert abs(float(r['J'])-200*(float(r['U'])-float(r['L']))/(float(r['U'])+float(r['L'])))<1e-11
    opt=load(OPT/'manifest.json');assert opt['N']==15 and opt['status']=='PASS_SAMPLED_N15_LOCAL_SQP'
    assert abs(opt['final_ibz45']['J']-opt['final_ibz153']['J'])<1e-12
    comp=rows(COMPARISON)
    fine=[r for r in comp if (int(r['mesh_level']),int(r['layers']),int(r['requested_eigenpairs']))==(2,4,24)]
    categories=[('initial','Gamma'),('initial','X'),('final','Gamma'),('final','X')]
    errors=[next(r for r in fine if r['shape']==s and r['point']==k and int(r['band'])==b) for s,k in categories for b in [4,5]]
    assert len(errors)==8
    for r in errors:
        assert abs(100*(float(r['frequency_pwe_N15_MHz'])/float(r['frequency_3d_MHz'])-1)-float(r['pwe_N15_minus_3d_percent']))<1e-12
    energy=[r for r in rows(ENERGY/'modal_energy_sensitivity.csv') if int(r['N'])==15 and (r['point'],int(r['band'])) in [('X',4),('X',5),('Gamma',5)]]
    assert len(energy)==12
    for r in energy:
        assert r['shape']=='ell1_terminal_frozen' and float(r['relative_residual'])<1e-7
        assert float(r['dense_first6_relative_drift'])<1e-7
        assert abs(float(r['R_mc'])-float(r['S_ell']))<1e-15 and 0<=float(r['R_mc'])<=1
        ell=float(r['ell_um']);p=(SCAN[1].parent/f'final_ell{ell:g}_N15_ibz45.csv' if ell==1.5 else SCAN[0].parent/f'final_ell{ell:g}_N15_ibz45.csv')
        k=rows(p)[0 if r['point']=='Gamma' else 36]
        assert abs(float(k[f"Omega{r['band']}"])-float(r['Omega']))<1e-11
        sources[rel(p)]=sha(p)

    derived=[]
    for r in scan:derived.append(dict(record_type='gap_transfer',panel='a',N=15,ell_um=r['ell_um'],shape=r['shape'],sampling_points=45,L_Omega=r['L'],U_Omega=r['U'],J_percent=r['J'],source=r['source'],source_sha256=r['source_sha256']))
    derived.append(dict(record_type='gap_reoptimized',panel='a',N=15,ell_um=2,shape='separately_optimized_ell2',sampling_points=45,L_Omega=opt['final_ibz45']['L'],U_Omega=opt['final_ibz45']['U'],J_percent=opt['final_ibz45']['J'],source=rel(OPT/'manifest.json'),source_sha256=sha(OPT/'manifest.json')))
    for r in errors:derived.append(dict(record_type='point_frequency_difference',panel='b',N=15,ell_um=2,shape=r['shape'],point=r['point'],band=r['band'],frequency_pwe_MHz=r['frequency_pwe_N15_MHz'],frequency_3d_MHz=r['frequency_3d_MHz'],difference_percent=r['pwe_N15_minus_3d_percent'],solid_mesh='M2L4',solid_requested_eigenpairs=24,source=rel(COMPARISON),source_sha256=sha(COMPARISON),source_pwe_spectrum=r['N15_source'],source_raw_3d_spectrum_sha256=r['raw_spectra_sha256']))
    for r in energy:derived.append(dict(record_type='fixed_geometry_modal_energy',panel='c',N=15,ell_um=r['ell_um'],shape='unchanged_ell1_design',point=r['point'],band=r['band'],Omega=r['Omega'],R_mc=r['R_mc'],S_ell=r['S_ell'],relative_residual=r['relative_residual'],dense_first6_relative_drift=r['dense_first6_relative_drift'],source=rel(ENERGY/'modal_energy_sensitivity.csv'),source_sha256=sha(ENERGY/'modal_energy_sensitivity.csv')))
    fields=list(dict.fromkeys(k for r in derived for k in r))
    with DATA.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(derived)
    assert len(derived)==29
    style={'font.family':'DejaVu Sans','font.size':7.2,'axes.labelsize':7.3,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
           'legend.fontsize':6.3,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,
           'pdf.fonttype':42,'ps.fonttype':42,'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.45,'figure.facecolor':'white','axes.facecolor':'white'}
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,3,figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4),layout='constrained',gridspec_kw={'width_ratios':[1.1,1.03,1.02]})
        a,b,c=axes
        for shape,color,marker,label in [('initial','#666666','o','Initial'),('final','#0072B2','s',r'$\ell=1$ design')]:
            data=sorted((r for r in scan if r['shape']==shape),key=lambda r:float(r['ell_um']))
            a.plot([float(r['ell_um']) for r in data],[float(r['J']) for r in data],color=color,marker=marker,ms=3.4,lw=.95,label=label)
        a.plot(2,opt['final_ibz45']['J'],marker='*',ms=8,color='#D55E00',markeredgewidth=.5,ls='none',label=r'$\ell=2$ reoptimized')
        a.set(xlabel=r'Epoxy length scale $\ell$ ($\mu$m)',ylabel=r'Relative gap $J_{45}$ (%)',xticks=[.5,1,1.5,2],xlim=(.4,2.25),ylim=(23.8,35.1),yticks=[24,26,28,30,32,34])
        a.legend(loc='lower center',bbox_to_anchor=(.5,1.02),frameon=False,labelspacing=.25,handlelength=1.6)
        for band,color,marker in [(4,'#0072B2','o'),(5,'#D55E00','s')]:
            vals=[float(next(r for r in errors if r['shape']==s and r['point']==k and int(r['band'])==band)['pwe_N15_minus_3d_percent']) for s,k in categories]
            b.plot(np.arange(4)+(band-4.5)*.15,vals,ls='none',color=color,marker=marker,ms=3.8,label=f'Band {band}')
        b.axhline(0,color='#444444',lw=.65,ls='--')
        b.set(xticks=np.arange(4),xticklabels=['Initial\n$\\Gamma$','Initial\n$X$','Reopt.\n$\\Gamma$','Reopt.\n$X$'],ylabel='PWE - 3D frequency\ndifference (%)',xlim=(-.45,3.45),ylim=(-1.15,1.5),yticks=[-1,-.5,0,.5,1,1.5])
        b.legend(loc='lower center',bbox_to_anchor=(.5,1.02),frameon=False,labelspacing=.25,handlelength=1.6)
        b.text(.5,1.21,r'$\ell=2\,\mu$m; 3D: M2L4',transform=b.transAxes,ha='center',va='bottom',fontsize=6.4)
        for point,band,color,marker,ls,label in [('X',4,'#D55E00','^','-',r'$X$, band 4'),('X',5,'#0072B2','o','--',r'$X$, band 5'),('Gamma',5,'#7B3294','s',':',r'$\Gamma$, band 5')]:
            data=sorted((r for r in energy if r['point']==point and int(r['band'])==band),key=lambda r:float(r['ell_um']))
            c.plot([float(r['ell_um']) for r in data],[float(r['R_mc']) for r in data],color=color,marker=marker,ms=3.3,lw=.95,ls=ls,label=label)
        c.set(xlabel=r'Epoxy length scale $\ell$ ($\mu$m)',ylabel=r'Couple-stress fraction $R_{\mathrm{mc}}$',xticks=[.5,1,1.5,2],xlim=(.4,2.1),ylim=(0,.30),yticks=[0,.1,.2,.3])
        c.legend(loc='lower center',bbox_to_anchor=(.5,1.02),frameon=False,labelspacing=.25,handlelength=1.8)
        for label,axis in zip('abc',axes):axis.text(-.20,1.22,label,transform=axis.transAxes,fontweight='bold',fontsize=9.5,va='bottom')
        fig.get_layout_engine().set(w_pad=.025,h_pad=.04,wspace=.06)
        fig.canvas.draw();renderer=fig.canvas.get_renderer();clipped=[]
        for artist in fig.findobj(match=matplotlib.text.Text):
            if artist.get_visible() and artist.get_text().strip():
                bb=artist.get_window_extent(renderer)
                if bb.x0<fig.bbox.x0-1 or bb.y0<fig.bbox.y0-1 or bb.x1>fig.bbox.x1+1 or bb.y1>fig.bbox.y1+1:clipped.append(artist.get_text())
        assert not clipped,clipped
        fig.savefig(OUT/(STEM+'.png'),dpi=450,facecolor='white',transparent=False)
        fig.savefig(OUT/(STEM+'.pdf'),facecolor='white',transparent=False)
        plt.close(fig)
    target=ROOT/'latex/figures'/(STEM+'.pdf');shutil.copy2(OUT/(STEM+'.pdf'),target)
    with Image.open(OUT/(STEM+'.png')) as im:
        assert im.mode!='RGBA' or im.getchannel('A').getextrema()==(255,255)
        raster=dict(size_px=list(im.size),dpi=list(im.info.get('dpi',[])),mode=im.mode,fully_opaque=True)
    page=PdfReader(target).pages[0]
    physical=dict(width_mm=float(page.mediabox.width)*25.4/72,height_mm=float(page.mediabox.height)*25.4/72)
    assert abs(physical['width_mm']-WIDTH_MM)<1e-8 and abs(physical['height_mm']-HEIGHT_MM)<1e-8
    fonts=[dict(base_font=str(v.get_object().get('/BaseFont')),subtype=str(v.get_object().get('/Subtype'))) for v in page['/Resources']['/Font'].get_object().values()]
    assert all(r['subtype']!='/Type3' for r in fonts)
    assert all(sha(ROOT/name)==digest for name,digest in sources.items())
    assert all(sha(ROOT/name)==digest for name,digest in preserved.items())
    outputs=[OUT/(STEM+'.png'),OUT/(STEM+'.pdf'),target,DATA]
    record=dict(status='GENERATED_REQUIRES_SEPARATE_ACTUAL_VISUAL_REVIEW',figure='Figure6',width_mm=WIDTH_MM,height_mm=HEIGHT_MM,
      N=15,new_physical_solver_runs=0,new_optimization_runs=0,new_COMSOL_runs=0,
      source_sha256=sources,old_assets_preserved_sha256=preserved,source_hash_check='PASS',
      plotting_script=rel(Path(__file__)),plotting_script_sha256=sha(Path(__file__)),
      output_sha256={rel(p):sha(p) for p in outputs},derived_csv_record_counts={'scan_gap':8,'reoptimized_gap':1,'signed_frequency_difference':8,'fixed_shape_modal_energy':12},
      panel_a_scope='Saved initial and unchanged ell1 design on45points at four prescribed ell; only one separately reoptimized point atell2. Lines guide the eye.',
      panel_b_scope='Eight M2L4/24-eigenpair frequencies at Gamma/X; signed100*(N15PWE/3D-1), ell2; no field diagnostic relabelled fromN21.',
      panel_c_scope='Twelve N15 modal energies of the same ell1 design atGamma5/X5/X4; S_ell=R_mc fixed-geometry/fixed-wavevector simple-mode identity. No morphology mechanism inferred.',
      semantics='45 denotes branches4/5 in J45; sampling_points=45 identifies the grid separately.',
      source_data=rel(DATA),source_data_rows=len(derived),normalization=load(REVIEW/'figure_numeric_check.json')['normalization'],
      PDF_page=physical,PDF_fonts=fonts,raster=raster,text_outside_page=[],
      lines='Only the four computed parameter points are joined; no smoothing, new interpolation samples, or uncertainty intervals.',
      marker='Parent reported successful edit/4/pdf marker before this authoring; no duplicate child marker.',
      skill='scientific-visualization, v1.4; general manuscript figure, not automated journal compliance',
      runtime=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),human_verified=False)
    (V/'figure_provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=record['status'],outputs=record['output_sha256'],records=len(derived),source_files=len(sources)),indent=2))

if __name__=='__main__':main()
