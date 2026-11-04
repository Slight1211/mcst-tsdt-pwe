"""Manuscript figures from completed ell2 spectra and physical field comparisons."""
from pathlib import Path
import csv,hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];V=ROOT/'05_validation/V01_ell2_20261031_v8'
OUT=ROOT/'01_manuscript/figures';OPT=ROOT/'05_validation/V07_ell2_reoptimization_20261030'
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=load(V/'analysis.json')
assert report['complete_requested_case_set'],'Do not plot a partial requested case set'
q=load(OPT/'restart/manifest.json');hi=load(OPT/'N21/manifest.json')
scan=rows(ROOT/'05_validation/V08_N15_20261030_v3/V08_summary.csv')+rows(ROOT/'05_validation/V08_N15_ell1p5_20261030/V08_summary.csv')
high=rows(ROOT/'05_validation/V08_N21_ell2_20261030/V08_summary.csv')
fine=[r for r in rows(V/'point_comparison.csv') if int(r['mesh_level'])==2 and int(r['layers'])==4 and int(r['requested_eigenpairs'])==24]
style={'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':8,'legend.fontsize':6.4,
       'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
       'axes.grid':True,'grid.alpha':.22}
with plt.rc_context(style):
    fig,ax=plt.subplots(1,2,figsize=(164/25.4,68/25.4),layout='constrained')
    # Main-text parameter transfer precedes its new solid-model check.
    ax=[ax[1],ax[0]]
    cats=[('initial','Gamma'),('initial','X'),('final','Gamma'),('final','X')]
    for b,color,marker in [(4,'#0072B2','o'),(5,'#D55E00','s')]:
        vals=[float(next(r for r in fine if r['shape']==s and r['point']==k and int(r['band'])==b)['pwe_minus_3d_percent']) for s,k in cats]
        ax[0].plot(np.arange(4)+(b-4.5)*.12,vals,marker,ls='none',color=color,label=f'Band {b}',ms=4.5)
    ax[0].axhline(0,color='#555555',lw=.7,ls='--')
    ax[0].set(xticks=np.arange(4),xticklabels=['Initial\n$\\Gamma$','Initial\n$X$','Reopt.\n$\\Gamma$','Reopt.\n$X$'],
              ylabel='PWE-3D frequency difference (%)')
    ax[0].legend(loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.16))
    ax[0].text(0,1.03,'b',transform=ax[0].transAxes,fontweight='bold',fontsize=10)
    for shape,color,marker,label in [('initial','#666666','o','Initial'),('final','#0072B2','s','$\\ell=1$ design')]:
        selected=sorted([r for r in scan if r['shape']==shape],key=lambda r:float(r['ell_um']))
        ax[1].plot([float(r['ell_um']) for r in selected],[float(r['J']) for r in selected],color=color,marker=marker,ms=4,lw=1,label=label+' ($N=15$)')
        r=next(r for r in high if r['shape']==shape)
        ax[1].plot(2,float(r['J']),ls='none',marker='D',mfc='white',mec=color,ms=5,label=label+' ($N=21$)')
    ax[1].plot(2,q['final_ibz45']['J'],marker='*',ms=9,color='#D55E00',ls='none',label='Reoptimized ($N=15$)')
    ax[1].plot(2,hi['final']['J'],marker='*',ms=9,mec='#D55E00',mfc='white',ls='none',label='Reoptimized ($N=21$)')
    for ell in [.5,1.,1.5,2.]:
        a=next(r for r in scan if r['shape']=='initial' and float(r['ell_um'])==ell)
        b=next(r for r in scan if r['shape']=='final' and float(r['ell_um'])==ell)
        ax[1].annotate(f'+{float(b["J"])-float(a["J"]):.2f}',(ell,(float(a['J'])+float(b['J']))/2),xytext=(4,0),textcoords='offset points',fontsize=6.3)
    ax[1].annotate(f'+{q["gain_from_circle_pp"]:.2f} pp',xy=(2,q['final_ibz45']['J']),xytext=(1.1,28.5),
                   arrowprops=dict(arrowstyle='-',lw=.7,color='#D55E00'),color='#A34200',fontsize=6.8)
    ax[1].set(xlabel='Epoxy length scale $\\ell$ ($\\mu$m)',ylabel='Sampled relative gap $J_{45}$ (%)',
              xticks=[.5,1,1.5,2],xlim=(.38,2.35),ylim=(23.8,35.1))
    ax[1].legend(loc='upper center',bbox_to_anchor=(.5,1.43),ncol=2,frameon=False,fontsize=6.1,handlelength=1.3,columnspacing=.8)
    ax[1].text(0,1.03,'a',transform=ax[1].transAxes,fontweight='bold',fontsize=10)
    # Tight bounding boxes intentionally include the external legend; TeX fixes final width.
    fig.canvas.draw()
    main=OUT/'figure6_ell2_3d_v8.pdf'
    fig.savefig(main,bbox_inches='tight',pad_inches=.035,facecolor='white')
    fig.savefig(main.with_suffix('.png'),dpi=260,bbox_inches='tight',pad_inches=.035,facecolor='white');plt.close(fig)
    gains=rows(V/'candidate_gains.csv')
    gains=sorted(gains,key=lambda r:(int(r['layers']),-int(r['mesh_level'])))
    fig,ax=plt.subplots(1,2,figsize=(164/25.4,68/25.4),layout='constrained')
    for key,label,marker,color in [('initial','Initial','o','#666666'),('final','Reoptimized','s','#0072B2')]:
        ax[0].plot(np.arange(len(gains)),[float(r[key]) for r in gains],marker=marker,label=label,color=color,lw=.9)
    for i,r in enumerate(gains):
        ax[0].annotate(f'{float(r["gain_pp"]):+.2f} pp',(i,(float(r['initial'])+float(r['final']))/2),
                       xytext=(-3 if i==len(gains)-1 else 3,0),ha='right' if i==len(gains)-1 else 'left',textcoords='offset points',fontsize=7)
    ax[0].set(xticks=np.arange(len(gains)),xticklabels=[f'M{r["mesh_level"]}L{r["layers"]}\n'+('$\\Gamma XM$' if r['points']=='GXM' else '$\\Gamma X$') for r in gains],
              ylabel='3D candidate relative separation (%)')
    ax[0].legend(loc='upper center',bbox_to_anchor=(.5,1.18),frameon=False,ncol=2)
    cats=[(s,k) for s in ['initial','final'] for k in ['Gamma','X','M']]
    for b,marker,color in [(4,'o','#0072B2'),(5,'s','#D55E00')]:
        vals=[float(next(r for r in fine if r['shape']==s and r['point']==k and int(r['band'])==b)['subspace_projection']) for s,k in cats]
        ax[1].plot(np.arange(6)+(b-4.5)*.12,vals,marker=marker,ls='none',label=f'Band {b}',color=color,ms=4)
    minimum=min(float(r['subspace_projection']) for r in fine)
    ax[1].set(xticks=np.arange(6),xticklabels=['I\n$\\Gamma$','I\n$X$','I\n$M$','R\n$\\Gamma$','R\n$X$','R\n$M$'],
              ylabel='Physical-field subspace projection',ylim=(min(.998,np.floor(minimum*1000)/1000-.001),1.0001))
    ax[1].ticklabel_format(axis='y',style='plain',useOffset=False);ax[1].legend(loc='lower left',frameon=False,ncol=2)
    for i in range(2):ax[i].text(0,1.03,chr(97+i),transform=ax[i].transAxes,fontweight='bold',fontsize=10)
    supp=OUT/'figureS8_ell2_3d_v8.pdf';fig.canvas.draw()
    fig.savefig(supp,bbox_inches='tight',pad_inches=.035,facecolor='white');fig.savefig(supp.with_suffix('.png'),dpi=260,bbox_inches='tight',pad_inches=.035,facecolor='white');plt.close(fig)
for p in [main,supp]:shutil.copy2(p,ROOT/'latex/figures'/p.name)
source_paths=[V/'analysis.json',V/'point_comparison.csv',V/'candidate_gains.csv',
              ROOT/'05_validation/V08_N15_20261030_v3/V08_summary.csv',
              ROOT/'05_validation/V08_N15_ell1p5_20261030/V08_summary.csv',OPT/'restart/manifest.json',OPT/'N21/manifest.json']
record=dict(assets={p.relative_to(ROOT).as_posix():sha(p) for p in [main,supp]},
            raw_sources={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths},
            transformations=['Raw deterministic values; no smoothing or inferred confidence intervals.',
              'Figure6a retains plate-only J45 transfer curves; Figure6b shows ell2 solid comparisons only.',
              'FigureS8a explicitly distinguishes Gamma/X/M from Gamma/X candidate sets.',
              'FigureS8b shows frequency-defined physical-field subspace projections; individual MAC remains in raw tables.'],
            intended_width_mm=164,external_legend_bbox_intentional=True,human_verified=False)
(ROOT/'03_review/new_figure_provenance_v8.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(main,supp)
