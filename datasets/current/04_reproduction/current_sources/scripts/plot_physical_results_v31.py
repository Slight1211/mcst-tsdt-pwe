"""Restyle the physical-gap curves and representative spectra, preserving v29 data."""
from pathlib import Path
import csv, hashlib, json, math, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator
from figure_style_v31 import COL, style_context, tidy_axes, export_figure

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'03_review/figures_v31/physical'
BASE=ROOT/'03_review/balance_v29/figures'
OUT=ROOT/'01_manuscript/figures'
HERE.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def axis(fig,x,y,w,h,H):return fig.add_axes([x/164,y/H,w/164,h/H])
def label(fig,x,y,t,H):return fig.text(x/164,y/H,t,fontsize=9,fontweight='bold',va='top')
def finish(fig,name):
    result=export_figure(fig,OUT/name)
    for ext in ['pdf','svg']:shutil.copy2(OUT/f'{name}.{ext}',ROOT/f'latex/figures/{name}.{ext}')
    plt.close(fig)
    return result

oldprov=json.loads((BASE/'PHYSICAL_FIGURES_PROVENANCE.json').read_text(encoding='utf8'))
for name,digest in oldprov['source_sha256'].items():assert sha(ROOT/name)==digest,name
records=rows(BASE/'physical_gap_records.csv')
representative=rows(BASE/'representative_spectra_records.csv')
gaprecords=rows(BASE/'representative_gap_checks.csv')
assert len(records)==54 and len(representative)==1248 and len(gaprecords)==16
for r in records:r.update(coordinate=float(r['coordinate']),J_percent=float(r['J_percent']))
for r in representative:
    for key in ['panel','path_index','band']:r[key]=int(r[key])
    for key in ['parameter','frequency_MHz','Omega']:r[key]=float(r[key])
    a_um=r['parameter'] if r['scan']=='size' else 75.
    expected=2*math.pi*a_um*math.sqrt(1180/4.35e9)*r['frequency_MHz']
    assert math.isclose(expected,r['Omega'],rel_tol=2e-15,abs_tol=1e-14)

with style_context():
    H=85.
    fig=plt.figure(figsize=(164/25.4,H/25.4))
    axes=[axis(fig,14,15,62,54,H),axis(fig,101,15,61,54,H)]
    for ax,kind,xticks,xticklabels in [(axes[0],'size',[40,100,500,2000],['40','100','500','2000']),
                                    (axes[1],'thickness',[.02,.04,.08,.16,.32],['0.02','0.04','0.08','0.16','0.32'])]:
        for model,color,marker,ls in [('Classic-TSDT',COL['classic'],None,'--'),('MCST-TSDT',COL['mcst'],'o','-'),('3D-MCST','black','o','none')]:
            rr=sorted([r for r in records if r['scan']==kind and r['model']==model],key=lambda r:r['coordinate'])
            ax.plot([r['coordinate'] for r in rr],[r['J_percent'] for r in rr],color=color,marker=marker,ls=ls,
                    lw=1.56 if model=='MCST-TSDT' else 1.2,ms=4 if model=='3D-MCST' else 3.7,
                    mfc='white' if model=='3D-MCST' else color,mew=.8,
                    label={'Classic-TSDT':'Classical TSDT','MCST-TSDT':'MCST TSDT','3D-MCST':'3D MCST'}[model],
                    zorder=4 if model=='3D-MCST' else 3)
        ax.set_xscale('log');ax.set_xticks(xticks,xticklabels)
        ax.xaxis.set_minor_locator(NullLocator());ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(min(xticks)*.91,max(xticks)*1.10)
        ax.set_ylabel(r'Relative gap $J_{45}$ (%)');tidy_axes(ax)
    a,b=axes
    a.set(ylim=(24.8,30.2),yticks=[25,26,27,28,29,30],xlabel='Cell size $a$ (µm)')
    b.set(ylim=(0,34),yticks=[0,10,20,30],xlabel='Thickness ratio $h/a$')
    a.set_title(r'$h/a=0.08$',pad=5);b.set_title(r'$a=75$ µm',pad=5)
    label(fig,1,74,'a',H);label(fig,87,74,'b',H)
    handles,names=a.get_legend_handles_labels()
    fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.54,.96),ncol=3,handlelength=1.7,borderaxespad=0)
    physical=finish(fig,'figure2_physical_gap_v31')

with style_context():
    H=128.
    fig=plt.figure(figsize=(164/25.4,H/25.4))
    axes=[]
    for index in range(4):
        col=index%2;row=index//2
        ax=axis(fig,14+85*col,73-59*row,64,40,H);axes.append(ax)
        ymax=0.
        for model in ['Classic-FSDT','Classic-TSDT','MCST-FSDT','MCST-TSDT']:
            rr=[r for r in representative if r['panel']==index+1 and r['model']==model]
            values=np.asarray([[next(r['Omega'] for r in rr if r['path_index']==k and r['band']==band) for band in range(1,7)] for k in range(13)])
            ymax=max(ymax,float(values.max()))
            color=COL['classic'] if model.startswith('Classic') else COL['mcst']
            ts=model.endswith('TSDT');ls='-' if ts else (0,(4,2.4));marker='o' if ts else 's'
            for band in range(6):
                ax.plot(np.arange(13)/4,values[:,band],color=color,ls=ls,lw=1.56 if ts else 1.2,
                        marker=marker,ms=1.8 if ts else 2.3,mfc=color if ts else 'white',mew=.6,
                        label=model.replace('Classic','Classical').replace('-',' ') if band==0 else None)
            if model=='MCST-TSDT':ax.axhspan(float(values[:,3].max()),float(values[:,4].min()),color=COL['gap'],alpha=.16,zorder=-3)
        ax.set(xlim=(0,3),ylim=(0,ymax*1.07),xticks=[0,1,2,3],xticklabels=[r'$\Gamma$','$X$','$M$',r'$\Gamma$'],xlabel='Bloch path')
        if col==0:ax.set_ylabel(r'$\Omega$')
        r=next(r for r in representative if r['panel']==index+1)
        title=(fr'$a={r["parameter"]:g}$ µm, $h/a=0.08$' if r['scan']=='size' else fr'$a=75$ µm, $h/a={r["parameter"]:g}$')
        ax.set_title(title,pad=5);tidy_axes(ax)
        label(fig,1+85*col,118-59*row,'abcd'[index],H)
    handles,names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.53,.97),ncol=4,handlelength=1.5,columnspacing=1.2,borderaxespad=0)
    rep=finish(fig,'figureS2_representative_v31')

for name in ['physical_gap_records.csv','representative_spectra_records.csv','representative_gap_checks.csv']:
    shutil.copy2(BASE/name,HERE/name)
    assert sha(BASE/name)==sha(HERE/name)
report=dict(status='GENERATED_PENDING_VISUAL',physical=physical,representative=rep,source_sha256=oldprov['source_sha256'],
            plotting_records_sha256={name:sha(HERE/name) for name in ['physical_gap_records.csv','representative_spectra_records.csv','representative_gap_checks.csv']},
            all_numeric_records_identical_to_v29=True,physical_points=54,representative_frequency_values=1248,
            gap_recomputations=16,new_solver_runs=0,new_optimizations=0,script_sha256=sha(Path(__file__)))
(HERE/'PHYSICAL_FIGURES_PROVENANCE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=report['status'],outputs=[physical['outputs']['pdf']['path'],rep['outputs']['pdf']['path']]),ensure_ascii=False,indent=2))
