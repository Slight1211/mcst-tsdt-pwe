"""Compact the six saved modal fields and16 local checks without changing data."""
from pathlib import Path
import csv, hashlib, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from pypdf import PdfReader
from figure_style_v31 import COL, ENERGY_CMAP, style_context, tidy_axes, export_figure

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'03_review/revision_v40/compact'
OUT=ROOT/'latex/figures'
REVIEW.mkdir(parents=True,exist_ok=True)
STEM='figure4_energy_compact_v40'
FIELDS=ROOT/'06_release/github_v20/repository/data/wave_motion_20261009_v1/assets/item-010.npz'
PRED=ROOT/'03_review/sensitivity_v26/plot_source.csv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected={str(p.relative_to(ROOT)):sha(p) for p in OUT.glob('*.pdf') if '_v40' not in p.name}
assert sha(FIELDS)=='9323dcc5b59433d60512de9ff2a54e78146e2012e27e9158d193a668501a460d'
with np.load(FIELDS) as z:arrays={k:np.array(z[k],copy=True) for k in z.files}
keys=['w0','w1','mc0','mc1','shear0','shear1']
assert all(arrays[k].shape==(512,512) for k in keys)
array_hashes={k:hashlib.sha256(arrays[k].tobytes()).hexdigest() for k in keys}
q=np.arange(512)/512-.5;xx,yy=np.meshgrid(q,q,indexing='xy');steel=xx*xx+yy*yy<.3**2
assert np.array_equal(steel,(arrays['mc0']==0)&(arrays['mc1']==0))
assert arrays['shear0'][steel].sum()/arrays['shear0'].sum()>.029
assert arrays['shear1'][steel].sum()/arrays['shear1'].sum()>.083
with PRED.open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
for r in records:
    r['band']=int(r['band'])
    for k in ['fractional_change','analytic_sensitivity','predicted_delta_ln_f','actual_delta_ln_f','absolute_log_error','mass_MAC']:
        r[k]=float(r[k])
    assert abs(r['analytic_sensitivity']*math.log1p(r['fractional_change'])-r['predicted_delta_ln_f'])<1e-15
    r['residual_direct_minus_first_order_x1e5']=(r['actual_delta_ln_f']-r['predicted_delta_ln_f'])*1e5
assert len(records)==16
maximum=max(r['absolute_log_error'] for r in records)
assert abs(maximum-8.334251106865628e-5)<1e-15
W,H=164.,120.
with style_context():
    fig=plt.figure(figsize=(W/25.4,H/25.4))
    def axmm(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
    columns=[12.,45.,78.];ys=[78.,42.];size=26.
    images=[]
    for col,title in enumerate(['Periodic amplitude','Couple-stress energy','Classical shear energy']):
        fig.text((columns[col]+size/2)/W,115./H,title,fontsize=8,ha='center',va='center')
    for row,band in enumerate([4,5]):
        fig.text(columns[0]/W,(ys[row]+size+2.)/H,f'Mode {band}',color=COL[f'mode{band}'],
                 fontsize=8,ha='left',va='bottom')
        for col,prefix in enumerate(['w','mc','shear']):
            ax=axmm(columns[col],ys[row],size,size)
            arr=arrays[f'{prefix}{row}']
            if col==0:
                im=ax.imshow(arr,origin='lower',extent=(-.5,.5,-.5,.5),cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest')
            elif col==1:
                cmap=ENERGY_CMAP.with_extremes(bad=COL['steel_mask'])
                im=ax.imshow(np.ma.masked_where(steel,arr),origin='lower',extent=(-.5,.5,-.5,.5),cmap=cmap,vmin=0,vmax=.75,interpolation='nearest')
            else:
                im=ax.imshow(arr,origin='lower',extent=(-.5,.5,-.5,.5),cmap=ENERGY_CMAP,vmin=0,vmax=2,interpolation='nearest')
            ax.add_patch(Circle((0,0),.3,fill=False,edgecolor=COL['classic'],lw=.6))
            ax.set(xlim=(-.5,.5),ylim=(-.5,.5),xticks=[-.5,0,.5],yticks=[-.5,0,.5],aspect='equal')
            ax.tick_params(labelleft=col==0,labelbottom=row==1,length=2,pad=1.5)
            if col==0:ax.set_ylabel('$y/a$',labelpad=1.5)
            if row==1:ax.set_xlabel('$x/a$',labelpad=1.5)
            ax.text(-.20,.98,'abcdef'[row*3+col],transform=ax.transAxes,fontsize=9,fontweight='bold',ha='left',va='top')
            if row==1:images.append(im)
    for col,(ticks,label) in enumerate([
        ([-1,0,1],r'$\mathrm{Re}\,\widetilde w$ (normalized)'),
        ([0,.25,.5,.75],r'$e_{\mathrm{mc}}/\bar U$'),
        ([0,1,2],r'$e_{\mathrm{sh}}/\bar U$')]):
        cax=axmm(columns[col],27.,size,2.2)
        bar=fig.colorbar(images[col],cax=cax,orientation='horizontal',ticks=ticks)
        bar.outline.set_visible(False);bar.ax.tick_params(labelsize=7,length=2,pad=1.5)
        bar.set_label(label,fontsize=8,labelpad=2)
    handles=[Line2D([],[],color=COL['mode4'],ls='-',marker='o',mfc='white',ms=3,lw=1.2,label='Mode 4'),
             Line2D([],[],color=COL['mode5'],ls='--',marker='s',mfc='white',ms=3,lw=1.2,label='Mode 5')]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(142/W,115/H),ncol=2,
               handlelength=1.1,columnspacing=.65,handletextpad=.35,borderaxespad=0,fontsize=7)
    fig.text(111./W,109./H,'g',fontsize=9,fontweight='bold',ha='left',va='center')
    for param,title,power,top,resy,lim,ticks in [
        ('ell',r'Length $\ell$',3,85.,69.,3.3,[-3,0,3]),
        ('h',r'Thickness $h$',2,35.,19.,2.48,[-2.4,0,2.4])]:
        ax=axmm(123.,top,38.,18.);res=axmm(123.,resy,38.,10.)
        for band,marker,ls in [(4,'o','-'),(5,'s','--')]:
            rr=sorted([r for r in records if r['parameter']==param and r['band']==band],key=lambda r:r['fractional_change'])
            color=COL[f'mode{band}'];delta=np.linspace(-.03,.03,401)
            slope=rr[0]['analytic_sensitivity'];assert all(r['analytic_sensitivity']==slope for r in rr)
            ax.plot(100*delta,10**power*slope*np.log1p(delta),color=color,ls=ls,lw=1.2)
            x=[100*r['fractional_change'] for r in rr]
            ax.plot(x,[10**power*r['actual_delta_ln_f'] for r in rr],color=color,marker=marker,mfc='white',ms=3,ls='none',mew=.7)
            res.plot(x,[r['residual_direct_minus_first_order_x1e5'] for r in rr],color=color,marker=marker,mfc='white',ms=3,ls=ls,lw=1.2,mew=.7)
        ax.set(xlim=(-3.3,3.3),ylim=(-lim,lim),xticks=[-3,-1,0,1,3],yticks=ticks)
        ax.tick_params(labelbottom=False,pad=1.5,length=2)
        ax.set_ylabel(fr'$10^{power}\Delta\ln f_n$',labelpad=1.5)
        ax.set_title(title,pad=5,fontsize=8)
        res.axhline(0,color=COL['zero'],ls=':',lw=.6)
        res.set(xlim=(-3.3,3.3),ylim=(-10,10),xticks=[-3,-1,0,1,3],yticks=[-10,0,10])
        res.set_xlabel(r'$\Delta p/p$ (%)',labelpad=2)
        res.set_ylabel(r'$10^5\epsilon_n$',labelpad=1.5)
        res.tick_params(pad=1.5,length=2)
        tidy_axes(ax);tidy_axes(res)
    meta=export_figure(fig,OUT/STEM)
    assert meta['minimum_visible_font_pt']>=7
    plt.close(fig)

with (REVIEW/'energy_residual_records_v40.csv').open('w',encoding='utf-8-sig',newline='') as f:
    names=['parameter','band','fractional_change','predicted_delta_ln_f','actual_delta_ln_f',
           'residual_direct_minus_first_order_x1e5','absolute_log_error','mass_MAC']
    w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows({k:r[k] for k in names} for r in records)
assert (REVIEW/'energy_residual_records_v40.csv').read_bytes()==(ROOT/'03_review/figures_v31/energy/residual_records.csv').read_bytes()
assert all(sha(ROOT/p)==v for p,v in protected.items())
meta.update(status='GENERATED_PENDING_VISUAL',source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [FIELDS,PRED]},
    array_sha256=array_hashes,all_six_native_arrays_unchanged=True,all16_checks_and_residuals_unchanged=True,
    field_size_mm=26,prediction_axes_size_mm=[38,18],residual_axes_size_mm=[38,10],
    field_color_limits={'amplitude':[-1,1],'couple_stress':[0,.75],'shear':[0,2]},
    mask='Zero couple-stress steel only; real steel shear and displacement retained',
    maximum_absolute_log_error=maximum,minimum_mass_MAC=min(r['mass_MAC'] for r in records),
    prediction_display_scaling={'ell':1000,'h':100},native_arrays_no_smoothing=True,
    old_artwork_preserved=True,new_eigensolves=0,new_optimizations=0,new_COMSOL_runs=0,
    script_sha256=sha(Path(__file__)))
(REVIEW/'energy_provenance_v40.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
caps={
'en':r'''Modal fields and first-order frequency checks at $X$ for the circular cell, $(a,h,\ell)=(75,6,1)\,\mu\mathrm m$ and $N=15$. The upper and lower field rows show modes four and five. (a,d) Phase-aligned real Bloch-periodic amplitude $\widetilde w$, excluding the Bloch phase and normalized by its peak. (b,e) Couple-stress energy density. (c,f) Classical shear energy density reconstructed from the dual field. Both energy columns use cividis with separate labelled ranges; densities are divided by each mode's area-mean total potential energy $\bar U$. Grey steel pixels in (b,e) have zero couple-stress energy; (c,f) retain nonzero steel shear. (g) First-order predictions (lines) and direct frequency changes (open symbols), for $\ell$ in the upper pair of axes and $h$ in the lower pair. Blue circles and vermilion squares denote modes four and five. The prediction axes display $10^3\Delta\ln f_n$ for $\ell$ and $10^2\Delta\ln f_n$ for $h$; each residual axis shows $10^5\epsilon_n$, where $\epsilon_n=\Delta\ln f_n^{\rm direct}-\Delta\ln f_n^{(1)}$. The maximum absolute logarithmic discrepancy is $8.3\times10^{-5}$.''',
'zh':r'''圆形单胞在$X$点的模态场与一阶频率核验，取$(a,h,\ell)=(75,6,1)\,\mu\mathrm m$、$N=15$。场图上、下行分别为第4、5模态。(a,d) 相位对齐、按峰值归一化的Bloch周期幅函数$\widetilde w$实部，不含Bloch相位。(b,e) 偶应力能密度。(c,f) 对偶场重构的经典剪切能密度。两个能量列采用cividis及分别标注的范围，密度均除以各模态的面积平均总势能$\bar U$。(b,e)中的灰色钢区偶应力能量为零，(c,f)保留钢中非零剪切能量。(g) 一阶预测（线）与直接频率变化（空心标记），上方两轴为$\ell$扰动，下方两轴为$h$扰动；蓝圆与朱红方表示第4、5模态。预测轴分别显示$10^3\Delta\ln f_n$与$10^2\Delta\ln f_n$，每组下方残差轴显示$10^5\epsilon_n$，其中$\epsilon_n=\Delta\ln f_n^{\rm direct}-\Delta\ln f_n^{(1)}$。最大绝对对数频差为$8.3\times10^{-5}$。'''}
for lang,c in caps.items():(REVIEW/f'energy_caption_{lang}_v40.tex').write_text(c+'\n',encoding='utf-8')
print(STEM,meta['width_mm'],meta['height_mm'])
