"""Plot only computed sampled paths; straight segments are not new samples."""
import sys,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
out=Path(sys.argv[1]);root=Path(__file__).resolve().parent;assets=root.parents[1]/'output/pdf/modal_selectivity_assets'
cfg=json.loads((out/'config.json').read_text());pr=list(csv.DictReader((out/'pwe.csv').open()));fr=list(csv.DictReader((out/'fem.csv').open()))
assets.mkdir(parents=True,exist_ok=True)
def array(rows,theory=None,ell=None):
 a=np.full((12,8),np.nan)
 for r in rows:
  if theory is not None and (r['theory']!=theory or float(r['ell_um'])!=ell):continue
  a[int(r['point']),int(r['band'])-1]=float(r['frequency_MHz'])
 assert np.isfinite(a).all()
 return np.vstack([a,a[0]])
models=[('FSDT',0.,'Classic-FSDT','#0072BD','--'),('TSDT',0.,'Classic-TSDT','#4DBEEE','-'),('FSDT',1.,'MCST-FSDT','#D95319','--'),('TSDT',1.,'MCST-TSDT','#7E2F8E','-')]
fem=array(fr);arrays={name:array(pr,t,l) for t,l,name,c,ls in models}
k=np.loadtxt(out/'path.csv',delimiter=',',skiprows=1);k=np.vstack([k,k[0]]);x=np.r_[0,np.cumsum(np.linalg.norm(np.diff(k,axis=0),axis=1))]/np.pi
plt.rcParams.update({'font.size':10,'savefig.dpi':220})
fig,ax=plt.subplots(figsize=(7.5,4.3),layout='constrained');handles=[];gaps=[]
for t,l,name,c,ls in models:
 a=arrays[name]
 for j in range(4):ax.plot(x,a[:,j],color=c,ls=ls,lw=1.25)
 handles.append(Line2D([0],[0],color=c,ls=ls,label=name))
 for j in range(3):
  lo=float(max(a[:,j]));hi=float(min(a[:,j+1]));gaps.append(dict(model=name,after_band=j+1,lower_MHz=lo,upper_MHz=hi,width_MHz=hi-lo,status='sampled_path_candidate' if hi>lo else 'no_positive_sampled_interval'))
for j in range(4):ax.plot(x,fem[:,j],ls='none',marker='o',ms=4,mfc='none',mec='black',mew=.85)
handles.append(Line2D([0],[0],ls='none',marker='o',mfc='none',mec='black',label='COMSOL 2D MCST-TSDT'))
for g in gaps:
 if g['model']=='MCST-TSDT' and g['width_MHz']>1e-5:
  ax.axhspan(g['lower_MHz'],g['upper_MHz'],color='#77AC30',alpha=.12,zorder=-2)
  ax.text(.03,.5*(g['lower_MHz']+g['upper_MHz']),f"{g['lower_MHz']:.2f}-{g['upper_MHz']:.2f} MHz",fontsize=9,va='center',color='#42691B')
for split in x[[4,8]]:ax.axvline(split,c='0.75',lw=.7)
ax.set(xlim=(x[0],x[-1]),ylim=(0,max(max(a[:,:4].ravel()) for a in [*arrays.values(),fem])*1.04),ylabel='Frequency (MHz)',xticks=x[[0,4,8,12]],xticklabels=[r'$\Gamma$','X','M',r'$\Gamma$'])
ax.legend(handles=handles,ncol=3,fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False);ax.grid(axis='y',alpha=.15)
fig.savefig(assets/'validation_bands.pdf');fig.savefig(out/'four_theory_bands.png');plt.close(fig)
for j in range(3):
 lo=float(max(fem[:,j]));hi=float(min(fem[:,j+1]));gaps.append(dict(model='COMSOL 2D MCST-TSDT',after_band=j+1,lower_MHz=lo,upper_MHz=hi,width_MHz=hi-lo,status='sampled_path_candidate' if hi>lo else 'no_positive_sampled_interval'))
with (out/'sampled_path_gaps.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=gaps[0]);w.writeheader();w.writerows(gaps)
err=100*abs(arrays['MCST-TSDT']-fem)/np.where(fem==0,np.nan,fem)
fig,ax=plt.subplots(figsize=(5.2,2.7),layout='constrained')
ax.bar(np.arange(1,9),err[4],width=.6,color='#0072BD')
for j,e in enumerate(err[4]):ax.text(j+1,e+.025,f'{e:.2f}',ha='center',fontsize=8)
ax.set(xlabel='Sorted mode at X',ylabel='Frequency difference (%)',xticks=np.arange(1,9),ylim=(0,max(err[4])*1.2));ax.grid(axis='y',alpha=.15)
fig.savefig(assets/'validation_error_path.pdf');plt.close(fig)
summary=dict(N=cfg['N'],unique_path_points=12,displayed_bands=4,X_first8_sorted_frequency_difference_percent=err[4].tolist(),max_first4_sorted_frequency_difference_percent=float(np.nanmax(err[:,:4])),max_first8_sorted_frequency_difference_percent=float(np.nanmax(err)),gap_scope='sampled bending path, not complete BZ or all-polarization gap',gaps=gaps)
(out/'plot_assessment.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
