"""Source-bound figures for parameter transfer and state-specific reoptimization."""
from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'01_manuscript/figures'
V1=ROOT/'05_validation/V01_20261030';OPT=ROOT/'05_validation/V07_ell2_reoptimization_20261030'
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
q=json.loads((OPT/'restart/manifest.json').read_text());hi=json.loads((OPT/'N21/manifest.json').read_text())
assert q['status']=='PASS_SAMPLED_N15_LOCAL_SQP' and hi['status']=='COMPLETED_N21_FROZEN_GEOMETRY'
scan=rows(ROOT/'05_validation/V08_N15_20261030_v3/V08_summary.csv')+rows(ROOT/'05_validation/V08_N15_ell1p5_20261030/V08_summary.csv')
high=rows(ROOT/'05_validation/V08_N21_ell2_20261030/V08_summary.csv')
data=rows(V1/'V01_point_comparison.csv')
style={'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':8,'legend.fontsize':6.4,
       'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
       'axes.grid':True,'grid.alpha':.22}
with plt.rc_context(style):
 fig,ax=plt.subplots(1,2,figsize=(164/25.4,68/25.4),layout='constrained')
 fine=[r for r in data if int(r['mesh_level'])==2 and int(r['layers'])==4 and int(r['requested_eigenpairs'])==24 and r['point'] in ['Gamma','X']]
 cats=[('initial','Gamma'),('initial','X'),('final','Gamma'),('final','X')]
 for b,color,marker in [(4,'#0072B2','o'),(5,'#D55E00','s')]:
  ys=[float(next(r for r in fine if r['shape']==s and r['point']==k and int(r['band'])==b)['pwe_minus_3d_percent']) for s,k in cats]
  ax[0].plot(np.arange(4)+(b-4.5)*.12,ys,marker,ls='none',color=color,label=f'Band {b}',ms=4.5)
 ax[0].axhline(0,color='#555555',lw=.7,ls='--')
 ax[0].set(xticks=np.arange(4),xticklabels=['Initial\n$\\Gamma$','Initial\n$X$','Final\n$\\Gamma$','Final\n$X$'],ylabel='PWE-3D frequency difference (%)')
 ax[0].legend(loc='upper center',ncol=2,frameon=False);ax[0].text(-.13,1.03,'a',transform=ax[0].transAxes,fontweight='bold',fontsize=10)
 for shape,color,marker,label in [('initial','#666666','o','Initial'),('final','#0072B2','s','$\\ell=1$ design')]:
  selected=sorted([r for r in scan if r['shape']==shape],key=lambda r:float(r['ell_um']))
  ax[1].plot([float(r['ell_um']) for r in selected],[float(r['J']) for r in selected],color=color,marker=marker,ms=4,lw=1,label=label+' ($N=15$)')
  r=next(r for r in high if r['shape']==shape)
  ax[1].plot(2,float(r['J']),ls='none',marker='D',mfc='white',mec=color,ms=5,label=label+' ($N=21$)')
 ax[1].plot(2,q['final_ibz45']['J'],marker='*',ms=9,color='#D55E00',ls='none',label='Reoptimized ($N=15$)')
 ax[1].plot(2,hi['final']['J'],marker='*',ms=9,mec='#D55E00',mfc='white',ls='none',label='Reoptimized ($N=21$)')
 for ell in [.5,1.,1.5,2.]:
  a=next(r for r in scan if r['shape']=='initial' and float(r['ell_um'])==ell);b=next(r for r in scan if r['shape']=='final' and float(r['ell_um'])==ell)
  ax[1].annotate(f'+{float(b["J"])-float(a["J"]):.2f}',(ell,(float(a['J'])+float(b['J']))/2),xytext=(4,0),textcoords='offset points',fontsize=6.3)
 ax[1].annotate(f'+{q["gain_from_circle_pp"]:.2f} pp',xy=(2,q['final_ibz45']['J']),xytext=(1.55,28.9),
                arrowprops=dict(arrowstyle='-',lw=.7,color='#D55E00'),color='#A34200',fontsize=6.8)
 ax[1].set(xlabel='Epoxy length scale $\\ell$ ($\\mu$m)',ylabel='Sampled relative gap $J_{45}$ (%)',xticks=[.5,1,1.5,2],xlim=(.38,2.35),ylim=(23.8,35.1))
 ax[1].legend(loc='upper center',bbox_to_anchor=(.5,1.43),ncol=2,frameon=False,fontsize=6.1,handlelength=1.3,columnspacing=.8)
 ax[1].text(-.16,1.03,'b',transform=ax[1].transAxes,fontweight='bold',fontsize=10)
 # Include the external legend in both exports; the first PDF layout pass
 # otherwise clips its top row even when the subsequent PNG is complete.
 fig.canvas.draw()
 output=OUT/'figure6_parameter_design_v7.pdf';fig.savefig(output,bbox_inches='tight',pad_inches=.035,facecolor='white');fig.savefig(output.with_suffix('.png'),dpi=240,bbox_inches='tight',pad_inches=.035,facecolor='white');plt.close(fig)
 # Geometry and path spectra for the independently rerun ell2 design.
 node=ROOT/'02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1'
 old=rows(ROOT/'05_validation/V08_N15_20261030_v3/final_ell2_N15_ibz45.csv')
 new=rows(OPT/'restart'/f'round{q["rounds"][-1]["optimizer"]["round"]}_ibz45.csv')
 fig,ax=plt.subplots(1,2,figsize=(164/25.4,68/25.4),layout='constrained')
 for p,color,label in [(node/'initial_vertices.csv','#666666','Initial'),(node/'final_vertices.csv','#0072B2','$\\ell=1$ design'),(OPT/'restart/final_vertices.csv','#D55E00','$\\ell=2$ design')]:
  v=np.genfromtxt(p,delimiter=',',names=True);xy=np.column_stack((v['x_over_a'],v['y_over_a']));xy=np.vstack((xy,xy[0]))
  ax[0].plot(xy[:,0],xy[:,1],color=color,lw=1,label=label)
 ax[0].set(xlabel='$x/a$',ylabel='$y/a$',aspect='equal',xlim=(-.45,.45),ylim=(-.45,.45));ax[0].legend(frameon=False,fontsize=6.5)
 ks=[(np.pi*i/8,0) for i in range(9)]+[(np.pi,np.pi*j/8) for j in range(1,9)]+[(np.pi*j/8,np.pi*j/8) for j in range(7,-1,-1)]
 conv=np.sqrt(4.35e9/1180)/(75e-6*2*np.pi*1e6)
 for rr,color,style,label in [(old,'#0072B2','--','$\\ell=1$ design at $\\ell=2$'),(new,'#D55E00','-','Reoptimized at $\\ell=2$')]:
  path=[min(rr,key=lambda r:(float(r['kx_a'])-k[0])**2+(float(r['ky_a'])-k[1])**2) for k in ks]
  for b in [4,5]:ax[1].plot(np.arange(25)/8,[float(r[f'Omega{b}'])*conv for r in path],color=color,ls=style,marker='o',ms=2,lw=.9,label=label if b==4 else None)
 ax[1].set(xticks=[0,1,2,3],xticklabels=['$\\Gamma$','$X$','$M$','$\\Gamma$'],xlabel='Sampled Bloch path',ylabel='Frequency (MHz)');ax[1].legend(frameon=False,fontsize=6.4)
 for i in range(2):ax[i].text(-.13,1.03,chr(97+i),transform=ax[i].transAxes,fontweight='bold',fontsize=10)
 supplementary=OUT/'figureS7_ell2_design_v7.pdf';fig.savefig(supplementary,facecolor='white');fig.savefig(supplementary.with_suffix('.png'),dpi=240,facecolor='white');plt.close(fig)
for p in [output,supplementary]:shutil.copy2(p,ROOT/'latex/figures'/p.name)
sources=[V1/'V01_point_comparison.csv',ROOT/'05_validation/V08_N15_20261030_v3/V08_summary.csv',ROOT/'05_validation/V08_N15_ell1p5_20261030/V08_summary.csv',OPT/'restart/manifest.json',OPT/'N21/manifest.json']
record=dict(assets={p.relative_to(ROOT).as_posix():sha(p) for p in [output,supplementary]},raw_sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
            transformations=['No smoothing; straight lines guide the eye; raw modal frequencies used.','Figure6a uses signed relative differences instead of redundant absolute-frequency series.','Frozen-shape curves and reoptimized ell2 points are distinguished.'],human_verified=False)
(ROOT/'03_review/new_figure_provenance_v7.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(output,supplementary)
