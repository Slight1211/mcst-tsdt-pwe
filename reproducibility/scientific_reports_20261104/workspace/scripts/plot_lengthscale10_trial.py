"""Display the fresh ell10 trial without editing manuscript v16 or clipping signed J."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
TRIAL=ROOT/'05_validation/ELL10_exploration_20261031'
OUT=ROOT/'output/ell10_trial';OUT.mkdir(parents=True,exist_ok=True)
read=lambda p:list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
factor=np.sqrt(4.35e9/1180)/(75e-6*2*np.pi*1e6)
frozen=load(TRIAL/'frozen/manifest.json')
optimization_runs={name:load(TRIAL/name/'manifest.json') for name in ['optimization','warmstart']}
assert all(m['status']=='PASS_SAMPLED_N15_LOCAL_SQP' for m in optimization_runs.values())
opt_name,opt=max(optimization_runs.items(),key=lambda pair:pair[1]['final_ibz153']['J'])
assert frozen['status']=='COMPLETED_SAMPLED_FROZEN_SCAN'
assert opt['status']=='PASS_SAMPLED_N15_LOCAL_SQP'
assert opt['ell_epoxy_um']==frozen['ell_epoxy_um']==10
assert opt['N']==frozen['N']==15

def gap(rows):
    lo=max(rows,key=lambda r:float(r['Omega4']))
    up=min(rows,key=lambda r:float(r['Omega5']))
    L,U=float(lo['Omega4']),float(up['Omega5'])
    return dict(L=L,U=U,J=200*(U-L)/(U+L),lower_MHz=L*factor,upper_MHz=U*factor,
        lower_kx_a=float(lo['kx_a']),lower_ky_a=float(lo['ky_a']),
        upper_kx_a=float(up['kx_a']),upper_ky_a=float(up['ky_a']))

designs=[('initial','Initial','#666666','--'),
         ('ell1_design',r'$\ell=1$ design','#0072B2',':'),
         ('ell2_design',r'$\ell=2$ design','#009E73','-.'),
         ('ell10_design',r'$\ell=10$ local design','#D55E00','-')]
rows={};summary=[];sources=[]
for name,label,color,ls in designs:
    if name=='ell10_design':
        rid=opt['rounds'][-1]['optimizer']['round']
        freq=TRIAL/f'{opt_name}/round{rid}_ibz45.csv'
        vertices=TRIAL/f'{opt_name}/final_vertices.csv'
    else:
        freq=TRIAL/f'frozen/{name}_ell10_N15_ibz45.csv'
        vertices=TRIAL/f'frozen/{name}_vertices.csv'
    rows[name]=read(freq);assert len(rows[name])==45
    summary.append(dict(shape=name,N=15,ell_epoxy_um=10,points=45,**gap(rows[name])))
    sources.extend([freq,vertices])
denser_path=TRIAL/f'{opt_name}/final_ibz153.csv'
denser_rows=read(denser_path);assert len(denser_rows)==153
sources.append(denser_path)
denser_gap=gap(denser_rows)
summary.append(dict(shape='ell10_design_153',N=15,ell_epoxy_um=10,points=153,**denser_gap))
with (OUT/'ell10_comparison.csv').open('w',encoding='utf-8',newline='') as h:
    w=csv.DictWriter(h,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)

style={'font.family':'DejaVu Sans','font.size':9.5,'axes.spines.top':False,
       'axes.spines.right':False,'axes.labelsize':10,'legend.fontsize':8,
       'axes.grid':True,'grid.alpha':.18,'svg.fonttype':'none'}
with plt.rc_context(style):
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.8),layout='constrained',gridspec_kw={'width_ratios':[1,1.15,.9]})
    fig.suptitle(r'All spectra at $\ell=10\,\mu$m; $a=75\,\mu$m, $h=6\,\mu$m, $N=15$',fontsize=12)
    for name,label,color,ls in designs:
        vertices=TRIAL/(f'{opt_name}/final_vertices.csv' if name=='ell10_design' else f'frozen/{name}_vertices.csv')
        v=np.genfromtxt(vertices,delimiter=',',names=True)
        xy=np.column_stack([v['x_over_a'],v['y_over_a']]);xy=np.vstack([xy,xy[0]])
        axes[0].plot(xy[:,0],xy[:,1],color=color,ls=ls,lw=2 if name=='ell10_design' else 1.3,label=label)
    axes[0].set(xlabel='$x/a$',ylabel='$y/a$',xlim=(-.45,.45),ylim=(-.45,.45))
    axes[0].set_aspect('equal',adjustable='box')
    axes[0].legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,handlelength=2.8)

    for name,color,lw in [('initial','#888888',1.1),('ell10_design','#D55E00',1.8)]:
        rr=rows[name] if name=='initial' else denser_rows
        grid=8 if name=='initial' else 16
        path=sorted([r for r in rr if abs(float(r['ky_a']))<1e-12],key=lambda r:float(r['kx_a']))
        path+=sorted([r for r in rr if abs(float(r['kx_a'])-np.pi)<1e-12 and float(r['ky_a'])>1e-12],key=lambda r:float(r['ky_a']))
        path+=sorted([r for r in rr if abs(float(r['kx_a'])-float(r['ky_a']))<1e-12 and float(r['kx_a'])<np.pi-1e-12],key=lambda r:-float(r['kx_a']))
        assert len(path)==3*grid+1
        x=np.linspace(0,3,len(path));who='Initial (45)' if name=='initial' else 'Local (153)'
        for band,ls,marker in [(4,'-','o'),(5,'--','s')]:
            axes[1].plot(x,[float(r[f'Omega{band}'])*factor for r in path],color=color,lw=lw,ls=ls,
                         marker=marker,ms=2.6,label=f'{who}, band {band}')
    axes[1].set(xlabel='Sampled Bloch path',ylabel='Frequency (MHz)',xticks=[0,1,2,3],xticklabels=[r'$\Gamma$','$X$','$M$',r'$\Gamma$'])
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)

    values=[r['J'] for r in summary]
    bars=axes[2].bar(range(5),values,color=[d[2] for d in designs]+['#D55E00'],width=.65)
    bars[-1].set_hatch('///')
    axes[2].axhline(0,color='black',lw=.85)
    low=min(0,min(values));high=max(0,max(values));span=max(1,high-low)
    axes[2].set(xticks=range(5),xticklabels=['Initial\n45',r'$\ell=1$'+'\n45',r'$\ell=2$'+'\n45','Local\n45','Local\n153'],
                ylabel='Signed sampled separation $J$ (%)',ylim=(low-.18*span,high+.13*span))
    for bar,value in zip(bars,values):
        axes[2].annotate(f'{value:.2f}',(bar.get_x()+bar.get_width()/2,value),xytext=(0,-10 if value<0 else 5),
                         textcoords='offset points',ha='center',va='top' if value<0 else 'bottom',fontsize=9)
    axes[2].text(.5,.97,'Negative = band overlap',transform=axes[2].transAxes,ha='center',va='top',fontsize=8.5)
    for ax,letter in zip(axes,'abc'):
        ax.text(-.12,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=13)
    fig.savefig(OUT/'ell10_shape_bands_comparison.png',dpi=220,facecolor='white')
    fig.savefig(OUT/'ell10_shape_bands_comparison.svg',facecolor='white')
    plt.close(fig)

with plt.rc_context(style):
    fig,ax=plt.subplots(figsize=(6,6),layout='constrained')
    for name,label,color,ls in [designs[0],designs[2],designs[3]]:
        vertices=TRIAL/(f'{opt_name}/final_vertices.csv' if name=='ell10_design' else f'frozen/{name}_vertices.csv')
        v=np.genfromtxt(vertices,delimiter=',',names=True);xy=np.column_stack([v['x_over_a'],v['y_over_a']]);xy=np.vstack([xy,xy[0]])
        ax.plot(xy[:,0],xy[:,1],color=color,ls=ls,lw=2 if name=='ell10_design' else 1.3,label=label)
        if name=='ell10_design':ax.scatter(xy[:-1,0],xy[:-1,1],s=10,color=color,zorder=3)
    ax.set(xlabel='$x/a$',ylabel='$y/a$',xlim=(-.45,.45),ylim=(-.45,.45),title=r'Local design at $\ell=10\,\mu$m ($N=15$)')
    ax.set_aspect('equal');ax.legend(loc='upper right',frameon=False)
    fig.savefig(OUT/'ell10_local_shape.png',dpi=220,facecolor='white');plt.close(fig)
manifest=dict(status='DERIVED_FROM_COMPLETED_TRIAL',normalization_MHz_per_Omega=float(factor),
    summary=summary,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
    transformations=['No smoothing; geometry draws the48straight edges.','Band lines join25initial and49local boundary samples; summaries use full45/153pointsets as labelled.',
                     'Signed negative separations retained; bars include zero.','Two independent local searches in the same six-variable space; neither certifies a global or153-point optimum.'],
    selected_local_run=opt_name,selection='Largest final153point signed separation among two completed local searches; no global optimality claim.',
    local_runs={name:dict(final_ibz45=m['final_ibz45'],final_ibz153=m['final_ibz153'],design=m['design']) for name,m in optimization_runs.items()},
    raw_frozen_manifest_sha256=sha(TRIAL/'frozen/manifest.json'),raw_optimization_manifest_sha256={name:sha(TRIAL/name/'manifest.json') for name in optimization_runs},
    output_sha256={name:sha(OUT/name) for name in ['ell10_comparison.csv','ell10_shape_bands_comparison.png','ell10_shape_bands_comparison.svg','ell10_local_shape.png']},
    new_3D_validation=False,manuscript_updated=False)
(OUT/'plot_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
