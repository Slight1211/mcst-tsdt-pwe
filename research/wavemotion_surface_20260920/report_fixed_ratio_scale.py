"""Audit and plot only the newly computed fixed-h/a scan (no legacy substitution)."""
from pathlib import Path
import csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(sys.argv[1]).resolve()
CFG=json.loads((OUT/'config.json').read_text())
ASSETS=Path(__file__).resolve().parents[2]/'output/pdf/modal_selectivity_assets'
STYLES=[('FSDT',0,'Classic-FSDT','#0072BD','--'),('TSDT',0,'Classic-TSDT','#56B4E9','-'),('FSDT',1,'MCST-FSDT','#D55E00','-.'),('TSDT',1,'MCST-TSDT','#7E2F8E',':')]
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':9,'axes.linewidth':.8,'pdf.fonttype':42,'xtick.direction':'in','ytick.direction':'in'})

def load(N,a,p,t,l):
    with np.load(OUT/'pwe'/f'N{N}_a{a}_p{p}_{t}_l{l}.npz') as z:
        assert abs(float(z['h_um'])/a-.08)<1e-12
        f=z['frequency_MHz'].copy(); start=int(p==0)
        assert np.all(np.isfinite(f)) and min(f[start:])>0
        assert max(z['residual'][start:8])<1e-5
        assert float(z['seed_agreement'])<1e-7 and float(z['mass_orthogonality'])<1e-7
        return f,float(max(z['residual'][start:8]))

def savecsv(name,rows):
    if rows:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def savefig(fig,name,publish):
    fig.savefig(OUT/(name+'.png'),dpi=250,bbox_inches='tight')
    if publish:
        ASSETS.mkdir(parents=True,exist_ok=True)
        fig.savefig(ASSETS/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

comparison=[];states=[];gaps=[];orders=[];bands={};maxres=0
for a in CFG['a_um']:
    folder=OUT/f'mcst_a{a}'
    try:
        assert json.loads((folder/'manifest.json').read_text())['exit_code']==0
        assert (folder/'model_Model.mph').exists()
        raw=list(csv.DictReader((folder/'frequencies.csv').open()))
        local=[]
        for p,name,ky in [(4,'X',0),(8,'M',1)]:
            modes=sorted([r for r in raw if abs(float(r['kx_pi_over_a'])-1)<1e-9 and abs(float(r['ky_pi_over_a'])-ky)<1e-9 and float(r['vertical_fraction'])>.8 and float(r['frequency_mhz'])>0],key=lambda r:float(r['frequency_mhz']))
            assert len(modes)>=6
            ref=np.array([float(r['frequency_mhz']) for r in modes[:6]])
            assert max(abs(float(r['imag_frequency_mhz']))/float(r['frequency_mhz']) for r in modes[:6])<1e-7
            for t,l,*_ in STYLES:
                f,res=load(21,a,p,t,l);maxres=max(maxres,res)
                for n in range(6):local.append(dict(a_um=a,h_um=.08*a,theory=t,ell_um=l,point=name,band=n+1,pwe_MHz=float(f[n]),fem_MHz=float(ref[n]),error_percent=float(100*abs(f[n]/ref[n]-1))))
        comparison.extend(local);states.append(dict(a_um=a,status='PASS_FIXED_MESH_CHECKS'))
    except (OSError,KeyError,AssertionError,ValueError) as e:
        states.append(dict(a_um=a,status='PENDING_OR_FAILED',reason=str(e)))
    for t,l,*_ in STYLES:
        try:
            fs=np.array([load(15,a,p,t,l)[0] for p in range(12)])
            bands[(a,t,l)]=fs
            low=float(max(fs[:,3]));high=float(min(fs[:,4]))
            gaps.append(dict(a_um=a,h_um=.08*a,theory=t,ell_um=l,lower_MHz=low,upper_MHz=high,J45_percent=200*(high-low)/(high+low)))
            for p in [4,8]:
                fine,_=load(21,a,p,t,l)
                for n in range(6):orders.append(dict(a_um=a,theory=t,ell_um=l,point=p,band=n+1,change_percent=float(100*abs(fs[p,n]/fine[n]-1))))
        except (OSError,KeyError,AssertionError,ValueError):pass

complete=len(comparison)==480 and len(gaps)==40 and len(orders)==480
maxima=[max([r for r in comparison if r['a_um']==a and r['theory']==t and r['ell_um']==l],key=lambda r:r['error_percent']) for a in CFG['a_um'] for t,l,*_ in STYLES if any(r['a_um']==a and r['theory']==t and r['ell_um']==l for r in comparison)]
for name,rows in [('comparison.csv',comparison),('maximum_errors.csv',maxima),('gaps.csv',gaps),('order_change.csv',orders)]:savecsv(name,rows)
consistency={}
for t in ['FSDT','TSDT']:
    available=[a for a in CFG['a_um'] if (a,t,0) in bands]
    if available:
        ref=bands[(available[0],t,0)]*available[0]
        consistency[t]=max(float(np.max(np.abs((bands[(a,t,0)]*a-ref)[1:]/ref[1:]))) for a in available)
audit=dict(status='COMPLETE_FIXED_MESH_CHECKS' if complete else 'INCOMPLETE',fem_scales=states,comparison_rows=len(comparison),gap_rows=len(gaps),order_rows=len(orders),max_control_residual=maxres,classic_similarity_relative=consistency,max_order_change_percent=max((r['change_percent'] for r in orders),default=None),NOT_RUN=['full-BZ error','all-scale 3D mesh convergence','cross-theory field MAC'])
(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
if maxima:
    fig,axes=plt.subplots(1,2,figsize=(7.0,3.15),sharey=True)
    for ax,point in zip(axes,['X','M']):
        for t,l,label,c,ls in STYLES:
            rr=[r for r in comparison if r['point']==point and r['theory']==t and r['ell_um']==l]
            rr=[max((r for r in rr if r['a_um']==a),key=lambda r:r['error_percent']) for a in CFG['a_um'] if any(r['a_um']==a for r in rr)]
            ax.plot([r['a_um'] for r in rr],[r['error_percent'] for r in rr],color=c,ls=ls,marker='s' if t=='FSDT' else 'o',ms=3.5,lw=1.25,label=label)
        ax.set(xscale='log',yscale='log',xlabel=r'$a$ ($\mu$m)',title=point)
        ax.grid(alpha=.18,which='both')
    axes[0].set_ylabel('Maximum error, branches 1-6 (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.03),fontsize=8)
    fig.subplots_adjust(top=.81,bottom=.18,left=.11,right=.98,wspace=.12)
    savefig(fig,'ratio_scale_max_error',complete)
if complete:
    # Six representative scales span gap closure, reopening and the
    # large-scale asymptote; all are precomputed members of CFG['a_um'].
    for a in [10,15,25,75,500,1000]:
        fig,ax=plt.subplots(figsize=(3.6,2.9))
        for t,l,label,c,ls in STYLES:
            f=bands[(a,t,l)]*2*np.pi*a*np.sqrt(1180/4.35e9)
            f=np.vstack([f,f[0]])
            for n in range(6):ax.plot(np.arange(13),f[:,n],color=c,ls=ls,lw=1,label=label if n==0 else None)
        # These are actual 3D M2L4 control frequencies, not interpolated bands.
        for point,index in [('X',4),('M',8)]:
            refs=[r for r in comparison if r['a_um']==a and r['theory']=='TSDT'
                  and r['ell_um']==1 and r['point']==point]
            assert len(refs)==6
            for j,r in enumerate(sorted(refs,key=lambda x:x['band'])):
                ax.plot(index,r['fem_MHz']*2*np.pi*a*np.sqrt(1180/4.35e9),
                        marker='D',ms=2.5,mfc='none',mec='.1',ls='None',
                        label='3D MCST M2L4' if point=='X' and j==0 else None)
        g=next(r for r in gaps if r['a_um']==a and r['theory']=='TSDT' and r['ell_um']==1)
        factor=2*np.pi*a*np.sqrt(1180/4.35e9)
        if g['J45_percent']>0:ax.axhspan(g['lower_MHz']*factor,g['upper_MHz']*factor,color='#77AC30',alpha=.15)
        ax.text(.02,.98,f"MCST-TSDT: J = {g['J45_percent']:.2f}%",transform=ax.transAxes,va='top',fontsize=7)
        ax.set(xticks=[0,4,8,12],xticklabels=[r'$\Gamma$','X','M',r'$\Gamma$'],xlim=(0,12),ylabel=r'$\Omega=\omega a\sqrt{\rho_B/E_B}$')
        ax.legend(ncol=2,fontsize=6,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.29));ax.grid(alpha=.12)
        savefig(fig,f'ratio_scale_bands_{a}',True)
    fig,ax=plt.subplots(figsize=(3.6,2.9))
    for t,l,label,c,ls in STYLES:
        rr=[r for r in gaps if r['theory']==t and r['ell_um']==l]
        ax.plot([r['a_um'] for r in rr],[r['J45_percent'] for r in rr],color=c,ls=ls,marker='o',ms=3,label=label)
    ax.axhline(0,color='.5',lw=.7);ax.set(xscale='log',xlabel=r'$a$ ($\mu$m)',ylabel=r'Signed $J_{45}$ (%)');ax.grid(alpha=.15)
    ax.legend(ncol=2,fontsize=6,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.25));savefig(fig,'ratio_scale_gap45',True)
lines=['# 固定厚跨比重新计算','',f"状态：{audit['status']}",'','固定 h/a=0.08，r/a=0.30，环氧 l=1 微米，钢 l=0；经典对照 l=0。旧固定板厚数据未代入。','', '|a (um)|h (um)|Classic-FSDT (%)|Classic-TSDT (%)|MCST-FSDT (%)|MCST-TSDT (%)|','|---:|---:|---:|---:|---:|---:|']
for a in CFG['a_um']:
    values=[next((f"{r['error_percent']:.5f}" for r in maxima if r['a_um']==a and r['theory']==t and r['ell_um']==l),'PENDING') for t,l,*_ in STYLES]
    lines.append('|'+f'{a}|{.08*a:g}|'+'|'.join(values)+'|')
lines+=['','三维参考为 M2 面内、厚向四层扫掠；不是网格无关精确解。统计 X/M 排序前六弯曲分支，经典曲线包含省略偶应力的模型差异。','', 'NOT_RUN：全布里渊区误差、全尺度三维网格收敛、跨模型场 MAC。','', '实际命令与环境/配置哈希分别见各 manifest.json 与 provenance.json；详细数值检查见 audit.json。']
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(audit,indent=2))
