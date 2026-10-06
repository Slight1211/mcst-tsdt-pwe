"""Report only exported and checked MCST FEM controls; no substituted data."""
from pathlib import Path
import csv,json,sys,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,ScalarFormatter,NullFormatter,FuncFormatter

root=Path(__file__).resolve().parent;out=Path(sys.argv[1]);assets=root.parents[1]/'output/pdf/modal_selectivity_assets'
cfg=json.loads((out/'config.json').read_text())
assets.mkdir(parents=True,exist_ok=True)
styles=[('FSDT',0,'Classic-FSDT','#0072BD','--'),('TSDT',0,'Classic-TSDT','#56B4E9','-'),('FSDT',1,'MCST-FSDT','#D55E00','-.'),('TSDT',1,'MCST-TSDT','#7E2F8E',':')]
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':10,'axes.linewidth':.8,'pdf.fonttype':42,'xtick.direction':'in','ytick.direction':'in'})
comparisons=[];status=[]
for a in cfg['a_um']:
 folder=out/f'mcst_a{a}_M2_sweep4';mf=folder/'manifest.json'
 if not mf.exists() or json.loads(mf.read_text()).get('exit_code')!=0 or not (folder/'model_Model.mph').exists():
  status.append(dict(a_um=a,status='NOT_AVAILABLE'));continue
 try:
  rr=list(csv.DictReader((folder/'frequencies.csv').open()));checked=[]
  for name,px,py in [('X',1,0),('M',1,1)]:
   part=[r for r in rr if abs(float(r['kx_pi_over_a'])-px)<1e-9 and abs(float(r['ky_pi_over_a'])-py)<1e-9]
   vals=sorted([r for r in part if float(r['vertical_fraction'])>.8 and float(r['frequency_mhz'])>0],key=lambda r:float(r['frequency_mhz']))
   assert len(vals)>=6,(name,'insufficient bending modes',len(vals))
   ref=np.array([float(r['frequency_mhz']) for r in vals[:6]])
   assert max(abs(float(r['imag_frequency_mhz']))/float(r['frequency_mhz']) for r in vals[:6])<1e-7
   for t,ell,label,c,ls in styles:
    with np.load(out/'pwe_controls'/f'N21_a{a}_h2_{name}_{t}_l{ell}.npz') as z:
     ff=z['frequency_MHz'][:6];res=float(max(z['residual'][:6]));assert res<1e-5
    for n,(f,g) in enumerate(zip(ff,ref),1):checked.append(dict(a_um=a,point=name,theory=t,ell_um=ell,band=n,pwe_MHz=float(f),mcst_3d_MHz=float(g),error_percent=float(100*abs(f/g-1)),pwe_residual=res))
  comparisons+=checked;status.append(dict(a_um=a,status='CHECKED_FIXED_M2_NOT_MESH_CONVERGED'))
 except (AssertionError,FileNotFoundError,ValueError) as e:status.append(dict(a_um=a,status='VALIDATION_PENDING_OR_FAILED',reason=str(e)))
with (out/'comparison.csv').open('w',newline='') as f:
 if comparisons:
  w=csv.DictWriter(f,fieldnames=comparisons[0]);w.writeheader();w.writerows(comparisons)
(out/'comparison_status.json').write_text(json.dumps(status,indent=2))
fig,ax=plt.subplots(figsize=(6.8,3.8));fig.subplots_adjust(left=.12,right=.98,bottom=.15,top=.73);maxima=[]
for t,ell,label,c,ls in styles:
 xx=[];yy=[]
 for a in cfg['a_um']:
  part=[r for r in comparisons if r['a_um']==a and r['theory']==t and r['ell_um']==ell]
  if len(part)==12:
   r=max(part,key=lambda r:r['error_percent']);xx.append(a);yy.append(r['error_percent']);maxima.append(r)
  else:xx.append(a);yy.append(np.nan)
 ax.plot(xx,yy,c=c,ls=ls,marker='s' if t=='FSDT' else 'o',mfc='none' if t=='FSDT' else c,ms=4.8 if t=='FSDT' else 3.5,label=label,zorder=2.2 if t=='FSDT' else 2)
ax.set_xscale('log');ax.set_yscale('log');ax.set(xlabel=r'Lattice constant $a$ ($\mu$m)',ylabel='Maximum frequency discrepancy (%)');ax.grid(alpha=.18)
ax.xaxis.set_major_locator(FixedLocator(cfg['a_um']));ax.xaxis.set_major_formatter(ScalarFormatter());ax.tick_params(axis='x',labelsize=8)
ax.yaxis.set_major_locator(FixedLocator([.5,1,2,5,10,20,30,50]));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'));ax.yaxis.set_minor_formatter(NullFormatter())
handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,frameon=False,ncol=2,fontsize=9,loc='upper center',bbox_to_anchor=(.55,.995))
ax.set_title(r'3D MCST-COMSOL: M2 in-plane, 4 swept layers; $X,M$, branches 1-6'+'\n'+r'$h=2\,\mu$m, $\ell=1\,\mu$m; inverse PWE $N=21$',fontsize=9,pad=8)
fig.savefig(out/'scale_mcst_max_error.png',dpi=300);fig.savefig(assets/'scale_mcst_max_error.pdf');plt.close(fig)
(out/'maximum_errors.json').write_text(json.dumps(maxima,indent=2))
lines=['# 尺度扩展与三维MCST对照','', '固定h=2微米，环氧l=1微米，钢l=0；r/a=0.3。四模型均与同一三维MCST对照，经典模型的差异包含省略偶应力的影响。','', '误差定义：X、M处前六条按频率排序的弯曲分支中，max(100*abs(f_PWE/f_3D-1))。PWE为N21，经典广义刚度逆因子化，MCST项直接组装。','', '三维网格：面内自动尺寸M2，厚度四层扫掠。不是原自由四面体M2；不得沿用其收敛结论。','', '| a (微米) | Classic-FSDT (%) | Classic-TSDT (%) | MCST-FSDT (%) | MCST-TSDT (%) |','|---:|---:|---:|---:|---:|']
for a in cfg['a_um']:
 vals=[next((r['error_percent'] for r in maxima if r['a_um']==a and r['theory']==t and r['ell_um']==ell),None) for t,ell,*_ in styles]
 lines.append('| '+str(a)+' | '+' | '.join('NOT_RUN / PENDING' if v is None else f'{v:.5f}' for v in vals)+' |')
lines+=['','## 运行与检查','Python、NumPy、SciPy版本、源码和配置哈希见provenance.json；COMSOL实际命令及耗时见mcst_a*_M2_sweep4/manifest.json。','命令：.venv/Scripts/python.exe research/wavemotion_surface_20260920/run_scale_mcst_reference.py '+str(out),'后处理：.venv/Scripts/python.exe research/wavemotion_surface_20260920/report_scale_mcst_reference.py '+str(out),'逐点状态见comparison_status.json，逐频率差异见comparison.csv。仅保留面外极化>0.8、相对虚部<1e-7且PWE残差<1e-5的受检分支。','', '## 失败记录','原a2000自由四面体M2生成3298594单元，内存不足而人工终止；同批后继也暂停。旧目录不用于比较。随后相对路径重启失败，已修正为绝对路径并隔离到sweep4目录。COMSOL实际保存文件名为model_Model.mph；首批导出包装器错误检查model.mph而标FAIL，后处理独立核查真实退出码、已保存模型、CSV分支数和极化，不以错误包装状态代替数据检查。','', '## 限制与下一步','NOT_RUN：全布里渊区最大误差、跨理论场MAC、全尺度三维网格收敛。本图是固定参考网格的频谱差异，不是精确解误差。干态问题不适用声功率守恒测试。','下一阶段：三维网格敏感性与PWE阶数差异应分别核查；不得以误差平台宣称已收敛。']
if any(r['a_um']==25 for r in comparisons):
 old_path=root/'runs/original_parameters_20260921T013511952049Z/mcst.csv'
 old_rows=list(csv.reader(line for line in old_path.read_text(encoding='utf-8-sig').splitlines() if not line.startswith('%')))
 old_f=np.array(sorted(complex(r[1].replace('i','j')).real for r in old_rows if float(r[2])>.8)[:6])
 new_f=np.array([r['mcst_3d_MHz'] for r in comparisons if r['a_um']==25 and r['point']=='X' and r['theory']=='TSDT' and r['ell_um']==1])
 assert len(old_f)==len(new_f)==6
 delta=100*abs(old_f/new_f-1)
 audit=dict(a_um=25,h_um=2,point='X',old_mesh='tetrahedral M3',new_mesh='in-plane M2, swept 4 layers',old_source=str(old_path),old_MHz=old_f.tolist(),new_MHz=new_f.tolist(),difference_percent=delta.tolist(),max_difference_percent=float(delta.max()),scope='single point cross-mesh check, not whole-scale convergence')
 (out/'cross_mesh_check.json').write_text(json.dumps(audit,indent=2))
 lines+=['','## 同几何网格类型交叉检查',f'a=25微米、h=2微米、X点前六条弯曲分支，旧四面体M3与新面内M2/厚向四层扫掠的最大频差为{delta.max():.4f}%。这不是全尺度网格收敛认证。详细频率及来源见cross_mesh_check.json。']
(out/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')

# Extend the existing N15 path scan using actual new checkpoints.
old=root/'runs/inverse_scan_verified_N15_20260921T160730024316Z'
data={}
for r in csv.DictReader((old/'spectra.csv').open()):
 if float(r['h_um'])!=2:continue
 key=(float(r['a_um']),r['theory'],int(float(r['ell_um'])))
 if key not in data:data[key]=np.full((12,8),np.nan)
 data[key][int(r['point']),int(r['band'])-1]=float(r['frequency_MHz'])
for a in [500,1000,2000]:
 for t,ell,*_ in styles:
  vv=[]
  for p in range(12):
   fn=out/'pwe_path'/f'N15_a{a}_h2_p{p}_{t}_l{ell}.npz'
   if fn.exists():
    with np.load(fn) as z:vv.append(z['frequency_MHz'])
  if len(vv)==12:data[a,t,ell]=np.array(vv)
gaps=[]
fig,ax=plt.subplots(figsize=(3.6,2.7),layout='constrained')
for t,ell,label,c,ls in styles:
 xx=[];yy=[]
 for a in cfg['a_um']:
  if (a,t,ell) not in data:continue
  v=data[a,t,ell];assert np.all(np.isfinite(v));L=v[:,3].max();U=v[:,4].min();J=200*(U-L)/(U+L)
  xx.append(a);yy.append(J);gaps.append(dict(a_um=a,theory=t,ell_um=ell,lower_MHz=float(L),upper_MHz=float(U),J45_percent=float(J)))
 ax.plot(xx,yy,c=c,ls=ls,marker='o',ms=3,label=label)
ax.set_xscale('log');ax.set(xlabel=r'$a$ ($\mu$m)',ylabel='Signed relative interval 4-5 (%)');ax.grid(alpha=.18);ax.legend(frameon=False,ncol=2,fontsize=7)
ax.xaxis.set_major_locator(FixedLocator([10,100,500,2000]));ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_formatter(NullFormatter())
fig.savefig(assets/'scale_extended_gap45.pdf');fig.savefig(out/'scale_extended_gap45.png',dpi=300);plt.close(fig)
(out/'extended_gaps.json').write_text(json.dumps(gaps,indent=2))
order=[]
for a in cfg['a_um']:
 for t,ell,*_ in styles:
  if (a,t,ell) not in data:continue
  for name,point in [('X',4),('M',8)]:
   fn=out/'pwe_controls'/f'N21_a{a}_h2_{name}_{t}_l{ell}.npz'
   if not fn.exists():continue
   with np.load(fn) as z:fine=z['frequency_MHz'][:6]
   rough=data[a,t,ell][point,:6]
   for n,v in enumerate(100*abs(rough/fine-1),1):order.append(dict(a_um=a,theory=t,ell_um=ell,point=name,band=n,N15_to_N21_percent=float(v)))
if order:
 with (out/'pwe_order_check.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=order[0]);w.writeheader();w.writerows(order)
 with (out/'REPORT_ZH.md').open('a',encoding='utf-8') as f:f.write('\n\n## PWE阶数检查\n同一X、M处前六条分支，N15至N21最大相对变化为'+f'{max(r["N15_to_N21_percent"] for r in order):.4f}'+'%。原始逐分支值见pwe_order_check.csv。该统计不等于全路径或N21极限收敛证明。\n')
if all(r['status']=='CHECKED_FIXED_M2_NOT_MESH_CONVERGED' for r in status):
 assert len(comparisons)==480 and len(maxima)==40 and len(data)==40 and len(gaps)==40 and len(order)==480
 (out/'completion_checks.json').write_text(json.dumps(dict(status='COMPUTED_FIXED_REFERENCE_NOT_CONVERGENCE_CERTIFIED',checked_scales=10,comparison_rows=480,maximum_error_rows=40,path_models=40,pwe_order_rows=480,max_N15_N21_percent=max(r['N15_to_N21_percent'] for r in order),max_PWE_residual=max(r['pwe_residual'] for r in comparisons),NOT_RUN=['full BZ maximum','full scale 3D mesh convergence','field MAC']),indent=2))
 files=[Path(__file__),root/'run_scale_mcst_reference.py',root/'run_verified_inverse_scans.py',root/'refine_model_map.py',root/'src/inverse_circle_bending.py',root/'src/real_circle_bending.py',root.parents[1]/'comsol/BuildTwoPhaseValidation.java',root.parents[1]/'comsol/BuildTwoPhaseValidation.class',out/'config.json',out/'comparison.csv',out/'pwe_order_check.csv']
 (out/'final_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print(json.dumps(dict(checked=sum(r['status']=='CHECKED_FIXED_M2_NOT_MESH_CONVERGED' for r in status),pending=[r for r in status if r['status']!='CHECKED_FIXED_M2_NOT_MESH_CONVERGED'],path_model_count=len(data)),indent=2))
