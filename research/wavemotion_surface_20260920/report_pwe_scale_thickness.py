"""Report actual sorted spectra; no invented gaps or mode pairing."""
import sys,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 out=Path(sys.argv[1]);rows=list(csv.DictReader((out/'spectra.csv').open()))
 for r in rows:
  for key in ['N','band']:r[key]=int(r[key])
  for key in ['a_um','h_um','ell_um','frequency_MHz','Omega','residual']:r[key]=float(r[key])
 orders=sorted(set(r['N'] for r in rows));N=orders[-1]
 colors=['#0072BD','#D95319','#77AC30','#7E2F8E']
 combos=[('FSDT',0.),('TSDT',0.),('FSDT',1.),('TSDT',1.)]
 plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':200})
 for quantity in ['frequency_MHz','Omega']:
  fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
  for i,(vary,fixed,val) in enumerate([('a_um','h_um',2.),('h_um','a_um',25.)]):
   for j,point in enumerate(['X','M']):
    ax=axs[i,j]
    for c,(theory,ell) in zip(colors,combos):
     rr=sorted([r for r in rows if r['N']==N and r['band']==1 and r['point']==point and r[fixed]==val and r['theory']==theory and r['ell_um']==ell],key=lambda r:r[vary])
     ax.plot([r[vary] for r in rr],[r[quantity] for r in rr],'-o',c=c,ms=4,label=('MCST' if ell else 'Classic')+'-'+theory)
    ax.set(xlabel=('a' if i==0 else 'h')+' (um)',ylabel='Frequency (MHz)' if quantity=='frequency_MHz' else 'Omega',title=f'{point}; '+('h = 2 um' if i==0 else 'a = 25 um'))
    ax.set_xscale('log')
    ticks=sorted(set(r[vary] for r in rr));ax.set_xticks(ticks,labels=[f'{x:g}' for x in ticks]);ax.minorticks_off()
    ax.legend(fontsize=8)
  fig.suptitle(f'Direct PWE N={N}: lowest sorted bending frequency (pilot)')
  fig.savefig(out/f'scan_{quantity}.png');fig.savefig(out/f'scan_{quantity}.pdf');plt.close(fig)
 key=lambda r:tuple(r[v] for v in ['a_um','h_um','point','theory','ell_um','band'])
 low={key(r):r for r in rows if r['N']==orders[0]};conv=[]
 for r in rows:
  if r['N']==N:
   delta=100*abs(r['frequency_MHz']/low[key(r)]['frequency_MHz']-1)
   conv.append(dict(**{k:r[k] for k in ['a_um','h_um','point','theory','ell_um','band']},sorted_frequency_step_percent=delta))
 with (out/'sorted_order_change.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=conv[0]);w.writeheader();w.writerows(conv)
 assessment=dict(solved_eigenproblems=len(rows)//8,stored_frequencies=len(rows),orders=orders,max_residual=max(r['residual'] for r in rows),max_sorted_frequency_step_percent=max(r['sorted_frequency_step_percent'] for r in conv),lowest_branch_max_step_percent=max(r['sorted_frequency_step_percent'] for r in conv if r['band']==1),scope='Sorted spectra only. Eigenvectors retained for later subspace matching. No convergence or full gap certification.')
 (out/'assessment.json').write_text(json.dumps(assessment,indent=2));print(json.dumps(assessment,indent=2))
 (out/'REPORT.md').write_text('# PWE scale/thickness pilot\n\n'+json.dumps(assessment,indent=2)+'\n\nFSDT uses 5/6 classical shear correction; full kinematic mass and MCST curvature. Host ell=1 um, steel ell=0; classic ell=0. Direct factorization. No COMSOL executed. No full-path bands. Numerical ell is not calibrated.\n',encoding='utf-8')
if __name__=='__main__':main()
