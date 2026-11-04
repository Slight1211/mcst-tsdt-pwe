"""Optional N=11 refinement after failed N=7->9 frequency gate."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,datetime,hashlib,time,sys
import numpy as np
from two_phase import assemble,solve
ROOT=Path(__file__).resolve().parents[1]
def main():
    source=Path(sys.argv[1]).resolve();base=json.loads((source/'metadata.json').read_text());assert base['execution_status']=='completed'
    out=ROOT/'runs'/('t03_N11_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    meta=dict(base);meta.update(run_id=out.name,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_status='running',parent_run=str(source),reason='N7->9 failed 0.5% frequency gate')
    meta['config']=dict(base['config'],N=[11]);meta['config_sha256']=hashlib.sha256(json.dumps(meta['config'],sort_keys=True).encode()).hexdigest();meta['source_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));rows=[];old=json.loads((source/'cases.json').read_text())
    print('OUTPUT',out,flush=True)
    for point,k in meta['config']['k'].items():
        start=time.perf_counter();s=assemble(11,k);d=solve(s);sec=time.perf_counter()-start
        prev=next(x for x in old if x['point']==point and x['N']==9);omega=np.sqrt(d['lam']);err=abs(omega[:12]/np.array(prev['omega'][:12])-1)
        # Union of near-degenerate adjacent gaps in either discretization.
        # Means are basis-invariant within fully retained clusters; matching
        # by spectral ordering is still provisional, not field MAC.
        groups=[];g=[0]
        for j in range(1,16):
            close=min(abs(omega[j]/omega[j-1]-1),abs(prev['omega'][j]/prev['omega'][j-1]-1))<1e-5
            if close:g.append(j)
            else:groups.append(g);g=[j]
        groups.append(g);energy=[]
        for group in groups:
            if group[-1]>=12:continue
            ids=np.array(group);change=max(abs(np.mean(d['RA'][ids])-np.mean(np.array(prev['RA'])[ids])),abs(np.mean(d['RB'][ids])-np.mean(np.array(prev['RB'])[ids])))
            energy.append(dict(sorted_indices=(ids+1).tolist(),mean_participation_change=float(change),gate=bool(change<=.01)))
        row=dict(point=point,N=11,ndof=len(s['M']),seconds=sec,omega=omega.tolist(),RA=d['RA'].tolist(),RB=d['RB'].tolist(),frequency_change_first12=err.tolist(),max_frequency_change=float(max(err)),frequency_gate=bool(max(err)<=.005),cluster_mean_participation=energy,matching='sorted spectral clusters; field MAC not done',residual=d['residual'],hermitian=d['hermitian'],closure=d['closure'])
        rows.append(row);np.savez_compressed(out/f'{point}_N11.npz',**s,lambda_raw=d['lam'],modes=d['v'],RA=d['RA'],RB=d['RB'])
        (out/'refinement.json').write_text(json.dumps(rows,indent=2));print(point,sec,'max frequency change',max(err),flush=True)
    meta['execution_status']='completed';meta['convergence_status']='see refinement.json; no global convergence claim';(out/'metadata.json').write_text(json.dumps(meta,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for a,point in zip(ax,['X','M']):
        x=[r for r in old if r['point']==point]+[r for r in rows if r['point']==point]
        for j in [0,3,7,11]:a.plot([r['N'] for r in x],[r['omega'][j] for r in x],'-o',ms=4,label=f'Band {j+1}')
        a.set(xlabel='Plane-wave order N',ylabel='Dimensionless frequency',title=f'{point}: ordered full-five-variable spectrum',xticks=[3,5,7,9,11]);a.grid(alpha=.2);a.legend(fontsize=8)
    fig.savefig(out/'convergence.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
