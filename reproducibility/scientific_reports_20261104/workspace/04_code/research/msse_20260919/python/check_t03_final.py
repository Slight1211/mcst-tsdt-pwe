"""N=13 refinement, embedded-basis mass subspaces, geometry eigen audit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import datetime,json,hashlib,time,gc
import numpy as np
from scipy.linalg import eigh,svdvals
from scipy.optimize import linear_sum_assignment
from two_phase import assemble,solve,polygon
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/t03_N11_20260919T194912206613Z'

def match(old,s,d):
    v0=old['modes'];idx={tuple(p):i for i,p in enumerate(s['ij'])};embedded=np.zeros((len(s['M']),v0.shape[1]),complex)
    for j,p in enumerate(old['ij']):embedded[5*idx[tuple(p)]:5*idx[tuple(p)]+5]=v0[5*j:5*j+5]
    inner=embedded.conj().T@s['M']@d['v'];norm=np.einsum('ij,ij->j',embedded.conj(),s['M']@embedded).real
    mac=abs(inner)**2/norm[:,None];i,j=linear_sum_assignment(-mac);mapping=dict(zip(i,j))
    # Union clusters based on near-degeneracy in either ordered spectrum.
    w0=np.sqrt(old['lambda_raw']);w=np.sqrt(d['lam']);groups=[];g=[0]
    for n in range(1,len(w0)):
        if min(abs(w0[n]/w0[n-1]-1),abs(w[n]/w[n-1]-1))<1e-5:g.append(n)
        else:groups.append(g);g=[n]
    groups.append(g);rows=[]
    for g in groups:
        if g[-1]>=12:continue
        target=sorted(mapping[x] for x in g);A=embedded[:,g];B=d['v'][:,target]
        gram=A.conj().T@s['M']@A;e,U=eigh(gram);A=A@(U@np.diag(1/np.sqrt(e))@U.conj().T)
        overlap=svdvals(A.conj().T@s['M']@B)**2
        delta=max(abs(np.mean(old['RA'][g])-np.mean(d['RA'][target])),abs(np.mean(old['RB'][g])-np.mean(d['RB'][target])))
        rows.append(dict(source_indices=(np.array(g)+1).tolist(),target_indices=(np.array(target)+1).tolist(),min_subspace_overlap=float(min(overlap)),mean_participation_change=float(delta),matching_gate=bool(min(overlap)>=.99),energy_gate=bool(delta<=.01)))
    return rows

def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'runs'/('t03_N13_'+stamp);out.mkdir();print('OUTPUT',out,flush=True)
    meta=json.loads((OLD/'metadata.json').read_text());meta.update(run_id=out.name,created_at=stamp,parent_run=str(OLD),execution_status='running',reason='N11 X band4 not frequency converged; added N13 without changing thresholds')
    meta['config']=dict(meta['config'],N=[13],subspace_overlap_gate=.99,geometry_N=5,geometry_nodes=[32,64,128,256]);meta['config_sha256']=hashlib.sha256(json.dumps(meta['config'],sort_keys=True).encode()).hexdigest();meta['source_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));rows=[];geom=[]
    for point,k in [('X',[np.pi,0]),('M',[np.pi,np.pi])]:
        t=time.perf_counter();s=assemble(13,k);d=solve(s)
        with np.load(OLD/f'{point}_N11.npz') as f:old={x:f[x] for x in ['modes','ij','lambda_raw','RA','RB']}
        groups=match(old,s,d);delta=abs(np.sqrt(d['lam'][:12]/old['lambda_raw'][:12])-1)
        row=dict(point=point,N=13,ndof=len(s['M']),seconds=time.perf_counter()-t,frequency_change=delta.tolist(),max_frequency_change=float(max(delta)),frequency_gate=bool(max(delta)<=.005),clusters=groups,residual=d['residual'],hermitian=d['hermitian'],closure=d['closure'])
        rows.append(row);(out/'refinement.json').write_text(json.dumps(rows,indent=2))
        np.savez_compressed(out/f'{point}_N13_modes.npz',ij=s['ij'],G=s['G'],lambda_raw=d['lam'],modes=d['v'],RA=d['RA'],RB=d['RB'])
        print(point,'frequency',max(delta),'overlap',min(g['min_subspace_overlap'] for g in groups),'energy',max(g['mean_participation_change'] for g in groups),flush=True)
        del s,d,old;gc.collect()
    r=np.sqrt(.3/np.pi)
    for point,k in [('X',[np.pi,0]),('M',[np.pi,np.pi])]:
        base=solve(assemble(5,k))
        for n in [32,64,128,256]:
            theta=2*np.pi*np.arange(n)/n;p=r*np.column_stack([np.cos(theta),np.sin(theta)])
            s=assemble(5,k,indicator=lambda g:polygon(g,p));d=solve(s)
            dw=abs(np.sqrt(d['lam'][:12]/base['lam'][:12])-1)
            # Fixed discretization geometry comparison, not cross-N matching.
            geom.append(dict(point=point,nodes=n,N=5,max_frequency_error=float(max(dw)),area=np.pi*r*r*n*np.sin(2*np.pi/n)/(2*np.pi),phase_trace_difference=max(float(abs(sum(d['RA'][:12])-sum(base['RA'][:12]))),float(abs(sum(d['RB'][:12])-sum(base['RB'][:12]))))))
            np.savez_compressed(out/f'{point}_polygon{n}.npz',nodes=p,lambda_raw=d['lam'],modes=d['v'],RA=d['RA'],RB=d['RB'])
            (out/'geometry.json').write_text(json.dumps(geom,indent=2));print('geometry',point,n,max(dw),flush=True)
    meta['execution_status']='completed';meta['convergence_status']='selected X/M points only; no BZ or independent validation'
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
