"""Audit raw three-dimensional spectra and physical displacement correspondence."""
from pathlib import Path
import csv,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'04_code/research/wavemotion_surface_20260920'
NODE=ROOT/'02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1'
OUT=ROOT/'05_validation/V01_20261030'
sys.path[:0]=[str(CODE/'src'),str(CODE.parent/'msse_20260919/python')]
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse.linalg import eigsh,LinearOperator
from threadpoolctl import threadpool_limits
from inverse_circle_bending import prepare,classical_inverse
from real_circle_bending import assemble_real
from node_shape_bending import fourier
from msse_core import operators
from matplotlib.path import Path as PolygonPath
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(p,rows):
    with p.open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def reconstruct(coords,V,ij,k):
    q=2*np.pi*ij+k;xy,inverse=np.unique(coords[:,:2]/75e-6,axis=0,return_inverse=True)
    phase=np.exp(1j*(xy@q.T));coef=V.reshape(len(ij),3,-1).astype(complex)*np.array([1,1j,1j])[None,:,None]
    w=(phase@coef[:,0])[inverse];bx=(phase@coef[:,1])[inverse];by=(phase@coef[:,2])[inverse]
    wx=(phase@(1j*q[:,0,None]*coef[:,0]))[inverse];wy=(phase@(1j*q[:,1,None]*coef[:,0]))[inverse]
    z=coords[:,2]/75e-6;tau=.08;A=z-4*z**3/(3*tau*tau);B=-4*z**3/(3*tau*tau)
    return np.stack([A[:,None]*bx+B[:,None]*wx,A[:,None]*by+B[:,None]*wy,w],axis=1).reshape(3*len(coords),-1)
def mac(A,B,w):
    return abs(A.conj().T@(w[:,None]*B))**2/(np.sum(w[:,None]*abs(A)**2,axis=0)[:,None]*np.sum(w[:,None]*abs(B)**2,axis=0)[None,:])
def read_fields(path):
    d=np.genfromtxt(path,delimiter=',',names=True)
    coords=np.column_stack([d[x] for x in ['x_m','y_m','z_m']])
    nm=sum(x.startswith('u') and x.endswith('_real') for x in d.dtype.names)
    f=np.stack([np.column_stack([d[f'{c}{b}_real']+1j*d[f'{c}{b}_imag'] for c in ['u','v','w']]) for b in range(1,nm+1)],axis=2).reshape(3*len(coords),nm)
    weights=np.repeat(d['rho_kg_m3']*d['volume_weight_m3'],3)
    return coords,weights,f,d['rho_kg_m3']
def pwe(shape,tag):
    folder=OUT/'pwe_N21_fields';folder.mkdir(exist_ok=True)
    path=folder/f'{shape}_{tag}.npz'
    if path.exists():
        with np.load(path) as d:return {k:d[k] for k in d.files}
    y=np.zeros(6) if shape=='initial' else np.array(json.loads((NODE/'manifest.json').read_text())['design'])
    k=np.array({'Gamma':[0,0],'X':[np.pi,0],'M':[np.pi,np.pi]}[tag]);F=fourier(y,21);cache=prepare(21,.08,theory='TSDT',indicator_matrix=F)
    s=assemble_real(21,k,h=.08,theory='TSDT',indicator_matrix=F);K=classical_inverse(cache,k)+s['KB']/36;M=s['M'];del s
    factor=cho_factor(K+.001*M,lower=True,check_finite=False)
    inv=LinearOperator(K.shape,matvec=lambda v:cho_solve(factor,v,check_finite=False),dtype=float)
    val,V=eigsh(K,k=12,M=M,sigma=-.001,OPinv=inv,tol=1e-10,v0=np.random.default_rng(20261030).normal(size=len(K)),ncv=60)
    idx=np.argsort(val);val=val[idx];V=V[:,idx];kv=K@V;mv=M@V
    residual=np.linalg.norm(kv-mv*val,axis=0)/(np.linalg.norm(kv,axis=0)+abs(val)*np.linalg.norm(mv,axis=0)+1e-30)
    assert np.max(residual[1:6])<1e-7
    result=dict(vectors=V,omega=np.sqrt(np.maximum(val,0)),ij=cache['ij'],k=k,residual=residual)
    np.savez_compressed(path,**result)
    (folder/f'{shape}_{tag}.json').write_text(json.dumps(dict(shape=shape,N=21,ell_um=1,geometry_file_sha256=sha(NODE/f'{shape}_vertices.csv'),source_sha256=sha(Path(__file__)),max_nonrigid_first6_residual=float(np.max(residual[1:6])),npz_sha256=sha(path)),indent=2),encoding='utf-8')
    print('PWE_FIELD_SOLVE',shape,tag,flush=True);return result
def main():
    records=[];case_gaps=[];matrix_rows=[];checked=[]
    with threadpool_limits(limits=2):
      for folder in sorted(OUT.iterdir()):
        if not folder.is_dir() or not (folder/'manifest.json').exists():continue
        meta=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if meta['status']!='COMPLETED_RAW_EXPORT':continue
        assert sha(folder/'ValidateShapeMcst.java')==meta['source_sha256']
        shape=meta['shape'];raw=list(csv.DictReader((folder/'spectra.csv').open(newline='',encoding='utf-8')))
        spectra=[]
        vertices=np.genfromtxt(NODE/f'{shape}_vertices.csv',delimiter=',',names=True)
        polygon=np.column_stack([vertices['x_over_a'],vertices['y_over_a']])
        for pi in sorted({int(r['point_index']) for r in raw}):
          rr=[r for r in raw if int(r['point_index'])==pi]
          px=float(rr[0]['kx_pi_over_a']);py=float(rr[0]['ky_pi_over_a']);tag='Gamma' if px==0 else 'X' if py==0 else 'M'
          choices={}
          for threshold in [.7,.8]:
            selected=sorted((r for r in rr if float(r['frequency_mhz'])>.001 and float(r['vertical_fraction'])>threshold),key=lambda r:float(r['frequency_mhz']))
            if tag=='Gamma':selected=[None]+selected
            assert len(selected)>=6,(folder,tag,'insufficient flexural count',len(selected));choices[threshold]=selected
          assert all(choices[.7][b]['mode_index']==choices[.8][b]['mode_index'] for b in [3,4]),(folder,tag,'threshold dependent target')
          selected=choices[.8];spectra.append(dict(tag=tag,L=float(selected[3]['frequency_mhz']),U=float(selected[4]['frequency_mhz'])))
          coords,weights,fe,rho=read_fields(folder/f'fields_{tag}.csv')
          expected=np.where(PolygonPath(polygon).contains_points(coords[:,:2]/75e-6),7780.,1180.)
          assert np.allclose(rho,expected,rtol=1e-10,atol=1e-8),'density/physical-grid mismatch'
          p=pwe(shape,tag);physical=reconstruct(coords,p['vectors'][:,:6],p['ij'],p['k']);mm=mac(physical[:,3:5],fe,weights)
          for b in [3,4]:
            r=selected[b];idx=int(r['mode_index'])-1;freq=float(r['frequency_mhz']);pwe_f=float(p['omega'][b]*np.sqrt(4.35e9/1180)/(75e-6*2*np.pi*1e6));overlap=float(mm[b-3,idx]);best=int(np.argmax(mm[b-3]))+1
            relative_imag=abs(float(r['imag_frequency_mhz']))/freq;assert relative_imag<1e-7
            # Near-degenerate modes have arbitrary individual basis vectors.
            # Use a frequency-defined cluster and a weighted subspace projection;
            # keep the individual MAC rather than replacing or hiding it.
            cluster=[int(x['mode_index'])-1 for x in rr if abs(float(x['frequency_mhz'])/freq-1)<1e-3 and float(x['vertical_fraction'])>.8]
            A=physical[:,b:b+1];B=fe[:,cluster];gram=B.conj().T@(weights[:,None]*B);cross=B.conj().T@(weights[:,None]*A)
            projection=float(np.real((cross.conj().T@np.linalg.solve(gram,cross))[0,0])/np.sum(weights*abs(A[:,0])**2))
            records.append(dict(case=folder.name,shape=shape,mesh_level=meta['mesh_level'],layers=meta['layers'],requested_eigenpairs=meta['eigen_count'],point=tag,band=b+1,raw_3d_mode=idx+1,frequency_3d_MHz=freq,frequency_pwe_N21_MHz=pwe_f,pwe_minus_3d_percent=100*(pwe_f/freq-1),MAC=overlap,cluster_modes=';'.join(str(x+1) for x in cluster),cluster_relative_tolerance=.001,subspace_projection=projection,best_MAC_mode=best,best_MAC=float(np.max(mm[b-3])),vertical_fraction=float(r['vertical_fraction']),relative_imaginary=relative_imag,threshold_0p7_0p8_same=True,field_grid='21x21x5',raw_spectra_sha256=sha(folder/'spectra.csv')))
            for j,x in enumerate(mm[b-3]):matrix_rows.append(dict(case=folder.name,point=tag,pwe_band=b+1,solid_mode=j+1,MAC=float(x)))
          print('AUDITED',folder.name,tag,[(b+1,float(mm[b-3,int(selected[b]['mode_index'])-1])) for b in [3,4]],flush=True)
        L=max(x['L'] for x in spectra);U=min(x['U'] for x in spectra)
        case_gaps.append(dict(case=folder.name,shape=shape,mesh_level=meta['mesh_level'],layers=meta['layers'],requested_eigenpairs=meta['eigen_count'],sample_set=meta['points'],points=len(spectra),candidate_L_MHz=L,candidate_U_MHz=U,candidate_J=200*(U-L)/(U+L),is_J45=False))
        checked.append(dict(case=folder.name,manifest_sha256=sha(folder/'manifest.json')))
    assert records,'no completed raw cases available'
    write_csv(OUT/'V01_point_comparison.csv',records);write_csv(OUT/'V01_candidate_gaps.csv',case_gaps);write_csv(OUT/'V01_MAC_matrix.csv',matrix_rows)
    gx=[r for r in records if r['point'] in ['Gamma','X']]
    report=dict(status='ANALYSED_AVAILABLE_CASES',cases=checked,minimum_individual_MAC_all_points=min(r['MAC'] for r in records),minimum_Gamma_X_target_MAC=min(r['MAC'] for r in gx),minimum_cluster_projection=min(r['subspace_projection'] for r in records),all_Gamma_X_targets_are_max_MAC=all(r['raw_3d_mode']==r['best_MAC_mode'] for r in gx),all_threshold_checks_pass=True,all_imaginary_checks_pass=True,not_continuous_BZ_validation=True,full_3D_gap_not_certified=True,code_sha256=sha(Path(__file__)),human_verified=False)
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
