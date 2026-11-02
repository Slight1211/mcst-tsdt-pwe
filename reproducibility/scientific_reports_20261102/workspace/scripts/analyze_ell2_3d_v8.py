"""Source-bound branch pairing, mesh and candidate-gap checks for the new ell2 run."""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'05_validation/V01_ell2_20261031_v8'
CODE=ROOT/'04_code/research/wavemotion_surface_20260920'
sys.path[:0]=[str(ROOT/'scripts'),str(CODE/'src'),str(CODE.parent/'msse_20260919/python')]
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse.linalg import eigsh,LinearOperator
from threadpoolctl import threadpool_limits
from inverse_circle_bending import prepare,classical_inverse
from real_circle_bending import assemble_real
from node_shape_bending import fourier
from matplotlib.path import Path as PolygonPath
from analyze_v01 import reconstruct,mac,read_fields
OPT=ROOT/'05_validation/V07_ell2_reoptimization_20261030'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def write(p,rr):
    if not rr:return
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def pwe(shape,tag):
    folder=OUT/'pwe_N21_fields';folder.mkdir(exist_ok=True);path=folder/f'{shape}_{tag}.npz'
    metadata=path.with_suffix('.json');geom=OUT/'inputs'/f'{shape}_vertices.csv'
    if path.exists():
        m=load(metadata);assert m['ell_um']==2 and m['geometry_sha256']==sha(geom) and m['npz_sha256']==sha(path)
        with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
    y=np.zeros(6) if shape=='initial' else np.array(load(OPT/'restart/manifest.json')['design'])
    k=np.array({'Gamma':[0,0],'X':[np.pi,0],'M':[np.pi,np.pi]}[tag])
    F=fourier(y,21);cache=prepare(21,.08,theory='TSDT',indicator_matrix=F)
    s=assemble_real(21,k,h=.08,theory='TSDT',indicator_matrix=F)
    K=classical_inverse(cache,k)+s['KB']/9;M=s['M'];del s
    factor=cho_factor(K+.001*M,lower=True,check_finite=False)
    inv=LinearOperator(K.shape,matvec=lambda v:cho_solve(factor,v,check_finite=False),dtype=float)
    val,V=eigsh(K,k=12,M=M,sigma=-.001,OPinv=inv,tol=1e-10,
                v0=np.random.default_rng(20261031).normal(size=len(K)),ncv=60)
    ix=np.argsort(val);val=val[ix];V=V[:,ix]
    kv=K@V;mv=M@V
    residual=np.linalg.norm(kv-mv*val,axis=0)/(np.linalg.norm(kv,axis=0)+abs(val)*np.linalg.norm(mv,axis=0)+1e-30)
    selected=slice(1,6) if tag=='Gamma' else slice(0,6)
    assert max(residual[selected])<1e-7
    omega=np.sqrt(np.maximum(val,0))
    baseline=(ROOT/'05_validation/V08_N21_ell2_20261030/initial_ell2_N21_ibz45.csv' if shape=='initial'
              else OPT/'N21/final_ell2_N21_ibz45.csv')
    archived=min(rows(baseline),key=lambda r:(float(r['kx_a'])-k[0])**2+(float(r['ky_a'])-k[1])**2)
    assert abs(float(archived['kx_a'])-k[0])+abs(float(archived['ky_a'])-k[1])<1e-12
    drift=max(abs(omega[b]/float(archived[f'Omega{b+1}'])-1) for b in [3,4])
    assert drift<1e-8,(shape,tag,'PWE differs from frozen N21 spectrum',drift)
    result=dict(vectors=V,omega=omega,ij=cache['ij'],k=k,residual=residual)
    np.savez_compressed(path,**result)
    dump(metadata,dict(shape=shape,N=21,ell_um=2,ell_over_h_squared=1/9,geometry_sha256=sha(geom),
        max_nonrigid_first6_residual=float(max(residual[selected])),npz_sha256=sha(path),
        baseline_path=baseline.relative_to(ROOT).as_posix(),baseline_sha256=sha(baseline),
        band45_drift_from_saved_N21=drift,source_sha256=sha(Path(__file__)),new_eigensolve=True))
    print('PWE_FIELD_SOLVE',shape,tag,'drift',drift,flush=True);return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only-pwe',action='store_true');args=ap.parse_args()
    records=[];gaps=[];matrix=[];cases=[]
    with threadpool_limits(limits=2):
        if args.only_pwe:
            for shape in ['initial','final']:
                for tag in ['Gamma','X','M']:pwe(shape,tag)
            return
        for folder in sorted(OUT.iterdir()):
            if not folder.is_dir() or not (folder/'manifest.json').exists():continue
            meta=load(folder/'manifest.json')
            if meta['status']!='COMPLETED_RAW_EXPORT':continue
            assert meta['ell_um']==2 and sha(folder/'ValidateShapeMcst.java')==meta['source_sha256']
            shape=meta['shape'];raw=rows(folder/'spectra.csv')
            vertices=np.genfromtxt(OUT/'inputs'/f'{shape}_vertices.csv',delimiter=',',names=True)
            polygon=np.column_stack([vertices['x_over_a'],vertices['y_over_a']])
            assert meta['vertices_sha256']==sha(OUT/'inputs'/f'{shape}_vertices.csv')
            edges=[]
            for pi in sorted({int(r['point_index']) for r in raw}):
                rr=[r for r in raw if int(r['point_index'])==pi]
                px=float(rr[0]['kx_pi_over_a']);py=float(rr[0]['ky_pi_over_a'])
                tag='Gamma' if px==0 else 'X' if py==0 else 'M';choices={}
                for threshold in [.7,.8]:
                    target=sorted([r for r in rr if float(r['frequency_mhz'])>.001 and float(r['vertical_fraction'])>threshold],
                                  key=lambda r:float(r['frequency_mhz']))
                    if tag=='Gamma':target=[None]+target
                    assert len(target)>=6,(folder.name,tag,'insufficient flexural modes')
                    choices[threshold]=target
                assert all(choices[.7][b]['mode_index']==choices[.8][b]['mode_index'] for b in [3,4]),(folder.name,tag,'threshold-sensitive target')
                target=choices[.8]
                coords,weights,fe,rho=read_fields(folder/f'fields_{tag}.csv')
                expected=np.where(PolygonPath(polygon).contains_points(coords[:,:2]/75e-6),7780.,1180.)
                assert np.allclose(rho,expected,rtol=1e-10,atol=1e-8),(folder.name,tag,'geometry/density grid')
                p=pwe(shape,tag);physical=reconstruct(coords,p['vectors'][:,:6],p['ij'],p['k'])
                mm=mac(physical[:,3:5],fe,weights)
                edges.append(dict(point=tag,L=float(target[3]['frequency_mhz']),U=float(target[4]['frequency_mhz'])))
                for b in [3,4]:
                    r=target[b];idx=int(r['mode_index'])-1;freq=float(r['frequency_mhz'])
                    pwe_f=float(p['omega'][b]*np.sqrt(4.35e9/1180)/(75e-6*2*np.pi*1e6))
                    cluster=[int(x['mode_index'])-1 for x in rr if abs(float(x['frequency_mhz'])/freq-1)<.001 and float(x['vertical_fraction'])>.8]
                    B=fe[:,cluster];A=physical[:,b]
                    gram=B.conj().T@(weights[:,None]*B);cross=B.conj().T@(weights*A)
                    projection=float(np.real(cross.conj()@np.linalg.pinv(gram,rcond=1e-12)@cross)/np.sum(weights*abs(A)**2))
                    imag=abs(float(r['imag_frequency_mhz']))/freq;assert imag<1e-7
                    records.append(dict(case=folder.name,shape=shape,ell_um=2,mesh_level=meta['mesh_level'],layers=meta['layers'],
                        requested_eigenpairs=meta['eigen_count'],point=tag,band=b+1,raw_3d_mode=idx+1,
                        frequency_3d_MHz=freq,frequency_pwe_N21_MHz=pwe_f,pwe_minus_3d_percent=100*(pwe_f/freq-1),
                        MAC=float(mm[b-3,idx]),cluster_modes=';'.join(str(c+1) for c in cluster),subspace_projection=projection,
                        best_MAC_mode=int(np.argmax(mm[b-3]))+1,best_MAC=float(max(mm[b-3])),
                        vertical_fraction=float(r['vertical_fraction']),relative_imaginary=imag,threshold_0p7_0p8_same=True,
                        field_grid='21x21x5',raw_spectra_sha256=sha(folder/'spectra.csv'),raw_field_sha256=sha(folder/f'fields_{tag}.csv')))
                    matrix.extend(dict(case=folder.name,point=tag,pwe_band=b+1,solid_mode=j+1,MAC=float(x)) for j,x in enumerate(mm[b-3]))
                print('AUDITED',folder.name,tag,flush=True)
            lo=max(edges,key=lambda e:e['L']);hi=min(edges,key=lambda e:e['U']);L=lo['L'];U=hi['U']
            gaps.append(dict(case=folder.name,shape=shape,ell_um=2,mesh_level=meta['mesh_level'],layers=meta['layers'],
                requested_eigenpairs=meta['eigen_count'],sample_set=meta['points'],points=len(edges),candidate_L_MHz=L,
                candidate_U_MHz=U,candidate_J=200*(U-L)/(U+L),lower_point=lo['point'],upper_point=hi['point'],is_full_BZ_gap=False))
            cases.append(dict(case=folder.name,manifest_sha256=sha(folder/'manifest.json')))
    assert records,'No completed raw cases yet'
    write(OUT/'point_comparison.csv',records);write(OUT/'candidate_gaps.csv',gaps);write(OUT/'MAC_matrix.csv',matrix)
    comparisons=[]
    for shape in ['initial','final']:
        for kind,mesh,layers,count,points in [('in_plane_mesh',3,4,24,['Gamma','X','M']),
                ('eigenpair_count',2,4,32,['X']),('thickness_mesh',2,6,24,['Gamma','X'])]:
            for tag in points:
                for band in [4,5]:
                    base=next((r for r in records if r['shape']==shape and r['mesh_level']==2 and r['layers']==4 and r['requested_eigenpairs']==24 and r['point']==tag and r['band']==band),None)
                    extra=next((r for r in records if r['shape']==shape and r['mesh_level']==mesh and r['layers']==layers and r['requested_eigenpairs']==count and r['point']==tag and r['band']==band),None)
                    if base and extra:comparisons.append(dict(kind=kind,shape=shape,point=tag,band=band,case_base=base['case'],
                        case_check=extra['case'],absolute_relative_change_percent=abs(100*(extra['frequency_3d_MHz']/base['frequency_3d_MHz']-1))))
    write(OUT/'mesh_count_checks.csv',comparisons)
    gains=[]
    for mesh,layers,points in [(2,4,'GXM'),(3,4,'GXM'),(2,6,'GX')]:
        pair=[next((g for g in gaps if g['shape']==s and g['mesh_level']==mesh and g['layers']==layers and g['requested_eigenpairs']==24 and g['sample_set']==points),None) for s in ['initial','final']]
        if all(pair):gains.append(dict(mesh_level=mesh,layers=layers,points=points,initial=pair[0]['candidate_J'],
            final=pair[1]['candidate_J'],gain_pp=pair[1]['candidate_J']-pair[0]['candidate_J']))
    write(OUT/'candidate_gains.csv',gains)
    complete=len(cases)==8 and len(records)==36 and len(comparisons)==24
    # Diagnostic limits were fixed before any raw case completed; they are not error bounds.
    limits=dict(in_plane_mesh=.5,eigenpair_count=.0001,thickness_mesh=.2)
    checks=dict(threshold_and_imaginary_checks=True,physical_field_projection=all(r['subspace_projection']>.98 for r in records),
        target_is_best_MAC_or_in_best_MAC_cluster=all(str(r['best_MAC_mode']) in r['cluster_modes'].split(';') for r in records),
        mesh_and_count_changes=all(r['absolute_relative_change_percent']<limits[r['kind']] for r in comparisons))
    report=dict(status=('PASS_SELECTED_POINT_ELL2_3D_CHECKS' if all(checks.values()) else 'REQUIRES_REVIEW') if complete else 'ANALYSED_AVAILABLE_CASES',
        complete_requested_case_set=complete,cases=cases,checks=checks,diagnostic_limits_percent=limits,candidate_gains=gains,
        minimum_individual_MAC=min(r['MAC'] for r in records),minimum_subspace_projection=min(r['subspace_projection'] for r in records),
        maximum_absolute_PWE_frequency_difference_percent=max(abs(r['pwe_minus_3d_percent']) for r in records),
        maximum_changes_percent={k:max((r['absolute_relative_change_percent'] for r in comparisons if r['kind']==k),default=None) for k in limits},
        full_3D_BZ_gap_certified=False,independent_MCST_surface_interface_benchmark=False,
        source_sha256=sha(Path(__file__)),field_helper_sha256=sha(ROOT/'scripts/analyze_v01.py'),human_verified=False,submission_ready=False)
    dump(OUT/'analysis.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
