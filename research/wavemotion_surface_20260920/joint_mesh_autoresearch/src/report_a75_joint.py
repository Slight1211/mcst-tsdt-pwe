"""Independent reconstruction of H3b from three immutable completed raw sources."""
import csv,json,math,re
from pathlib import Path
import numpy as np
from run_joint_baseline import ROOT,WORK,HERE,sha,utc,write_json

FIRST=ROOT/'runs/endpoint_a75_joint_M1L6_20261001_v1'
PRIOR=ROOT/'runs/endpoint_a75_joint_M1L6_n15_20261001_v1'
RUN=ROOT/'runs/endpoint_a75_joint_M1L6_n15_reboot_20261002_v1'
OLD=ROOT/'runs/thickness_ratio75_fullpath_M2L4_20260926/mcst_ratio0.32_M2L4'
SHIFT=ROOT/'runs/endpoint_a75_gamma_shift_20261001_v1'
PWE=ROOT/'runs/thickness_ratio75_20260925/pwe'

def rows(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))

def save_csv(path,records):
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=records[0].keys());w.writeheader();w.writerows(records)

def spectrum(raw):
    assert sorted({int(r['point_index']) for r in raw})==list(range(13))
    spectra=[];chosen=[];coverage=[]
    for p in range(13):
        group=[r for r in raw if int(r['point_index'])==p]
        xy=(p/4,0) if p<=4 else ((1,(p-4)/4) if p<=8 else ((12-p)/4,(12-p)/4))
        assert sorted(int(r['mode_index']) for r in group)==list(range(1,len(group)+1))
        for r in group:
            assert all(math.isfinite(float(v)) for v in r.values())
            assert (float(r['kx_pi_over_a']),float(r['ky_pi_over_a']))==xy
            assert 0<=float(r['vertical_fraction'])<=1+1e-10
        eligible=sorted([r for r in group if float(r['frequency_mhz'])>1e-5 and float(r['vertical_fraction'])>.7],key=lambda r:float(r['frequency_mhz']))
        n=5 if p in (0,12) else 6;assert len(eligible)>=n,(p,len(eligible))
        selected=eligible[:n];chosen.extend(selected)
        coverage.append(dict(point=p,raw_count=len(group),eligible_count=len(eligible),selected_modes=[int(r['mode_index']) for r in selected]))
        spectra.append(([0.] if n==5 else [])+[float(r['frequency_mhz']) for r in selected])
    assert all(abs(float(r['imag_frequency_mhz']))/float(r['frequency_mhz'])<1e-7 for r in chosen)
    closure=[abs(a-b)*1e6 for a,b in zip(spectra[0],spectra[12])]
    assert all(math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-8) for a,b in zip(spectra[0],spectra[12]))
    lp=max(range(12),key=lambda p:spectra[p][3]);up=min(range(12),key=lambda p:spectra[p][4])
    return dict(L_MHz=spectra[lp][3],U_MHz=spectra[up][4],L_point=lp,U_point=up,
      spectra_MHz=spectra,coverage=coverage,gamma_differences_Hz=closure,
      min_vertical=min(float(r['vertical_fraction']) for r in chosen),
      max_relative_imaginary=max(abs(float(r['imag_frequency_mhz']))/float(r['frequency_mhz']) for r in chosen))

def main():
    done=json.loads((RUN/'completion.json').read_text());assert done==json.loads((RUN/'manifest.json').read_text())
    assert done['status']=='PASS_DERIVED_PATH_MIXED_20_15_REBOOT_RECOVERY' and done['exit_code']==0
    hashes={}
    def record(path,expected=None):
        digest=sha(path)
        if expected is not None:assert digest==expected,str(path)
        hashes[str(path)]=digest
    # Full frozen chain proves the source model/material/weak-form lineage.
    experiments=[('endpoint_a75_joint',FIRST),('endpoint_a75_joint_n15',PRIOR),('endpoint_a75_reboot_resume',RUN)]
    for name,folder in experiments:
        exp=WORK/'experiments'/name;frozen=json.loads((exp/'inputs.json').read_text())
        rec=json.loads((folder/'manifest.json').read_text())
        record(exp/'inputs.json',rec['inputs_sha256']);record(exp/'protocol.md',frozen['protocol_sha256'])
        for path,digest in frozen['input_hashes'].items():record(Path(path),digest)
        for source,digest in frozen['source_hashes'].items():record(folder/'sources'/source,digest)
    settings=done['settings']
    for key,val in [('COMSOL_LATTICE_UM',75),('COMSOL_THICKNESS_UM',24),('COMSOL_RING_RADIUS_UM',22.5),('COMSOL_MCST_LENGTH_UM',1),('COMSOL_MESH_LEVEL',1),('COMSOL_DRY_SWEEP_LAYERS',6),('COMSOL_EIGEN_COUNT',15)]:assert float(settings[key])==val
    mapping=rows(RUN/'point_sources.csv');assert [int(r['point']) for r in mapping]==list(range(13))
    merged=[]
    for p,item in enumerate(mapping):
        folder=FIRST if p<2 else (PRIOR if p<11 else RUN)
        rawpath=folder/('frequencies.csv' if p<2 else ('frequencies_remaining.csv' if p<11 else 'frequencies_recovery.csv'))
        checkpoint=folder/f'checkpoints/point_{p}.mph';count=20 if p<2 else 15
        assert Path(item['raw_source'])==rawpath and Path(item['checkpoint'])==checkpoint and int(item['eigenpairs'])==count
        record(rawpath,item['raw_sha256']);record(checkpoint,item['checkpoint_sha256'])
        subset=[r for r in rows(rawpath) if int(r['point_index'])==p];assert len(subset)==count;merged.extend(subset)
        log=(folder/'console.log').read_text(errors='replace')
        assert re.search(r'JOINT_POINT_COMPLETE point='+str(p)+r'\s',log)
        hit=re.search(r'POINT_SETTINGS point='+str(p)+r' kx=([^ ]+) ky=([^ ]+) shift=([^ ]+) rtol=([^ ]+) neigs=([^\s]+)',log)
        assert hit and float(hit[5])==count
        xy=(p/4,0) if p<=4 else ((1,(p-4)/4) if p<=8 else ((12-p)/4,(12-p)/4))
        assert (float(hit[1]),float(hit[2]))==xy
        assert float(hit[3].replace('[MHz]',''))==(1 if p in (0,12) else 1e-4)
        assert float(hit[4])==(1e-9 if p in (0,12) else 1e-6)
    assert len(merged)==205 and merged==rows(RUN/'frequencies_derived_fullpath.csv')
    record(RUN/'frequencies_derived_fullpath.csv',done['derived_sha256']);record(RUN/'point_sources.csv',done['source_map_sha256'])
    reference=rows(FIRST/'reference_M2L4_gamma_recomputed/frequencies_derived.csv');assert len(reference)==260
    for p in range(13):
        source=rows(SHIFT/f'shift1_repeat{1 if p==0 else 2}/frequencies.csv') if p in (0,12) else [r for r in rows(OLD/'frequencies.csv') if int(r['point_index'])==p]
        actual=[r for r in reference if int(r['point_index'])==p];assert len(actual)==len(source)==20
        for a,b in zip(actual,source):assert all(a[k]==b[k] for k in a if k not in ('point_index','kx_pi_over_a','ky_pi_over_a'))
    audits={'M2L4':spectrum(reference),'M1L6':spectrum(merged)}
    for key in ('L_MHz','U_MHz','L_point','U_point','spectra_MHz','min_vertical','max_relative_imaginary'):assert audits['M1L6'][key]==done['spectrum_audit'][key]
    config=json.loads((PWE/'config.json').read_text());assert config['a_um']==75 and 24 in config['h_um'] and .32 in config['h_over_a']
    assert config['N_path']==15 and config['segments_path']==4 and config['r_over_a']==.3 and config['ell_host_um']==1 and config['ell_steel_um']==0
    record(PWE/'config.json');comparisons=[]
    for theory in ('FSDT','TSDT'):
        bands=[]
        for p in range(12):
            path=PWE/'N15'/f'N15_a75_h24_p{p}_{theory}_l1.npz'
            with np.load(path) as d:
                f=d['frequency_MHz'][:6];assert len(f)==6 and np.isfinite(f).all()
                assert np.max(d['residual'][int(p==0):8])<1e-5
                xy=(p/4,0) if p<=4 else ((1,(p-4)/4) if p<=8 else ((12-p)/4,(12-p)/4))
                assert np.allclose(d['k'],np.array(xy)*np.pi,rtol=0,atol=1e-12)
                bands.append(f)
            record(path)
        bands=np.array(bands);lower=float(bands[:,3].max());upper=float(bands[:,4].min())
        for mesh,a in audits.items():comparisons.append(dict(model='MCST-'+theory,N=15,mesh=mesh,PWE_L_MHz=lower,PWE_U_MHz=upper,FEM_L_MHz=a['L_MHz'],FEM_U_MHz=a['U_MHz'],eL_percent=100*abs(lower/a['L_MHz']-1),eU_percent=100*abs(upper/a['U_MHz']-1),FEM_L_point=a['L_point'],FEM_U_point=a['U_point']))
    drift={e:100*(audits['M1L6'][e+'_MHz']/audits['M2L4'][e+'_MHz']-1) for e in ('L','U')}
    ranking=all(next(r for r in comparisons if r['mesh']==m and r['model']=='MCST-TSDT')[key]<next(r for r in comparisons if r['mesh']==m and r['model']=='MCST-FSDT')[key] for m in audits for key in ('eL_percent','eU_percent'))
    out=RUN/'report_v1';out.mkdir(exist_ok=False)
    save_csv(out/'pwe_mesh_edge_errors.csv',comparisons)
    save_csv(out/'selected_path_spectra.csv',[dict(mesh=m,point=p,branch=b+1,frequency_MHz=f) for m,a in audits.items() for p,fs in enumerate(a['spectra_MHz']) for b,f in enumerate(fs)])
    record(RUN/'completion.json');record(HERE/'report_a75_joint.py')
    report=dict(status='PASS_RAW_PATH_AND_PWE_RECOMPUTATION',created_utc=utc(),a_um=75,h_um=24,r_over_a=.3,polarization_threshold=.7,raw_rows=205,counts='20 at0/1;15 at2..12',audits=audits,comparisons=comparisons,signed_M2L4_to_M1L6_edge_change_percent=drift,TSDT_closer_at_both_edges_both_meshes=ranking,predefined_screen='PASS' if max(abs(v) for v in drift.values())<=.5 and ranking else 'INVESTIGATE',total_completed_solve_seconds=done['total_completed_solve_seconds'],recovery_wall_seconds=done['recovery_wall_seconds'],cross_mesh_MAC='NOT_RUN',full_path_15_vs20='NOT_RUN',independent_3D_algebraic_residual='NOT_RUN',full_BZ='NOT_RUN',scope='Finite-grid sampled-path comparison; not continuum error certification; cross-mesh physical pairing pending',source_hashes=hashes)
    write_json(out/'audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('audits','source_hashes')},indent=2))

if __name__=='__main__':main()
