"""Standard-library arithmetic/provenance audit; not another physical solve."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run.resolve();meta=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    assert meta['status']=='COMPLETED_SAMPLED_FROZEN_SCAN'
    assert meta['completed_eigensolves']==135 and meta['N']==15
    assert all(sha(ROOT/path)==digest for path,digest in meta['source_sha256'].items())
    assert all(sha(ROOT/path)==digest for path,digest in meta['input_sha256'].items())
    assert all(sha(run/path)==digest for path,digest in meta['output_sha256'].items())
    factor=math.sqrt(4.35e9/1180)/(75e-6*2*math.pi*1e6)
    summaries=[];all_records=[]
    for name in ['initial','ell1_design','ell2_design']:
        data=rows(run/f'{name}_ell{meta["ell_epoxy_um"]:g}_N15_ibz45.csv')
        assert len(data)==45
        assert len({(r['kx_a'],r['ky_a']) for r in data})==45
        assert all(all(float(r[f'Omega{b}'])<=float(r[f'Omega{b+1}']) for b in range(1,6)) for r in data)
        normerror=max(abs(float(r[f'f{b}_MHz'])-float(r[f'Omega{b}'])*factor) for r in data for b in range(1,7))
        assert normerror<1e-12
        lower=max(data,key=lambda r:float(r['Omega4']));upper=min(data,key=lambda r:float(r['Omega5']))
        L,U=float(lower['Omega4']),float(upper['Omega5']);J=200*(U-L)/(U+L)
        reference=next(r for r in rows(run/'summary.csv') if r['shape']==name)
        assert abs(J-float(reference['J']))<1e-12
        summaries.append(dict(shape=name,points=45,L=L,U=U,J_signed_percent=J,
            positive_gap_present=U>L,L_MHz=float(lower['f4_MHz']),U_MHz=float(upper['f5_MHz']),
            kx_L=float(lower['kx_a']),ky_L=float(lower['ky_a']),kx_U=float(upper['kx_a']),ky_U=float(upper['ky_a']),
            saved_max_nonrigid_residual=max(float(r['max_nonrigid_relative_residual']) for r in data),
            saved_max_mass_orthogonality_error=max(float(r['mass_orthogonality_max']) for r in data),
            saved_max_dense_relative_drift=max(float(r['dense_first6_relative_drift']) for r in data if r['dense_first6_relative_drift'])))
        all_records.extend(data)
    for r in summaries:r['gain_from_initial_pp']=r['J_signed_percent']-summaries[0]['J_signed_percent']
    dense=rows(run/'dense_checks.csv');assert len(dense)==54
    drifts=[]
    for row in dense:
        if row['point']=='Gamma' and int(row['band'])==1:continue
        drift=float(row['Omega_sparse'])/float(row['Omega_dense'])-1
        assert abs(drift-float(row['relative_drift']))<1e-15
        drifts.append(abs(drift))
    report=dict(status='PASS_SAVED_FROZEN_ARITHMETIC_AND_PROVENANCE',N=15,ell_epoxy_um=meta['ell_epoxy_um'],
        shapes=summaries,points_per_shape=45,total_spectra_rows=len(all_records),
        dense_comparison_rows=54,dense_nonrigid_rows=len(drifts),max_dense_nonrigid_relative_drift=max(drifts),
        source_hashes_verified=len(meta['source_sha256']),input_hashes_verified=len(meta['input_sha256']),
        output_hashes_verified=len(meta['output_sha256']),solver_reruns=False,human_verified=False,
        normalization_MHz_per_Omega=factor,
        interpretation='Negative signedJ means fourth/fifth-band overlap on the finite sampled set, not a positive gap. Positive gain alone does not establish a reopened gap.')
    best=max(summaries,key=lambda r:r['J_signed_percent'])
    report['best_frozen_shape']=best['shape']
    seed=dict(selection='Maximum signed relative separation among the three frozen shapes on45points',
        shape=best['shape'],design=meta['shape_designs'][best['shape']],
        N=15,ell_epoxy_um=meta['ell_epoxy_um'],J_signed_percent=best['J_signed_percent'],
        positive_gap_present=best['positive_gap_present'],
        source_run=str(run.relative_to(ROOT)),source_run_manifest_sha256=sha(run/'manifest.json'),
        not_a_new_optimization=True)
    (run/'best_frozen_seed.json').write_text(json.dumps(seed,indent=2)+'\n',encoding='utf-8')
    (run/'saved_data_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
