"""Bind current manuscript numbers to b=.24 optimization and validation outputs."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/uniform_b024_v47'
PWE = ROOT / '05_validation/UNIFORM_B024_PWE_V47_20261103'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))


def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def gap(p):
    rr = rows(p)
    L, U = max(float(r['Omega4']) for r in rr), min(float(r['Omega5']) for r in rr)
    return dict(L=L, U=U, J=200 * (U - L) / (U + L), points=len(rr))


def main():
    summary = load(OUT / 'summary.json')
    solid = load(OUT / 'solid_analysis.json')
    queue = load(ROOT / '05_validation/UNIFORM_B024_CHECK_QUEUE_V47_20261103/manifest.json')
    assert summary['status'] == 'PASS_ALL_FIVE_B024_SEARCHES'
    assert solid['status'] == 'PASS_NEW_B024_SELECTED_POINT_SOLID_CHECKS'
    assert queue['status'] == 'COMPLETED_ALL_CHECKS'
    cases = {c['ell_um']: c for c in summary['cases']}
    sources = dict(solid['source_sha256'])
    for job in queue['jobs']:
        if job['kind'] != 'pwe':
            continue
        mp = PWE / job['tag'] / 'manifest.json'
        m = load(mp)
        assert m['status'] == 'PASS_EXACT_B024_GEOMETRY_PWE' and m['shape_bound'] == .24
        sources[mp.relative_to(ROOT).as_posix()] = sha(mp)
        for name, digest in m['output_sha256'].items():
            assert sha(mp.parent / name) == digest
            sources[(mp.parent / name).relative_to(ROOT).as_posix()] = digest
    for c in cases.values():
        for path, key in [(c['manifest'], 'manifest_sha256'), (c['vertices'], 'vertices_sha256'), (c['spectrum'], 'spectrum_sha256')]:
            assert sha(ROOT / path) == c[key]
            sources[path] = c[key]
    mechanism = rows(ROOT / 'output/data/Figure6_design_story_v19.csv')
    initial = {float(r['ell_um']): dict(L=float(r['L_Omega']), U=float(r['U_Omega']), J=float(r['J_percent']), points=45)
               for r in mechanism if r['record_type'] == 'gap_transfer' and r['shape'] == 'initial'}
    for r in mechanism:
        if r['record_type'] == 'gap_transfer' and r['shape'] == 'initial':
            assert sha(ROOT / r['source']) == r['source_sha256']
            sources[r['source']] = r['source_sha256']
    primary = []
    for ell in [1, 2]:
        case = cases[ell]
        n15path = (ROOT / case['manifest']).parent / 'final_ibz45.csv'
        n15 = gap(n15path)
        assert abs(n15['J'] - case['J153_percent']) < 1e-6
        sources[n15path.relative_to(ROOT).as_posix()] = sha(n15path)
        n21m = load(PWE / f'g{ell}_N21_own/manifest.json')
        assert n21m['shape_vertices_sha256'] == case['vertices_sha256']
        n21 = n21m['gap']
        ip = ROOT / ('02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1/independent_validation/initial_N21_ibz45.csv' if ell == 1 else '05_validation/V08_N21_ell2_20261030/initial_ell2_N21_ibz45.csv')
        initial21 = gap(ip)
        sources[ip.relative_to(ROOT).as_posix()] = sha(ip)
        g3 = next(r for r in solid['baseline_gains'] if r['ell_um'] == ell)
        primary.extend([dict(ell_um=ell, model='PWE N15', initial=initial[ell]['J'], optimized=n15['J'], gain=n15['J'] - initial[ell]['J'], points=45),
                        dict(ell_um=ell, model='PWE N21', initial=initial21['J'], optimized=n21['J'], gain=n21['J'] - initial21['J'], points=45),
                        dict(ell_um=ell, model='3D M2L4', initial=g3['initial_J'], optimized=g3['final_J'], gain=g3['gain_pp'], points=3)])
    transfer, energies = [], []
    for ell in [.5, 1, 1.5, 2]:
        tag = 'g1_N15_own_GXM' if ell == 1 else 'g1_at' + str(ell).replace('.', 'p')
        folder = PWE / tag
        if ell == 1:
            g = dict(L=cases[1]['L'], U=cases[1]['U'], J=cases[1]['J153_percent'])
        else:
            g = load(folder / 'manifest.json')['gap']
        transfer.append(dict(ell_um=ell, initial_J=initial[ell]['J'], final_J=g['J'], gain_pp=g['J'] - initial[ell]['J']))
        for r in rows(folder / 'diagnostics.csv'):
            point = 'Gamma' if abs(float(r['kx_a'])) < 1e-12 and abs(float(r['ky_a'])) < 1e-12 else 'X' if abs(float(r['kx_a']) - np.pi) < 1e-12 and abs(float(r['ky_a'])) < 1e-12 else None
            if point:
                for band in [4, 5]:
                    energies.append(dict(ell_um=ell, point=point, band=band, R_mc=float(r[f'Rmc{band}'])))
    endpoint_old = rows(ROOT / '03_review/bound024_adoption_v42/endpoint_comparison_v42.csv')
    g0at5 = next(r for r in endpoint_old if r['shape'] == 'g0_b024' and float(r['ell_um']) == 5)
    assert sha(ROOT / g0at5['spectrum']) == g0at5['sha256']
    assert sha((ROOT / cases[0]['manifest']).parent / 'final_ibz153.csv') == next(r['sha256'] for r in endpoint_old if r['shape'] == 'g0_b024' and float(r['ell_um']) == 0)
    endpoints = [dict(shape='g0_b024', ell_um=0, **gap(ROOT / cases[0]['spectrum'])),
                 dict(shape='g0_b024', ell_um=5, **gap(ROOT / g0at5['spectrum'])),
                 dict(shape='g5_b024', ell_um=0, **gap(PWE / 'g5_at0/spectra.csv')),
                 dict(shape='g5_b024', ell_um=5, **gap(ROOT / cases[5]['spectrum']))]
    sources[g0at5['spectrum']] = g0at5['sha256']
    changes = []
    for ell in [0, 5]:
        a = next(r for r in endpoints if r['ell_um'] == ell and r['shape'] == 'g0_b024')
        b = next(r for r in endpoints if r['ell_um'] == ell and r['shape'] == 'g5_b024')
        changes.append(dict(ell_um=ell, lower_percent=100 * (b['L'] / a['L'] - 1), upper_percent=100 * (b['U'] / a['U'] - 1), gap_initial=a['J'], gap_replaced=b['J']))
    gates = []
    for case in cases.values():
        for start in case['starts']:
            m = load(ROOT / start['manifest'])
            for g in m['gradient_gates']:
                gates.extend(r['relative_error'] for r in g['records'])
    result = dict(status='PASS_CURRENT_B024_RESULT_BINDING', revision='v47', bound=.24, cases=summary['cases'],
                  primary_table=primary, transfer=transfer, energies=energies, endpoint_comparison=endpoints,
                  endpoint_changes=changes, max_gradient_check_error=max(gates), solid_analysis=solid,
                  source_sha256=sources, script_sha256=sha(Path(__file__)),
                  all_current_designs_b024=True, original_narrow_bound_results_not_used_as_current_optimization_results=True,
                  original_regular_polygon_controls_reused=True, human_verified=False)
    (OUT / 'current_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for stem, data in [('primary_table', primary), ('transfer', transfer), ('energies', energies), ('endpoint_comparison', endpoints), ('endpoint_changes', changes)]:
        with (OUT / f'{stem}.csv').open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(json.dumps(dict(status=result['status'], primary=primary, endpoints=changes), ensure_ascii=False))


if __name__ == '__main__':
    main()
