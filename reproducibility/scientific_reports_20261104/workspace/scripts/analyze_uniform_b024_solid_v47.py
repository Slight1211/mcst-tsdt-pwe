"""Pair fresh b=.24 solid fields with matching N21 plate fields and exact controls."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from matplotlib.path import Path as PolygonPath
from threadpoolctl import threadpool_limits
from analyze_v01 import reconstruct, mac, read_fields

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/uniform_b024_v47'
NEW = ROOT / '05_validation/UNIFORM_B024_SOLID_V47_20261103'
PWE = ROOT / '05_validation/UNIFORM_B024_PWE_V47_20261103'
load = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(p, records):
    if records:
        with p.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)


def assess(folder, ell, shape, fields_folder, polygon, sources):
    meta = load(folder / 'manifest.json')
    assert meta['status'] == 'COMPLETED_RAW_EXPORT'
    assert sha(folder / 'ValidateShapeMcst.java') == meta['source_sha256']
    assert float(meta['settings']['COMSOL_MCST_LENGTH_UM']) == ell
    assert float(meta['settings']['COMSOL_LATTICE_UM']) == 75 and float(meta['settings']['COMSOL_THICKNESS_UM']) == 6
    raw = rows(folder / 'spectra.csv')
    sources[str((folder / 'manifest.json').relative_to(ROOT))] = sha(folder / 'manifest.json')
    sources[str((folder / 'spectra.csv').relative_to(ROOT))] = sha(folder / 'spectra.csv')
    records = []
    for pi in sorted({int(r['point_index']) for r in raw}):
        rr = [r for r in raw if int(r['point_index']) == pi]
        px, py = float(rr[0]['kx_pi_over_a']), float(rr[0]['ky_pi_over_a'])
        tag = 'Gamma' if px == 0 else 'X' if py == 0 else 'M'
        selected = {}
        for threshold in [.7, .8]:
            keep = sorted([r for r in rr if float(r['frequency_mhz']) > .001 and float(r['vertical_fraction']) > threshold], key=lambda r: float(r['frequency_mhz']))
            selected[threshold] = [None] + keep if tag == 'Gamma' else keep
            assert len(selected[threshold]) >= 6
        assert all(selected[.7][b]['mode_index'] == selected[.8][b]['mode_index'] for b in [3, 4])
        fieldfile = folder / f'fields_{tag}.csv'
        coords, weights, solid_fields, rho = read_fields(fieldfile)
        expected = np.where(PolygonPath(polygon).contains_points(coords[:, :2] / 75e-6), 7780., 1180.)
        assert np.allclose(rho, expected, rtol=1e-10, atol=1e-8)
        pwepath = fields_folder / (f'fields_{tag}.npz' if shape == 'final' else f'initial_{tag}.npz')
        sources[str(pwepath.relative_to(ROOT))] = sha(pwepath)
        sources[str(fieldfile.relative_to(ROOT))] = sha(fieldfile)
        with np.load(pwepath, allow_pickle=False) as p:
            physical = reconstruct(coords, p['vectors'][:, :6], p['ij'], p['k'])
            frequencies = p['omega'][:6] * np.sqrt(4.35e9 / 1180) / (75e-6 * 2 * np.pi * 1e6)
        matrix = mac(physical[:, 3:5], solid_fields, weights)
        for b in [3, 4]:
            row = selected[.8][b]
            mode = int(row['mode_index']) - 1
            freq = float(row['frequency_mhz'])
            imag = abs(float(row['imag_frequency_mhz'])) / freq
            assert imag < 1e-7
            cluster = [int(r['mode_index']) - 1 for r in rr if abs(float(r['frequency_mhz']) / freq - 1) < .001 and float(r['vertical_fraction']) > .8]
            A, B = physical[:, b], solid_fields[:, cluster]
            gram = B.conj().T @ (weights[:, None] * B)
            cross = B.conj().T @ (weights * A)
            projection = float(np.real(cross.conj() @ np.linalg.pinv(gram, rcond=1e-12) @ cross) / np.sum(weights * abs(A) ** 2))
            records.append(dict(ell_um=ell, shape=shape, case=folder.relative_to(ROOT).as_posix(),
                                mesh=meta['mesh_level'], layers=meta['layers'], eigen_count=meta['eigen_count'],
                                point=tag, band=b + 1, frequency_3d_MHz=freq, frequency_pwe_N21_MHz=float(frequencies[b]),
                                pwe_N21_minus_solid_percent=float(100 * (frequencies[b] / freq - 1)),
                                raw_mode=mode + 1, vertical_fraction=float(row['vertical_fraction']),
                                threshold_0p7_0p8_same=True, relative_imaginary=imag,
                                individual_MAC=float(matrix[b - 3, mode]), subspace_projection=projection,
                                cluster_modes=';'.join(str(j + 1) for j in cluster),
                                raw_spectrum_sha256=sha(folder / 'spectra.csv'), raw_field_sha256=sha(fieldfile)))
        print('AUDITED', ell, shape, folder.name, tag, flush=True)
    return records


def main():
    selected = load(OUT / 'summary.json')['cases']
    sources, all_rows, final_manifests = {}, [], []
    initial_source = ROOT / '02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1/initial_vertices.csv'
    initial = np.loadtxt(initial_source, delimiter=',', skiprows=1)
    with threadpool_limits(2):
        for ell in [1, 2]:
            if not (NEW / f'ell_{ell}').exists() or not (PWE / f'g{ell}_N21_own/fields_M.npz').exists():
                continue
            case = next(c for c in selected if c['ell_um'] == ell)
            polygon = np.loadtxt(ROOT / case['vertices'], delimiter=',', skiprows=1)
            for folder in sorted((NEW / f'ell_{ell}').iterdir()):
                mp = folder / 'manifest.json'
                if not folder.is_dir() or not mp.exists() or load(mp)['status'] != 'COMPLETED_RAW_EXPORT':
                    continue
                meta = load(mp)
                assert meta['vertices_sha256'] == case['vertices_sha256']
                final_manifests.append(meta)
                all_rows += assess(folder, ell, 'final', PWE / f'g{ell}_N21_own', polygon, sources)
            oldroot = ROOT / ('05_validation/V01_20261030' if ell == 1 else '05_validation/V01_ell2_20261031_v8')
            names = ['initial_M2L4_n24_GXM', 'initial_M3L4_n24_GXM', 'initial_M2L4_n32_X']
            if ell == 2:
                names.append('initial_M2L6_n24_GX')
            for name in names:
                folder = oldroot / name
                if name == 'initial_M3L4_n24_GXM' and (oldroot / (name + '_v2')).exists():
                    folder = oldroot / (name + '_v2')
                assert load(folder / 'manifest.json')['status'] == 'COMPLETED_RAW_EXPORT'
                all_rows += assess(folder, ell, 'initial', oldroot / 'pwe_N21_fields', initial, sources)
    assert all_rows
    write(OUT / 'solid_point_comparison.csv', all_rows)
    gaps, gains, checks = [], [], []
    for ell in [1, 2]:
        for shape in ['initial', 'final']:
            for mesh, layers, count in [(2, 4, 24), (3, 4, 24), (2, 4, 32), (2, 6, 24)]:
                rr = [r for r in all_rows if (r['ell_um'], r['shape'], r['mesh'], r['layers'], r['eigen_count']) == (ell, shape, mesh, layers, count)]
                if not rr:
                    continue
                L = max(r['frequency_3d_MHz'] for r in rr if r['band'] == 4)
                U = min(r['frequency_3d_MHz'] for r in rr if r['band'] == 5)
                points = ';'.join(sorted({r['point'] for r in rr}))
                gaps.append(dict(ell_um=ell, shape=shape, mesh=mesh, layers=layers, eigen_count=count, points=points,
                                 L_MHz=L, U_MHz=U, J_percent=200 * (U - L) / (U + L)))
        for mesh, layers, count in [(2, 4, 24), (3, 4, 24), (2, 4, 32), (2, 6, 24)]:
            rr = [r for r in gaps if (r['ell_um'], r['mesh'], r['layers'], r['eigen_count']) == (ell, mesh, layers, count)]
            if {r['shape'] for r in rr} == {'initial', 'final'} and len({r['points'] for r in rr}) == 1:
                a, b = next(r for r in rr if r['shape'] == 'initial'), next(r for r in rr if r['shape'] == 'final')
                gains.append(dict(ell_um=ell, mesh=mesh, layers=layers, eigen_count=count, points=a['points'], initial_J=a['J_percent'], final_J=b['J_percent'], gain_pp=b['J_percent'] - a['J_percent']))
        for row in [r for r in all_rows if r['ell_um'] == ell and r['shape'] == 'final' and (r['mesh'], r['layers'], r['eigen_count']) != (2, 4, 24)]:
            prior = next(r for r in all_rows if (r['ell_um'], r['shape'], r['mesh'], r['layers'], r['eigen_count'], r['point'], r['band']) == (ell, 'final', 2, 4, 24, row['point'], row['band']))
            checks.append(dict(ell_um=ell, mesh=row['mesh'], layers=row['layers'], eigen_count=row['eigen_count'], point=row['point'], band=row['band'], absolute_change_percent=abs(100 * (row['frequency_3d_MHz'] / prior['frequency_3d_MHz'] - 1))))
    write(OUT / 'solid_gaps.csv', gaps)
    write(OUT / 'solid_gains.csv', gains)
    write(OUT / 'solid_mesh_count_changes.csv', checks)
    complete = all(len([r for r in all_rows if r['ell_um'] == ell and r['shape'] == 'final']) == (16 if ell == 1 else 18) for ell in [1, 2])
    projection = min(r['subspace_projection'] for r in all_rows)
    assert projection > .95
    result = dict(status='PASS_NEW_B024_SELECTED_POINT_SOLID_CHECKS' if complete else 'PARTIAL_COMPLETED_SOLID_CASES',
                  selected_geometry_sources={str(c['ell_um']): c['vertices_sha256'] for c in selected if c['ell_um'] in [1, 2]},
                  minimum_subspace_projection=projection, minimum_individual_MAC=min(r['individual_MAC'] for r in all_rows),
                  baseline_gains=[r for r in gains if (r['mesh'], r['layers'], r['eigen_count']) == (2, 4, 24)],
                  source_sha256=sources, script_sha256=sha(Path(__file__)),
                  new_final_solid_cases=sum(not m.get('reused_archived_export', False) for m in final_manifests),
                  reused_final_solid_cases=sum(m.get('reused_archived_export', False) for m in final_manifests),
                  new_solid_wavevector_solves=sum({'GXM':3,'GX':2,'X':1}[m['points']] for m in final_manifests if not m.get('reused_archived_export',False)),
                  original_initial_controls_reused=True, full_BZ_gap_certified=False, human_verified=False)
    (OUT / 'solid_analysis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'minimum_subspace_projection', 'baseline_gains']}), flush=True)


if __name__ == '__main__':
    main()
