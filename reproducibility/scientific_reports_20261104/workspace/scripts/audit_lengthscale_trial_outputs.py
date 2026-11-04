"""Independently check saved ell10 results; never run an eigensolver.

Run after the frozen scan and both local searches have finished. A RUNNING
manifest is reported as pending, never promoted to a successful optimum.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
AREA = math.pi * 0.3**2
F_MHZ_PER_OMEGA = math.sqrt(4.35e9 / 1180) / (2 * math.pi * 75e-6 * 1e6)


def read_json(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def coord(r):
    return (float(r['kx_a']), float(r['ky_a']))


def key(k):
    return tuple(round(x, 12) for x in k)


def spectrum(p, grid):
    data = rows(p)
    expected = {(i, j) for i in range(grid + 1) for j in range(i + 1)}
    observed = {(round(coord(r)[0] * grid / math.pi), round(coord(r)[1] * grid / math.pi)) for r in data}
    assert len(data) == len(expected) and observed == expected, (p, 'grid')
    assert len({key(coord(r)) for r in data}) == len(data), (p, 'duplicate wavevectors')
    for r in data:
        f = [float(r[f'Omega{i}']) for i in range(1, 7)]
        assert all(math.isfinite(v) for v in f), (p, 'nonfinite')
        assert all(a <= b + 1e-9 for a, b in zip(f, f[1:])), (p, 'unordered frequencies')
        assert all(x > 0 for x in (f[1:] if key(coord(r)) == (0., 0.) else f)), (p, 'nonpositive nonrigid frequency')
        k = coord(r)
        assert all(abs(x * grid / math.pi - round(x * grid / math.pi)) < 1e-10 for x in k), (p, 'off-grid')
    return data


def gap(data):
    lo = max(data, key=lambda r: float(r['Omega4']))
    up = min(data, key=lambda r: float(r['Omega5']))
    L, U = float(lo['Omega4']), float(up['Omega5'])
    return dict(L=L, U=U, J=200*(U-L)/(U+L), gap_open=bool(U > L),
                lower_k_a=list(coord(lo)), upper_k_a=list(coord(up)),
                L_MHz=L*F_MHZ_PER_OMEGA, U_MHz=U*F_MHZ_PER_OMEGA)


def compare_spectra(small, large):
    lookup = {key(coord(r)): r for r in large}
    drift = 0.
    for row in small:
        other = lookup[key(coord(row))]
        for band in range(1, 7):
            if band == 1 and key(coord(row)) == (0., 0.):
                continue
            a, b = float(row[f'Omega{band}']), float(other[f'Omega{band}'])
            drift = max(drift, abs(a/b - 1))
    return drift


def polygon_area(p):
    return .5*sum(a[0]*b[1]-a[1]*b[0] for a, b in zip(p, p[1:]+p[:1]))


def reconstructed_polygon(y):
    # Independent explicit Helmert construction, without importing production code.
    sector = []
    for node in range(7):
        z = 0.
        for c in range(6):
            i = c+1
            v = (1. if node < i else -float(i) if node == i else 0.) / math.sqrt(i*(i+1))
            z += v*y[c]
        sector.append(math.exp(z))
    raw = []
    for j in range(48):
        radius = sector[min(j % 12, 12-j % 12)]
        angle = j*math.pi/24
        raw.append([radius*math.cos(angle), radius*math.sin(angle)])
    scale = math.sqrt(AREA/polygon_area(raw))
    return [[x*scale, y*scale] for x, y in raw]


def check_search(name, base, frozen):
    folder = base/name
    manifest = read_json(folder/'manifest.json')
    if manifest['status'] != 'PASS_SAMPLED_N15_LOCAL_SQP':
        return dict(status='PENDING_OR_FAILED', manifest_status=manifest['status'])
    checks = {}
    checks['source_hashes'] = all(sha(Path(p)) == digest for p, digest in manifest['source_sha256'].items())
    checks['output_hashes'] = all(sha(folder/p) == digest for p, digest in manifest['output_sha256'].items())
    checks['parameters'] = all(manifest[k] == v for k, v in [('N',15), ('a_um',75.), ('h_um',6.), ('ell_epoxy_um',10.), ('ell_steel_um',0.), ('threads',2)])
    final_round = len(manifest['rounds'])
    final45 = spectrum(folder/f'round{final_round}_ibz45.csv', 8)
    final153 = spectrum(folder/'final_ibz153.csv', 16)
    initial45 = spectrum(folder/'initial_ibz45.csv', 8)
    initial_reference = spectrum(frozen/'initial_ell10_N15_ibz45.csv', 8)
    computed = {k:gap(v) for k,v in [('initial_ibz45',initial45), ('final_ibz45',final45), ('final_ibz153',final153)]}
    for label, result in computed.items():
        checks[label+'_manifest_arithmetic'] = all(abs(result[v]-manifest[label][v]) < 1e-10 for v in ['L','U','J'])
    nested_drift = compare_spectra(final45, final153)
    baseline_drift = compare_spectra(initial45, initial_reference)
    checks['nested45_153_point_frequencies'] = nested_drift < 1e-7
    checks['common_initial_baseline_frequencies'] = baseline_drift < 1e-7
    checks['gradient_gates'] = len(manifest['gradient_gates']) == 2 and all(g['status'] == 'PASS' and max(r['relative_error'] for r in g['records']) < 1e-4 for g in manifest['gradient_gates'])
    last = manifest['rounds'][-1]
    checks['SLSQP_success_and_feasibility'] = bool(last['optimizer']['success']) and last['optimizer']['constraint_violation'] < 1e-6 and last['full_constraint_violation'] < 1e-6
    checks['final_design_matches_optimizer'] = max(abs(a-b) for a,b in zip(manifest['design'],last['optimizer']['design'])) < 1e-14
    vertices = rows(folder/'final_vertices.csv')
    polygon = [[float(r['x_over_a']),float(r['y_over_a'])] for r in vertices]
    area = polygon_area(polygon)
    reconstructed = reconstructed_polygon(manifest['design'])
    geometry_drift = max(abs(a-b) for p,q in zip(polygon,reconstructed) for a,b in zip(p,q))
    checks['48_vertices'] = len(polygon) == 48
    checks['fixed_area'] = abs(area-AREA) < 1e-13 and abs(area-manifest['area_fraction']) < 1e-13
    checks['six_variable_bounds'] = len(manifest['design']) == 6 and max(abs(x) for x in manifest['design']) <= .16+1e-10
    checks['cell_clearance'] = max(abs(x) for p in polygon for x in p) < .49
    checks['independent_geometry_reconstruction'] = geometry_drift < 1e-13
    trajectory = read_json(folder/'trajectory.json')
    if name == 'warmstart':
        seed = ROOT/manifest['seed_source']
        seed_y = read_json(seed)['design']
        seed_reference = spectrum(frozen/'ell1_design_ell10_N15_ibz45.csv', 8)
        seed_points = [r for r in seed_reference if key(coord(r)) in [(0.,0.),key((math.pi,0.)),key((math.pi,math.pi))]]
        seed_gap = gap(seed_points)
        checks['warm_seed_source_hash'] = sha(seed) == manifest['seed_source_sha256']
        checks['warm_seed_source_design'] = manifest['seed_design'] == seed_y == trajectory[0]['design']
        checks['warm_seed_event0_spectrum'] = all(abs(seed_gap[v]-trajectory[0][v]) < 1e-10 for v in ['L','U','J'])
    else:
        checks['circle_seed_event0'] = max(abs(x) for x in trajectory[0]['design']) == 0.
    checks['dense_drift_manifest_arithmetic'] = abs(computed['final_ibz153']['J']-computed['final_ibz45']['J']-manifest['dense_gap_drift_pp']) < 1e-10
    checks['gain_manifest_arithmetic'] = abs(computed['final_ibz45']['J']-computed['initial_ibz45']['J']-manifest['gain_from_initial_pp']) < 1e-10
    return dict(status='PASS_SAVED_DATA_AUDIT' if all(checks.values()) else 'FAIL', checks=checks,
        issues=[k for k,v in checks.items() if not v], independent_results=computed,
        final45_153_max_relative_frequency_drift=nested_drift,
        initial_vs_independent_frozen_max_relative_frequency_drift=baseline_drift,
        independent_area_fraction=area,independent_geometry_coordinate_drift=geometry_drift,
        optimization_seed='archived ell1 terminal geometry' if name=='warmstart' else 'regular48gon',
        initial_ibz45_semantics='Common regular48gon baseline, not necessarily optimization seed.',
        accepted_iterations=manifest['accepted_iterations'],rounds=len(manifest['rounds']),
        optimization_wavevector_points=45,denser_check_wavevector_points=153,
        dense_gap_drift_pp=computed['final_ibz153']['J']-computed['final_ibz45']['J'],
        denser_sampling_finds_tighter_separation=computed['final_ibz153']['J']<computed['final_ibz45']['J']-1e-8,
        stationarity_on_153_points_not_established=True,
        audit_pass_means_saved_data_consistency_not_zero_truncation_or_sampling_error=True,
        projected_lagrangian_gradient_inf=last['optimizer']['projected_lagrangian_gradient_inf'],
        complementarity_inf=last['optimizer']['complementarity_inf'],
        max_gradient_gate_relative_error=max(r['relative_error'] for g in manifest['gradient_gates'] for r in g['records']),
        output_manifest_sha256=sha(folder/'manifest.json'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=ROOT/'05_validation/ELL10_exploration_20261031')
    parser.add_argument('--out',type=Path,default=ROOT/'03_review/ell10_exploration_20261031/terminal_saved_data_audit.json')
    args=parser.parse_args()
    base=args.base.resolve(); frozen=base/'frozen'
    results={name:check_search(name,base,frozen) for name in ['optimization','warmstart']}
    complete=all(v['status']=='PASS_SAVED_DATA_AUDIT' for v in results.values())
    report=dict(status='PASS_BOTH_SAVED_SEARCHES' if complete else 'PENDING_OR_FAILED', searches=results,
        no_eigensolves_in_this_audit=True,not_a_physical_model_validation=True,
        residual_scope='Search CSVs do not preserve eigenvectors, so residuals cannot be recomputed here. The patched search enforces target-band residuals; separate frozen and terminal matrix checks provide independent diagnostics.',
        no_global_optimality_or_full_zone_3D_claim=True,human_verified=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'], searches={k:v['status'] for k,v in results.items()}),indent=2))
    raise SystemExit(0 if complete else 2)


if __name__=='__main__':
    main()
