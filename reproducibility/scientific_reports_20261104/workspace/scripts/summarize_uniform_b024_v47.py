"""Audit and select completed b=.24 searches without changing any source results."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / '05_validation/UNIFORM_B024_V47_20261103'
OUT = ROOT / 'output/uniform_b024_v47'
load = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def read_case(mp):
    m = load(mp)
    assert m['status'].startswith('PASS') and m['N'] == 15
    assert m['shape_bounds'] == [-.24, .24]
    for name, digest in m['output_sha256'].items():
        assert sha(mp.parent / name) == digest
    assert all(g['status'] == 'PASS' for g in m['gradient_gates'])
    terminal = m['rounds'][-1]
    opt = terminal['optimizer']
    assert opt['success'] and opt['constraint_violation'] < 1e-6 and terminal['full_constraint_violation'] < 1e-6
    assert opt['projected_lagrangian_gradient_inf'] < 1e-3 and opt['complementarity_inf'] < 1e-4
    with (mp.parent / 'final_ibz153.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 153
    frequencies = np.array([[float(r[f'Omega{i}']) for i in range(1, 7)] for r in rows])
    assert np.isfinite(frequencies).all() and frequencies.min() >= 0
    assert np.diff(frequencies, axis=1).min() > -1e-10
    il, iu = np.argmax(frequencies[:, 3]), np.argmin(frequencies[:, 4])
    lower, upper = float(frequencies[il, 3]), float(frequencies[iu, 4])
    gap = 200 * (upper - lower) / (upper + lower)
    assert all(abs(m['final_ibz153'][n] - v) < 1e-10 for n, v in [('L', lower), ('U', upper), ('J', gap)])
    vertices = np.loadtxt(mp.parent / 'final_vertices.csv', delimiter=',', skiprows=1)
    assert vertices.shape == (48, 2) and abs(vertices).max() < .49
    area = .5 * np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) - vertices[:, 1] * np.roll(vertices[:, 0], -1))
    assert abs(area - .09 * np.pi) < 1e-12
    assert np.min(np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)) >= 1e-5
    y = np.asarray(m['design'])
    assert abs(y).max() <= .24 + 1e-10
    r = np.linalg.norm(vertices, axis=1)
    return dict(ell_um=m['ell_epoxy_um'], bound=.24, N=15, points=153, J153_percent=gap, L=lower, U=upper,
                kL=[float(rows[il][k]) for k in ['kx_a', 'ky_a']], kU=[float(rows[iu][k]) for k in ['kx_a', 'ky_a']],
                manifest=mp.relative_to(ROOT).as_posix(), manifest_sha256=sha(mp),
                vertices=(mp.parent / 'final_vertices.csv').relative_to(ROOT).as_posix(),
                vertices_sha256=sha(mp.parent / 'final_vertices.csv'),
                spectrum=(mp.parent / 'final_ibz153.csv').relative_to(ROOT).as_posix(),
                spectrum_sha256=sha(mp.parent / 'final_ibz153.csv'),
                design=y.tolist(), max_abs_design=float(abs(y).max()),
                bound_active_variables=int(np.sum(abs(y) >= .24 - 1e-7)), radial_ratio=float(r.max() / r.min()),
                projected_stationarity=opt['projected_lagrangian_gradient_inf'], complementarity=opt['complementarity_inf'],
                max_constraint_violation=max(opt['constraint_violation'], terminal['full_constraint_violation']),
                elapsed_seconds=m['elapsed_seconds'], iterations=m['accepted_iterations'])


def main():
    state = load(STUDY / 'manifest.json')
    OUT.mkdir(exist_ok=True)
    cases = []
    for ell in [0., .5, 1., 2., 5.]:
        jobs = state['existing_low_ell_b024_jobs'] if ell < 1 else state['new_jobs']
        jobs = [j for j in jobs if j['ell_um'] == ell and j['bound'] == .24]
        if len(jobs) != 2 or any(j['status'] != 'PASS' for j in jobs):
            continue
        runs = [read_case(ROOT / j['final_manifest']) for j in jobs]
        selected = dict(max(runs, key=lambda r: r['J153_percent']))
        selected['two_start_spread_pp'] = max(r['J153_percent'] for r in runs) - min(r['J153_percent'] for r in runs)
        selected['starts'] = [dict(seed=j['seed'], manifest=r['manifest'], manifest_sha256=r['manifest_sha256'], J153_percent=r['J153_percent'], iterations=r['iterations']) for j, r in zip(jobs, runs)]
        # A changed boundary never inherits another boundary's solid validation.
        if ell >= 1:
            seedmp = next(ROOT / j['seed_manifest'] for j in jobs if j.get('seed_manifest'))
            oldvertices = np.loadtxt(seedmp.parent / 'final_vertices.csv', delimiter=',', skiprows=1)
            vertices = np.loadtxt(ROOT / selected['vertices'], delimiter=',', skiprows=1)
            selected['previous_geometry_bitwise_identical'] = np.array_equal(vertices, oldvertices)
            selected['maximum_vertex_change_from_previous_um'] = float(75 * np.max(np.linalg.norm(vertices - oldvertices, axis=1)))
        cases.append(selected)
    summary = dict(status='PASS_ALL_FIVE_B024_SEARCHES' if len(cases) == 5 else 'PARTIAL_COMPLETED_CASES',
                   revision='v47', N=15, shape_bound=.24, cases=cases,
                   new_solid_validation='PENDING_EXACT_SELECTED_GEOMETRIES',
                   source_manifest_sha256=sha(STUDY / 'manifest.json'), script_sha256=sha(Path(__file__)),
                   no_global_optimality_claim=True, human_verified=False)
    target = OUT / 'summary.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(target)
    columns = ['ell_um', 'bound', 'N', 'points', 'J153_percent', 'L', 'U', 'max_abs_design', 'bound_active_variables', 'radial_ratio', 'two_start_spread_pp', 'manifest', 'manifest_sha256', 'vertices', 'vertices_sha256', 'spectrum', 'spectrum_sha256']
    with (OUT / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(cases)
    print(json.dumps(dict(status=summary['status'], cases=[{k: c.get(k) for k in ['ell_um', 'J153_percent', 'bound_active_variables', 'maximum_vertex_change_from_previous_um']} for c in cases]), ensure_ascii=False))


if __name__ == '__main__':
    main()
