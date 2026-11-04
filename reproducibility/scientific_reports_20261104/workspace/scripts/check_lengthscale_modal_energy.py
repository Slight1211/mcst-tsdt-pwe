"""Fresh active-edge MCST modal energies at five saved N15 geometries.

Only the active points from each existing full153 spectrum are solved. This
does not optimize, scan the zone, calculate a dense reference, or use COMSOL.
The five shapes differ: reported sensitivities hold their own geometry fixed.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / '04_code/research/wavemotion_surface_20260920'
CORE = CODE.parent / 'msse_20260919/python'
sys.path[:0] = [str(CODE / 'src'), str(CORE)]
import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import eigsh, LinearOperator
from threadpoolctl import threadpool_info, threadpool_limits
from node_shape_bending import AREA, boundary, fourier, signed_area
from inverse_circle_bending import prepare, classical_inverse
from real_circle_bending import assemble_real

CASES = [
    (2., 'ell2_restart',
     '05_validation/V07_ell2_reoptimization_20261030/restart/manifest.json'),
    (5., 'ell5_warm',
     '05_validation/GAP_CLOSURE_SCAN_20261031/ell_5/warm/manifest.json'),
    (0., 'ell0_multistart_lhs05',
     '05_validation/ELL0_MULTISTART_20261101/lhs_05/manifest.json'),
    (.5, 'ell0p5_regular',
     '05_validation/LOW_ELL_SHAPES_20261101/ell_0p5/regular/manifest.json'),
    (1., 'ell1_saved_terminal',
     '02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/'
     'node_gradient_a75_h6_N15_20261004_v1/manifest.json'),
]
TIE_TOL = 1e-8
CHECK_TOL = 1e-7
SIMPLE_EIGENGAP_TOL = 1e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def save(path, data):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def load_cases():
    cases, inputs = [], {}
    for ell, label, name in CASES:
        manifest = ROOT / name
        worker = json.loads(manifest.read_text(encoding='utf-8'))
        assert worker['status'].startswith('PASS'), (name, worker['status'])
        assert worker['N'] == 15 and worker['a_um'] == 75 and worker['h_um'] == 6
        assert worker['ell_epoxy_um'] == ell
        # The old ell2 manifest omits ell_steel_um; construction below explicitly
        # uses the epoxy-only KB and therefore sets steel ell=0 independently.
        assert worker.get('ell_steel_um', 0.) == 0.
        if 'ell_over_h' in worker:
            assert abs(worker['ell_over_h'] - ell / 6) < 1e-14
        spectrum = manifest.parent / 'final_ibz153.csv'
        digest = sha(spectrum)
        if 'output_sha256' in worker:
            assert digest == worker['output_sha256'][spectrum.name]
        with spectrum.open(encoding='utf-8-sig', newline='') as handle:
            records = list(csv.DictReader(handle))
        assert len(records) == 153
        k = np.array([[float(r['kx_a']), float(r['ky_a'])] for r in records])
        expected = np.array([(i, j) for i in range(17) for j in range(i + 1)])
        assert np.max(abs(k * 16 / np.pi - expected)) < 1e-10
        f = np.array([[float(r[f'Omega{b}']) for b in range(1, 7)] for r in records])
        assert np.isfinite(f).all() and np.all(np.diff(f, axis=1) >= -1e-9)
        assert np.all(f[:, 1:] > 0) and np.all(f[1:, 0] > 0)
        L, U = float(np.max(f[:, 3])), float(np.min(f[:, 4]))
        gap = dict(L=L, U=U, J=200 * (U - L) / (U + L))
        assert all(abs(gap[key] - worker['final_ibz153'][key]) < 1e-9 for key in gap)
        lower = list(map(int, np.flatnonzero(abs(f[:, 3] - L) <= TIE_TOL)))
        upper = list(map(int, np.flatnonzero(abs(f[:, 4] - U) <= TIE_TOL)))
        y = np.asarray(worker['design'], float)
        assert y.shape == (6,) and np.isfinite(y).all() and np.max(abs(y)) <= .16 + 1e-10
        vertices = boundary(y)
        area = float(signed_area(vertices))
        assert vertices.shape == (48, 2) and abs(area - AREA) < 1e-13
        point_plan = []
        for index in sorted(set(lower + upper)):
            label_k = ('Gamma' if index == 0 else 'X' if index == 136
                       else 'M' if index == 152 else f'Q_{index}')
            bands = []
            if index in lower:
                bands.append(dict(band=4, role='lower_edge'))
            if index in upper:
                bands.append(dict(band=5, role='upper_edge'))
            point_plan.append(dict(index=index, label=label_k, k_a=k[index].tolist(), bands=bands))
        case = dict(ell_um=ell, shape_label=label, design=y.tolist(),
                    worker_manifest=name, spectrum=rel(spectrum), spectrum_sha256=digest,
                    full153_gap=gap, lower_tied_indices=lower, upper_tied_indices=upper,
                    point_plan=point_plan, area_fraction=area, area_error=area - AREA,
                    max_abs_vertex_over_a=float(np.max(abs(vertices))))
        cases.append((case, y, f))
        inputs.update({rel(p): sha(p) for p in [manifest, spectrum]})
    assert [len(c['point_plan']) for c, _, _ in cases] == [2, 2, 2, 2, 2]
    return cases, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / '03_review/lengthscale_mechanism_20261101')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    cases, inputs = load_cases()
    if args.plan_only:
        print(json.dumps(dict(status='PLAN_ONLY_NO_EIGENSOLVES',
                              cases=[c for c, _, _ in cases], input_sha256=inputs), indent=2))
        return
    out = args.out.resolve()
    assert out.is_relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), CODE / 'src/node_shape_bending.py',
               CODE / 'src/inverse_circle_bending.py', CODE / 'src/real_circle_bending.py',
               CORE / 'msse_core.py', CORE / 'two_phase.py']
    source_hashes = {rel(p): sha(p) for p in sources}
    meta = dict(status='RUNNING', pid=os.getpid(),
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                N=15, a_um=75., h_um=6., h_over_a=.08, ell_steel_um=0.,
                theory='TSDT', threads=2, sparse_eigen_count=12,
                sparse_shift=-.001, sparse_tolerance=1e-10, ncv=60, seed=20261030,
                active_edge_Omega_tolerance=TIE_TOL,
                frequency_and_residual_gate=CHECK_TOL,
                near_degeneracy_relative_lambda_gap_threshold=SIMPLE_EIGENGAP_TOL,
                source_sha256=source_hashes, input_sha256=inputs,
                construction='Kcl=classical_inverse(prepare(N,h/a,indicatorF),k); '
                             'K=Kcl+(ell_um/h_um)^2*KB; M=assemble_real(...)[M]; KB epoxy only',
                energy_convention='quadratic forms q.T Kpart q and elastic energies 0.5*q.T Kpart q',
                sensitivity_identity='R_cs=dln(Omega)/dln(ell) for ell>0, fixed own geometry/k, simple eigenvalue; ell0 undefined',
                distinct_geometry_per_ell=True,
                tie_semantics='All controls within 1e-8 Omega retained; ties do not imply a same-k repeated eigenvalue.',
                scope='Local modal responses only. No derivative of the reoptimized shape/gap curve; '
                      'no spatial localization or interface-curvature mechanism inferred.',
                new_optimization=False, new_full153_scan=False, new_dense_reference=False, new_COMSOL_runs=False,
                environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                 platform=platform.platform()),
                cases=[c for c, _, _ in cases], point_checks=[])
    save(out / 'manifest.json', meta)
    clock = time.perf_counter()
    mode_rows = []
    fields = ['ell_um', 'shape_label', 'source153_index', 'point_label', 'kx_a', 'ky_a',
              'edge_role', 'band', 'Omega_saved', 'Omega_fresh', 'lambda',
              'relative_saved_frequency_difference', 'relative_residual', 'mass_norm',
              'classical_quadratic_form', 'KB_unscaled_quadratic_form', 'MCST_quadratic_form',
              'total_quadratic_form', 'classical_elastic_energy', 'MCST_elastic_energy',
              'total_elastic_energy', 'R_cs', 'dlnOmega_dlnell',
              'nearest_relative_lambda_gap', 'nearest_band_in_12', 'near_degeneracy',
              'single_mode_energy_interpretation', 'energy_decomposition_relative_error',
              'Rayleigh_eigenvalue_relative_error']
    try:
        with threadpool_limits(2):
            meta['threadpool'] = threadpool_info()
            with (out / 'active_modal_energy.csv').open('x', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for case, y, saved in cases:
                    ell = case['ell_um']
                    p = (ell / 6.) ** 2
                    F = fourier(y, 15)
                    cache = prepare(15, .08, theory='TSDT', indicator_matrix=F)
                    for point in case['point_plan']:
                        tick = time.perf_counter()
                        k = np.asarray(point['k_a'])
                        index = point['index']
                        parts = assemble_real(15, k, h=.08, theory='TSDT', indicator_matrix=F)
                        M, KB = parts['M'], parts['KB']
                        del parts
                        Kcl = classical_inverse(cache, k)
                        K = Kcl + p * KB
                        factor = cho_factor(K + .001 * M, lower=True, check_finite=False)
                        inverse = LinearOperator(K.shape,
                            matvec=lambda v: cho_solve(factor, v, check_finite=False), dtype=float)
                        values, vectors = eigsh(K, k=12, M=M, sigma=-.001, OPinv=inverse,
                            tol=1e-10, ncv=60, v0=np.random.default_rng(20261030).normal(size=len(K)))
                        order = np.argsort(values)
                        values, vectors = values[order], vectors[:, order]
                        gamma = np.max(abs(k)) < 1e-14
                        first = 1 if gamma else 0
                        assert np.all(values[first:] > 0)
                        if gamma:
                            assert abs(values[0]) < 1e-7
                        frequencies = np.sqrt(np.maximum(values, 0))
                        Kv, Mv = K @ vectors, M @ vectors
                        residual = np.linalg.norm(Kv - Mv * values, axis=0) / (
                            np.linalg.norm(Kv, axis=0) + abs(values) * np.linalg.norm(Mv, axis=0) + 1e-30)
                        maxres = float(np.max(residual[first:6]))
                        rawdelta = float(np.max(abs(frequencies[first:6] / saved[index, first:6] - 1)))
                        orth = float(np.max(abs(vectors.T @ Mv - np.eye(12))))
                        assert maxres < CHECK_TOL and rawdelta < CHECK_TOL and orth < CHECK_TOL
                        check = dict(ell_um=ell, shape_label=case['shape_label'], **point,
                            max_first6_nonrigid_relative_residual=maxres,
                            max_first6_nonrigid_raw_frequency_relative_difference=rawdelta,
                            mass_orthogonality_max=orth, solve_seconds=time.perf_counter() - tick)
                        for target in point['bands']:
                            b = target['band'] - 1
                            q = vectors[:, b]
                            lam = float(values[b])
                            mass = float(q @ Mv[:, b])
                            cl = float(q @ Kcl @ q)
                            kb = float(q @ KB @ q)
                            mc = p * kb
                            total = float(q @ Kv[:, b])
                            R = mc / total
                            closure = abs(cl + mc - total) / total
                            rayleigh = abs(total / mass / lam - 1)
                            distances = abs(values - lam) / abs(lam)
                            distances[b] = np.inf
                            nearest = int(np.argmin(distances))
                            eigengap = float(distances[nearest])
                            near = eigengap < SIMPLE_EIGENGAP_TOL
                            assert abs(mass - 1) < 1e-7 and closure < 1e-9 and rayleigh < 1e-7
                            assert cl >= -1e-10 * total and kb >= -1e-10 * total
                            assert -1e-10 <= R <= 1 + 1e-10
                            if ell == 0:
                                assert R == 0.
                            row = dict(ell_um=ell, shape_label=case['shape_label'], source153_index=index,
                                point_label=point['label'], kx_a=float(k[0]), ky_a=float(k[1]),
                                edge_role=target['role'], band=b + 1, Omega_saved=float(saved[index, b]),
                                Omega_fresh=float(frequencies[b]), **{'lambda': lam},
                                relative_saved_frequency_difference=float(frequencies[b] / saved[index, b] - 1),
                                relative_residual=float(residual[b]), mass_norm=mass,
                                classical_quadratic_form=cl, KB_unscaled_quadratic_form=kb,
                                MCST_quadratic_form=mc, total_quadratic_form=total,
                                classical_elastic_energy=.5 * cl, MCST_elastic_energy=.5 * mc,
                                total_elastic_energy=.5 * total, R_cs=R,
                                dlnOmega_dlnell=R if ell > 0 and not near else None,
                                nearest_relative_lambda_gap=eigengap, nearest_band_in_12=nearest + 1,
                                near_degeneracy=near, single_mode_energy_interpretation=not near,
                                energy_decomposition_relative_error=closure,
                                Rayleigh_eigenvalue_relative_error=rayleigh)
                            writer.writerow(row)
                            mode_rows.append(row)
                        handle.flush()
                        meta['point_checks'].append(check)
                        meta['active_modes'] = mode_rows
                        save(out / 'manifest.json', meta)
                        print('ENERGY', ell, point['label'],
                              [(r['band'], r['R_cs']) for r in mode_rows
                               if r['ell_um'] == ell and r['source153_index'] == index],
                              'maxres', maxres, 'rawdelta', rawdelta, flush=True)
                        del K, Kcl, KB, M, Kv, Mv, factor, inverse, vectors
                    del F, cache
        source_post = {name: sha(ROOT / name) for name in source_hashes}
        input_post = {name: sha(ROOT / name) for name in inputs}
        assert source_post == source_hashes, 'Construction source changed during solve'
        assert input_post == inputs, 'Saved evidence inputs changed during solve'
        save(out / 'active_modal_energy.json', dict(
            scope=meta['scope'], energy_convention=meta['energy_convention'],
            sensitivity_identity=meta['sensitivity_identity'], modes=mode_rows))
        outputs = [out / 'active_modal_energy.csv', out / 'active_modal_energy.json']
        meta.update(status='PASS_ACTIVE_EDGE_MODAL_ENERGY_N15',
                    source_hash_check='PASS', input_hash_check='PASS',
                    source_sha256_post=source_post, input_sha256_post=input_post,
                    output_sha256={p.name: sha(p) for p in outputs},
                    shape_count=len(cases), point_count=len(meta['point_checks']),
                    active_mode_count=len(mode_rows),
                    max_first6_nonrigid_relative_residual=max(
                        c['max_first6_nonrigid_relative_residual'] for c in meta['point_checks']),
                    max_first6_nonrigid_raw_frequency_relative_difference=max(
                        c['max_first6_nonrigid_raw_frequency_relative_difference'] for c in meta['point_checks']),
                    max_mass_orthogonality_error=max(c['mass_orthogonality_max'] for c in meta['point_checks']),
                    minimum_target_relative_lambda_gap=min(r['nearest_relative_lambda_gap'] for r in mode_rows),
                    near_degenerate_target_count=sum(r['near_degeneracy'] for r in mode_rows),
                    elapsed_seconds=time.perf_counter() - clock,
                    finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        save(out / 'manifest.json', meta)
        print('PASS', json.dumps({k: meta[k] for k in [
            'shape_count', 'point_count', 'active_mode_count',
            'max_first6_nonrigid_relative_residual',
            'max_first6_nonrigid_raw_frequency_relative_difference',
            'minimum_target_relative_lambda_gap', 'near_degenerate_target_count']}), flush=True)
    except BaseException:
        meta.update(status='FAILED_MODAL_ENERGY_CHECK', error=traceback.format_exc(),
                    elapsed_seconds=time.perf_counter() - clock)
        save(out / 'manifest.json', meta)
        raise


if __name__ == '__main__':
    main()
