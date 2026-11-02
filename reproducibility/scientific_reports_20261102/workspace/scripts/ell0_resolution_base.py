"""Order-explicit ell=0 helpers; retain the archived SLSQP implementation."""
from pathlib import Path
import csv
import numpy as np
import run_lengthscale_optimization as legacy
from ell0_resolution_sensitivity import shape_point, sampled_gap

AREA = legacy.AREA
boundary = legacy.boundary
signed_area = legacy.signed_area
fourier = legacy.fourier
prepare = legacy.prepare
ibz = legacy.ibz
save_json = legacy.save_json
optimize_round = legacy.optimize_round
LoggedEvaluator = legacy.LoggedEvaluator
# Evaluator resolves this function in the imported pilot module at evaluation.
legacy.pilot.shape_point = shape_point
ELL_UM = 0.
ELL_OVER_H = 0.


def spectrum(y, kpoints, out, stem, *, N):
    assert ELL_UM == 0 and ELL_OVER_H == 0
    F = fourier(y, N)
    cache = prepare(N, .08, theory='TSDT', indicator_matrix=F)
    values = []
    with (Path(out) / f'{stem}.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['point', 'kx_a', 'ky_a'] + [f'Omega{i}' for i in range(1, 7)] +
                        ['nonrigid_first6_residual', 'mass_orthogonality'])
        for i, k in enumerate(kpoints):
            f = shape_point(y, N, k, h_over_a=.08, ell_over_h=0., gradient=False,
                            geometry=(F, None), prepared=cache)
            values.append(f)
            checks = shape_point.last_checks
            assert checks['N'] == N
            writer.writerow([i, *k, *f, checks['nonrigid_first6_residual'], checks['mass_orthogonality']])
            handle.flush()
            print(stem, i + 1, '/', len(kpoints), 'N', N, flush=True)
    return np.asarray(values)


def gradient_gate(y, name, *, N):
    assert ELL_UM == 0 and ELL_OVER_H == 0
    k = np.array([2.7, .63])
    direction = np.arange(1., 7.)
    direction /= np.linalg.norm(direction)
    f, g, _ = shape_point(y, N, k, h_over_a=.08, ell_over_h=0.)
    records = []
    for step in [1e-4, 5e-5]:
        fp = shape_point(y + step * direction, N, k, h_over_a=.08, ell_over_h=0., gradient=False)
        fm = shape_point(y - step * direction, N, k, h_over_a=.08, ell_over_h=0., gradient=False)
        fd = (fp - fm) / (2 * step)
        exact = (g @ direction)[3:5]
        error = np.linalg.norm(exact - fd[3:5]) / max(np.linalg.norm(fd[3:5]), 1e-12)
        records.append(dict(step=step, analytic=exact.tolist(), central=fd[3:5].tolist(),
                            relative_error=float(error)))
    assert max(r['relative_error'] for r in records) < 1e-4, 'Order-specific gradient gate failed'
    print('GRADIENT_GATE', name, 'N', N, 'PASS', records, flush=True)
    return dict(name=name, N=N, k=k.tolist(), direction=direction.tolist(), records=records, status='PASS')
