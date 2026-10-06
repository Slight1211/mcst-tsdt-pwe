"""Check the actual inverse-factorized classical stiffness thickness derivative.

The dense convolution is used only at N=2 as an independent algebraic test.
Production spectra keep the existing grouped-compliance implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/manuscript_revision_20260925"
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "msse_20260919/python")]
from inverse_circle_bending import classical_inverse, generalized_D, prepare, strain  # noqa: E402


def local_derivative(material: str, h: float) -> tuple[np.ndarray, np.ndarray]:
    E, nu = ((210.6 / 4.35, .3) if material == "steel"
             else (1., 4.35 / (2 * 1.59) - 1))
    h1, h2 = .07, .13
    d1 = generalized_D(E, nu, h1, "TSDT")
    d2 = generalized_D(E, nu, h2, "TSDT")
    q = np.linalg.solve(np.array([[h1, h1**3], [h2, h2**3]]),
                        np.stack((d1, d2)).reshape(2, -1))
    first, third = q.reshape(2, *d1.shape)
    d = h * first + h**3 * third
    derivative = h * first + 3 * h**3 * third
    assert np.linalg.norm(d-generalized_D(E, nu, h, "TSDT"))/np.linalg.norm(d) < 1e-12
    return d, derivative


def independent_k_and_derivative(cache: dict, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(cache["G"])
    h = cache["h"]
    da, dap = local_derivative("steel", h)
    db, dbp = local_derivative("epoxy", h)
    ia, ib = np.linalg.inv(da), np.linalg.inv(db)
    iap, ibp = -ia @ dap @ ia, -ib @ dbp @ ib
    f = cache["F"]
    c = np.kron(np.eye(n), ib) + np.kron(f, ia-ib)
    cp = np.kron(np.eye(n), ibp) + np.kron(f, iap-ibp)
    # B is h-independent in the original unscaled DOFs at fixed physical k*a.
    local_b = strain(cache["G"] + k, "TSDT")
    b = np.zeros((8*n, 3*n))
    for i in range(n):
        b[8*i:8*i+8, 3*i:3*i+3] = local_b[i]
    r = np.linalg.solve(c, b)
    return b.T @ r, -r.T @ cp @ r


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    h0 = .08
    N = 2
    k = np.array([np.pi, .4])
    with threadpool_limits(limits=1):
        cache = prepare(N, h0, .3, "TSDT")
        k0, kp = independent_k_and_derivative(cache, k)
        actual = classical_inverse(cache, k)
        matrix_error = float(np.linalg.norm(k0-actual)/np.linalg.norm(actual))
        assert matrix_error < 1e-10, matrix_error
        checks = []
        for step in (.001, .002, .005):
            hi, lo = h0*np.exp(step), h0*np.exp(-step)
            hi_k = classical_inverse(prepare(N, hi, .3, "TSDT"), k)
            lo_k = classical_inverse(prepare(N, lo, .3, "TSDT"), k)
            fd = (hi_k-lo_k)/(2*step)
            error = float(np.linalg.norm(kp-fd)/np.linalg.norm(kp))
            checks.append({"log_step": step, "relative_matrix_derivative_error": error})
            assert error < 2e-4, (step, error)
    result = {
        "status": "PASS_N2_INVERSE_CLASSICAL_MATRIX_DERIVATIVE",
        "N": N, "h_over_a": h0, "kx_a": float(k[0]), "ky_a": float(k[1]),
        "inverse_stiffness_matrix_error": matrix_error, "steps": checks,
        "formula": "d(C^-1)/dlnh=-C^-1(dC/dlnh)C^-1; B h-independent in original DOFs",
        "scope": "N2 classical inverse term only; not inverse modal energy or full-band sensitivity",
        "NOT_RUN": ["inverse modal sensitivity field tracking", "inverse energy-map consistency"],
    }
    (OUT / "inverse_derivative_check.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
