"""Eigenfrequency and signed-gap shape derivatives for node-based inverse PWE."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.sparse.linalg import eigsh, LinearOperator
from inverse_circle_bending import prepare, strain, classical_inverse
from real_circle_bending import assemble_real
from node_shape_bending import fourier


def shape_point(design, N, k, h_over_a=.08, ell_over_h=1/6, count=6,
                theory="TSDT", gradient=True, geometry=None, prepared=None):
    """Return sorted flexural frequencies and analytic matrix shape derivatives.

    Frequencies are nondimensional sqrt(E_epoxy/rho_epoxy)/a units.
    The inverse-compliance derivative is evaluated as -C(dS)C, without
    differentiating an eigensolver or replacing inverse factorization.
    """
    F, dF = fourier(design, N, derivative=True) if geometry is None else geometry
    cache = (prepare(N, h_over_a, theory=theory, indicator_matrix=F)
             if prepared is None else prepared)
    parts = assemble_real(N, k, h=h_over_a, theory=theory,
                          indicator_matrix=F, return_shape_operators=gradient)
    K = classical_inverse(cache, k) + ell_over_h**2 * parts["KB"]
    M = parts["M"]
    if N >= 11:
        shift = -0.001
        factor = cho_factor(K-shift*M, lower=True, check_finite=False)
        inverse = LinearOperator(
            K.shape, matvec=lambda v:cho_solve(factor,v,check_finite=False), dtype=float)
        values, vectors = eigsh(K, k=count, M=M, sigma=shift, OPinv=inverse,
                                tol=1e-9, v0=np.ones(len(K)))
        order = np.argsort(values)
        values, vectors = values[order], vectors[:,order]
        residual = np.linalg.norm(K@vectors-M@vectors*values,axis=0) / (
            np.linalg.norm(K@vectors,axis=0)
            + np.abs(values)*np.linalg.norm(M@vectors,axis=0)+1e-30)
        if np.max(residual[1:]) > 1e-5:
            raise ArithmeticError("Shift-invert eigenpair residual exceeds 1e-5")
    else:
        values, vectors = eigh(K, M, subset_by_index=(0, count-1), driver="gvx")
    if np.min(values[1:]) <= 0:
        raise ArithmeticError("Nonpositive non-rigid flexural eigenvalue")
    freq = np.sqrt(np.maximum(values, 0))
    if not gradient:
        return freq
    G = cache["G"]
    B = strain(G + np.asarray(k), theory)
    Z = np.einsum("ab,nbd->nad", cache["W"].T, B)
    df = np.empty((count, len(dF)))
    components = []
    for band in range(count):
        q = vectors[:, band]
        if freq[band] < 1e-8:
            df[band] = np.nan  # Gamma's exact rigid translation is not a log-frequency mode.
            components.append(dict(classical=None, mcst=None, inertia=None))
            continue
        dclassical = np.zeros(len(dF))
        qblocks = q.reshape(len(G), 3)
        y = np.einsum("nid,nd->ni", Z, qblocks)
        for (ids, C), alpha in zip(cache["groups"], cache["alphas"]):
            v = C @ y[:, ids]
            dclassical -= alpha * np.einsum("ni,jnp,pi->j", v, dF, v, optimize=True)
        pfield = np.einsum("snid,nd->sni", parts["P_operator"], qblocks)
        cfield = np.einsum("snid,nd->sni", parts["C_operator"], qblocks)
        dmass = (7780/1180-1) * np.einsum(
            "sni,jnp,spi->j", pfield, dF, pfield, optimize=True)
        nu_epoxy = 4.35/(2*1.59)-1
        dmc = -ell_over_h**2*h_over_a**2/(1+nu_epoxy) * np.einsum(
            "sni,jnp,spi->j", cfield, dF, cfield, optimize=True)
        df[band] = (dclassical + dmc - values[band]*dmass) / (2*freq[band])
        components.append(dict(classical=dclassical, mcst=dmc,
                               inertia=-values[band]*dmass))
    return freq, df, components


def sampled_gap(spectra, gradients=None):
    """Signed relative gap of sorted bands 4 and 5 on a supplied k sample."""
    spectra = np.asarray(spectra)
    iL = int(np.argmax(spectra[:, 3]))
    iU = int(np.argmin(spectra[:, 4]))
    L, U = spectra[iL, 3], spectra[iU, 4]
    J = 200*(U-L)/(U+L)
    result = dict(L=float(L), U=float(U), J=float(J),
                  center=float((L+U)/2), active_L=iL, active_U=iU)
    if gradients is not None:
        gradients = np.asarray(gradients)
        dL, dU = gradients[iL, 3], gradients[iU, 4]
        result["gradient"] = 400*(L*dU-U*dL)/(U+L)**2
    return result
