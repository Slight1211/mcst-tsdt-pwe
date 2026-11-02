"""C4v steel-in-epoxy node geometry for the existing PWE.

All coordinates are divided by the lattice constant.  The boundary has seven
nodes in [0, pi/4] and 48 distinct vertices after C4v completion.  The six
design coordinates are zero-mean logarithmic radial contrasts.  An optional
seventh coordinate scales the area from pi*(0.3)**2. Fourier coefficients and
their shape derivatives use exact straight-edge integrals, not rasterization.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import helmert
from two_phase import polygon, reciprocal

SECTOR_NODES = 7
VERTICES = 8 * (SECTOR_NODES - 1)
AREA = np.pi * 0.3**2
H = helmert(SECTOR_NODES, full=False).T
ANGLES = np.arange(VERTICES) * np.pi / VERTICES * 2
INDEX = np.minimum(np.arange(VERTICES) % 12, 12 - np.arange(VERTICES) % 12)
UNIT = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))


def signed_area(vertices):
    p = np.asarray(vertices)
    q = np.roll(p, -1, axis=0)
    return 0.5 * np.sum(p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0])


def convex_turns(design, derivative=False):
    """Signed consecutive-edge turns; nonnegative values define convexity."""
    if derivative:
        p, dp = boundary(design, True)
        e = np.roll(p,-1,axis=0)-p
        de = np.roll(dp,-1,axis=0)-dp
        en = np.roll(e,-1,axis=0)
        den = np.roll(de,-1,axis=0)
        turns = e[:,0]*en[:,1]-e[:,1]*en[:,0]
        jac = (de[:,0,:]*en[:,1,None]+e[:,0,None]*den[:,1,:]
               -de[:,1,:]*en[:,0,None]-e[:,1,None]*den[:,0,:])
        return turns, jac
    p = boundary(design)
    e = np.roll(p,-1,axis=0)-p
    en = np.roll(e,-1,axis=0)
    return e[:,0]*en[:,1]-e[:,1]*en[:,0]


def boundary(design, derivative=False):
    """Return vertices; a seventh log-scale coordinate releases the area."""
    y = np.asarray(design, float)
    if y.shape not in ((6,), (7,)) or not np.isfinite(y).all():
        raise ValueError("Six shape coordinates and optional log-scale are required")
    raw = np.exp(H @ y[:6])
    p0 = raw[INDEX, None] * UNIT
    area0 = signed_area(p0)
    scale = np.sqrt(AREA / area0) * (np.exp(y[6]) if len(y) == 7 else 1.)
    p = scale * p0
    # Free-area SLSQP may evaluate infeasible trial polygons while satisfying
    # its explicit vertex-clearance inequalities. Final designs are checked.
    if len(y) == 6 and np.max(np.abs(p)) >= 0.49:
        raise ValueError("Steel boundary approaches the unit-cell edge")
    # Positive radii at strictly increasing angles give a simple star-shaped
    # polygon.  Mild non-convexity is allowed; it is part of the design space.
    if np.min(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)) < 1e-5:
        raise ValueError("Adjacent boundary nodes are too close")
    if not derivative:
        return p
    dp0 = p0[:, :, None] * H[INDEX, None, :]
    nextp = np.roll(p0, -1, axis=0)
    nextdp = np.roll(dp0, -1, axis=0)
    darea = 0.5 * np.sum(
        dp0[:, 0, :] * nextp[:, 1, None] + p0[:, 0, None] * nextdp[:, 1, :]
        - dp0[:, 1, :] * nextp[:, 0, None] - p0[:, 1, None] * nextdp[:, 0, :],
        axis=0,
    )
    dp = scale * (dp0 - 0.5 * p0[:, :, None] * darea[None, None, :] / area0)
    if len(y) == 7:
        dp = np.concatenate((dp, p[:, :, None]), axis=2)
    return p, dp


def _line_moments(z):
    """Integral of exp(-i*z*t) and t*exp(-i*z*t), t in [0,1]."""
    z = np.asarray(z, float)
    i0 = np.empty(z.shape, complex)
    i1 = np.empty(z.shape, complex)
    small = np.abs(z) < 1e-4
    zz = z[small]
    i0[small] = 1 - 1j*zz/2 - zz**2/6 + 1j*zz**3/24 + zz**4/120
    i1[small] = .5 - 1j*zz/3 - zz**2/8 + 1j*zz**3/30 + zz**4/144
    zz = z[~small]
    i0[~small] = -np.expm1(-1j*zz)/(1j*zz)
    i1[~small] = (i0[~small] - np.exp(-1j*zz))/(1j*zz)
    return i0, i1


def scale_interval(shape):
    """Geometrically admissible scale interval and its piecewise derivatives."""
    p, dp = boundary(np.asarray(shape), derivative=True)
    index = np.unravel_index(np.argmax(np.abs(p)), p.shape)
    radius = abs(p[index])
    upper = .49/radius
    dupper = -upper*np.sign(p[index])*dp[index]/radius
    edges = np.roll(p, -1, axis=0)-p
    dedges = np.roll(dp, -1, axis=0)-dp
    lengths = np.linalg.norm(edges, axis=1)
    shortest = int(np.argmin(lengths))
    edge = lengths[shortest]
    # Tiny roundoff margin at the existing 1e-5 numerical edge-length gate.
    lower = (1e-5*(1+1e-10))/edge
    dlower = -lower*(edges[shortest]@dedges[shortest])/edge**2
    return lower, upper, dlower, dupper


def clearance_coordinates(design):
    """Physical log-area coordinates -> bounded scale-fraction coordinates."""
    y = np.asarray(design, float)
    lower, upper, _, _ = scale_interval(y[:6])
    return np.r_[y[:6], (np.exp(y[6])-lower)/(upper-lower)]


def clearance_design(coordinates, derivative=False):
    """q[6] in [0,1] enforces existing clearance and minimum-edge gates.

    No area cap is introduced: every physical geometry satisfying both
    original geometric gates has a corresponding scale-fraction coordinate.
    Derivatives are piecewise smooth at active-node/shortest-edge switches.
    """
    q = np.asarray(coordinates, float)
    lower, upper, dlower, dupper = scale_interval(q[:6])
    scale = lower+q[6]*(upper-lower)
    y = np.r_[q[:6], np.log(scale)]
    if not derivative:
        return y
    jac = np.eye(7)
    jac[6, :6] = (dlower+q[6]*(dupper-dlower))/scale
    jac[6, 6] = (upper-lower)/scale
    return y, jac


def _polygon_trial_fourier(wave, p):
    """Analytic polygon integral extended to infeasible SLSQP trial points.

    This extension is never accepted as a physical design; final results must
    satisfy the independent 0.49 cell-clearance check.
    """
    e = np.roll(p, -1, axis=0) - p
    area = signed_area(p)
    g2 = np.sum(wave**2, axis=1)
    phase = wave @ e.T
    integral = np.exp(-.5j*phase) * np.sinc(phase/(2*np.pi))
    normal = wave[:, 0, None]*e[None, :, 1] - wave[:, 1, None]*e[None, :, 0]
    value = np.full(len(wave), area, complex)
    nz = g2 > 0
    value[nz] = 1j*np.sum(normal[nz] * np.exp(-1j*wave[nz] @ p.T)
                          * integral[nz], axis=1)/g2[nz]
    return value


def fourier(design, N, derivative=False):
    """Steel indicator Toeplitz matrix; optional exact boundary-shape Jacobian.

    The first six coordinates preserve area; the optional seventh changes it.
    """
    p, dp = boundary(design, derivative=True)
    nvar = len(design)
    _, G = reciprocal(N)
    ij = np.rint(G / (2*np.pi)).astype(int)
    size = 4*N + 1
    grid = np.array([(i, j) for i in range(-2*N, 2*N+1)
                              for j in range(-2*N, 2*N+1)])
    wave = 2*np.pi*grid
    values = (polygon(wave, p) if np.max(np.abs(p)) < .5 else
              _polygon_trial_fourier(wave, p)).real.reshape(size, size)
    delta = ij[:, None, :] - ij[None, :, :]
    rows = delta[..., 0] + 2*N
    cols = delta[..., 1] + 2*N
    F = values[rows, cols]
    if not derivative:
        return F
    e = np.roll(p, -1, axis=0) - p
    de = np.roll(dp, -1, axis=0) - dp
    normals = np.column_stack((e[:, 1], -e[:, 0]))
    dnormals = np.stack((de[:, 1, :], -de[:, 0, :]), axis=1)
    z = wave @ e.T
    i0, i1 = _line_moments(z)
    phase = np.exp(-1j * wave @ p.T)
    base = np.einsum("ed,edj->ej", normals, dp)
    slope = np.einsum("ed,edj->ej", normals, de)
    dvalues = np.einsum("ge,ge,ej->gj", phase, i0, base, optimize=True)
    dvalues += np.einsum("ge,ge,ej->gj", phase, i1, slope, optimize=True)
    dvalues = dvalues.T.reshape(nvar, size, size)
    if np.max(np.abs(dvalues.imag)) > 1e-8:
        raise ArithmeticError("C4v Fourier derivative lost real symmetry")
    dF = dvalues.real[:, rows, cols]
    expected_area = AREA * (np.exp(2*np.asarray(design)[6]) if nvar == 7 else 1.)
    if abs(signed_area(p) - expected_area) > 1e-12:
        raise ArithmeticError("Polygon area identity failed")
    diagonal = dF[:, np.diag_indices(len(G))[0], np.diag_indices(len(G))[1]]
    if np.max(abs(diagonal[:6])) > 1e-9:
        raise ArithmeticError("Area-preserving shape Fourier identity failed")
    if nvar == 7 and np.max(abs(diagonal[6] - 2*expected_area)) > 1e-9:
        raise ArithmeticError("Variable-area Fourier derivative failed")
    return F, dF
