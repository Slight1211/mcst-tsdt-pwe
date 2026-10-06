"""Density-weighted sampled-volume displacement MAC, not FE-vector MAC."""
import argparse
import json
from pathlib import Path
import numpy as np


def overlaps(left, right, weights):
    a, b = np.asarray(left, complex), np.asarray(right, complex)
    w = np.asarray(weights, float)
    assert a.ndim == b.ndim == 2 and a.shape[0] == b.shape[0] == w.size
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert np.all(np.isfinite(w)) and np.all(w > 0)
    a, b = a * np.sqrt(w[:, None]), b * np.sqrt(w[:, None])
    na, nb = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    assert np.all(na > 0) and np.all(nb > 0), 'Zero displacement norm'
    a, b = a / na, b / nb
    mac = np.abs(a.conj().T @ b)**2
    assert np.all(mac >= 0) and np.max(mac) <= 1 + 1e-10
    # SVD orthonormal bases handle rotated near-degenerate eigenspaces.
    ua, sa, _ = np.linalg.svd(a, full_matrices=False)
    ub, sb, _ = np.linalg.svd(b, full_matrices=False)
    rank_a, rank_b = int(np.sum(sa > sa[0]*1e-12)), int(np.sum(sb > sb[0]*1e-12))
    assert rank_a == a.shape[1] and rank_b == b.shape[1], 'Linearly dependent selected modes'
    singular = np.linalg.svd(ua[:, :rank_a].conj().T @ ub[:, :rank_b], compute_uv=False)
    # No clipping, hand symmetrization, or correction of computed overlaps.
    return dict(MAC=mac.tolist(), subspace_singular_values=singular.tolist(),
                left_rank=rank_a, right_rank=rank_b)


def read_fields(path, modes):
    data = np.genfromtxt(path, delimiter=',', names=True)
    coords = np.column_stack([data[n] for n in ('x_m','y_m','z_m')])
    w = np.repeat(data['rho_kg_m3'] * data['volume_weight_m3'], 3)
    field = np.column_stack([
        np.column_stack([data[f'{c}{m}_real'] + 1j*data[f'{c}{m}_imag'] for c in ('u','v','w')]).ravel()
        for m in modes])
    meta = np.genfromtxt(str(path)+'.modes.csv', delimiter=',', names=True)
    meta = np.atleast_1d(meta)
    selected = [meta[meta['mode'] == m] for m in modes]
    assert all(len(x) == 1 for x in selected)
    k = np.column_stack((meta['kx_pi_over_a'],meta['ky_pi_over_a']))
    assert np.allclose(k, k[0], rtol=0, atol=1e-12)
    return coords, w, field, k[0], [float(x[0]['frequency_mhz']) for x in selected]


def compare(left, right, lm, rm):
    ca, wa, a, ka, fa = read_fields(left, lm)
    cb, wb, b, kb, fb = read_fields(right, rm)
    assert ca.shape == cb.shape and np.allclose(ca, cb, rtol=1e-12, atol=1e-15), 'Different sample coordinates'
    assert np.allclose(wa, wb, rtol=1e-12, atol=0), 'Different density or volume weights'
    assert np.allclose(ka, kb, rtol=0, atol=1e-12), 'Different Bloch wavevectors'
    return dict(left=str(left), right=str(right), left_modes=lm, right_modes=rm,
                left_frequencies_MHz=fa, right_frequencies_MHz=fb, wavevector=ka.tolist(),
                sampled_points=len(ca), **overlaps(a, b, wa))


def self_test():
    a = np.eye(6, dtype=complex)[:, :2]
    phase = a * np.exp(1j*np.array([.67, -1.2]))
    assert np.allclose(overlaps(a, phase, np.ones(6))['MAC'], np.eye(2), atol=1e-14)
    rotation = np.array([[1,1],[-1,1]])/np.sqrt(2)
    rotated = overlaps(a, a@rotation, np.ones(6))
    assert np.allclose(rotated['MAC'], .5, atol=1e-14)
    assert np.allclose(rotated['subspace_singular_values'], 1, atol=1e-14)
    assert np.allclose(overlaps(a, a[:,::-1], np.arange(1,7))['MAC'], [[0,1],[1,0]])
    try:
        overlaps(a, a, np.zeros(6))
    except AssertionError:
        pass
    else:
        raise AssertionError('Invalid weights accepted')
    print('PASS_SYNTHETIC_ALGEBRA_ONLY: phase, degenerate rotation, permutation, invalid weights')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--left', type=Path); ap.add_argument('--right', type=Path)
    ap.add_argument('--left-modes', nargs='+', type=int); ap.add_argument('--right-modes', nargs='+', type=int)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        assert args.output and not args.output.exists(), 'Choose a new derived output file'
        result = compare(args.left, args.right, args.left_modes, args.right_modes)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
