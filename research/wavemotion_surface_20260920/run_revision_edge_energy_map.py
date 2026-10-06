"""Reconstruct N9 inverse-PWE X-edge displacement and MCST energy density.

The inverse classical quadratic form is localized through its dual stress
field, not by substituting the raw displacement into the direct functional.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
OUT = ROOT / "runs/manuscript_revision_20260925"
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "msse_20260919/python")]
from inverse_circle_bending import classical_inverse, generalized_D, prepare, strain  # noqa: E402
from msse_core import operators  # noqa: E402
from real_circle_bending import assemble_real  # noqa: E402


def periodic_field(coeff: np.ndarray, ij: np.ndarray, size: int) -> np.ndarray:
    spectral = np.zeros((size, size) + coeff.shape[1:], complex)
    for index, (i, j) in enumerate(ij):
        spectral[i % size, j % size] = coeff[index]
    return np.fft.ifft2(spectral, axes=(0, 1)) * size**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, choices=(9, 15), default=15)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    N, ratio, a_um, ell_um, radius, size = args.N, .08, 75., 1., .3, 512
    k = np.array([np.pi, 0.])
    input_file = OUT / f"ibz_N{N}_tau0.08_ix4_iy0.npz"
    with np.load(input_file) as data:
        vectors = data["vectors"][:, 3:5].copy()
        frequencies = data["frequency_MHz"][3:5].copy()
    with threadpool_limits(limits=1):
        cache = prepare(N, ratio, radius, "TSDT")
        s = assemble_real(N, k, ratio, radius, theory="TSDT")
        Kmc = (ell_um / (a_um * ratio))**2 * s["KB"]
        K = classical_inverse(cache, k) + Kmc
    ij, G = s["ij"], s["G"]
    axis = np.arange(size) / size
    xs = np.minimum(axis, 1-axis)
    epoxy = xs[:, None]**2 + xs[None, :]**2 >= radius**2
    z, weight = leggauss(8)
    panels = []
    for column, band in enumerate((4, 5)):
        q = vectors[:, column]
        denominator = float(q @ K @ q)
        qphysical = q.reshape(-1, 3) * np.array([1., 1j, 1j])
        q5 = np.zeros((len(ij), 5), complex)
        q5[:, 2:] = qphysical
        w = periodic_field(q5[:, 2:3], ij, size)[:, :, 0]
        mc = np.zeros((size, size), float)
        for zi, wi in zip(z, weight):
            op = np.array([operators("TSDT", g+k, zi*ratio/2, ratio)[2]
                           for g in G])
            curv = np.einsum("gij,gj->gi", op, q5)
            field = periodic_field(curv, ij, size)
            mc += wi * np.sum(abs(field)**2, axis=2) * ratio / 2
        # q^T Kmc q = (ell/a)^2/(1+nu) * integral |chi|^2.
        nu = 4.35 / (2*1.59) - 1
        density = (ell_um/a_um)**2/(1+nu) * mc * epoxy / denominator
        matrix_ratio = float(q @ Kmc @ q) / denominator
        spatial_ratio = float(np.mean(density))
        relative_error = abs(spatial_ratio/matrix_ratio-1)
        assert relative_error < .005, (band, relative_error)
        # Dual reconstruction: sigma=Cconv^-1 Bq, eta(x)=D(x)^-1 sigma(x).
        b = strain(G+k, "TSDT")
        transformed = np.einsum("ab,nbd,nd->na", cache["W"].T, b,
                                 q.reshape(-1, 3))
        dual = np.zeros_like(transformed)
        for ids, inverse_block in cache["groups"]:
            dual[:, ids] = inverse_block @ transformed[:, ids]
        sigma = periodic_field(dual @ cache["W"].T, ij, size)
        da, db = cache["DA"], cache["DB"]
        eta_a = sigma @ np.linalg.inv(da).T
        eta_b = sigma @ np.linalg.inv(db).T
        local_cl = np.where(
            epoxy,
            np.einsum("...i,ij,...j->...", eta_b.conj(), db, eta_b).real,
            np.einsum("...i,ij,...j->...", eta_a.conj(), da, eta_a).real,
        ) / denominator
        local_sh = np.where(
            epoxy,
            np.einsum("...i,ij,...j->...", eta_b[..., 6:].conj(),
                      db[6:, 6:], eta_b[..., 6:]).real,
            np.einsum("...i,ij,...j->...", eta_a[..., 6:].conj(),
                      da[6:, 6:], eta_a[..., 6:]).real,
        ) / denominator
        classic_matrix_ratio = float(q @ classical_inverse(cache, k) @ q)/denominator
        classic_spatial_ratio = float(np.mean(local_cl))
        classic_error = abs(classic_spatial_ratio/classic_matrix_ratio-1)
        assert classic_error < .005, (band, classic_error)
        assert abs(float(np.mean(local_sh)) + float(np.mean(local_cl-local_sh))
                   - classic_spatial_ratio) < 1e-10
        panels.append((w, density, local_sh, matrix_ratio, spatial_ratio,
                       relative_error, classic_matrix_ratio, classic_spatial_ratio,
                       classic_error))
    maximum = max(float(np.max(item[1])) for item in panels)
    shear_maximum = max(float(np.max(item[2])) for item in panels)
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.8), constrained_layout=True)
    extent = (-.5, .5, -.5, .5)
    for row, (w, density, shear_density, matrix_ratio, spatial_ratio, error,
              classic_matrix_ratio, classic_spatial_ratio, classic_error) in enumerate(panels):
        phase = np.exp(-1j*np.angle(w[np.unravel_index(np.argmax(abs(w)), w.shape)]))
        displacement = np.fft.fftshift((w*phase).real)
        displacement /= max(abs(displacement).max(), 1e-30)
        plotted_energy = np.fft.fftshift(density)
        left = axes[row, 0].imshow(displacement.T, origin="lower", extent=extent,
                                   cmap="RdBu_r", vmin=-1, vmax=1)
        right = axes[row, 1].imshow(plotted_energy.T, origin="lower", extent=extent,
                                    cmap="viridis", vmin=0, vmax=maximum)
        shear = axes[row, 2].imshow(np.fft.fftshift(shear_density).T, origin="lower",
                                    extent=extent, cmap="magma", vmin=0,
                                    vmax=shear_maximum)
        for ax in axes[row]:
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, ec="black", lw=.8))
            ax.set_aspect("equal")
            ax.set_xlabel("x/a")
            ax.set_ylabel("y/a")
        axes[row, 0].set_title(f"Band {row+4}: {frequencies[row]:.4f} MHz", fontsize=10)
        axes[row, 1].set_title(f"MCST (R={matrix_ratio:.4f})", fontsize=10)
        axes[row, 2].set_title(f"Shear (R={np.mean(shear_density):.4f})", fontsize=10)
    fig.colorbar(left, ax=axes[:, 0], shrink=.68, label="w/max|w|")
    fig.colorbar(right, ax=axes[:, 1], shrink=.68, label="MCST density / total energy")
    fig.colorbar(shear, ax=axes[:, 2], shrink=.68, label="Shear density / total energy")
    image = OUT / f"figures/edge_mode_mcst_energy_N{N}_tau0p08.png"
    image.parent.mkdir(exist_ok=True)
    fig.savefig(image, dpi=220)
    pdf_image = PROJECT / f"output/pdf/modal_selectivity_assets/edge_energy_N{N}.pdf"
    fig.savefig(pdf_image)
    plt.close(fig)
    result = {
        "status": "PASS_INVERSE_DUAL_CLASSICAL_AND_MCST_SPATIAL_INTEGRATION",
        "case": f"a=75um, h/a=0.08, N{N} inverse MCST-TSDT, X, bands4-5",
        "phase": "each w rotated so its maximum-magnitude point is real positive",
        "normalization": "w per-mode max; energy densities / same-mode total inverse quadratic form",
        "classical_localization": "dual stress sigma=Cconv^-1 Bq then eta(x)=D(x)^-1 sigma(x)",
        "grid": size, "gauss_z": 8,
        "bands": [{
            "band": row+4, "frequency_MHz": float(frequencies[row]),
            "R_mc_matrix": float(item[3]), "R_mc_spatial": float(item[4]),
            "mc_relative_integration_error": float(item[5]),
            "R_classic_matrix": float(item[6]),
            "R_classic_spatial": float(item[7]),
            "classic_relative_integration_error": float(item[8]),
            "R_shear_spatial": float(np.mean(item[2])),
        } for row, item in enumerate(panels)],
        "image": str(image),
        "pdf_image": str(pdf_image),
        "NOT_RUN": ["N21 energy map convergence",
                    "3D spatial field correspondence"],
    }
    (OUT / f"edge_energy_map_N{N}_assessment.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
