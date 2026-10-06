"""Journal-style comparison of path and finite-IBZ band-edge searches."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RUN = ROOT / "runs/manuscript_revision_20260925"
OLD = ROOT / "runs/thickness_ratio75_20260925/gaps.csv"


def records(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    path = sorted((r for r in records(OLD)
                   if r["plate"] == "TSDT" and r["ell_um"] == "1"),
                  key=lambda r: float(r["h_over_a"]))
    grid5 = records(RUN / "ibz_grid5_band_edge_results.csv")
    grid9 = records(RUN / "ibz_refinement_grid8/band_edge_results.csv")
    for family in (grid5, grid9):
        assert len(family) == 6
        for row in (r for r in family if r["N"] == "15"):
            matching = next(r for r in path if abs(float(r["h_over_a"])
                                                  - float(row["h_over_a"])) < 1e-10)
            assert abs(float(row["L45_MHz"])-float(matching["lower_MHz"])) < 1e-9
            assert abs(float(row["U45_MHz"])-float(matching["upper_MHz"])) < 1e-9
    for row in grid5:
        refined = next(r for r in grid9 if r["N"] == row["N"]
                       and r["h_over_a"] == row["h_over_a"])
        assert abs(float(row["L45_MHz"])-float(refined["L45_MHz"])) < 1e-9
        assert abs(float(row["U45_MHz"])-float(refined["U45_MHz"])) < 1e-9
    n15 = sorted((r for r in grid9 if r["N"] == "15"),
                 key=lambda r: float(r["h_over_a"]))
    assert [r["L_ix"] for r in n15] == ["8", "8", "8"]
    assert [r["L_iy"] for r in n15] == ["0", "0", "0"]
    assert [(r["U_ix"], r["U_iy"]) for r in n15] == [
        ("0", "0"), ("8", "0"), ("8", "0")
    ]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .8, "lines.linewidth": 1.6,
        "savefig.bbox": "tight",
    })
    fig, ax = plt.subplots(1, 3, figsize=(10.8, 3.35), constrained_layout=True)
    triangle = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
    ax[0].fill(triangle[:, 0], triangle[:, 1], color="#eaf1f7")
    ax[0].plot(triangle[:, 0], triangle[:, 1], color="#50616f")
    for i in range(9):
        for j in range(i+1):
            ax[0].plot(i/8, j/8, ".", color="#bac7d0", ms=2)
    ax[0].scatter([1], [0], marker="s", color="#0072B2", s=53, zorder=4,
                  label="Lower edge: X (all 3)")
    ax[0].scatter([0], [0], marker="o", facecolors="none", edgecolors="#D55E00",
                  s=65, zorder=4, label="Upper edge: Γ (0.02)")
    ax[0].scatter([1], [0], marker="o", facecolors="none", edgecolors="#D55E00",
                  s=100, zorder=5, label="Upper edge: X (0.08, 0.20)")
    ax[0].set(xlabel=r"$k_x a/\pi$", ylabel=r"$k_y a/\pi$", xlim=(-.06, 1.07),
              ylim=(-.06, 1.07), title="(a) Sampled irreducible zone")
    ax[0].set_aspect("equal")
    ax[0].legend(fontsize=6.7, loc="upper left", frameon=False)
    tau = np.array([float(r["h_over_a"]) for r in path])
    lower = np.array([float(r["lower_MHz"]) for r in path])
    upper = np.array([float(r["upper_MHz"]) for r in path])
    gap = np.array([float(r["J45_percent"]) for r in path])
    x9 = np.array([float(r["h_over_a"]) for r in n15])
    ax[1].plot(tau, lower, "o-", color="#0072B2", label="Path L₄₅")
    ax[1].plot(tau, upper, "o-", color="#D55E00", label="Path U₄₅")
    ax[1].scatter(x9, [float(r["L45_MHz"]) for r in n15], marker="s",
                  color="#0072B2", facecolors="none", s=70, zorder=4,
                  label="9×9 IBZ estimate")
    ax[1].scatter(x9, [float(r["U45_MHz"]) for r in n15], marker="s",
                  color="#D55E00", facecolors="none", s=70, zorder=4)
    ax[1].set(xlabel=r"$h/a$", ylabel="Frequency (MHz)",
              title="(b) Band-edge frequencies")
    ax[1].legend(fontsize=7, loc="upper left", frameon=False)
    ax[2].plot(tau, gap, "o-", color="#244b71", label="N15 path")
    ax[2].scatter(x9, [float(r["J45_percent"]) for r in n15], marker="s",
                  facecolors="none", edgecolors="#D55E00", s=80,
                  label="N15 9×9 IBZ")
    ax[2].set(xlabel=r"$h/a$", ylabel=r"$J_{45}$ (percentage points)",
              title="(c) Signed interval")
    ax[2].legend(fontsize=7, loc="lower right", frameon=False)
    for panel in ax:
        panel.grid(alpha=.18)
    png = RUN / "figures/band_edge_ibz.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf = PROJECT / "output/pdf/modal_selectivity_assets/band_edge_ibz.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    result = {
        "status": "PASS_PATH_MATCH_ON_SAMPLED_IBZ_GRIDS",
        "N15_grid9_active_points": [{
            "h_over_a": float(r["h_over_a"]),
            "L_kx_a": float(r["L_kx_a"]), "L_ky_a": float(r["L_ky_a"]),
            "U_kx_a": float(r["U_kx_a"]), "U_ky_a": float(r["U_ky_a"]),
            "L45_MHz": float(r["L45_MHz"]), "U45_MHz": float(r["U45_MHz"]),
            "J45_percent": float(r["J45_percent"]),
        } for r in n15],
        "meaning": "both finite 5x5 and 9x9 IBZ grids reproduce old sampled path edges",
        "NOT_RUN": ["continuous k-domain proof", "perturbed full-IBZ",
                    "3D full-IBZ branch matching"],
        "png": str(png), "pdf": str(pdf),
    }
    (RUN / "band_edge_plot_assessment.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
