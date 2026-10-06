"""Plot audited inverse-PWE path-convergence data for manuscript section 3.1."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATA = ROOT / "runs/pwe_path_convergence_dense_20260927"
ASSETS = REPO / "output/pdf/modal_selectivity_assets"
STEM = "pwe_path_convergence_20260927"


def load_verified_rows() -> list[dict[str, str]]:
    audit = json.loads((DATA / "audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS_SAME_GEOMETRY_12_POINT_PATH_EVERY_N7_TO_N21"
    with (DATA / "pwe_path_convergence_dense.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["N"]) for row in rows] == list(range(7, 22))
    assert all(
        float(row["a_um"]) == 75
        and float(row["h_um"]) == 6
        and float(row["ell_epoxy_um"]) == 1
        and int(row["path_points"]) == 12
        for row in rows
    )
    ref_l = float(rows[-1]["lower_MHz"])
    ref_u = float(rows[-1]["upper_MHz"])
    for row in rows:
        lower = float(row["lower_MHz"])
        upper = float(row["upper_MHz"])
        gap = 200 * (upper - lower) / (upper + lower)
        error_l = 100 * abs(lower / ref_l - 1)
        error_u = 100 * abs(upper / ref_u - 1)
        assert math.isclose(gap, float(row["J45_percent"]), abs_tol=1e-9)
        assert math.isclose(error_l, float(row["lower_difference_to_N21_percent"]), abs_tol=1e-9)
        assert math.isclose(error_u, float(row["upper_difference_to_N21_percent"]), abs_tol=1e-9)
    return rows


def main() -> None:
    rows = load_verified_rows()
    ASSETS.mkdir(parents=True, exist_ok=True)
    n = np.array([int(row["N"]) for row in rows])
    lower_error = np.array([float(row["lower_difference_to_N21_percent"]) for row in rows])
    upper_error = np.array([float(row["upper_difference_to_N21_percent"]) for row in rows])
    gap = np.array([float(row["J45_percent"]) for row in rows])

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
        }
    )
    fig, (ax_err, ax_gap) = plt.subplots(1, 2, figsize=(6.9, 2.55))
    ax_err.plot(n, lower_error, color="#0072BD", marker="o", ms=4.5, lw=1.45, label=r"$\delta_L$")
    ax_err.plot(n, upper_error, color="#D55E00", marker="s", ms=4.2, lw=1.45, label=r"$\delta_U$")
    ax_err.set_ylabel(r"Difference from $N=21$ (%)")
    ax_err.set_ylim(-0.05 * upper_error.max(), 1.12 * upper_error.max())
    ax_err.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92,
                  edgecolor="none", ncol=1, handlelength=1.8)

    ax_gap.plot(n, gap, color="#7E2F8E", marker="D", ms=4.3, lw=1.45)
    ax_gap.set_ylabel(r"Signed $J_{45}$ (%)")
    gap_span = gap.max() - gap.min()
    ax_gap.set_ylim(gap.min() - 0.08 * gap_span, gap.max() + 0.08 * gap_span)

    for ax, letter in ((ax_err, "a"), (ax_gap, "b")):
        ax.set_xlim(6.6, 21.4)
        ax.set_xticks(np.arange(7, 22, 2))
        ax.set_xticks(n, minor=True)
        ax.set_xlabel("Fourier truncation order $N$")
        ax.grid(alpha=0.16)
        ax.tick_params(direction="in", top=True, right=True)
        ax.text(0.02, 1.025, f"({letter})", transform=ax.transAxes,
                va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.90, bottom=0.235, wspace=0.34)
    fig.savefig(ASSETS / f"{STEM}.pdf")
    fig.savefig(DATA / f"{STEM}.png", dpi=240)
    plt.close(fig)
    print(f"PASS: {ASSETS / (STEM + '.pdf')}")
    print(f"PASS: {DATA / (STEM + '.png')}")


if __name__ == "__main__":
    main()
