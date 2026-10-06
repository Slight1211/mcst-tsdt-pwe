"""Replot completed sampled band-edge data without fixed-point 3D claims."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

import report_section41_m2l4 as full_path
import report_ratio75_thickness as thickness_report


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ASSETS = REPO / "output/pdf/modal_selectivity_assets"
OUT = ROOT / "runs/edge_focus_20260926"
SCALE = ROOT / "runs/fixed_ratio_scale_20260922T175850279921Z"
FEM_SCALE = ROOT / "runs/ratio_scale_fullpath_M2L4_20260926"
ADD_SCALE = ROOT / "runs/scale_addendum_20260927"
THICK = ROOT / "runs/thickness_ratio75_20260925"
FEM_THICK = ROOT / "runs/thickness_ratio75_fullpath_M2L4_20260926"
ADD_THICK = ROOT / "runs/thickness_ratio75_h016_20260927"
STYLES = (
    ("FSDT", 0, "Classic-FSDT", "#0072BD", "--"),
    ("TSDT", 0, "Classic-TSDT", "#56B4E9", "-"),
    ("FSDT", 1, "MCST-FSDT", "#D55E00", "-."),
    ("TSDT", 1, "MCST-TSDT", "#7E2F8E", ":"),
)
BASE_DISPLAY_SCALES = (40, 75, 100, 200, 500, 1000, 2000)
DISPLAY_SCALES = (
    (40, 75, 100, 150, 200, 500, 750, 1000, 1500, 2000)
    if (ADD_SCALE / "audit.json").is_file() else BASE_DISPLAY_SCALES
)
MAIN_BAND_SCALES = (40, 75, 100, 200, 500, 2000)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def check_gap(row: dict[str, str]) -> None:
    lower = float(row["lower_MHz"])
    upper = float(row["upper_MHz"])
    reported = float(row["J45_percent"])
    assert lower > 0 and upper > 0
    assert abs(200 * (upper - lower) / (upper + lower) - reported) < 1e-8


def plot_scale_metrics(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True)
    lower_ax, upper_ax, gap_ax = axes
    lower_values = []
    for plate, ell, label, color, linestyle in STYLES:
        selected = sorted(
            (r for r in rows if r["theory"] == plate and int(r["ell_um"]) == ell
             and int(float(r["a_um"])) in DISPLAY_SCALES),
            key=lambda r: float(r["a_um"]),
        )
        assert len(selected) == len(DISPLAY_SCALES)
        x = np.array([float(r["a_um"]) for r in selected])
        factor = 2 * math.pi * x * math.sqrt(1180 / 4.35e9)
        for ax, key in ((lower_ax, "lower_MHz"), (upper_ax, "upper_MHz")):
            values = np.array([float(r[key]) for r in selected]) * factor
            ax.plot(x, values, color=color, ls=linestyle, marker="o",
                    ms=3.0, lw=1.15, label=label)
            if ax is lower_ax:
                lower_values.extend(values.tolist())
        if plate == "TSDT" and ell == 1:
            gap_ax.plot(x,
                        [float(r["J45_percent"]) for r in selected],
                        color=color, ls=linestyle, marker="o", ms=3.0, lw=1.15)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$a$ ($\mu$m)")
        ax.grid(alpha=0.16, which="both")
        ax.tick_params(direction="in", top=True, right=True)
    span = max(lower_values) - min(lower_values)
    lower_ax.set_ylim(min(lower_values) - 0.06 * span,
                      max(lower_values) + 0.06 * span)
    lower_ax.set_ylabel(r"$\Omega_L$")
    upper_ax.set_ylabel(r"$\Omega_U$")
    assert len(gap_ax.lines) == 1
    gap_ax.set_ylabel(r"Signed $J_{45}$ (%)")
    gap_ax.axhline(0, color=".5", lw=0.7)
    lower_ax.legend(ncol=1, frameon=True, facecolor="white",
                    framealpha=0.92, edgecolor="none", loc="upper right",
                    fontsize=7.4, handlelength=1.6, labelspacing=0.12)
    for ax, letter in zip(axes, "abc"):
        ax.text(0.02, 1.015, f"({letter})", transform=ax.transAxes,
                va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.94,
                        bottom=0.25, wspace=0.33)
    fig.savefig(ASSETS / "ratio_scale_edges_J.pdf")
    fig.savefig(OUT / "ratio_scale_edges_J.png", dpi=220)
    plt.close(fig)


def plot_scale_edge_errors(rows: list[dict[str, str]],
                           common_ref: list[dict[str, str]]) -> None:
    assert len(rows) == len(common_ref) == 4 * len(DISPLAY_SCALES)
    display_a = set(DISPLAY_SCALES)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True)
    lower_ax, upper_ax, gap_ax = axes
    for plate, ell, label, color, linestyle in STYLES:
        ref_rows = sorted(
            (row for row in common_ref if row["PWE_model"] == label),
            key=lambda row: float(row["a_um"]),
        )
        gap_rows = sorted(
            (row for row in rows if row["theory"] == plate
             and int(row["ell_um"]) == ell
             and int(float(row["a_um"])) in display_a),
            key=lambda row: float(row["a_um"]),
        )
        assert len(gap_rows) == len(ref_rows) == len(display_a)
        assert [int(float(row["a_um"])) for row in ref_rows] == [
            int(float(row["a_um"])) for row in gap_rows]
        x = [float(row["a_um"]) for row in ref_rows]
        for ax, key in ((lower_ax, "L_to_3D_MCST_percent"),
                        (upper_ax, "U_to_3D_MCST_percent")):
            ax.plot(x, [float(row[key]) for row in ref_rows],
                     color=color, ls=linestyle, marker="o", ms=3, lw=1.15,
                     label=label)
        assert all(abs(float(source["J45_percent"])
                       - float(edge["PWE_J45_percent"])) < 1e-8
                   for source, edge in zip(gap_rows, ref_rows))
        gap_ax.plot(x, [float(row["J_difference_pp"]) for row in ref_rows],
                    color=color, ls=linestyle, marker="o", ms=3, lw=1.15,
                    label=label)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$a$ ($\mu$m)")
        ax.grid(alpha=0.16, which="both")
        ax.tick_params(direction="in", top=True, right=True)
    lower_ax.set_ylabel(r"Lower-edge difference $e_L$ (%)")
    upper_ax.set_ylabel(r"Upper-edge difference $e_U$ (%)")
    for ax in (lower_ax, upper_ax):
        ax.set_yscale("log")
        ax.set_ylim(0.2, 20)
    assert len(gap_ax.lines) == 4
    gap_ax.set_ylabel(r"$\Delta J_{45}$ (pp)")
    gap_ax.axhline(0, color=".5", lw=0.7)
    lower_ax.legend(ncol=1, frameon=True, facecolor="white",
                    framealpha=0.92, edgecolor="none", loc="upper right",
                    fontsize=7.4, handlelength=1.6, labelspacing=0.12)
    for ax, letter in zip(axes, "abc"):
        ax.text(0.02, 1.015, f"({letter})", transform=ax.transAxes,
                va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.94,
                        bottom=0.25, wspace=0.33)
    fig.savefig(ASSETS / "ratio_scale_edges_J.pdf")
    fig.savefig(OUT / "ratio_scale_edges_J.png", dpi=220)
    plt.close(fig)


def scale_all_vs_3d_mcst_errors(
    comparison: list[dict[str, str]],
) -> list[dict[str, float | int | str]]:
    """Compare all four PWE models with the same-scale 3D MCST edges."""
    by_key = {(int(float(row["a_um"])), row["model"]): row
              for row in comparison}
    display_a = BASE_DISPLAY_SCALES
    assert len(comparison) == 40 and len(by_key) == 40
    output = []
    for a in display_a:
        mcst = by_key[a, "MCST-TSDT"]
        mcst_fsdt = by_key[a, "MCST-FSDT"]
        for edge in ("L", "U"):
            assert abs(float(mcst[f"FEM_{edge}_MHz"])
                       - float(mcst_fsdt[f"FEM_{edge}_MHz"])) < 1e-10
        for model in ("Classic-FSDT", "Classic-TSDT",
                      "MCST-FSDT", "MCST-TSDT"):
            pwe = by_key[a, model]
            fem_l = float(mcst["FEM_L_MHz"])
            fem_u = float(mcst["FEM_U_MHz"])
            output.append({
                "a_um": a,
                "PWE_model": model,
                "reference": "3D MCST M2L4",
                "PWE_L_MHz": float(pwe["PWE_L_MHz"]),
                "FEM_MCST_L_MHz": fem_l,
                "L_to_3D_MCST_percent": 100 * abs(
                    float(pwe["PWE_L_MHz"]) - fem_l) / fem_l,
                "PWE_U_MHz": float(pwe["PWE_U_MHz"]),
                "FEM_MCST_U_MHz": fem_u,
                "U_to_3D_MCST_percent": 100 * abs(
                    float(pwe["PWE_U_MHz"]) - fem_u) / fem_u,
                "PWE_J45_percent": float(pwe["PWE_J45_percent"]),
                "FEM_MCST_J45_percent": float(mcst["FEM_J45_percent"]),
                "J_difference_pp": (float(pwe["PWE_J45_percent"])
                                    - float(mcst["FEM_J45_percent"])),
            })
    assert len(output) == 4 * len(display_a)
    return output


def plot_thickness_metrics(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=True)
    lower_ax, upper_ax, gap_ax = axes
    for plate, ell, label, color, linestyle in STYLES:
        selected = sorted(
            (r for r in rows if r["plate"] == plate and int(r["ell_um"]) == ell),
            key=lambda r: float(r["h_over_a"]),
        )
        assert len(selected) == len({float(r["h_over_a"]) for r in rows})
        x = np.array([float(r["h_over_a"]) for r in selected])
        for ax, key in ((lower_ax, "lower_MHz"), (upper_ax, "upper_MHz")):
            values = np.array([float(r[key]) for r in selected])
            ax.plot(x, values,
                    color=color, ls=linestyle, marker="o", ms=3.0, lw=1.15,
                    label=label)
        if plate == "TSDT" and ell == 1:
            gap_ax.plot(x, [float(r["J45_percent"]) for r in selected],
                        color=color, ls=linestyle, marker="o", ms=3.0, lw=1.15)
    for ax in (lower_ax, upper_ax, gap_ax):
        ax.set_xscale("log")
        ax.grid(alpha=0.16, which="both")
        ax.tick_params(direction="in", top=True, right=True)
    lower_ax.set_ylim(bottom=0)
    lower_ax.legend(ncol=1, frameon=True, facecolor="white",
                    framealpha=0.92, edgecolor="none",
                    loc="upper left",
                    fontsize=8.2, handlelength=1.6, labelspacing=0.15)
    ticks = (0.02, 0.04, 0.08, 0.16, 0.32)
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels((".02", ".04", ".08", ".16", ".32"))
        ax.xaxis.set_minor_locator(NullLocator())
    lower_ax.set_ylabel(r"$L_{45}$ (MHz)")
    upper_ax.set_ylabel(r"$U_{45}$ (MHz)")
    assert len(gap_ax.lines) == 1
    gap_ax.set_ylabel(r"Signed $J_{45}$ (%)")
    for ax in axes:
        ax.set_xlabel(r"$h/a$")
    gap_ax.axhline(0, color=".5", lw=0.7)
    for ax, letter in ((lower_ax, "a"), (upper_ax, "b"), (gap_ax, "c")):
        ax.text(0.02, 1.015, f"({letter})", transform=ax.transAxes,
                va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.94,
                        bottom=0.25, wspace=0.33)
    fig.savefig(ASSETS / "ratio75_thickness_edges_J.pdf")
    fig.savefig(OUT / "ratio75_thickness_edges_J.png", dpi=220)
    plt.close(fig)


def plot_thickness_edge_errors(rows: list[dict[str, str]],
                               comparison: list[dict[str, str]]) -> None:
    ratios = {float(row["h_over_a"]) for row in rows}
    assert len(rows) == len(comparison) == 4 * len(ratios)
    common_ref = thickness_all_vs_3d_mcst_errors(comparison)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=True)
    lower_ax, upper_ax, gap_ax = axes
    for plate, ell, label, color, linestyle in STYLES:
        edge_rows = sorted(
            (row for row in comparison if row["model"] == label),
            key=lambda row: float(row["h_over_a"]),
        )
        ref_rows = sorted(
            (row for row in common_ref if row["PWE_model"] == label),
            key=lambda row: float(row["h_over_a"]),
        )
        gap_rows = sorted(
            (row for row in rows if row["plate"] == plate
             and int(row["ell_um"]) == ell),
            key=lambda row: float(row["h_over_a"]),
        )
        assert len(edge_rows) == len(ref_rows) == len(gap_rows) == len(ratios)
        assert np.allclose(
            [float(row["h_over_a"]) for row in edge_rows],
            [float(row["h_over_a"]) for row in gap_rows],
            atol=1e-12, rtol=0,
        )
        assert np.allclose(
            [float(row["h_over_a"]) for row in edge_rows],
            [float(row["h_over_a"]) for row in ref_rows],
            atol=1e-12, rtol=0,
        )
        x = [float(row["h_over_a"]) for row in edge_rows]
        for ax, key in ((lower_ax, "L_to_3D_MCST_percent"),
                        (upper_ax, "U_to_3D_MCST_percent")):
            ax.plot(x, [float(row[key]) for row in ref_rows],
                    color=color, ls=linestyle, marker="o", ms=3, lw=1.15,
                    label=label)
        assert all(abs(float(source["J45_percent"])
                       - float(edge["PWE_J45_percent"])) < 1e-8
                   for source, edge in zip(gap_rows, ref_rows))
        gap_ax.plot(x, [float(row["J_difference_pp"]) for row in ref_rows],
                    color=color, ls=linestyle, marker="o", ms=3, lw=1.15,
                    label=label)
    ticks = (0.02, 0.04, 0.08, 0.16, 0.32)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels((".02", ".04", ".08", ".16", ".32"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"$h/a$")
        ax.grid(alpha=0.16, which="both")
        ax.tick_params(direction="in", top=True, right=True)
    lower_ax.set_ylabel(r"Lower-edge error $e_L$ (%)")
    upper_ax.set_ylabel(r"Upper-edge error $e_U$ (%)")
    for ax in (lower_ax, upper_ax):
        ax.set_yscale("log")
        ax.set_ylim(0.07, 45)
    assert len(gap_ax.lines) == 4
    gap_ax.set_yscale("symlog", linthresh=1)
    gap_ax.set_ylim(-2, 25)
    gap_ax.set_yticks((-1, 0, 1, 10, 20))
    gap_ax.set_yticklabels(("-1", "0", "1", "10", "20"))
    gap_ax.tick_params(axis="y", labelsize=7)
    gap_ax.set_ylabel(r"$\Delta J_{45}$ (pp)")
    gap_ax.axhline(0, color=".5", lw=0.7)
    lower_ax.legend(ncol=1, frameon=True, facecolor="white",
                    framealpha=0.92, edgecolor="none", loc="upper right",
                    fontsize=7.4, handlelength=1.6, labelspacing=0.12)
    for ax, letter in zip(axes, "abc"):
        ax.text(0.02, 1.015, f"({letter})", transform=ax.transAxes,
                va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.94,
                        bottom=0.25, wspace=0.33)
    fig.savefig(ASSETS / "ratio75_thickness_edges_J.pdf")
    fig.savefig(OUT / "ratio75_thickness_edges_J.png", dpi=220)
    plt.close(fig)


def thickness_all_vs_3d_mcst_errors(
    comparison: list[dict[str, str]],
) -> list[dict[str, float | str]]:
    """Use one 3D MCST path-edge pair at each thickness for all PWE models."""
    by_key = {(float(row["h_over_a"]), row["model"]): row
              for row in comparison}
    ratios = sorted({float(row["h_over_a"]) for row in comparison})
    assert len(comparison) == len(by_key) == 4 * len(ratios)
    output = []
    for ratio in ratios:
        mcst = by_key[ratio, "MCST-TSDT"]
        mcst_fsdt = by_key[ratio, "MCST-FSDT"]
        for edge in ("L", "U"):
            assert abs(float(mcst[f"FEM_{edge}_MHz"])
                       - float(mcst_fsdt[f"FEM_{edge}_MHz"])) < 1e-10
        for model in ("Classic-FSDT", "Classic-TSDT",
                      "MCST-FSDT", "MCST-TSDT"):
            pwe = by_key[ratio, model]
            fem_l = float(mcst["FEM_L_MHz"])
            fem_u = float(mcst["FEM_U_MHz"])
            output.append({
                "h_over_a": ratio,
                "h_um": float(pwe["h_um"]),
                "PWE_model": model,
                "reference": "3D MCST M2L4",
                "PWE_L_MHz": float(pwe["PWE_L_MHz"]),
                "FEM_MCST_L_MHz": fem_l,
                "L_to_3D_MCST_percent": 100 * abs(
                    float(pwe["PWE_L_MHz"]) - fem_l) / fem_l,
                "PWE_U_MHz": float(pwe["PWE_U_MHz"]),
                "FEM_MCST_U_MHz": fem_u,
                "U_to_3D_MCST_percent": 100 * abs(
                    float(pwe["PWE_U_MHz"]) - fem_u) / fem_u,
                "PWE_J45_percent": float(pwe["PWE_J45_percent"]),
                "FEM_MCST_J45_percent": float(mcst["FEM_J45_percent"]),
                "J_difference_pp": (float(pwe["PWE_J45_percent"])
                                    - float(mcst["FEM_J45_percent"])),
            })
    return output


def plot_scale_bands(a: int, gap_rows: list[dict[str, str]],
                     show_legend: bool = False, show_ylabel: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(2.18, 1.92))
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.19, top=0.97)
    factor = 2 * math.pi * a * math.sqrt(1180 / 4.35e9)
    max_frequency = 0.0
    for plate, ell, label, color, linestyle in STYLES:
        spectra = []
        for point in range(12):
            path = SCALE / "pwe" / f"N15_a{a}_p{point}_{plate}_l{ell}.npz"
            with np.load(path) as record:
                freq = record["frequency_MHz"]
                assert len(freq) >= 6 and np.all(np.isfinite(freq[:6]))
                spectra.append(freq[:6])
        values = np.array(spectra)
        values = np.vstack((values, values[0]))
        max_frequency = max(max_frequency, float(values.max() * factor))
        for band in range(6):
            ax.plot(np.arange(13), values[:, band] * factor, color=color,
                    ls=linestyle, lw=0.95,
                    label=label if band == 0 and show_legend else None)
        gap = next(r for r in gap_rows if int(r["a_um"]) == a
                   and r["theory"] == plate and int(r["ell_um"]) == ell)
        assert abs(float(gap["lower_MHz"]) - float(values[:, 3].max())) < 1e-9
        assert abs(float(gap["upper_MHz"]) - float(values[:, 4].min())) < 1e-9
        if plate == "TSDT" and ell == 1:
            if float(gap["J45_percent"]) > 0:
                ax.axhspan(float(gap["lower_MHz"]) * factor,
                           float(gap["upper_MHz"]) * factor,
                           color="#77AC30", alpha=0.15)
    ax.set(xticks=(0, 4, 8, 12),
           xticklabels=(r"$\Gamma$", "X", "M", r"$\Gamma$"),
           xlim=(0, 12), ylim=(0, max_frequency * 1.20))
    if show_ylabel:
        ax.set_ylabel(r"$\Omega$", fontsize=9)
    ax.tick_params(direction="in", top=True, right=True, labelsize=8.2)
    ax.grid(alpha=0.12)
    if show_legend:
        ax.legend(ncol=2, fontsize=7.2, frameon=True, facecolor="white",
                  framealpha=0.9, edgecolor="none", loc="upper left",
                  handlelength=1.0, handletextpad=0.3, columnspacing=0.45,
                  labelspacing=0.08, borderpad=0.15)
    name = f"ratio_scale_bands_no3d_{a}"
    fig.savefig(ASSETS / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def plot_thickness_bands(index: int, ratio: float,
                         gap_rows: list[dict[str, str]],
                         show_legend: bool = False,
                         show_ylabel: bool = False) -> None:
    h = 75.0 * ratio
    fig, ax = plt.subplots(figsize=(2.18, 1.92))
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.19, top=0.97)
    max_frequency = 0.0
    for plate, ell, label, color, linestyle in STYLES:
        if ratio == 0.16:
            values = []
            for point in range(12):
                path = (ADD_THICK / "pwe/N15" /
                        f"N15_a75_h12_p{point}_{plate}_l{ell}.npz")
                with np.load(path) as record:
                    assert float(record["seed_agreement"]) < 1e-7
                    values.append(record["frequency_MHz"][:6].copy())
            values = np.array(values)
        else:
            values = np.array([
                thickness_report.pwe_frequency(15, h, point, plate, ell)[:6]
                for point in range(12)
            ])
        values = np.vstack((values, values[0]))
        max_frequency = max(max_frequency, float(values.max()))
        for band in range(6):
            ax.plot(np.arange(13), values[:, band], color=color,
                    ls=linestyle, lw=0.95,
                    label=label if band == 0 and show_legend else None)
        gap = next(r for r in gap_rows
                   if abs(float(r["h_over_a"]) - ratio) < 1e-10
                   and r["plate"] == plate and int(r["ell_um"]) == ell)
        assert abs(float(gap["lower_MHz"]) - float(values[:, 3].max())) < 1e-9
        assert abs(float(gap["upper_MHz"]) - float(values[:, 4].min())) < 1e-9
        if plate == "TSDT" and ell == 1 and float(gap["J45_percent"]) > 0:
            ax.axhspan(float(gap["lower_MHz"]), float(gap["upper_MHz"]),
                       color="#77AC30", alpha=0.15)
    ax.set(xticks=(0, 4, 8, 12),
           xticklabels=(r"$\Gamma$", "X", "M", r"$\Gamma$"),
           xlim=(0, 12), ylim=(0, max_frequency * 1.20))
    if show_ylabel:
        ax.set_ylabel("Frequency (MHz)", fontsize=9)
    ax.tick_params(direction="in", top=True, right=True, labelsize=8.2)
    ax.grid(alpha=0.12)
    if show_legend:
        ax.legend(ncol=2, fontsize=7.2, frameon=True, facecolor="white",
                  framealpha=0.9, edgecolor="none", loc="upper left",
                  handlelength=1.0, handletextpad=0.3, columnspacing=0.45,
                  labelspacing=0.08, borderpad=0.15)
    name = (f"ratio75_thickness_bands_{index}" if index >= 0
            else ("ratio75_thickness_bands_012" if index == -1
                  else "ratio75_thickness_bands_020"))
    fig.savefig(ASSETS / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)


def full_path_edges() -> list[dict]:
    assessment = json.loads((full_path.PATHS / "assessment.json").read_text())
    assert assessment["status"] == "M2L4_FULL_PATH_VALIDATED"
    rows = full_path.csv_rows(full_path.PWE / "pwe.csv")
    results = []
    for plate, ell, label, _, _ in STYLES:
        spectrum = np.array(
            [full_path.pwe_spectrum(rows, plate, float(ell), "inverse", p)
             for p in range(12)]
        )
        reference = full_path.path_spectrum("classic" if ell == 0 else "mcst", 2, 4)
        lower = float(spectrum[:, 3].max())
        upper = float(spectrum[:, 4].min())
        lower_3d = float(reference[:, 3].max())
        upper_3d = float(reference[:, 4].min())
        assert spectrum[:, 3].argmax() == spectrum[:, 4].argmin() == 4
        assert reference[:, 3].argmax() == reference[:, 4].argmin() == 4
        results.append({
            "model": label, "PWE_L_kHz": lower, "3D_L_kHz": lower_3d,
            "L_error_percent": 100 * abs(lower / lower_3d - 1),
            "PWE_U_kHz": upper, "3D_U_kHz": upper_3d,
            "U_error_percent": 100 * abs(upper / upper_3d - 1),
            "PWE_J_percent": 200 * (upper - lower) / (upper + lower),
            "3D_J_percent": 200 * (upper_3d - lower_3d) / (upper_3d + lower_3d),
        })
    return results


def mesh_edge_sensitivity() -> list[dict]:
    pairs = (((3, 2), (3, 4)), ((3, 4), (2, 4)),
             ((2, 2), (2, 4)), ((2, 4), (1, 4)),
             ((2, 4), (2, 6)))
    results = []
    for theory in ("classic", "mcst"):
        for coarse, finer in pairs:
            a = full_path.path_spectrum(theory, *coarse)
            b = full_path.path_spectrum(theory, *finer)
            results.append({
                "theory": theory,
                "coarse_mesh": f"M{coarse[0]}L{coarse[1]}",
                "finer_mesh": f"M{finer[0]}L{finer[1]}",
                "L_change_percent": 100 * abs(a[:, 3].max() / b[:, 3].max() - 1),
                "U_change_percent": 100 * abs(a[:, 4].min() / b[:, 4].min() - 1),
            })
    return results


def main() -> None:
    assert json.loads((SCALE / "audit.json").read_text())["status"] == "COMPLETE_FIXED_MESH_CHECKS"
    assert json.loads((THICK / "assessment.json").read_text())["status"] == "PASS_ACTUAL_A75_PWE_AND_M2L4_XM"
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9,
                         "mathtext.fontset": "stix", "axes.linewidth": 0.8,
                         "pdf.fonttype": 42})
    source_scale_rows = read_csv(SCALE / "gaps.csv")
    add_scale_audit_path = ADD_SCALE / "audit.json"
    if add_scale_audit_path.is_file():
        assert json.loads(add_scale_audit_path.read_text())["status"] == \
            "PASS_TEN_SCALE_SAMPLED_PATH_COMPARISON"
        scale_rows = read_csv(ADD_SCALE / "combined_gaps_10scales.csv")
    else:
        scale_rows = [row for row in source_scale_rows
                      if int(float(row["a_um"])) in DISPLAY_SCALES]
    add_audit_path = ADD_THICK / "audit.json"
    if add_audit_path.is_file():
        assert json.loads(add_audit_path.read_text())["status"] == \
            "PASS_EIGHT_POINT_SAMPLED_PATH_EDGE_COMPARISON"
        thick_rows = read_csv(ADD_THICK / "combined_gaps.csv")
    else:
        thick_rows = read_csv(THICK / "gaps.csv")
    assert len(source_scale_rows) == 40
    assert len(scale_rows) == 4 * len(DISPLAY_SCALES) and len(thick_rows) in (28, 32)
    for row in source_scale_rows + scale_rows + thick_rows:
        check_gap(row)
    path_rows = full_path_edges()
    mesh_rows = mesh_edge_sensitivity()
    write_csv(OUT / "scale_edges.csv", scale_rows)
    write_csv(OUT / "thickness_edges.csv", thick_rows)
    write_csv(OUT / "section41_path_edges.csv", path_rows)
    write_csv(OUT / "section41_mesh_edge_changes.csv", mesh_rows)
    fem_audit_path = FEM_SCALE / "edge_audit.json"
    if fem_audit_path.is_file():
        fem_audit = json.loads(fem_audit_path.read_text())
        assert fem_audit["status"] == "PASS_FULL_PATH_EDGE_COMPARISON"
        if add_scale_audit_path.is_file():
            common_ref = read_csv(
                ADD_SCALE / "combined_common_reference_10scales.csv")
            write_csv(OUT / "scale_all_vs_3d_mcst_10scales.csv", common_ref)
        else:
            comparison = read_csv(FEM_SCALE / "pwe_fem_edge_errors.csv")
            common_ref = scale_all_vs_3d_mcst_errors(comparison)
            write_csv(OUT / "scale_all_vs_3d_mcst.csv", common_ref)
        plot_scale_edge_errors(scale_rows, common_ref)
        figure4_reference = "same-geometry 3D MCST M2L4 sampled-path edges for all four PWE models"
    else:
        plot_scale_metrics(scale_rows)
        figure4_reference = "PWE sampled-path edges only; 3D full paths pending"
    thick_audit_path = (add_audit_path if add_audit_path.is_file()
                        else FEM_THICK / "edge_audit.json")
    if thick_audit_path.is_file():
        thick_audit = json.loads(thick_audit_path.read_text())
        expected_status = ("PASS_EIGHT_POINT_SAMPLED_PATH_EDGE_COMPARISON"
                           if add_audit_path.is_file()
                           else "PASS_FULL_PATH_EDGE_COMPARISON")
        assert thick_audit["status"] == expected_status
        comparison = read_csv(
            ADD_THICK / "combined_pwe_fem_edge_errors.csv"
            if add_audit_path.is_file()
            else FEM_THICK / "pwe_fem_edge_errors.csv"
        )
        plot_thickness_edge_errors(thick_rows, comparison)
        write_csv(OUT / "thickness_all_vs_3d_mcst.csv",
                  thickness_all_vs_3d_mcst_errors(comparison))
        figure6_reference = "same-geometry 3D MCST M2L4 sampled-path edges for all four PWE models"
    else:
        plot_thickness_metrics(thick_rows)
        figure6_reference = "PWE sampled-path edges only; 3D full paths pending"
    for index, a in enumerate(MAIN_BAND_SCALES):
        plot_scale_bands(a, scale_rows, show_legend=(index == 0),
                          show_ylabel=(index % 3 == 0))
    display_ratios = ((0.02, 0.04, 0.06, 0.08, 0.16, 0.32)
                      if add_audit_path.is_file()
                      else (0.02, 0.04, 0.06, 0.08, 0.20, 0.32))
    for index, ratio in enumerate(display_ratios):
        plot_thickness_bands(index, ratio, thick_rows,
                             show_legend=(index == 0),
                             show_ylabel=(index % 3 == 0))
    plot_thickness_bands(-1, 0.12, thick_rows, show_ylabel=True)
    if add_audit_path.is_file():
        plot_thickness_bands(-2, 0.20, thick_rows, show_ylabel=True)
    audit = {
        "status": ("PASS_EIGHT_POINT_PATH_DATA_REPLOTTED"
                   if add_audit_path.is_file()
                   else "PASS_EXISTING_PATH_DATA_REPLOTTED"),
        "scale_rows": len(scale_rows), "thickness_rows": len(thick_rows),
        "figure4_reference": figure4_reference,
        "figure6_reference": figure6_reference,
        "J45_trend_source": "Scale and thickness panel (c): four PWE J45 values minus the matching 3D MCST M2L4 J45",
        "figure4_J45_points": len(DISPLAY_SCALES),
        "figure4_display_a_um": list(DISPLAY_SCALES),
        "figure5_display_a_um": list(MAIN_BAND_SCALES),
        "scale_all_vs_3d_mcst_rows": 4 * len(DISPLAY_SCALES) if fem_audit_path.is_file() else 0,
        "thickness_all_vs_3d_mcst_rows": len(thick_rows) if thick_audit_path.is_file() else 0,
        "figure6_J45_points": len(thick_rows) // 4,
        "figure6_J45_curves": 4 if thick_audit_path.is_file() else 1,
        "section41_path_models": len(path_rows),
        "section41_mesh_edge_comparisons": len(mesh_rows),
        "section41_path_edge_location": "X for all four PWE models and two 3D models",
        "not_run": (["3D full paths for scale scan"] if not fem_audit_path.is_file()
                    else []) +
                   (["3D full paths for thickness scan"] if not thick_audit_path.is_file()
                    else []) +
                   ["continuous irreducible-zone extremum certification",
                    "3D field MAC and joint mesh refinement"],
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    for row in path_rows:
        print(row["model"], round(row["L_error_percent"], 4),
              round(row["U_error_percent"], 4))


if __name__ == "__main__":
    main()
