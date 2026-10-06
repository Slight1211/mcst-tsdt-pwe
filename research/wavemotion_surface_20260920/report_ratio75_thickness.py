"""Audit actual a=75 um inverse-PWE and 3D M2L4 thickness results; then plot."""

from pathlib import Path
import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator
import numpy as np


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "runs/thickness_ratio75_20260925"
PWE = BASE / "pwe"
FEM = BASE / "comsol_M2L4"
RATIO = ROOT / "runs/fixed_ratio_scale_20260922T175850279921Z"
ASSETS = ROOT.parents[1] / "output/pdf/modal_selectivity_assets"
A_UM = 75.0
RATIOS = (0.02, 0.04, 0.06, 0.08, 0.12, 0.20, 0.32)
DISPLAY = (0.02, 0.04, 0.06, 0.08, 0.20, 0.32)
CUTOFF = 0.7
STYLES = (("FSDT", 0, "Classic-FSDT", "#0072BD", "--"),
          ("TSDT", 0, "Classic-TSDT", "#56B4E9", "-"),
          ("FSDT", 1, "MCST-FSDT", "#D55E00", "-."),
          ("TSDT", 1, "MCST-TSDT", "#7E2F8E", ":"))


def save_csv(path, rows):
    assert rows
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def pwe_frequency(N, h, point, plate, ell):
    path = PWE / f"N{N}" / f"N{N}_a75_h{h:g}_p{point}_{plate}_l{ell:g}.npz"
    with np.load(path) as record:
        frequency = record["frequency_MHz"].copy()
        start = int(point == 0)
        assert len(frequency) == 8 and np.all(np.isfinite(frequency))
        assert np.min(frequency[start:]) > 0
        assert float(np.max(record["residual"][start:8])) < 1e-5
        assert float(record["seed_agreement"]) < 1e-7
        assert float(record["mass_orthogonality"]) < 1e-7
    return frequency


def fem_controls(folder, h, theory):
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["exit_code"] == 0 and (folder / "model_Model.mph").is_file()
    assert manifest.get("status") not in ("FAIL", "RUNNING")
    settings = manifest["settings"]
    assert float(settings["COMSOL_LATTICE_UM"]) == A_UM
    assert abs(float(settings["COMSOL_THICKNESS_UM"]) - h) < 1e-9
    assert float(settings["COMSOL_MCST_LENGTH_UM"]) == (0 if theory == "classic" else 1)
    assert settings["COMSOL_MESH_LEVEL"] == "2"
    assert settings["COMSOL_DRY_SWEEP_LAYERS"] == "4"
    rows = list(csv.DictReader((folder / "frequencies.csv").open(newline="")))
    controls, changed = {}, []
    for point, ky in (("X", 0), ("M", 1)):
        modes = sorted((row for row in rows
                        if abs(float(row["kx_pi_over_a"]) - 1) < 1e-9
                        and abs(float(row["ky_pi_over_a"]) - ky) < 1e-9
                        and float(row["frequency_mhz"]) > 0),
                       key=lambda row:float(row["frequency_mhz"]))
        selected = [row for row in modes if float(row["vertical_fraction"]) > CUTOFF][:6]
        strict = [row for row in modes if float(row["vertical_fraction"]) > 0.8][:6]
        assert len(selected) == 6, (folder, point, len(selected))
        assert all(abs(float(row["imag_frequency_mhz"])) /
                   float(row["frequency_mhz"]) < 1e-7 for row in selected)
        controls[point] = np.array([float(row["frequency_mhz"]) for row in selected])
        if len(strict) < 6 or any(left is not right for left, right in zip(selected, strict)):
            changed.append(point)
    return controls, changed


def plot_bands(h, spectra, gap, name):
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for plate, ell, label, color, linestyle in STYLES:
        values = np.vstack((spectra[h, plate, ell], spectra[h, plate, ell][0]))
        for band in range(6):
            ax.plot(range(13), values[:, band], color=color, ls=linestyle,
                    lw=1.0, label=label if band == 0 else None)
    if gap["J45_percent"] > 0:
        ax.axhspan(gap["lower_MHz"], gap["upper_MHz"], color="#77AC30", alpha=.15)
    ax.text(.02, .98, f"MCST-TSDT: J = {gap['J45_percent']:.2f}%",
            transform=ax.transAxes, va="top", fontsize=7)
    ax.set(xticks=(0, 4, 8, 12), xticklabels=(r"$\Gamma$", "X", "M", r"$\Gamma$"),
           xlim=(0, 12), ylim=(0, None), ylabel="Frequency (MHz)")
    ax.legend(ncol=2, fontsize=6, frameon=False, loc="upper center",
              bbox_to_anchor=(.5, 1.29))
    ax.grid(alpha=.12)
    fig.savefig(ASSETS / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metric_gap(gaps):
    fig, ax = plt.subplots(figsize=(2.55, 2.12))
    for plate, ell, label, color, linestyle in STYLES:
        series = [next(row["J45_percent"] for row in gaps
                       if row["h_over_a"] == ratio and row["plate"] == plate
                       and row["ell_um"] == ell) for ratio in RATIOS]
        ax.plot(RATIOS, series, color=color, ls=linestyle, marker="o",
                lw=1.05, ms=2.5, label=label)
    ax.axhline(0, color=".5", lw=.7)
    ax.set_xscale("log")
    ax.set(xlabel=r"$h/a$", ylabel=r"Signed $J_{45}$ (%)")
    ax.xaxis.set_major_locator(FixedLocator((.02, .04, .08, .20, .32)))
    ax.xaxis.set_major_formatter(FixedFormatter((".02", ".04", ".08", ".20", ".32")))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.get_offset_text().set_visible(False)
    ax.grid(alpha=.18)
    ax.legend(ncol=2, fontsize=5.5, frameon=False, loc="upper center",
              bbox_to_anchor=(.5, 1.27))
    fig.tight_layout(pad=.25)
    fig.savefig(ASSETS / "ratio75_thickness_gap45.pdf")
    plt.close(fig)


def plot_metric_error(comparison, point):
    fig, ax = plt.subplots(figsize=(2.55, 2.12))
    for plate, ell, label, color, linestyle in STYLES:
        theory = "classic" if ell == 0 else "mcst"
        series = [max(row["error_percent"] for row in comparison
                      if row["h_over_a"] == ratio and row["theory"] == theory
                      and row["plate"] == plate and row["point"] == point)
                  for ratio in RATIOS]
        assert all(math.isfinite(value) and value > 0 for value in series)
        ax.plot(RATIOS, series, color=color, ls=linestyle,
                marker="s" if plate == "FSDT" else "o", ms=2.5, lw=1.05)
    ax.set_xscale("log")
    ax.set(xlabel=r"$h/a$", ylabel="Max. frequency error (%)")
    ax.xaxis.set_major_locator(FixedLocator((.02, .04, .08, .20, .32)))
    ax.xaxis.set_major_formatter(FixedFormatter((".02", ".04", ".08", ".20", ".32")))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.get_offset_text().set_visible(False)
    ax.grid(alpha=.18)
    fig.tight_layout(pad=.25)
    fig.savefig(ASSETS / f"ratio75_thickness_error_{point}.pdf")
    plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    assert json.loads((PWE / "provenance.json").read_text())["status"] == "PASS_PWE_COMPUTED"
    assert json.loads((FEM / "completion.json").read_text())["status"] == "ALL_RATIO75_XM_VALIDATED"
    assert json.loads((PWE / "N21/completion.json").read_text())["status"] == "PASS_N21_XM_ONLY"
    cfg = json.loads((PWE / "config.json").read_text())
    assert cfg["a_um"] == A_UM and cfg["h_over_a"] == list(RATIOS)
    assert all(json.loads((PWE / f"N{N}/completion.json").read_text())["frequency_count"]
               == 7 * 3 * segments * 4 * 8 for N, segments in ((9, 1), (15, 4)))
    with (PWE / "N15/spectra.csv").open(newline="") as stream:
        raw_pwe = list(csv.DictReader(stream))
    assert len(raw_pwe) == 2688
    nonzero = [row for row in raw_pwe if float(row["frequency_MHz"]) > 0]
    max_residual = max(float(row["residual"]) for row in nonzero)
    max_seed_difference = max(float(row["seed_agreement"]) for row in raw_pwe)
    max_mass_orthogonality = max(float(row["mass_orthogonality"]) for row in raw_pwe)
    assert max_residual < 1e-5 and max_seed_difference < 1e-7
    assert max_mass_orthogonality < 1e-7

    spectra, gaps, comparison, order_changes, higher_order_changes, selection_changes = {}, [], [], [], [], []
    for ratio in RATIOS:
        h = A_UM * ratio
        for plate, ell, label, color, linestyle in STYLES:
            points = np.array([pwe_frequency(15, h, point, plate, ell)
                               for point in range(12)])
            spectra[h, plate, ell] = points
            lower, upper = float(max(points[:, 3])), float(min(points[:, 4]))
            gaps.append(dict(h_over_a=ratio, h_um=h, plate=plate, ell_um=ell,
                             lower_MHz=lower, upper_MHz=upper,
                             J45_percent=200 * (upper-lower)/(upper+lower)))
            for name, p15, p9 in (("X", 4, 1), ("M", 8, 2)):
                coarse = pwe_frequency(9, h, p9, plate, ell)[:6]
                fine = points[p15, :6]
                for band, (f9, f15) in enumerate(zip(coarse, fine), 1):
                    order_changes.append(dict(h_over_a=ratio, point=name,
                                              plate=plate, ell_um=ell, band=band,
                                              N9_MHz=float(f9), N15_MHz=float(f15),
                                              change_percent=100*abs(float(f9/f15)-1)))
                higher = pwe_frequency(21, h, p15, plate, ell)[:6]
                for band, (f15, f21) in enumerate(zip(fine, higher), 1):
                    higher_order_changes.append(dict(h_over_a=ratio, point=name,
                                                     plate=plate, ell_um=ell, band=band,
                                                     N15_MHz=float(f15), N21_MHz=float(f21),
                                                     change_percent=100*abs(float(f15/f21)-1)))
        for theory in ("classic", "mcst"):
            reused = ratio == .08 and theory == "mcst"
            folder = RATIO / "mcst_a75" if reused else FEM / f"{theory}_ratio{ratio:g}"
            controls, changed = fem_controls(folder, h, theory)
            for point in changed:
                selection_changes.append(dict(h_over_a=ratio, theory=theory,
                                              point=point, threshold_0_8_changes_selection=True))
            ell = 0 if theory == "classic" else 1
            for plate in ("FSDT", "TSDT"):
                for point, index in (("X", 4), ("M", 8)):
                    pwe = pwe_frequency(21, h, index, plate, ell)[:6]
                    fem = controls[point]
                    for band, (fp, ff) in enumerate(zip(pwe, fem), 1):
                        comparison.append(dict(h_over_a=ratio, h_um=h,
                                               theory=theory, plate=plate,
                                               point=point, band=band,
                                               pwe_MHz=float(fp), fem_MHz=float(ff),
                                               error_percent=100*abs(float(fp/ff)-1)))
    assert (len(gaps) == 28 and len(comparison) == 336
            and len(order_changes) == 336 and len(higher_order_changes) == 336)
    save_csv(BASE / "gaps.csv", gaps)
    save_csv(BASE / "comparison.csv", comparison)
    save_csv(BASE / "order_change.csv", order_changes)
    save_csv(BASE / "higher_order_change.csv", higher_order_changes)
    if selection_changes:
        save_csv(BASE / "selection_changes.csv", selection_changes)

    plt.rcParams.update({"font.family":"Times New Roman", "mathtext.fontset":"stix",
                         "font.size":8.3, "axes.linewidth":.75, "pdf.fonttype":42,
                         "xtick.direction":"in", "ytick.direction":"in"})
    for index, ratio in enumerate(DISPLAY):
        h = A_UM * ratio
        gap = next(row for row in gaps if row["h_over_a"] == ratio
                   and row["plate"] == "TSDT" and row["ell_um"] == 1)
        plot_bands(h, spectra, gap, f"ratio75_thickness_bands_{index}")
    plot_metric_gap(gaps)
    for point in ("X", "M"):
        plot_metric_error(comparison, point)
    assessment = dict(status="PASS_ACTUAL_A75_PWE_AND_M2L4_XM", a_um=A_UM,
                      h_over_a=list(RATIOS), displayed_ratios=list(DISPLAY),
                      pwe_N15_path_points=12, comparison_rows=len(comparison),
                      gap_rows=len(gaps),
                      max_pwe_nonzero_residual=max_residual,
                      max_independent_start_difference=max_seed_difference,
                      max_mass_orthogonality_error=max_mass_orthogonality,
                      max_N9_to_N15_XM_first6_percent=max(row["change_percent"]
                                                            for row in order_changes),
                      max_N15_to_N21_XM_first6_percent=max(row["change_percent"]
                                                             for row in higher_order_changes),
                      max_XM_frequency_error_percent=max(row["error_percent"]
                                                         for row in comparison),
                      vertical_fraction_cutoff=CUTOFF,
                      cutoff_0_8_selection_changes=selection_changes,
                      NOT_RUN=["3D full-path band edges", "full Brillouin zone",
                               "per-thickness joint 3D mesh refinement",
                               "higher-N full-path convergence"])
    (BASE / "assessment.json").write_text(json.dumps(assessment, indent=2))
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
