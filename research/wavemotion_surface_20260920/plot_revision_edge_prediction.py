"""Plot actual inverse-PWE energy fractions and frozen X-edge predictions."""

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


def main():
    frozen = json.loads((RUN / "edge_prediction_frozen.json").read_text(encoding="utf-8"))
    with (RUN / "prediction_checks.csv").open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [r for r in all_rows if r["baseline_band"] == "4"]
    assert len(rows) == 8
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .8, "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    baseline = frozen["baseline_X_J45_percent"]
    for panel, parameter, title in (
        (axes[0], "ell", "(a) Internal length"),
        (axes[1], "h", "(b) Thickness"),
    ):
        subset = sorted((r for r in rows if r["parameter"] == parameter),
                        key=lambda r: float(r["fractional_change"]))
        xx = np.array([100*float(r["fractional_change"]) for r in subset])
        yy_pred = np.array([float(r["predicted_X_J45_percent"])-baseline for r in subset])
        yy_actual = np.array([float(r["actual_X_J45_percent"])-baseline for r in subset])
        panel.plot(xx, yy_pred, "--o", color="#0072B2", markersize=4,
                   label="Frozen linear prediction")
        panel.plot(xx, yy_actual, "-s", color="#D55E00", markersize=4,
                   label="Re-solved X modes")
        panel.axhline(0, color="#69737b", linewidth=.7)
        panel.set(xlabel="Parameter change (%)",
                  ylabel=r"$\Delta J_{45}(X)$ (percentage points)",
                  title=title, xlim=(-3.4, 3.4))
        if parameter == "ell":
            panel.legend(frameon=False, fontsize=7, loc="best")
    for panel in axes:
        panel.grid(alpha=.18, axis="y")
    png = RUN / "figures/edge_prediction.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf = PROJECT / "output/pdf/modal_selectivity_assets/edge_prediction.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    parameter_errors = {}
    for parameter in ("ell", "h"):
        subset = [r for r in rows if r["parameter"] == parameter]
        absolute = [abs(float(r["error_X_J45_percentage_points"])) for r in subset]
        relative = [
            100 * abs(float(r["error_X_J45_percentage_points"]))
            / abs(float(r["actual_X_J45_percent"]) - baseline)
            for r in subset
        ]
        parameter_errors[parameter] = {
            "max_abs_error_percentage_points": max(absolute),
            "max_change_relative_error_percent": max(relative),
            "signs_agree": all(
                np.sign(float(r["predicted_X_J45_percent"]) - baseline)
                == np.sign(float(r["actual_X_J45_percent"]) - baseline)
                for r in subset
            ),
        }
    result = {
        "status": "PASS_PLOTTED_FROM_FROZEN_AND_RECOMPUTED",
        "source_rows": len(all_rows),
        "parameter_errors": parameter_errors,
        "max_abs_delta_J_error_percentage_points": max(
            abs(float(r["error_X_J45_percentage_points"])) for r in rows),
        "scope": "X-point pair only; full IBZ not re-searched under perturbation",
        "png": str(png), "pdf": str(pdf),
    }
    (RUN / "edge_prediction_plot_assessment.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
