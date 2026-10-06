"""Make compact X/M panels from validated comparison CSVs; no eigenproblem is solved."""

from pathlib import Path
import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RATIO = ROOT / "runs/fixed_ratio_scale_20260922T175850279921Z"
THICKNESS = ROOT / "runs/thickness_M2L4_20260924"
ASSETS = PROJECT / "output/pdf/modal_selectivity_assets"
STYLES = (
    ("classic", "FSDT", "#0072BD", "--", "s"),
    ("classic", "TSDT", "#56B4E9", "-", "o"),
    ("mcst", "FSDT", "#D55E00", "-.", "s"),
    ("mcst", "TSDT", "#7E2F8E", ":", "o"),
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_panel(rows, point, axis, output):
    fig, ax = plt.subplots(figsize=(2.55, 2.12))
    extrema = []
    for theory, plate, color, linestyle, marker in STYLES:
        series = []
        for value in axis:
            group = [row for row in rows
                     if row["point"] == point
                     and row["three_dimensional"] == theory
                     and row["plate"] == plate
                     and float(row["coordinate"]) == value]
            assert len(group) == 6, (output, point, theory, plate, value, len(group))
            series.append(max(float(row["error_percent"]) for row in group))
        assert all(math.isfinite(value) and value > 0 for value in series)
        extrema.extend(series)
        ax.plot(axis, series, color=color, ls=linestyle, marker=marker,
                ms=2.5, lw=1.05)
    if output.name.startswith("ratio_"):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks((10, 25, 100, 500, 2000))
        ax.set_xticklabels(("10", "25", "100", "500", "2000"))
        ax.set_xlabel(r"$a$ ($\mu$m)")
    else:
        ax.set_xlim(0.4, 8.3)
        ax.set_xticks((0.5, 1, 2, 3, 5, 8))
        ax.set_xlabel(r"$h$ ($\mu$m)")
    ax.set_ylabel("Max. frequency error (%)")
    ax.grid(alpha=.18, which="both")
    fig.tight_layout(pad=.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return max(extrema)


def main():
    ratio_audit = json.loads((RATIO / "audit.json").read_text(encoding="utf-8"))
    thick_audit = json.loads((THICKNESS / "assessment.json").read_text(encoding="utf-8"))
    assert ratio_audit["status"] == "COMPLETE_FIXED_MESH_CHECKS"
    assert thick_audit["status"] == "PASS_ACTUAL_M2L4_XM_COMPARISON"
    assert thick_audit["vertical_fraction_cutoff"] == .7
    ratio_raw = read_csv(RATIO / "comparison.csv")
    thick_raw = read_csv(THICKNESS / "comparison.csv")
    assert len(ratio_raw) == 480 and len(thick_raw) == 336

    ratio = [dict(point=row["point"], three_dimensional=("classic" if int(row["ell_um"]) == 0 else "mcst"),
                  plate=row["theory"], coordinate=row["a_um"],
                  error_percent=row["error_percent"]) for row in ratio_raw]
    thickness = [dict(point=row["point"], three_dimensional=row["three_dimensional"],
                      plate=row["plate"], coordinate=row["h_um"],
                      error_percent=row["absolute_error_percent"]) for row in thick_raw]
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                         "font.size": 8.3, "axes.linewidth": .75, "pdf.fonttype": 42,
                         "xtick.direction": "in", "ytick.direction": "in"})
    produced = {}
    for prefix, rows, axis in (("ratio_scale", ratio, (10, 15, 25, 40, 75, 100, 200, 500, 1000, 2000)),
                               ("thickness", thickness, (.5, 1, 1.5, 2, 3, 5, 8))):
        for point in ("X", "M"):
            name = f"{prefix}_error_{point}"
            produced[name] = plot_panel(rows, point, axis, ASSETS / f"{name}.pdf")
    audit = {"status": "PASS_EXISTING_COMPARISONS_ONLY", "ratio_rows": len(ratio),
             "thickness_rows": len(thickness), "panel_max_percent": produced,
             "NOT_RUN": ["new PWE solve", "new COMSOL solve"]}
    (ASSETS / "six_panel_control_errors_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
