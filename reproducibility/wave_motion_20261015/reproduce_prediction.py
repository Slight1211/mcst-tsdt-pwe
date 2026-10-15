#!/usr/bin/env python3
"""Reproduce the numerical predictions underlying Figure 9 and Table A1.

This is a newly authored, portable postprocessing implementation, not a byte
copy of an original research script. It uses only four files already published
in data/wave_motion_20261009_v1 at Git commit
998833149606a0a9324d133f324eb3ceae6f78b2. Python 3.10+ and its standard library
are sufficient. No network access, installation, subprocess, or eigensolve is
performed; this program does not render the figure.

Example (run from the repository root):
    python reproducibility/wave_motion_20261015/reproduce_prediction.py \
        --data-root data/wave_motion_20261009_v1 --output prediction_results

The output directory must not exist. The input directory is never modified.
Historical prediction columns in the input files are deliberately unused.
Analytic S values are fixed; T is fitted only to the actual -1%/+1% frequencies.
The existing +/-2% and +/-3% frequencies are excluded from this fit. This is a
post-hoc evaluation, not a new prospective prediction or solver validation.

All results concern fixed X, bands 4 and 5, inverse MCST-TSDT N=15, for the
circular reference geometry (a=75 um, h=6 um, r/a=0.3, epoxy ell=1 um).
The perturbed full-IBZ extrema are not re-searched. J is the fixed-X relative
bandwidth in percent; changes/errors in J are percentage points (pp).
"""

import argparse
import csv
import json
import math
from pathlib import Path


PUBLIC_COMMIT = "998833149606a0a9324d133f324eb3ceae6f78b2"
INPUTS = {
    "baseline": "assets/item-032.json",
    "analytic_thickness_slopes": "assets/item-039.json",
    "frequencies_1_and_3_percent": "assets/item-043.csv",
    "thickness_frequencies_2_percent": "assets/item-015.csv",
}
PARAMETERS = ("ell", "h")
BANDS = (4, 5)
CALIBRATION_CHANGES = (-0.01, 0.01)
RAW_CHANGES = (-0.03, -0.01, 0.01, 0.03)


def finite(value, name):
    """Parse a finite number without accepting JSON booleans as numbers."""
    if isinstance(value, bool):
        raise ValueError(f"{name}: expected a number, not a boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}: non-finite number")
    return result


def close(actual, expected, name, tolerance=1e-12):
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{name}: inconsistent input values")


def frequency_pair(values, name):
    if len(values) != 2:
        raise ValueError(f"{name}: exactly two frequencies required")
    pair = tuple(finite(value, name) for value in values)
    if not 0 < pair[0] < pair[1]:
        raise ValueError(f"{name}: require 0 < f4 < f5")
    return pair


def bandwidth(frequencies):
    f4, f5 = frequencies
    return 200.0 * (f5 - f4) / (f5 + f4)


def read_csv(path, required):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name}: required CSV columns missing")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path.name}: malformed CSV row")
    return rows


def read_inputs(data_root):
    """Load public actual frequencies and analytic slopes, not old predictions."""
    with (data_root / INPUTS["baseline"]).open(encoding="utf-8-sig") as stream:
        baseline = json.load(stream)
    with (data_root / INPUTS["analytic_thickness_slopes"]).open(encoding="utf-8-sig") as stream:
        analytic = json.load(stream)
    f0 = frequency_pair(baseline["baseline_f4_f5_MHz"], "baseline frequencies")
    j0 = bandwidth(f0)
    close(j0, finite(baseline["baseline_X_J45_percent"], "baseline J"), "baseline J")
    ell_slopes = baseline["R_mc_f4_f5"]
    if len(ell_slopes) != 2:
        raise ValueError("exactly two MCST participation ratios required")
    slopes = {"ell": tuple(finite(s, "S_ell") for s in ell_slopes)}
    if not all(0 <= s <= 1 for s in slopes["ell"]):
        raise ValueError("MCST participation ratios must lie in [0, 1]")
    by_band = {}
    for entry in analytic["bands"]:
        band = finite(entry["band"], "analytic band")
        if band not in BANDS or band in by_band:
            raise ValueError("analytic bands must be exactly 4 and 5, without duplicates")
        slope = finite(entry["S_h_Laurent"], "analytic S_h")
        identity = (finite(entry["kappa_3"], "kappa_3")
                    - finite(entry["kappa_minus1"], "kappa_minus1")
                    - finite(entry["eta_3"], "eta_3"))
        close(slope, identity, "Laurent identity", tolerance=1e-14)
        by_band[band] = slope
    if set(by_band) != set(BANDS):
        raise ValueError("analytic bands 4 and 5 are required")
    slopes["h"] = tuple(by_band[band] for band in BANDS)

    raw_rows = read_csv(data_root / INPUTS["frequencies_1_and_3_percent"], {
        "parameter", "fractional_change", "baseline_band", "matched_band",
        "baseline_f_MHz", "actual_f_MHz", "actual_delta_ln_f",
        "mass_MAC", "max_residual", "seed_agreement",
    })
    raw = {}
    for row in raw_rows:
        parameter = row["parameter"]
        change = finite(row["fractional_change"], "fractional change")
        band = finite(row["baseline_band"], "baseline band")
        key = (parameter, change, band)
        if parameter not in PARAMETERS or change not in RAW_CHANGES or band not in BANDS:
            raise ValueError("unexpected parameter, change, or band in item-043.csv")
        if key in raw:
            raise ValueError("duplicate parameter/change/band in item-043.csv")
        if finite(row["matched_band"], "matched band") != band:
            raise ValueError("input band matching differs from the requested band")
        index = BANDS.index(band)
        close(finite(row["baseline_f_MHz"], "row baseline"), f0[index], "row baseline")
        actual = finite(row["actual_f_MHz"], "actual frequency")
        if actual <= 0:
            raise ValueError("actual frequency must be positive")
        close(math.log(actual / f0[index]), finite(row["actual_delta_ln_f"], "actual log shift"),
              "actual log shift")
        if not 0.99 <= finite(row["mass_MAC"], "mass MAC") <= 1.0 + 1e-12:
            raise ValueError("recorded mass MAC fails the input quality check")
        for name in ("max_residual", "seed_agreement"):
            if not 0 <= finite(row[name], name) <= 1e-7:
                raise ValueError(f"recorded {name} fails the input quality check")
        raw[key] = actual
    expected = {(p, d, b) for p in PARAMETERS for d in RAW_CHANGES for b in BANDS}
    if set(raw) != expected:
        raise ValueError("item-043.csv must contain all 16 unique parameter/change/band rows")
    cases = []
    for parameter in PARAMETERS:
        for change in RAW_CHANGES:
            actual = frequency_pair([raw[(parameter, change, band)] for band in BANDS],
                                    "actual frequencies")
            split = "calibration_1pct" if change in CALIBRATION_CHANGES else "evaluation_3pct"
            cases.append((parameter, change, split, actual))

    h2_rows = read_csv(data_root / INPUTS["thickness_frequencies_2_percent"], {
        "fractional_change", "actual_f4_MHz", "actual_f5_MHz", "actual_J_percent",
        "actual_delta_J_pp", "min_mass_MAC", "max_two_seed_residual",
    })
    h2 = {}
    for row in h2_rows:
        change = finite(row["fractional_change"], "h2 change")
        if change not in (-0.02, 0.02) or change in h2:
            raise ValueError("item-015.csv must contain unique -2% and +2% rows")
        actual = frequency_pair([row["actual_f4_MHz"], row["actual_f5_MHz"]], "h2 frequencies")
        close(bandwidth(actual), finite(row["actual_J_percent"], "h2 J"), "h2 J")
        close(bandwidth(actual) - j0, finite(row["actual_delta_J_pp"], "h2 delta J"), "h2 delta J")
        if not 0.99 <= finite(row["min_mass_MAC"], "h2 mass MAC") <= 1.0 + 1e-12:
            raise ValueError("recorded h2 mass MAC fails the input quality check")
        if not 0 <= finite(row["max_two_seed_residual"], "h2 residual") <= 1e-7:
            raise ValueError("recorded h2 residual fails the input quality check")
        h2[change] = actual
    if set(h2) != {-0.02, 0.02}:
        raise ValueError("item-015.csv must contain both -2% and +2% rows")
    cases.extend(("h", change, "evaluation_2pct", h2[change]) for change in (-0.02, 0.02))
    return f0, slopes, raw, cases


def fit_curvatures(f0, slopes, raw):
    """Fit y=S*x+T*x*x/2, with fixed S and no fitted intercept, at +/-1% only."""
    curvatures = {}
    for parameter in PARAMETERS:
        fitted = []
        for index, band in enumerate(BANDS):
            numerator, denominator = 0.0, 0.0
            for change in CALIBRATION_CHANGES:
                x = math.log1p(change)
                y = math.log(raw[(parameter, change, band)] / f0[index])
                numerator += x * x * (y - slopes[parameter][index] * x)
                denominator += x ** 4
            fitted.append(2.0 * numerator / denominator)
        curvatures[parameter] = tuple(fitted)
    return curvatures


def prediction_row(f0, slopes, curvatures, case):
    parameter, change, split, actual = case
    x = math.log1p(change)
    j0, ja = bandwidth(f0), bandwidth(actual)
    actual_increment = ja - j0
    if actual_increment == 0:
        raise ValueError("relative increment error is undefined for zero actual bandwidth change")
    row = {"parameter": parameter, "fractional_change": change, "split": split,
           "actual_f4_MHz": actual[0], "actual_f5_MHz": actual[1],
           "baseline_J_percent": j0, "actual_J_percent": ja,
           "actual_delta_J_pp": actual_increment}
    for index, band in enumerate(BANDS):
        row[f"baseline_f{band}_MHz"] = f0[index]
        row[f"actual_delta_ln_f{band}"] = math.log(actual[index] / f0[index])
    for order, quadratic in (("first", False), ("second", True)):
        shifts = tuple(slopes[parameter][i] * x
                       + (0.5 * curvatures[parameter][i] * x * x if quadratic else 0.0)
                       for i in range(2))
        predicted = tuple(f0[i] * math.exp(shifts[i]) for i in range(2))
        jp = bandwidth(predicted)
        increment, error = jp - j0, jp - ja
        row.update({f"{order}_J_percent": jp, f"{order}_delta_J_pp": increment,
                    f"{order}_signed_J_error_pp": error, f"{order}_abs_J_error_pp": abs(error),
                    f"{order}_relative_increment_error_percent": 100.0 * abs(error) / abs(actual_increment),
                    f"{order}_direction_agrees": (increment > 0) == (actual_increment > 0)
                    and increment != 0})
        for index, band in enumerate(BANDS):
            frequency_error = predicted[index] - actual[index]
            row.update({f"{order}_f{band}_MHz": predicted[index],
                        f"{order}_delta_ln_f{band}": shifts[index],
                        f"{order}_log_f{band}_error": math.log(predicted[index] / actual[index]),
                        f"{order}_signed_f{band}_error_MHz": frequency_error,
                        f"{order}_abs_f{band}_error_MHz": abs(frequency_error),
                        f"{order}_relative_f{band}_error_percent":
                        100.0 * abs(frequency_error) / actual[index]})
    if any(not math.isfinite(v) for v in row.values() if isinstance(v, (int, float))):
        raise ValueError("non-finite derived result")
    return row


def summarize(rows):
    summaries = {}
    for parameter in PARAMETERS:
        summaries[parameter] = {}
        splits = sorted({row["split"] for row in rows if row["parameter"] == parameter})
        for split in splits:
            selected = [r for r in rows if r["parameter"] == parameter and r["split"] == split]
            summaries[parameter][split] = {}
            for order in ("first", "second"):
                summaries[parameter][split][order] = {
                    "max_abs_J_error_pp": max(r[f"{order}_abs_J_error_pp"] for r in selected),
                    "max_relative_increment_error_percent":
                        max(r[f"{order}_relative_increment_error_percent"] for r in selected),
                    "max_abs_log_frequency_error":
                        max(abs(r[f"{order}_log_f{b}_error"]) for r in selected for b in BANDS),
                    "max_relative_frequency_error_percent":
                        max(r[f"{order}_relative_f{b}_error_percent"] for r in selected for b in BANDS),
                    "all_directions_agree": all(r[f"{order}_direction_agrees"] for r in selected),
                }
    return summaries


def derive(data_root):
    f0, slopes, raw, cases = read_inputs(data_root)
    curvatures = fit_curvatures(f0, slopes, raw)
    rows = [prediction_row(f0, slopes, curvatures, case) for case in cases]
    coefficients = [{"parameter": p, "band": b, "S_dimensionless": slopes[p][i],
                     "T_dimensionless": curvatures[p][i]}
                    for p in PARAMETERS for i, b in enumerate(BANDS)]
    report = {
        "implementation": "New portable postprocessing implementation authored 2026-10-15",
        "public_input_commit": PUBLIC_COMMIT,
        "input_collection": "data/wave_motion_20261009_v1",
        "input_files": INPUTS,
        "scope": "Fixed X, bands 4/5, circular reference, inverse MCST-TSDT N15; a=75 um, h=6 um, r/a=0.3, epoxy ell=1 um",
        "interpretation": "Post-hoc analytic-slope evaluation of existing frequencies; no new prospective freeze, eigensolve, or full-IBZ search",
        "quality_check_scope": "Checks existing published MAC/residual/seed records, without recomputing eigenvectors or solver diagnostics",
        "slope_sources": {"h": "item-039.json bands[].S_h_Laurent",
                          "ell": "item-032.json R_mc_f4_f5"},
        "calibration_fractional_changes": list(CALIBRATION_CHANGES),
        "excluded_from_fit": "All +/-2% and +/-3% frequencies",
        "historical_prediction_columns": "Not used",
        "units": {"frequency": "MHz", "J": "percent", "delta_J_and_J_error": "percentage points (pp)",
                  "S_T_and_log_frequency_changes": "dimensionless", "relative_errors": "percent"},
        "formulas": {
            "x": "ln(1+fractional_change)", "y": "ln(f_actual/f0)",
            "S_h": "kappa_3-kappa_minus1-eta_3",
            "T": "2*sum(x_i^2*(y_i-S*x_i))/sum(x_i^4), i=-1%,+1%",
            "first_frequency": "f0*exp(S*x)", "second_frequency": "f0*exp(S*x+T*x^2/2)",
            "J_percent": "200*(f5-f4)/(f5+f4)", "delta_J_pp": "J-J0",
            "signed_J_error_pp": "J_prediction-J_actual",
            "relative_increment_error_percent": "100*abs(J_prediction-J_actual)/abs(J_actual-J0)",
            "log_frequency_error": "ln(f_prediction/f_actual)",
            "relative_frequency_error_percent": "100*abs(f_prediction-f_actual)/f_actual",
        },
        "baseline_f4_f5_MHz": list(f0), "baseline_J_percent": bandwidth(f0),
        "coefficients": coefficients, "summaries": summarize(rows),
        "row_count": len(rows),
    }
    return rows, coefficients, report


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Path to the published data/wave_motion_20261009_v1 directory (containing assets/)")
    parser.add_argument("--output", required=True, type=Path, help="New, non-existing output directory")
    args = parser.parse_args(argv)
    try:
        data_root = args.data_root.expanduser().resolve(strict=True)
        output_requested = args.output.expanduser()
        if output_requested.exists() or output_requested.is_symlink():
            raise ValueError("output directory already exists; choose a new directory")
        output = output_requested.resolve()
        if output == data_root or data_root in output.parents:
            raise ValueError("output must be outside the input data directory")
        rows, coefficients, report = derive(data_root)
        # Validate serializability before creating anything; never replace existing files.
        report_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        output.mkdir(parents=True, exist_ok=False)
        write_csv(output / "predictions.csv", rows)
        write_csv(output / "coefficients.csv", coefficients)
        with (output / "summary.json").open("x", encoding="utf-8") as stream:
            stream.write(report_text)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        parser.error(str(error))
    print(f"Wrote {len(rows)} prediction rows, {len(coefficients)} coefficient rows, and summary.json.")
    print("No eigensolves or network access were performed.")


if __name__ == "__main__":
    main()
