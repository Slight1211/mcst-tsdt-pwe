#!/usr/bin/env python3
"""Check saved homogeneous MCST results; no COMSOL or eigensolver is invoked.

The input files are complete, unchanged exports of all four prescribed cases.
This publication postprocessor is newly authored; it is not the historical
solver or a byte-for-byte copy of the private validation runner.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError("Non-finite numeric input: " + key)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).parent / "uniform")
    parser.add_argument("--output", required=True, type=Path,
                        help="New directory; existing paths are refused")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new directory")

    a, young, nu, rho = 100e-6, 4.35e9, 4.35 / (2 * 1.59) - 1, 1180.0
    mu = young / (2 * (1 + nu))
    lame = young * nu / ((1 + nu) * (1 - 2 * nu))
    k2 = sum((math.pi * x / a) ** 2 for x in (0.18, 0.11, 0.07))
    case_results, summary = [], []
    accepted = True
    for n in (4, 6):
        both = {}
        frequency_errors, constraints = [], []
        for suffix, ratio in (("0p05", 0.05), ("0p10", 0.10)):
            prefix = args.data_root / ("bulk_n%d_ell%s" % (n, suffix))
            selected = rows(Path(str(prefix) + ".selected.csv"))
            checks = rows(Path(str(prefix) + ".checks.csv"))
            mesh = rows(Path(str(prefix) + ".mesh.csv"))
            selections = rows(Path(str(prefix) + ".selection.csv"))
            discretization = rows(Path(str(prefix) + ".discretization.csv"))
            if len(checks) != 1 or len(selected) != 3:
                raise ValueError("Expected one check row and three selected modes")
            if {r["type"] for r in selected} != {"T1", "T2", "L"}:
                raise ValueError("Missing or duplicate selected mode type")
            check = checks[0]
            tests = []
            tests.append(all(int(r["n"]) == n and
                             number(r, "ell_over_a") == ratio
                             for r in selected + checks))
            tests.append(int(check["actual_hex_elements"]) == n ** 3 ==
                         int(check["expected_hex_elements"]))
            tests.append(int(check["other_volume_elements"]) == 0)
            tests.append(int(check["integration_order"]) == 8)
            tests.append(sum(int(r["actual_count"]) for r in mesh
                             if r["element_type"] == "hex") == n ** 3)
            tests.append(all(int(r["actual_count"]) == 0 for r in mesh
                             if r["topological_dimension"] == "3" and
                             r["element_type"] != "hex"))
            expected_selection = {"lo" + x: (2, 1) for x in "xyz"}
            expected_selection.update({"hi" + x: (2, 1) for x in "xyz"})
            expected_selection.update({"pair" + x: (2, 2) for x in "xyz"})
            expected_selection["bottomEdges"] = (1, 4)
            tests.append(len(selections) == len(expected_selection) and
                         {r["selection"]: (int(r["dimension"]), int(r["actual_count"]))
                          for r in selections} == expected_selection)
            tests.append(all(r["passed"].lower() == "true" and
                             r["actual_count"] == r["expected_count"] for r in selections))
            orders = {r["physics"]: r["actual_value"] for r in discretization
                      if r["shape_property"] == "order_displacement"}
            tests.append(orders == {"solid": "2", "mcstRot": "2", "mcstLag": "1"})
            tests.append(check["passed"].lower() == "true")
            tests.append(number(check, "transverse_split") <=
                         number(check, "transverse_split_tolerance"))
            tests.append(1 <= number(check, "transverse_amplitude_gram_condition") <=
                         number(check, "transverse_amplitude_gram_condition_tolerance"))
            ell = ratio * a
            omega_t2 = mu / rho * (k2 + ell ** 2 * k2 ** 2 / 4)
            omega_l2 = (lame + 2 * mu) / rho * k2
            for row in selected:
                transverse = row["type"].startswith("T")
                f = number(row, "frequency_hz")
                exact = math.sqrt(omega_t2 if transverse else omega_l2) / (2 * math.pi)
                error = abs(f / exact - 1)
                frequency_errors.append(error)
                tests.extend((f > 0, error <= number(check, "frequency_tolerance"),
                              math.isclose(number(row, "analytic_frequency_hz"), exact,
                                           rel_tol=1e-13),
                              math.isclose(number(row, "relative_frequency_error"), error,
                                           rel_tol=1e-6, abs_tol=1e-15),
                              number(row, "relative_imaginary_frequency") <= 1e-8,
                              row["passed"].lower() == "true",
                              number(check, "minimum_g0_projection_fraction") <=
                              number(row, "g0_projection_fraction") <= 1 + 1e-8))
                polarization = number(row, "longitudinal_fraction")
                if transverse:
                    ec = number(row, "constraint_error_rotation_normalized")
                    constraints.append(ec)
                    individual_increment = abs(((2 * math.pi * f) ** 2 - mu / rho * k2)
                                               / (omega_t2 - mu / rho * k2) - 1)
                    tests.extend((0 <= polarization <= number(check, "polarization_tolerance"),
                                  0 <= ec <= number(check, "transverse_constraint_tolerance"),
                                  individual_increment <= number(check, "mcst_increment_tolerance"),
                                  math.isclose(individual_increment,
                                               number(row, "mcst_increment_relative_error"),
                                               rel_tol=1e-7, abs_tol=1e-12)))
                else:
                    tests.extend((1 - number(check, "polarization_tolerance") <= polarization <= 1,
                                  row["constraint_error_rotation_normalized"] == "",
                                  row["mcst_increment_relative_error"] == "",
                                  number(row, "constraint_error_k_u_normalized") <=
                                  number(check, "longitudinal_scaled_tolerance"),
                                  number(row, "rotation_over_k_u") <=
                                  number(check, "longitudinal_scaled_tolerance")))
            both[ratio] = sum((2 * math.pi * number(r, "frequency_hz")) ** 2
                              for r in selected if r["type"].startswith("T")) / 2
            passed = all(tests)
            accepted = accepted and passed
            case_results.append({"n": n, "ell_over_a": ratio,
                                 "checks_evaluated": len(tests), "passed": passed})
        analytic_increment = mu * k2 ** 2 * ((0.10 * a) ** 2 - (0.05 * a) ** 2) / (4 * rho)
        paired_error = abs((both[0.10] - both[0.05]) / analytic_increment - 1)
        summary.append({"hexahedra": n ** 3, "maximum_relative_frequency_error": max(frequency_errors),
                        "paired_length_increment_relative_error": paired_error,
                        "maximum_transverse_constraint_percent": 100 * max(constraints)})

    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "uniform_summary_recomputed.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (args.output / "verification.json").open("x", encoding="utf-8") as stream:
        json.dump({"scope": "Saved homogeneous-solid exports; no new eigenproblem solved",
                   "cases": case_results, "passed": accepted}, stream, indent=2)
        stream.write("\n")
    print("PASS" if accepted else "FAIL", "four saved cases; summary written")
    raise SystemExit(0 if accepted else 1)


if __name__ == "__main__":
    main()
