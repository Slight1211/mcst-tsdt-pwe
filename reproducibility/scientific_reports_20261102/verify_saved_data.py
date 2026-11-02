"""Verify deposited spectra and Figure 6 without solving or extracting a ZIP.

The default path uses only Python's standard library. NumPy, SciPy and COMSOL
are not imported. Optional figure regeneration imports Matplotlib separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

ARCHIVE_SHA256 = "3d88c5882cd2e12cc5e824caf96e56f9b3ea19341d3c8ba3f2b68a52c21b269f"
MEMBER_INDEX_SHA256 = "87d881582d61425d3b5da233ec2afe22485e92abe2cd00fbec3abeec673b151a"
DOI = "10.5281/zenodo.23888401"
FIGURE_CSV = "Data5/Figure6_design_story_v19.csv"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_member(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return p


class DataReader:
    """Read the exact published ZIP, or its payload-hash-verified extraction."""

    def __init__(self, *, archive=None, data_dir=None):
        self.archive = None
        self.data_dir = None
        if archive is not None:
            path = Path(archive).resolve()
            if file_digest(path) != ARCHIVE_SHA256:
                raise ValueError("Archive SHA256 does not match the published version 1.0.0")
            self.archive = zipfile.ZipFile(path)
            names = self.archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP member names")
            for name in names:
                safe_member(name)
            bad = self.archive.testzip()
            if bad is not None:
                raise ValueError(f"ZIP CRC failed: {bad}")
        else:
            self.data_dir = Path(data_dir).resolve()
            if not self.data_dir.is_dir():
                raise ValueError("--data-dir must be an extracted Data1–Data8 directory")
        index_bytes = self.read("MEMBER_SHA256.csv")
        if digest(index_bytes) != MEMBER_INDEX_SHA256:
            raise ValueError("Member index does not match the published archive")
        self.index = list(csv.DictReader(io.StringIO(index_bytes.decode("utf-8-sig"))))
        if len(self.index) != 966:
            raise ValueError("Expected exactly 966 scientific/provenance payload members")
        self.by_digest = defaultdict(list)
        for row in self.index:
            name = row["destination"]
            safe_member(name)
            data = self.read(name)
            if len(data) != int(row["bytes"]) or digest(data) != row["sha256"]:
                raise ValueError(f"Payload bytes/SHA256 mismatch: {name}")
            if self.archive is not None and f"{self.archive.getinfo(name).CRC:08x}" != row["crc32"]:
                raise ValueError(f"Member CRC record mismatch: {name}")
            self.by_digest[row["sha256"]].append(name)

    def read(self, name):
        p = safe_member(name)
        if self.archive is not None:
            return self.archive.read(name)
        path = self.data_dir.joinpath(*p.parts).resolve()
        if not path.is_relative_to(self.data_dir):
            raise ValueError(f"Member resolves outside --data-dir: {name}")
        return path.read_bytes()

    def rows(self, name):
        return list(csv.DictReader(io.StringIO(self.read(name).decode("utf-8-sig"))))

    def json(self, name):
        return json.loads(self.read(name).decode("utf-8-sig"))

    def bound_source(self, original_relative_path, sha256):
        # Renamed public CSV aliases are deliberately resolved by verified digest.
        candidates = self.by_digest.get(sha256, [])
        if not candidates:
            raise ValueError(f"Bound source is absent: {original_relative_path} ({sha256})")
        preferred = [n for n in candidates if n.endswith("/" + original_relative_path)]
        name = sorted(preferred or candidates)[0]
        if digest(self.read(name)) != sha256:
            raise ValueError(f"Source binding failed: {name}")
        return name

    def close(self):
        if self.archive is not None:
            self.archive.close()


def require_close(actual, expected, label, tolerance=1e-9):
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise ValueError(f"{label}: {actual!r} differs from {expected!r}")


def sampled_gap(reader, name, points):
    rows = reader.rows(name)
    if len(rows) != points or points not in (45, 153):
        raise ValueError(f"Unexpected point count: {name}")
    grid = 8 if points == 45 else 16
    expected = [(i, j) for i in range(grid + 1) for j in range(i + 1)]
    for row, (i, j) in zip(rows, expected):
        require_close(float(row["kx_a"]), math.pi * i / grid, name + " kx", 1e-12)
        require_close(float(row["ky_a"]), math.pi * j / grid, name + " ky", 1e-12)
        frequencies = [float(row[f"Omega{b}"]) for b in range(1, 7)]
        if any(not math.isfinite(v) or v < 0 for v in frequencies):
            raise ValueError(f"Invalid frequencies: {name}")
        if any(a > b + 1e-9 for a, b in zip(frequencies, frequencies[1:])):
            raise ValueError(f"Unsorted frequencies: {name}")
    lower = max(float(row["Omega4"]) for row in rows)
    upper = min(float(row["Omega5"]) for row in rows)
    if lower + upper <= 0:
        raise ValueError(f"Invalid band-edge denominator: {name}")
    return {"L": lower, "U": upper, "J_percent": 200 * (upper - lower) / (upper + lower),
            "points": points, "source": name}


def verify(reader):
    results = {"status": "PASS_SAVED_DATA_ARITHMETIC_AND_BINDINGS", "doi": DOI,
               "archive_version": "1.0.0", "verified_payload_members": len(reader.index),
               "new_eigensolves": 0, "new_optimizations": 0, "new_COMSOL_runs": 0,
               "human_verified": False, "submission_ready": False}
    initial = sampled_gap(reader, "Data2/05_validation/V08_N15_20261030_v3/initial_ell1_N15_ibz45.csv", 45)
    final = sampled_gap(reader, "Data2/05_validation/V08_N15_20261030_v3/final_ell1_N15_ibz45.csv", 45)
    require_close(initial["J_percent"], 27.990629312699348, "ell1 initial")
    require_close(final["J_percent"], 33.22452528639608, "ell1 optimized")
    results["ell1_primary"] = {"initial": initial, "optimized": final,
                              "gain_percentage_points": final["J_percent"] - initial["J_percent"]}
    new = sampled_gap(reader, "Data3/05_validation/V07_ell2_reoptimization_20261030/restart/final_ibz153.csv", 153)
    require_close(new["J_percent"], 27.047699138915647, "ell2 reoptimized")
    manifest = reader.json("Data3/05_validation/V07_ell2_reoptimization_20261030/restart/manifest.json")
    require_close(new["J_percent"], float(manifest["final_ibz153"]["J"]), "ell2 manifest")
    results["ell2_reoptimized"] = new

    five = []
    for row in reader.rows("Data6/Supplementary_Data_6_size_shape_N15.csv"):
        if row["record_type"] != "gap_summary":
            continue
        spectrum = reader.bound_source(row["source_spectrum"], row["source_spectrum_sha256"])
        value = sampled_gap(reader, spectrum, 153)
        require_close(value["J_percent"], float(row["J153"]), "five-state gap")
        five.append({"ell_um": float(row["ell_um"]), "N": int(row["N"]),
                     "search_points": int(row["shape_search_points"]), **value})
    if sorted(x["ell_um"] for x in five) != [0, .5, 1, 2, 5] or any(x["N"] != 15 for x in five):
        raise ValueError("Expected the five identified N15 states")
    results["exploratory_five_state_gaps"] = sorted(five, key=lambda x: x["ell_um"])
    results["five_state_scope"] = "Different search protocols and starts; common 153-point terminal evaluation; not a uniform global optimization study."

    cross = []
    prefix = "Data7/05_validation/ELL0_N21_REOPT_20261101/order_cross_comparison/"
    summary = reader.json(prefix + "order_cross_summary.json")
    for cell in summary["cells"]:
        value = sampled_gap(reader, prefix + cell["spectrum"], 153)
        require_close(value["J_percent"], float(cell["J"]), "N15/N21 cross")
        cross.append({"geometry": cell["shape_label"], "evaluation_N": cell["evaluation_N"], **value})
    if {(x["geometry"], x["evaluation_N"]) for x in cross} != {("g15", 15), ("g15", 21), ("g21", 15), ("g21", 21)}:
        raise ValueError("N15/N21 comparison is incomplete")
    results["ell0_shape_order_cross"] = cross

    endpoint = []
    prefix = "Data8/05_validation/SHAPE_LENGTHSCALE_CROSS_V18_20261101/"
    maps = {("g0", 0): "g0_saved_diagonal_ibz153.csv", ("g0", 5): "g0_at5/g0_at5_ibz153.csv",
            ("g5", 5): "g5_saved_diagonal_ibz153.csv", ("g5", 0): "g5_at0/g5_at0_ibz153.csv"}
    for row in reader.rows(prefix + "cross_2x2_summary.csv"):
        ell = float(row["evaluation_ell_um"])
        value = sampled_gap(reader, prefix + maps[(row["shape"], ell)], 153)
        require_close(value["J_percent"], float(row["J"]), "endpoint shape/material cross")
        endpoint.append({"shape": row["shape"], "ell_um": ell, **value})
    if len(endpoint) != 4:
        raise ValueError("Endpoint 2x2 is incomplete")
    results["endpoint_shape_material_cross"] = endpoint

    figure = reader.rows(FIGURE_CSV)
    counts = Counter(r["record_type"] for r in figure)
    if counts != {"gap_transfer": 8, "gap_reoptimized": 1, "point_frequency_difference": 8,
                  "fixed_geometry_modal_energy": 12}:
        raise ValueError("Figure 6 record counts differ from 8+1+8+12")
    resolved_sources = {}
    for row in figure:
        source = reader.bound_source(row["source"], row["source_sha256"])
        resolved_sources[row["source"]] = source
        kind = row["record_type"]
        if kind in {"gap_transfer", "gap_reoptimized"}:
            lower, upper = float(row["L_Omega"]), float(row["U_Omega"])
            require_close(float(row["J_percent"]), 200 * (upper - lower) / (upper + lower), "Figure6 gap")
            if kind == "gap_transfer":
                original = [r for r in reader.rows(source) if r["shape"] == row["shape"] and float(r["ell_um"]) == float(row["ell_um"])]
                if len(original) != 1:
                    raise ValueError("Figure6 transfer source is not unique")
                original = original[0]
            else:
                original = reader.json(source)["final_ibz45"]
            for public_key, original_key in [("L_Omega", "L"), ("U_Omega", "U"), ("J_percent", "J")]:
                require_close(float(row[public_key]), float(original[original_key]), "Figure6 gap source")
        elif kind == "point_frequency_difference":
            expected = 100 * (float(row["frequency_pwe_MHz"]) / float(row["frequency_3d_MHz"]) - 1)
            require_close(float(row["difference_percent"]), expected, "Figure6 PWE/3D difference")
            comparison = [r for r in reader.rows(source) if r["shape"] == row["shape"] and r["point"] == row["point"] and r["band"] == row["band"] and r["mesh_level"] == "2" and r["layers"] == "4" and r["requested_eigenpairs"] == "24"]
            if len(comparison) != 1:
                raise ValueError("Figure6 comparison row is not unique")
            require_close(float(row["frequency_pwe_MHz"]), float(comparison[0]["frequency_pwe_N15_MHz"]), "Figure6 PWE source")
            require_close(float(row["frequency_3d_MHz"]), float(comparison[0]["frequency_3d_MHz"]), "Figure6 solid source")
        else:
            fraction = float(row["R_mc"])
            if not 0 <= fraction <= 1 or row["shape"] != "unchanged_ell1_design":
                raise ValueError("Figure6 modal energy scope/fraction mismatch")
            require_close(fraction, float(row["S_ell"]), "Figure6 energy sensitivity identity")
            if float(row["relative_residual"]) >= 1e-7 or float(row["dense_first6_relative_drift"]) >= 1e-7:
                raise ValueError("Figure6 stored eigensolver diagnostics exceed thresholds")
            energy_source = [r for r in reader.rows(source) if r["N"] == row["N"] and r["point"] == row["point"] and r["band"] == row["band"] and float(r["ell_um"]) == float(row["ell_um"])]
            if len(energy_source) != 1 or energy_source[0]["shape"] != "ell1_terminal_frozen":
                raise ValueError("Figure6 fixed-geometry energy source is not unique")
            for key in ["Omega", "R_mc", "S_ell", "relative_residual", "dense_first6_relative_drift"]:
                require_close(float(row[key]), float(energy_source[0][key]), "Figure6 modal source " + key, 1e-12)
    results["figure6"] = {"source_csv": FIGURE_CSV, "record_counts": dict(counts),
                          "source_bindings": resolved_sources, "status": "PASS_29_ROWS"}
    return results, figure


def plot_figure(figure, directory):
    """Regenerate the three panels from saved rows, with no numerical solve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), layout="constrained")
    for shape, label, color in [("initial", "Initial polygon", "#555555"), ("final", "Unchanged ell=1 design", "#0072B2")]:
        rows = sorted((r for r in figure if r["record_type"] == "gap_transfer" and r["shape"] == shape), key=lambda r: float(r["ell_um"]))
        axes[0].plot([float(r["ell_um"]) for r in rows], [float(r["J_percent"]) for r in rows], "o-", label=label, color=color)
    star = next(r for r in figure if r["record_type"] == "gap_reoptimized")
    axes[0].plot([2], [float(star["J_percent"])], "*", markersize=13, color="#D55E00", label="ell=2 reoptimized")
    axes[0].set(xlabel="Epoxy length scale (micrometres)", ylabel="Relative gap (%)")
    axes[0].legend(fontsize=7)
    categories = [("initial", "Gamma"), ("initial", "X"), ("final", "Gamma"), ("final", "X")]
    for band, offset, color in [("4", -.08, "#0072B2"), ("5", .08, "#E69F00")]:
        values = [float(next(r for r in figure if r["record_type"] == "point_frequency_difference" and r["shape"] == shape and r["point"] == point and r["band"] == band)["difference_percent"]) for shape, point in categories]
        axes[1].plot([i + offset for i in range(4)], values, "s", color=color, label="Band " + band)
    axes[1].axhline(0, color=".5", linewidth=.8)
    axes[1].set_xticks(range(4), ["Initial Gamma", "Initial X", "Redesigned Gamma", "Redesigned X"], rotation=35, ha="right", fontsize=7)
    axes[1].set_ylabel("100 (PWE / solid - 1) (%)")
    axes[1].legend(fontsize=7)
    for point, band, color in [("X", "4", "#0072B2"), ("X", "5", "#E69F00"), ("Gamma", "5", "#7B3294")]:
        rows = sorted((r for r in figure if r["record_type"] == "fixed_geometry_modal_energy" and r["point"] == point and r["band"] == band), key=lambda r: float(r["ell_um"]))
        axes[2].plot([float(r["ell_um"]) for r in rows], [float(r["R_mc"]) for r in rows], "o-", label=point + " band " + band, color=color)
    axes[2].set(xlabel="Epoxy length scale (micrometres)", ylabel="Couple-stress energy fraction")
    axes[2].legend(fontsize=7)
    for ax, letter in zip(axes, "abc"):
        ax.text(0, 1.04, "(" + letter + ")", transform=ax.transAxes, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(directory / "figure6_from_deposited_csv.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--archive", type=Path, help="Exact Supplementary_Data_v19.zip from Zenodo")
    group.add_argument("--data-dir", type=Path, help="Existing extraction with Data1–Data8 and MEMBER_SHA256.csv")
    parser.add_argument("--out", type=Path, help="Write JSON to a new file; otherwise print it")
    parser.add_argument("--plot-dir", type=Path, help="Create a new directory for a Figure6 PNG from verified saved rows")
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        raise FileExistsError("Use a new --out file; existing reports are not overwritten")
    if args.plot_dir is not None and args.plot_dir.exists():
        raise FileExistsError("Use a new --plot-dir directory")
    reader = DataReader(archive=args.archive, data_dir=args.data_dir)
    try:
        results, figure = verify(reader)
        if args.plot_dir is not None:
            plot_figure(figure, args.plot_dir)
        content = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        if args.out is None:
            print(content, end="")
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as stream:
                stream.write(content)
            print(json.dumps({"status": results["status"], "output": str(args.out.resolve())}))
    finally:
        reader.close()


if __name__ == "__main__":
    main()
