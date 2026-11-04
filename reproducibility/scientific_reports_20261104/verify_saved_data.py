"""Verify the current b=0.24 saved data using Python's standard library only.

No eigensolver, optimizer, COMSOL, network access or extraction is used. Data9
contains complete current design spectra, N21 checks and selected solid spectra.
Its reused N15 regular-polygon controls are saved as L/U source CSV rows; the
optional original Data1--Data8 archive checks those controls against raw spectra.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

ARCHIVE_SHA256 = "79b8cca7dad260cd84be9bcb752221ec83fc79c3b3d80a5c965a6fae57dad8d0"
CHECKSUMS_SHA256 = "576668507712e59bec67068a269d71a3356c3104a9c4028f09dd044a800ce6dc"
BASELINE_SHA256 = "3d88c5882cd2e12cc5e824caf96e56f9b3ea19341d3c8ba3f2b68a52c21b269f"
BASELINE_INDEX_SHA256 = "87d881582d61425d3b5da233ec2afe22485e92abe2cd00fbec3abeec673b151a"
PWE = "05_validation/UNIFORM_B024_PWE_V47_20261103/"
OUT = "output/uniform_b024_v47/"
SELECTED = {
    0: "05_validation/SHAPE_BOUND_SENSITIVITY_20261103/ell_0p0_b0p24_regular/",
    .5: "05_validation/SHAPE_BOUND_SENSITIVITY_20261103/ell_0p5_b0p24_old_best/",
    1: "05_validation/UNIFORM_B024_V47_20261103/ell_1_warm/",
    2: "05_validation/UNIFORM_B024_V47_20261103/ell_2_warm/",
    5: "05_validation/UNIFORM_B024_V47_20261103/ell_5_warm/",
}
EXPECTED_FIVE = [34.57724889982054, 34.31172213068446, 33.22452528639601,
                 27.047699138915647, 2.9559181605435336]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_member(name):
    """Reject names that are ambiguous or unsafe on POSIX or Windows."""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe member: {name!r}")
    parts = name.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if any(p in ("", ".", "..") or p.endswith((" ", ".")) or
           any(ord(c) < 32 or c in '<>\"|?*' for c in p) or
           p.split(".", 1)[0].upper() in reserved for p in parts):
        raise ValueError(f"Unsafe member: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute():
        raise ValueError(f"Absolute member: {name!r}")
    return path


def require_close(actual, expected, label, tolerance=1e-9):
    if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual - expected) > tolerance:
        raise ValueError(f"{label}: {actual!r} differs from {expected!r}")


class DataReader:
    """Authenticate a ZIP, all its indexed bytes, and its safe member names."""

    def __init__(self, archive, *, baseline=False):
        self.path = Path(archive).resolve()
        self.archive = None
        expected = BASELINE_SHA256 if baseline else ARCHIVE_SHA256
        if file_digest(self.path) != expected:
            raise ValueError("Archive SHA256 does not match the pinned public payload")
        try:
            self.archive = zipfile.ZipFile(self.path)
            infos = self.archive.infolist()
            self.names = [entry.filename for entry in infos]
            folded = [name.casefold() for name in self.names]
            if len(folded) != len(set(folded)):
                raise ValueError("Duplicate/case-colliding ZIP member names")
            for entry in infos:
                safe_member(entry.filename)
                kind = stat.S_IFMT(entry.external_attr >> 16)
                if entry.is_dir() or kind not in (0, stat.S_IFREG):
                    raise ValueError(f"Non-regular ZIP member: {entry.filename}")
            if baseline:
                index = self.read("MEMBER_SHA256.csv")
                if digest(index) != BASELINE_INDEX_SHA256:
                    raise ValueError("Original Data1--Data8 index mismatch")
                rows = list(csv.DictReader(io.StringIO(index.decode("utf-8-sig"))))
                if len(rows) != 966:
                    raise ValueError("Expected 966 original indexed members")
                self.checksums = {r["destination"]: r["sha256"] for r in rows}
                if len(self.checksums) != len(rows):
                    raise ValueError("Duplicate original index entries")
                for row in rows:
                    if len(self.read(row["destination"])) != int(row["bytes"]):
                        raise ValueError("Original payload byte count mismatch")
            else:
                index = self.read("CHECKSUMS.sha256")
                if digest(index) != CHECKSUMS_SHA256:
                    raise ValueError("Data9 checksum manifest mismatch")
                self.checksums = {}
                for line in index.decode("utf-8-sig").splitlines():
                    if not line:
                        continue
                    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
                    if not match or match[2] in self.checksums:
                        raise ValueError("Malformed/duplicate Data9 checksum entry")
                    safe_member(match[2])
                    self.checksums[match[2]] = match[1]
                if len(self.names) != 740 or len(self.checksums) != 739:
                    raise ValueError("Expected 740 Data9 members, including 739 indexed payloads")
                if set(self.names) != set(self.checksums) | {"CHECKSUMS.sha256"}:
                    raise ValueError("Data9 manifest does not cover exactly its payload")
            for name, expected_hash in self.checksums.items():
                safe_member(name)
                if digest(self.read(name)) != expected_hash:
                    raise ValueError(f"Payload SHA256 mismatch: {name}")
        except BaseException:
            self.close()
            raise

    def read(self, name):
        safe_member(name)
        return self.archive.read(name)

    def rows(self, name):
        return list(csv.DictReader(io.StringIO(self.read(name).decode("utf-8-sig"))))

    def json(self, name):
        return json.loads(self.read(name).decode("utf-8-sig"))

    def close(self):
        if self.archive is not None:
            self.archive.close()


def gap(lower, upper, *, points, source):
    if not all(math.isfinite(x) and x >= 0 for x in (lower, upper)) or lower + upper <= 0:
        raise ValueError(f"Invalid band edges: {source}")
    return {"L": lower, "U": upper, "J_percent": 200 * (upper - lower) / (upper + lower),
            "points": points, "source": source}


def sampled_gap(reader, name, points):
    rows = reader.rows(name)
    if len(rows) != points or points not in (45, 153):
        raise ValueError(f"Unexpected sampled grid: {name}")
    grid = 8 if points == 45 else 16
    for row, (i, j) in zip(rows, ((i, j) for i in range(grid + 1) for j in range(i + 1))):
        require_close(float(row["kx_a"]), math.pi * i / grid, name + " kx", 1e-12)
        require_close(float(row["ky_a"]), math.pi * j / grid, name + " ky", 1e-12)
        frequencies = [float(row[f"Omega{b}"]) for b in range(1, 7)]
        if any(not math.isfinite(v) or v < 0 for v in frequencies) or any(
                a > b + 1e-9 for a, b in zip(frequencies, frequencies[1:])):
            raise ValueError(f"Invalid/unsorted frequencies: {name}")
    return gap(max(float(r["Omega4"]) for r in rows), min(float(r["Omega5"]) for r in rows),
               points=points, source=name)


def solid_gap(reader, name):
    """Repeat saved flexural-mode selection, not the field/MAC calculation."""
    by_point = defaultdict(list)
    for row in reader.rows(name):
        by_point[(float(row["kx_pi_over_a"]), float(row["ky_pi_over_a"]))].append(row)
    if set(by_point) != {(0., 0.), (1., 0.), (1., 1.)}:
        raise ValueError("Selected solid comparison must have exactly Gamma, X and M")
    edges, selected_modes = [], []
    for point, rows in sorted(by_point.items()):
        selected = {}
        for threshold in (.7, .8):
            keep = sorted((r for r in rows if float(r["frequency_mhz"]) > .001 and
                           float(r["vertical_fraction"]) > threshold), key=lambda r: float(r["frequency_mhz"]))
            selected[threshold] = ([None] + keep) if point == (0., 0.) else keep
            if len(selected[threshold]) < 6:
                raise ValueError("Insufficient saved flexural modes")
        pair = []
        for band in (4, 5):
            row = selected[.8][band - 1]
            if row["mode_index"] != selected[.7][band - 1]["mode_index"]:
                raise ValueError("0.7/0.8 flexural thresholds give different edge modes")
            freq = float(row["frequency_mhz"])
            if abs(float(row["imag_frequency_mhz"])) / freq >= 1e-7:
                raise ValueError("Saved solid edge has excessive imaginary component")
            pair.append(freq)
            selected_modes.append({"point_pi_over_a": point, "band": band,
                                   "raw_mode": int(row["mode_index"]), "frequency_MHz": freq})
        edges.append(pair)
    return {**gap(max(x[0] for x in edges), min(x[1] for x in edges), points=3, source=name),
            "selected_modes": selected_modes}


def verify(reader, baseline=None):
    report = {"status": "PASS_DATA9_SAVED_DATA_ARITHMETIC_AND_BINDINGS",
              "data9_sha256": ARCHIVE_SHA256, "zip_members": len(reader.names),
              "payload_checksums_verified": len(reader.checksums),
              "checksum_manifest_sha256": CHECKSUMS_SHA256, "new_eigensolves": 0,
              "new_optimizations": 0, "new_COMSOL_runs": 0,
              "scope": "Stored spectra and arithmetic only; no independent PDE, field/MAC or global-optimality validation."}
    five, selected = [], {}
    summaries = {float(r["ell_um"]): r for r in reader.rows(OUT + "summary.csv")}
    case_records = {float(r["ell_um"]): r for r in reader.json(OUT + "summary.json")["cases"]}
    if set(summaries) != set(SELECTED):
        raise ValueError("Five-state summary scope mismatch")
    for (ell, folder), expected in zip(SELECTED.items(), EXPECTED_FIVE):
        value = sampled_gap(reader, folder + "final_ibz153.csv", 153)
        require_close(value["J_percent"], expected, "current five-state gap")
        manifest = reader.json(folder + "manifest.json")
        if manifest["shape_bounds"] != [-.24, .24] or manifest["N"] != 15 or manifest["optimization_points"] != 153:
            raise ValueError("Current design is not an N15, b=.24, 153-point optimization")
        require_close(float(manifest["ell_epoxy_um"]), ell, "design length scale")
        require_close(float(manifest["final_ibz153"]["J"]), value["J_percent"], "selected manifest gap")
        summary = summaries[ell]
        if summary["spectrum"] != value["source"] or summary["manifest"] != folder + "manifest.json":
            raise ValueError("Selected design path binding mismatch")
        for path, key in [(value["source"], "spectrum_sha256"), (folder + "manifest.json", "manifest_sha256"),
                          (folder + "final_vertices.csv", "vertices_sha256")]:
            if digest(reader.read(path)) != summary[key]:
                raise ValueError("Selected geometry/spectrum binding mismatch")
        vertices = reader.rows(folder + "final_vertices.csv")
        xy = [(float(r["x_over_a"]), float(r["y_over_a"])) for r in vertices]
        if len(xy) != 48 or any(not math.isfinite(v) for pair in xy for v in pair):
            raise ValueError("Expected 48 finite normalized vertices")
        area = abs(sum(x * xy[(i + 1) % 48][1] - y * xy[(i + 1) % 48][0]
                       for i, (x, y) in enumerate(xy))) / 2
        require_close(area, .09 * math.pi, "fixed steel area", 1e-10)
        starts = case_records[ell]["starts"]
        if len(starts) != 2:
            raise ValueError("Expected two saved b=.24 starts at every length scale")
        start_gaps = []
        for start in starts:
            mp = start["manifest"]
            if digest(reader.read(mp)) != start["manifest_sha256"]:
                raise ValueError("Start manifest binding mismatch")
            sm = reader.json(mp)
            if sm["shape_bounds"] != [-.24, .24] or sm["N"] != 15 or sm["optimization_points"] != 153:
                raise ValueError("Two-start search protocols differ")
            require_close(float(sm["ell_epoxy_um"]), ell, "start length scale")
            sg = sampled_gap(reader, str(PurePosixPath(mp).parent / "final_ibz153.csv"), 153)
            require_close(sg["J_percent"], float(start["J153_percent"]), "start final gap")
            start_gaps.append(sg["J_percent"])
        spread = max(start_gaps) - min(start_gaps)
        require_close(spread, float(summary["two_start_spread_pp"]), "two-start gap spread")
        require_close(value["J_percent"], max(start_gaps), "selected higher-gap start", 1e-8)
        require_close(value["J_percent"], float(summary["J153_percent"]), "five-state summary")
        five.append({"ell_um": ell, "N": 15, "bound": .24, "area_fraction": area,
                     "two_start_spread_pp": spread, **value})
        selected[ell] = value
    report["five_optimized_153_point_gaps"] = five

    controls = {}
    for name in ["05_validation/V08_N15_20261030_v3/V08_summary.csv",
                 "05_validation/V08_N15_ell1p5_20261030/V08_summary.csv"]:
        for row in reader.rows(name):
            if row["shape"] == "initial":
                ell = float(row["ell_um"])
                controls[ell] = gap(float(row["L"]), float(row["U"]), points=int(row["points"]), source=name)
                require_close(controls[ell]["J_percent"], float(row["J"]), "source control edge arithmetic")
    if set(controls) != {.5, 1, 1.5, 2}:
        raise ValueError("Incomplete saved regular-polygon controls")
    if baseline is not None:
        for ell, tag, prefix in [(.5, "0.5", "Data2/05_validation/V08_N15_20261030_v3/"),
                                 (1, "1", "Data2/05_validation/V08_N15_20261030_v3/"),
                                 (1.5, "1.5", "Data3/05_validation/V08_N15_ell1p5_20261030/"),
                                 (2, "2", "Data2/05_validation/V08_N15_20261030_v3/")]:
            raw = sampled_gap(baseline, prefix + f"initial_ell{tag}_N15_ibz45.csv", 45)
            for key in ("L", "U", "J_percent"):
                require_close(raw[key], controls[ell][key], "original raw control " + key)
            controls[ell] = raw
        report["original_control_archive"] = {"sha256": BASELINE_SHA256, "verified_indexed_members": len(baseline.checksums)}
    report["regular_polygon_controls"] = {"source_scope": "raw 45-point spectra in the original Data1--Data8 archive" if baseline else
        "source L/U rows in archived V08_summary.csv; complete N15 control spectra are not in Data9",
        "rows": [{"ell_um": ell, **value} for ell, value in sorted(controls.items())]}

    primary = []
    table = {(float(r["ell_um"]), r["model"]): r for r in reader.rows(OUT + "primary_table.csv")}
    for ell in (1, 2):
        initial21 = ("02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/"
                     "node_gradient_a75_h6_N15_20261004_v1/independent_validation/initial_N21_ibz45.csv") if ell == 1 else (
                     "05_validation/V08_N21_ell2_20261030/initial_ell2_N21_ibz45.csv")
        initial_solid = "05_validation/" + ("V01_20261030/" if ell == 1 else "V01_ell2_20261031_v8/") + "initial_M2L4_n24_GXM/spectra.csv"
        final_solid = f"05_validation/UNIFORM_B024_SOLID_V47_20261103/ell_{ell}/final_M2L4_n24_GXM/spectra.csv"
        n21meta = reader.json(PWE + f"g{ell}_N21_own/manifest.json")
        solidmeta = reader.json(str(PurePosixPath(final_solid).parent / "manifest.json"))
        if n21meta["shape_vertices_sha256"] != summaries[ell]["vertices_sha256"] or solidmeta["vertices_sha256"] != summaries[ell]["vertices_sha256"]:
            raise ValueError("N21/solid comparison is not bound to the selected b=.24 geometry")
        if n21meta["N"] != 21 or n21meta["points"] != 45 or (solidmeta["mesh_level"], solidmeta["layers"], solidmeta["eigen_count"]) != (2, 4, 24):
            raise ValueError("N21/solid comparison scope mismatch")
        require_close(float(n21meta["evaluation_ell_um"]), ell, "N21 length scale")
        require_close(float(solidmeta["settings"]["COMSOL_MCST_LENGTH_UM"]), ell, "solid length scale")
        comparisons = [
            ("PWE N15", controls[ell], sampled_gap(reader, SELECTED[ell] + "final_ibz45.csv", 45)),
            ("PWE N21", sampled_gap(reader, initial21, 45), sampled_gap(reader, PWE + f"g{ell}_N21_own/spectra.csv", 45)),
            ("3D M2L4", solid_gap(reader, initial_solid), solid_gap(reader, final_solid)),
        ]
        for model, initial, final in comparisons:
            gain = final["J_percent"] - initial["J_percent"]
            expected = table[(ell, model)]
            for key, val in [("initial", initial["J_percent"]), ("optimized", final["J_percent"]), ("gain", gain)]:
                require_close(val, float(expected[key]), "primary " + key)
            if initial["points"] != int(expected["points"]) or final["points"] != int(expected["points"]):
                raise ValueError("Primary point-count scope mismatch")
            primary.append({"ell_um": ell, "model": model, "initial": initial, "optimized": final, "gain_pp": gain})
    if len(table) != 6:
        raise ValueError("Expected six primary comparisons")
    report["primary_gains"] = primary

    transfer = []
    for row in reader.rows(OUT + "transfer.csv"):
        ell = float(row["ell_um"])
        tag = {.5: "g1_at0p5", 1.5: "g1_at1p5", 2: "g1_at2"}.get(ell)
        final = selected[1] if ell == 1 else sampled_gap(reader, PWE + tag + "/spectra.csv", 45)
        if tag:
            meta = reader.json(PWE + tag + "/manifest.json")
            if meta["shape_vertices_sha256"] != summaries[1]["vertices_sha256"]:
                raise ValueError("g1 transfer changed geometry")
            require_close(float(meta["evaluation_ell_um"]), ell, "transfer length scale")
        gain = final["J_percent"] - controls[ell]["J_percent"]
        for key, val in [("initial_J", controls[ell]["J_percent"]), ("final_J", final["J_percent"]), ("gain_pp", gain)]:
            require_close(val, float(row[key]), "g1 transfer " + key)
        transfer.append({"ell_um": ell, "initial_J": controls[ell]["J_percent"], "optimized": final, "gain_pp": gain})
    if sorted(r["ell_um"] for r in transfer) != [.5, 1, 1.5, 2]:
        raise ValueError("g1 transfer scope mismatch")
    report["g1_geometry_preserving_transfer"] = transfer

    paths = {("g0_b024", 0): SELECTED[0] + "final_ibz153.csv",
             ("g0_b024", 5): "05_validation/BOUND024_ENDPOINT_CROSS_V42_20261103/g0_at5_ibz153.csv",
             ("g5_b024", 0): PWE + "g5_at0/spectra.csv",
             ("g5_b024", 5): SELECTED[5] + "final_ibz153.csv"}
    endpoints = []
    for row in reader.rows(OUT + "endpoint_comparison.csv"):
        key = (row["shape"], float(row["ell_um"]))
        value = sampled_gap(reader, paths[key], 153)
        for k, k2 in [("L", "L"), ("U", "U"), ("J_percent", "J")]:
            require_close(value[k], float(row[k2]), "endpoint cross " + k)
        endpoints.append({"shape": key[0], "ell_um": key[1], **value})
    if len(endpoints) != 4 or len({(r["shape"], r["ell_um"]) for r in endpoints}) != 4:
        raise ValueError("Endpoint comparison scope mismatch")
    report["endpoint_cross_evaluations"] = endpoints
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Exact current Supplementary_Data_9_uniform_b024_v61.zip")
    parser.add_argument("--baseline-archive", type=Path, help="Optional original Supplementary_Data_v19.zip for raw N15 control spectra")
    parser.add_argument("--out", type=Path, help="Write a new JSON report; existing files are never overwritten")
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        raise FileExistsError("Use a new --out file")
    reader = DataReader(args.archive)
    baseline = None
    try:
        if args.baseline_archive is not None:
            baseline = DataReader(args.baseline_archive, baseline=True)
        result = verify(reader, baseline)
        content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.out is None:
            print(content, end="")
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as stream:
                stream.write(content)
            print(json.dumps({"status": result["status"], "output": str(args.out.resolve())}))
    finally:
        reader.close()
        if baseline is not None:
            baseline.close()


if __name__ == "__main__":
    main()
