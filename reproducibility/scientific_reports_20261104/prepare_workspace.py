"""Assemble a new b=0.24 workspace from exact public sources and verified Data9.

Only standard-library modules are used. Nothing is downloaded or solved. All
input checks and collision checks complete before the new directory is created.
The destination must not exist; existing files and directories are never changed.
Historical Data1--Data8 inputs are not overlaid by this command.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from verify_saved_data import ARCHIVE_SHA256, DataReader, digest, file_digest, safe_member, verify

HERE = Path(__file__).resolve().parent
WORKSPACE_INDEX_SHA256 = "d0d6a3ee93b202b535dc48b4711b7280321c185d22110af8056ee77ac1ee60e7"
ALLOWED_ROOTS = ("04_code/", "scripts/", "05_validation/", "portable_seed_inputs/",
                 "output/uniform_b024_v47/", "02_evidence/scientific_data/")
STYLE = "FABLE_REVIEW/style/mcst_paper.mplstyle"


def filesystem_path(path):
    """Support preserved long scientific paths on Windows without registry edits."""
    path = Path(path).absolute()
    text = str(path)
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        return Path("\\\\?\\UNC\\" + text[2:] if text.startswith("\\\\") else "\\\\?\\" + text)
    return path


def no_symlink_ancestors(path):
    for parent in [path, *path.parents]:
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise ValueError(f"Symbolic link/junction in path: {parent}")


def workspace_plan(source):
    no_symlink_ancestors(source)
    meta = json.loads((HERE / "SOURCE_MANIFEST.json").read_text(encoding="utf-8-sig"))
    if meta["current_source_archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Workspace manifest refers to a different Data9 archive")
    rows = meta["files"]
    index = [{k: row[k] for k in ("workspace_relative_path", "bytes", "sha256")} for row in rows]
    if digest(json.dumps(index, sort_keys=True, separators=(",", ":")).encode()) != WORKSPACE_INDEX_SHA256:
        raise ValueError("Bundled source index differs from this exact public addendum")
    if len(rows) != 110:
        raise ValueError("Expected 110 exact source/seed/style files")
    plan = {}
    for row in rows:
        name = row["workspace_relative_path"]
        parts = safe_member(name).parts
        if row["path"] != "workspace/" + name or name in plan:
            raise ValueError("Duplicate/inconsistent source index path")
        path = source.joinpath(*parts)
        no_symlink_ancestors(path)
        if not path.is_file() or path.stat().st_size != row["bytes"] or file_digest(path) != row["sha256"]:
            raise ValueError(f"Bundled source bytes differ: {name}")
        plan[name] = {"source_kind": "public_addendum", "source": path, "sha256": row["sha256"]}
    actual = set()
    for entry in source.rglob("*"):
        if entry.is_symlink() or (hasattr(entry, "is_junction") and entry.is_junction()):
            raise ValueError(f"Non-regular source path: {entry}")
        if entry.is_file():
            actual.add(entry.relative_to(source).as_posix())
    if actual != set(plan):
        raise ValueError("Bundled workspace contains unindexed or missing files")
    return plan


def assemble(reader, source, destination, baseline=None):
    destination = Path(destination).absolute()
    no_symlink_ancestors(destination)
    if destination.exists() or not destination.parent.is_dir():
        raise FileExistsError("--out must be a new directory whose parent already exists")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Destination and bundled source workspace must be separate")
    results = verify(reader, baseline)
    plan = workspace_plan(source)
    duplicates, excluded = [], []
    for name in reader.names:
        if not (name.startswith(ALLOWED_ROOTS) or name == STYLE):
            excluded.append(name)
            continue
        safe_member(name)
        expected = reader.checksums[name]
        if name in plan:
            if plan[name]["sha256"] != expected:
                raise ValueError(f"Data9/source collision has different bytes: {name}")
            duplicates.append(name)
        else:
            plan[name] = {"source_kind": "Data9", "source": name, "sha256": expected}
    folded = {name.casefold(): name for name in plan}
    if len(folded) != len(plan):
        raise ValueError("Case-colliding output paths")
    for name in plan:
        for parent in safe_member(name).parents:
            if str(parent) != "." and str(parent).casefold() in folded:
                raise ValueError("A file path is also used as an output directory")
    if "workspace_assembly.json" in folded:
        raise ValueError("Assembly report name collides with a source member")
    # Exclusive creation happens only after complete source/data preflight.
    destination.mkdir(exist_ok=False)
    for name, record in sorted(plan.items()):
        target = filesystem_path(destination.joinpath(*safe_member(name).parts))
        target.parent.mkdir(parents=True, exist_ok=True)
        no_symlink_ancestors(target.parent)
        data = reader.read(record["source"]) if record["source_kind"] == "Data9" else record["source"].read_bytes()
        if digest(data) != record["sha256"]:
            raise ValueError(f"Input changed after preflight: {name}")
        with target.open("xb") as stream:
            stream.write(data)
    report = {"status": "PASS_NEW_EXACT_WORKSPACE_ASSEMBLED", "data9_sha256": ARCHIVE_SHA256,
              "source_index_sha256": WORKSPACE_INDEX_SHA256, "files": len(plan),
              "identical_duplicate_paths": duplicates, "excluded_data9_members": excluded,
              "legacy_baseline_inputs_overlaid": False, "new_eigensolves": 0,
              "new_optimizations": 0, "new_COMSOL_runs": 0,
              "saved_data_verification": results}
    with (destination / "WORKSPACE_ASSEMBLY.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Exact current Data9 ZIP")
    parser.add_argument("--baseline-archive", type=Path, help="Optional original Data1--Data8 ZIP for extra control verification only")
    parser.add_argument("--out", type=Path, required=True, help="New workspace directory; its parent must exist")
    args = parser.parse_args()
    reader, baseline = DataReader(args.archive), None
    try:
        if args.baseline_archive is not None:
            baseline = DataReader(args.baseline_archive, baseline=True)
        result = assemble(reader, HERE / "workspace", args.out, baseline)
        print(json.dumps({"status": result["status"], "files": result["files"],
                          "output": str(args.out.resolve())}))
    finally:
        reader.close()
        if baseline is not None:
            baseline.close()


if __name__ == "__main__":
    main()
