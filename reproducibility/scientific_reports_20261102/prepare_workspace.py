"""Create an isolated historical-code workspace using verified Zenodo inputs.

Only the deposited 02_evidence and 05_validation trees are overlaid. Private
review/authoring folders are not imported. No solver or optimization is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from verify_saved_data import DataReader, safe_member, verify

HERE = Path(__file__).resolve().parent


def destination_for(name):
    parts = safe_member(name).parts
    if not parts or not parts[0].startswith("Data"):
        return None
    parts = parts[1:]
    if parts and parts[0] in ("part_a", "part_b"):
        parts = parts[1:]
    if not parts or parts[0] not in ("02_evidence", "05_validation"):
        return None
    return Path(*parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--archive", type=Path)
    group.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="A directory that does not yet exist")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("--out must be new; historical/current workspaces are never overwritten")
    source = HERE / "workspace"
    if out == source or out.is_relative_to(source):
        raise ValueError("Place --out outside the distributed workspace source tree")
    reader = DataReader(archive=args.archive, data_dir=args.data_dir)
    try:
        report, _ = verify(reader)
        plan = {}
        for row in reader.index:
            destination = destination_for(row["destination"])
            if destination is None:
                continue
            key = destination.as_posix()
            if key in plan and plan[key]["sha256"] != row["sha256"]:
                raise ValueError(f"Different historical payload versions collide at {key}; keep archive folders separate")
            plan[key] = row
            existing = source / destination
            if existing.exists() and hashlib.sha256(existing.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Deposited source and distributed source differ at {key}; no overwrite attempted")
        # All conflict/scope checks finish before creating the user's new tree.
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, out)
        for key, row in sorted(plan.items()):
            destination = (out / key).resolve()
            if not destination.is_relative_to(out):
                raise ValueError(f"Path escaped the fresh workspace: {key}")
            if destination.exists():
                continue  # Preflight proved the bytes are identical.
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(reader.read(row["destination"]))
        record = {"status": "PREPARED_FRESH_WORKSPACE_NO_SOLVES", "source_record": report["doi"],
                  "copied_payload_paths": len(plan), "source_code": "Exact copies listed in SOURCE_MANIFEST.json",
                  "input_scope": "02_evidence and 05_validation only; review/authoring/output folders excluded",
                  "new_eigensolves": 0, "new_optimizations": 0, "new_COMSOL_runs": 0,
                  "next_step": "Inspect source/paths; use new output directories for prospective solves. Preserved COMSOL drivers require local installation path configuration.",
                  "limitations": "Some historical drivers use fixed destinations or archived inputs not supplied for every experiment. This bootstrap does not certify a complete rerun of every workflow."}
        (out / "PUBLIC_WORKSPACE_PREPARATION.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": record["status"], "workspace": str(out), "payload_paths": len(plan)}, indent=2))
    finally:
        reader.close()


if __name__ == "__main__":
    main()
