#!/usr/bin/env python3
"""Read-only container checks for the selected numerical collection.

No original-source index or hash manifest is required or published.
No network, software installation, research model import or solver execution.
"""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import stat
import sys
from zipfile import ZipFile

FORBIDDEN = re.compile(
    r"PROJECT_ROOT|research[/\\]|[A-Za-z]:[/\\]|sha256|sha1|"
    r"solve_seconds|elapsed_seconds|runtime_seconds|threads_each|checkpoint",
    re.IGNORECASE,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Non-finite JSON constant: " + value)


def read_json(content):
    return json.loads(content, object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def inspect_npz(path):
    import numpy as np
    with ZipFile(path) as archive:
        members = archive.infolist()
        require(0 < len(members) <= 256, "Invalid NPZ member count")
        require(sum(m.file_size for m in members) <= 128 * 1024 * 1024,
                "NPZ expanded size exceeds limit")
        names = set()
        for member in members:
            name = member.filename
            require(name.endswith(".npy") and "/" not in name and "\\" not in name,
                    "NPZ member must be a flat NPY file")
            require(name not in names and not member.is_dir(), "Duplicate/directory member")
            require(not member.flag_bits & 1, "Encrypted NPZ member")
            require(member.file_size / max(member.compress_size, 1) <= 100,
                    "NPZ compression ratio exceeds limit")
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10_000)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=10_000)
                else:
                    raise ValueError("Unsupported NPY header version")
                require(not dtype.hasobject and dtype.kind in "biufc", "Non-numeric NPZ")
                require(all(isinstance(n, int) and n >= 0 for n in shape), "Invalid array shape")
                require(stream.tell() + math.prod(shape) * dtype.itemsize == member.file_size,
                        "NPY shape/dtype size mismatch")
            names.add(name)
    with np.load(path, allow_pickle=False, max_header_size=10_000) as arrays:
        for name in arrays.files:
            array = arrays[name]
            require(not array.dtype.hasobject and array.dtype.kind in "biufc", "Non-numeric NPZ")


def verify(root):
    require(root.is_dir() and not root.is_symlink(), "Invalid collection directory")
    catalog_path = root / "catalog.json"
    require(not catalog_path.is_symlink() and catalog_path.stat().st_size <= 1024 * 1024,
            "Invalid catalog")
    catalog = read_json(catalog_path.read_text(encoding="utf-8-sig"))
    entries = catalog["files"]
    require(isinstance(entries, list) and len(entries) == 58, "Expected 58 numerical files")
    formats, seen, total = Counter(), set(), 0
    for entry in entries:
        name = entry["path"]
        require(isinstance(name, str) and bool(re.fullmatch(r"assets/item-\d{3}\.(csv|json|npz|png)", name)),
                "Invalid collection asset name")
        require(name not in seen, "Duplicate asset")
        seen.add(name)
        require(isinstance(entry["label"], str) and bool(entry["label"]), "Label missing")
        require(isinstance(entry["role"], str) and bool(entry["role"]), "Scientific role missing")
        require(isinstance(entry["supports"], list) and all(isinstance(x, str) and x for x in entry["supports"]),
                "Invalid figure/table mapping")
        path = root / name
        require(not path.is_symlink() and not path.parent.is_symlink(), "Symlink rejected")
        require(path.resolve().is_relative_to(root.resolve()), "Asset escapes collection")
        info = path.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Asset must be a regular unlinked file")
        expected = entry["bytes"]
        require(isinstance(expected, int) and not isinstance(expected, bool) and 0 <= expected <= 64 * 1024 * 1024,
                "Invalid byte count")
        require(info.st_size == expected, "Asset byte count mismatch")
        suffix = path.suffix
        if suffix in (".csv", ".json"):
            content = path.read_text(encoding="utf-8-sig")
            require(not FORBIDDEN.search(content), "Excluded provenance/runtime field in asset")
            if suffix == ".json":
                read_json(content)
            else:
                csv.field_size_limit(1024 * 1024)
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream, strict=True)
                    header = next(reader, None)
                    require(header and len(header) <= 1024 and len(header) == len(set(header)), "Invalid CSV header")
                    for row_number, row in enumerate(reader, 1):
                        require(row_number <= 1_000_000 and len(row) == len(header), "Invalid CSV row")
        elif suffix == ".npz":
            inspect_npz(path)
        else:
            with path.open("rb") as stream:
                require(stream.read(8) == b"\x89PNG\r\n\x1a\n", "Invalid PNG signature")
        formats[suffix] += 1
        total += expected
    require(dict(formats) == {".csv": 39, ".json": 8, ".npz": 4, ".png": 7}, "Unexpected format counts")
    return {"status": "PASS", "data_files": len(entries), "bytes": total,
            "formats": dict(formats), "read_only": True,
            "original_source_hash_check": "NOT_RUN: index not distributed",
            "new_numerical_calculations": "NOT_RUN"}


if __name__ == "__main__":
    try:
        result = verify(Path(__file__).resolve().parent)
    except (ValueError, OSError, KeyError, TypeError, ImportError, csv.Error) as exc:
        print("FAIL: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, indent=2, sort_keys=True))
