"""Verify a released dataset with Python's standard library; no numerical solve."""
from pathlib import Path
import argparse
import csv
import hashlib
import json


def verify(root):
    root = root.resolve()
    manifest = root / '06_provenance/FILE_MANIFEST.csv'
    failures = []
    checked = 0
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len({r['path'] for r in rows}) == len(rows), 'Duplicate manifest paths'
    for row in rows:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append({'path': row['path'], 'reason': 'missing or unsafe path'})
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if path.stat().st_size != int(row['bytes']) or digest.hexdigest() != row['sha256']:
            failures.append({'path': row['path'], 'reason': 'size or SHA256 mismatch'})
        checked += 1
    checksum_path = root / '06_provenance/SHA256SUMS.txt'
    for line in checksum_path.read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append({'path': relative, 'reason': 'checksum-list mismatch'})
    result = {'status': 'PASS' if not failures else 'FAIL', 'manifest_files_checked': checked,
              'scientific_claims_recomputed': False, 'failures': failures}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    raise SystemExit(verify(parser.parse_args().root))
