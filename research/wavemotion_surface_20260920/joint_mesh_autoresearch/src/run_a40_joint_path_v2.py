"""Reuse locked controller with an isolated accessor-only Java recovery."""
import argparse
import json
from pathlib import Path
import run_a40_joint_path as base
from run_joint_baseline import WORK, ROOT, sha, write_json

base.EXP = WORK/'experiments/endpoint_a40_joint_v2'
base.HERE = base.EXP/'code'
base.DEST = ROOT/'runs/endpoint_a40_joint_M1L6_20261001_v2'
base.FILES = ('SolveA40JointPath.java','SolveA40JointPath.class','run_a40_joint_path.py',
              'run_joint_baseline.py','run_a40_joint_path_v2.py')

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');args=ap.parse_args()
    if args.freeze:
        base.freeze()
        p=base.EXP/'inputs.json'; frozen=json.loads(p.read_text())
        original=ROOT/'runs/endpoint_a40_joint_M1L6_20261001'
        for source in (original/'failure.json',original/'batch.log',WORK/'experiments/endpoint_a40_joint/protocol.md'):
            frozen['input_hashes'][str(source)] = sha(source)
        write_json(p,frozen)
    else:
        base.run()
