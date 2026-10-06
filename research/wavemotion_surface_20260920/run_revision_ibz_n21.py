"""Checkpointed N=21 finite-IBZ cross-check for the a=75 um, h/a=.08 case.

Uses the unchanged production inverse MCST--TSDT solver. This tests Fourier
order at all 45 sampled k points, not a continuous-domain or 3D band gap.
"""

from pathlib import Path

import run_revision_ibz_scan as scan


scan.OUT = Path(__file__).resolve().parent / "runs/manuscript_revision_20260925/ibz_N21_tau008"
scan.RATIOS = (0.08,)
scan.N_VALUES = (21,)
scan.GRID_DIVISIONS = 8


if __name__ == "__main__":
    scan.main()
