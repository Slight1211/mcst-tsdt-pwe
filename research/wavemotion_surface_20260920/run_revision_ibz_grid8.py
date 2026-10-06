"""Refine the already completed finite-IBZ search on a distinct 9x9 grid."""

from pathlib import Path

import run_revision_ibz_scan as scan

scan.OUT = Path(__file__).resolve().parent / "runs/manuscript_revision_20260925/ibz_refinement_grid8"
scan.GRID_DIVISIONS = 8


if __name__ == "__main__":
    scan.main()
