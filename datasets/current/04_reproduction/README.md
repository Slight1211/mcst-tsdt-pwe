# Reproduction entry points

`verify_dataset.py` is self-contained and checks the complete packaged file inventory.
It reads existing files and does not start PWE or COMSOL calculations.

`original_code/` is the exact public code tree at commit
e0a2184c59c62583ce448076903f14f69fd21877. Its requirements, copyright notices,
and original instructions are retained. The scientific_reports_20261104 addendum
contains the uniform b=0.24 optimizer, two initial coordinate sets for each material
length, saved-data checks, and an isolated-workspace extraction helper.

`current_sources/` contains exact plotting/derivation sources used to prepare the
current figures and sensitivity values, plus the table-export source. These scripts
retain their original project-relative paths and, in some cases, historical output
guards. They are provenance sources rather than portable top-level launchers.
Do not run them over the immutable dataset. Use the display map and source map to
assemble their required inputs in a separate working directory. The table-export
script reads the manuscript's TeX; the manuscript text is not part of this dataset.
Its already exported CSV tables are in `02_current_tables/`.

Saved CSV/JSON data can be read directly without a proprietary solver. New PWE
calculations require the code's NumPy/SciPy dependencies. New solid calculations
require a separately installed and licensed COMSOL environment. No COMSOL program,
license, or model checkpoint is distributed here.

For the recovered Table S5(a) records, run python 04_reproduction/verify_mac_records.py --root . from the complete dataset root. This checks original file hashes and reported table arithmetic without launching a solver.
