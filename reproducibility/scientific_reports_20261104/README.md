# Scientific Reports v62 reproducibility addendum — 4 November 2026

This additive source snapshot accompanies the current five-length-scale,
fixed-area interface study. Manuscript v62 makes editorial changes only; its
scientific sources and Data 9 payload are unchanged from v61.
It preserves the existing repository and earlier
addenda. Every file under `workspace/` is an unchanged source or initialization
input from the separately prepared Data 9 package or the prior public addendum.
`SOURCE_MANIFEST.json` records each origin, byte count and SHA256.

The current Data 9 package is `Supplementary_Data_9_uniform_b024_v61.zip`
(104,369,238 bytes; SHA256
`79b8cca7dad260cd84be9bcb752221ec83fc79c3b3d80a5c965a6fae57dad8d0`).
At assembly of this addendum it is ready locally for deposition. Zenodo version
1.1.0 has **not yet been published**. The historical version 1.0.0 record is
<https://zenodo.org/records/23888401>; it does not contain this new Data 9 package.

## Current calculation and source layout

The five reported states use epoxy material lengths of 0, 0.5, 1, 2 and
5 micrometres, coordinate half-width `b=0.24`, Fourier order `N=15`, fixed
cell dimensions `(a,h)=(75,6)` micrometres, 48 interface vertices and six shape
coordinates. Each state has two starts and is checked on 153 wavevectors.
The selected relative gaps are 34.58%, 34.31%, 33.22%, 27.05% and 2.96%.
The length-1 and length-2 designs also have fixed-geometry `N=21` evaluations
and selected-wavevector solid comparisons. These are local, sampled-spectrum
results; they do not assert a global optimum or complete-zone solid gap.

* `workspace/04_code/` contains the original PWE and shape-derivative kernels.
* `workspace/scripts/run_relaxed_shape_case.py` is the current configurable-box
  search worker. Other `run_uniform_b024_*` scripts retain the original study
  orchestration and immutable-record checks.
* `workspace/portable_seed_inputs/` contains the exact ten starting-coordinate
  JSON files, without historical optimizer outputs.
* `workspace/05_validation/.../source/` supplies the original sensitivity and
  generic COMSOL builders needed by the drivers. Additional old drivers are
  included as dependencies; their filenames do not designate new v61 runs.
* `workspace/FABLE_REVIEW/style/mcst_paper.mplstyle` is an author-owned plotting
  preset required at that exact path by the unchanged plotting helper. It is
  the sole file included from that folder; no third-party review reports or
  correspondence are included.

The large saved spectra, fields, manifests and run-specific generated Java
sources remain in Data 9. Historical numerical inputs are in Data 1–8 and the
earlier public source snapshot. Some original audit/orchestration scripts
reference those historical inputs; their inclusion does not promise that an
arbitrary subset of the archives can replay every earlier workflow.
For example, the plotting collector expects `output/data/Figure6_design_story_v19.csv`,
supplied as `Data5/Figure6_design_story_v19.csv` in the historical Data 1–8 archive.
That legacy plotting input is not needed for a fresh single-state shape search.

## Working with the source

From this addendum directory, check the saved data without numerical dependencies:

```sh
python verify_saved_data.py --archive /path/to/Supplementary_Data_9_uniform_b024_v61.zip
```

The checker verifies all archive members and recalculates the reported gaps,
gains and cross-evaluations. For the regular-polygon N=15 controls, the default
check uses the source lower/upper-edge CSV rows. Add the original published
archive to check the corresponding complete saved spectra as well:

```sh
python verify_saved_data.py --archive /path/to/Supplementary_Data_9_uniform_b024_v61.zip --baseline-archive /path/to/Supplementary_Data_v19.zip
python prepare_workspace.py --archive /path/to/Supplementary_Data_9_uniform_b024_v61.zip --out /new/path/mcst-rerun
```

The bootstrap requires a new output directory, verifies the archive and exact
source manifests, and rejects unsafe paths and conflicting duplicates. These
tools check saved-data arithmetic and file integrity; they start no eigensolver,
optimization or COMSOL process.

Preserve the project-relative directory layout. The separate bootstrap and
saved-data verifier provided with this addendum handle data preparation and
consistency checks; source staging itself launches no optimization or COMSOL
calculation. Reading saved data and performing a new eigensolve are separate
operations. Do not execute the historical study controllers over the saved
result directories.

For a fresh single-state search, use an empty output path inside a disposable
workspace. For example, from `workspace/`:

```text
python scripts/run_relaxed_shape_case.py --bound .24 --ell-um 1 --out runs/ell1_regular_new
python scripts/run_relaxed_shape_case.py --bound .24 --ell-um 1 --seed-manifest portable_seed_inputs/ell_1_warm.json --out runs/ell1_warm_new
```

The worker refuses existing output directories. The first command uses the
regular 48-gon; the second uses the saved same-length-scale initial geometry.
Both commands perform new calculations. They are examples, not operations run
during this release preparation.

Current numerical dependencies are Python 3.14, NumPy 2.4.3, SciPy 1.17.1,
Matplotlib 3.10.8 and threadpoolctl. The source archive retains original runtime
metadata; legacy results were not relabelled as fresh calculations. Fresh
solid calculations require a separately licensed COMSOL installation, suitable
Java tools and local installation/output-path configuration. Commercial
executables, licence files, compiled classes and solved `.mph` files are absent.

Original code remains MIT-licensed; author-owned numerical data and
documentation remain CC BY 4.0, as specified in the repository licences and
archive records. Keep pre-existing attribution and notices. The dataset DOI
identifies a data/code record, not a published journal article.
