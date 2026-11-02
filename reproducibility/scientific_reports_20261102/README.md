# Scientific Reports numerical-code addendum — 2 November 2026

This addendum contains the later scientific code used for manuscript revision
v20, alongside a portable check of the published numerical records. The exact
data/code archive is [Zenodo version 1.0.0](https://doi.org/10.5281/zenodo.23888401).
The original repository layout and its older public adaptations are retained
outside this directory.

The study uses modified couple stress theory, Reddy third-order plate
kinematics, inverse-factorized plane wave expansion and fixed-area interface
design. The later work includes selected-shape three-dimensional comparisons
at length scales of 1 and 2 micrometres, a classical Rayleigh–Lamb free-surface
benchmark, length-scale-specific local searches, the zero-length-scale N=21
reoptimization, and endpoint-shape cross-evaluation.

## Start with the saved-data check

Download **Supplementary_Data_v19.zip** from the
[published record's file list](https://zenodo.org/records/23888401).
Its SHA256 is:

```text
3d88c5882cd2e12cc5e824caf96e56f9b3ea19341d3c8ba3f2b68a52c21b269f
```

From this addendum directory:

```sh
python verify_saved_data.py --archive /path/to/Supplementary_Data_v19.zip
```

This command uses Python's standard library. It verifies the archive SHA256,
all ZIP CRCs, the published member index and all 966 payload hashes. It then
recalculates the following quantities directly from saved spectra:

- the initial and optimized N=15 gaps at length scale 1 micrometre;
- the reoptimized gap at 2 micrometres;
- the five exploratory length-scale states, retaining their different search protocols;
- all four N=15/N=21 shape/order cross-evaluations;
- all four endpoint-shape/material-state cross-evaluations, including negative separation;
- all 29 Figure 6 records, source-file bindings, gap arithmetic, signed
  plate/solid frequency differences and modal-energy sensitivity identities.

No eigensolver, optimization or COMSOL process is started. This arithmetic and
integrity check does not substitute for new physical or numerical validation.
For an existing extraction, use `--data-dir /path/to/extraction`; the published
member-index digest and all payload files must still match.

Optional output must have a new filename. Optional figure output requires
Matplotlib and a new directory:

```sh
python verify_saved_data.py --archive /path/to/Supplementary_Data_v19.zip --out /new/path/check.json --plot-dir /new/path/figure6
```

The generated PNG reproduces the three panels from the deposited CSV. It is a
portable rendering of the saved data; it does not run a new simulation.

## Scientific source layout

`workspace/` preserves project-relative paths because the recorded drivers
resolve their imports and source hashes using that layout:

```text
workspace/
  scripts/                         later numerical drivers, checks and scientific plotting sources
  04_code/research/
    wavemotion_surface_20260920/
      run_node_shape_pilot.py       shared evaluator and Brillouin-zone sampling helper
      src/                         PWE assembly, nodal geometry and shape sensitivities
    msse_20260919/python/           analytical through-thickness operators
  05_validation/
    V07_ell2_reoptimization_20261030/source/
                                   the newer shape-sensitivity kernel
    V01_20261030/source/            exact-polygon COMSOL Java builder
    V01_ell2_20261031_v8/source/     retained builder for the length-scale-2 checks
```

`SOURCE_MANIFEST.json` binds every included original project path to its public
path, byte count and SHA256. These scientific source copies are unchanged.
The new portable verifier/bootstrap are separately authored public utilities.
Original scientific hash assertions remain enabled. A fresh run creates its
own manifests and does not inherit an archived experiment's success status.

The 20 PWE-source modules and nine analytical-source modules are included as a
set because production workflows record hashes for the complete source sets,
including modules not called by every run. Several source files also expose
optional classical-solid/FEM experiments outside the default saved-data check.
Those optional Python FEM modules need the root repository's
`requirements-fem.txt`. The older `msse_20260919/python/run_uniform.py` utility
also refers to a historical `check_t01_python` module that is not included;
that auxiliary workflow is outside the supplied rerun scope. The current PWE
kernels and the portable saved-data check do not import it.

## Optional isolated workspace

To combine the exact sources with the deposited numerical inputs, create a
fresh directory outside this source tree:

```sh
python prepare_workspace.py --archive /path/to/Supplementary_Data_v19.zip --out /new/path/mcst-rerun
```

The bootstrap verifies the data first, preserves scientific source bytes and
imports only the deposited `02_evidence/` and `05_validation/` trees. It checks
duplicate destinations and aborts on unequal content before creating output.
Private review and manuscript-authoring folders are not imported. Data remain
in a separate local workspace; they are not added to this Git repository.

Inspect a driver before executing it. Some retained historical controllers
have fixed output paths and intentionally reject an existing directory.
Choose a fresh output directory where supported, or use a separate fresh
workspace for a new experiment. Some exploratory length-scale-3,
length-scale-10 and gap-closure inputs fall outside the manuscript data
selection; a complete rerun of every historical workflow is not claimed.

Important entry points include:

| Purpose | Sources under `workspace/scripts/` |
| --- | --- |
| Selected-shape 3D checks | `run_v01.py`, `run_ell2_3d_v8.py`, `analyze_v01.py`, `analyze_ell2_3d_v8.py` |
| Classical free-surface benchmark | `run_free_plate_v7.py`, `analyze_free_plate_v7.py` |
| Length-scale-specific N=15 design | `run_lengthscale_optimization.py`, `run_shape_comparison_case.py`, `run_gap_closure_case.py` |
| Zero-length-scale multistart and N=21 redesign | `run_ell0_multistart.py`, `run_ell0_N21_reoptimization.py`, `run_ell0_resolution_case.py` |
| Fixed-shape order/material cross-evaluation | `check_ell0_order_cross.py`, `check_shape_lengthscale_cross_v18.py` |
| Fixed-geometry modal response | `check_lengthscale_modal_energy.py` |

Some original analysis/plotting sources also read historical provenance
indices or prior figure assets that are not part of this public source
selection. Their original path/hash checks are preserved rather than silently
weakened. For the current main Figure 6 use the portable verifier's
`--plot-dir` option. The manuscript-copying `prepare_v01_source.py` and the
superseded Figure 6 v16 writer are excluded; prepared Java templates and the
current CSV renderer are supplied instead.

## Environments and computational cost

The newer computations used Python 3.14.0, NumPy 2.4.3 and SciPy 1.17.1.
`requirements.txt` records the observed newer Python plotting/checking
environment. Install these dependencies only if running numerical workflows
or the optional plot; the default saved-data check needs none of them.
The older root-level repository retains its original Python 3.12 requirements.
These are different recorded environments, not an all-version compatibility claim.

Original solid scans used COMSOL 6.0.0.318; later design/free-surface checks
used COMSOL 6.2.0.290. A new solid solve requires separately licensed COMSOL
and Java compilation tools. The preserved later COMSOL drivers use the
original Windows installation-path constant `BIN`; configure it for the local
installation in a separate working copy and record the adapted source hash.
No COMSOL executable, licence file, saved `.mph` model, private preference file
or compiled Java class is included. Full optimizations and N=15/N=21 solves
are substantial computations and are not smoke tests.

## Interpretation and licensing

The reported quantities concern sampled flexural-band gaps and constrained
local designs. Selected-wavevector solid comparisons do not certify a
complete-zone three-dimensional gap. The classical homogeneous free-surface
benchmark does not establish all nonzero-length MCST free-face/interface
conditions. Length-scale-specific searches use different starts and sampling
histories; the common terminal grid does not make them a uniform optimization
protocol. Two-order shape changes are sensitivity evidence, not shape convergence.

Original software is under MIT; original data and documentation are under
CC BY 4.0, following the rights-holder authorization for the Zenodo release.
See the repository-level licence files and the deposited `README_ARCHIVE.txt`.
Third-party dependencies and COMSOL retain their own terms.
