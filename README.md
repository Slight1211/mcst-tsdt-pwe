# MCST–TSDT PWE source code

Research code for flexural-wave dispersion in a periodic steel–epoxy plate, using modified couple-stress theory (MCST), third-order shear deformation theory (TSDT), and a plane-wave expansion (PWE). The repository also includes FSDT comparisons, circular-inclusion parameter scans, local band-edge prediction, physical displacement-field overlap, and gradient-based optimization of a nodal inclusion boundary.

This is a **source-only release**. Manuscripts, publication PDFs, saved spectra, field exports, COMSOL checkpoints, commercial software, third-party reference code, and machine-specific preferences are not included. The folder name `mcst_piezo_tsd_plate` in the original project is historical; the study represented here uses a two-phase steel–epoxy plate.

## Quick check

Use Python 3.12 with the packages listed in `requirements.txt`. The versions are those installed in the original research environment, not a claim of compatibility with all package versions.

```sh
python -m pip install -r requirements.txt
python checks/smoke_check.py
python checks/smoke_check.py --kernel frozen
python examples/circular_point.py
```

The smoke check parses the Python sources and runs small-matrix algebraic regressions. The example solves one wavevector with a small PWE basis. These are software checks, **not** the N=15/N=21 manuscript calculations or evidence of numerical convergence.

## Code layout

The original relative layout is retained because the research scripts use sibling imports:

```text
research/
  msse_20260919/python/             analytical thickness operators and Fourier geometry
  wavemotion_surface_20260920/
    src/                          PWE assembly, nodal geometry and analytic sensitivities
    run_*.py                      individual research workflows
    plot_*.py, report_*.py         analysis and scientific plotting
    runs/scale_addendum_20260927/
      sources/                    frozen circular PWE kernels used by selected validations
    joint_mesh_autoresearch/
      src/                        3D mesh/field controllers and overlap utilities
      experiments/                second-order prediction and PWE/3D displacement pairing
comsol/                           Java model-building source
checks/, examples/                lightweight public-release checks and example
```

Current nodal kernels and the frozen circular kernels are intentionally separate. `SOURCE_MANIFEST.json` records the SHA256 of the original and released files and identifies publication adaptations. Do not replace the frozen kernels with newer files when rerunning an audit that checks their hashes.

## Gradient optimization

The nodal production workflow uses a 48-vertex C4v boundary, six fixed-area shape variables, analytic shape sensitivities, and epigraph SLSQP with active-wavevector enrichment. Its settings are N=15, a=75 µm, h=6 µm, steel inner scale 0, and epoxy inner scale 1 µm.

From the repository root, use a **new output directory**:

```sh
python research/wavemotion_surface_20260920/run_node_gradient_process_20261004.py --out results/node_gradient_new
python research/wavemotion_surface_20260920/plot_node_gradient_process_20261004.py --run results/node_gradient_new --output results/node_gradient_new/process
```

This is a substantial numerical calculation; it is not run by the smoke check. The result is a local optimum of the sampled flexural-band objective, not a global optimum or a full-polarization band-gap certificate. The plotting script writes `iteration_table.csv` in its run directory and exports PDF/PNG figures. Use a copy of an immutable run when preparing plots.

## Other workflows and required data

| Workflow | Source entry | Prerequisites |
|---|---|---|
| Circular PWE scans | `run_ratio75_thickness_pwe.py`, `run_ratio75_thickness_n21.py`, `run_scale_addendum_20260927.py` | Python dependencies; inspect the script's output and resume rules before running |
| PWE path/IBZ checks | `run_pwe_path_convergence_dense_20260927.py`, `run_revision_ibz_*.py` | Some workflows consume earlier saved spectra/configuration |
| Local second-order prediction | `joint_mesh_autoresearch/experiments/second_order_local_prediction_20261003/second_order.py` | Original prediction CSV/JSON inputs with matching scientific hashes |
| Prospective thickness checks | `joint_mesh_autoresearch/experiments/second_order_h2_validation_20261003/recompute.py` | Frozen coefficients, baseline NPZ and prediction inputs |
| Physical PWE/3D overlap | `joint_mesh_autoresearch/experiments/pwe_3d_mac_20261004/compute_v2.py` | Saved PWE eigenvectors and completed COMSOL field exports on both sampling grids |
| Scientific plots | `plot_*.py`, `report_*.py`, `run_revision_edge_energy_map.py` | The CSV/NPZ/JSON inputs expected by each script |
| 3D models and mesh comparisons | `comsol/*.java`, `joint_mesh_autoresearch/src/` | A licensed COMSOL installation, locally compiled Java classes and, for continuation scripts, the original completed checkpoints |

Paths in the table are relative to `research/wavemotion_surface_20260920/` except `comsol/`. Input data are deliberately not included; their absence must not be interpreted as a successful reproduction. Publication-specific manuscript/PDF checks and dependencies on the original private Git history are separated from scientific-input checks in the public adaptations. See `PUBLIC_RELEASE_NOTES.md` and the source manifest.

For data-dependent adapted experiments, `MCST_DATA_ROOT` points to a data tree laid out like `research/wavemotion_surface_20260920/`. The saved research kernels remain in this source repository. Inspect each experiment's command-line help and protocol before running; historical protocols describe the original experiment, not a new public execution.

In the relevant experiment directory, the stages are:

```sh
# Local second-order prediction
python second_order.py freeze
python second_order.py evaluate
python independent_audit.py
python gen_fig_second_order.py

# Thickness perturbation checks
python recompute.py --stage freeze
python recompute.py --stage compute
python audit_v2.py
python verify_saved.py

# Physical PWE/3D overlap
python compute_v2.py --stage freeze
python compute_v2.py --stage compute
python compute_v2.py --stage verify
```

The thickness checks require the original historical `coefficients_frozen.json` with SHA256 beginning `e633d4ef`. A coefficient file newly generated by the public prediction workflow has different provenance metadata and must not replace this locked historical input.

COMSOL jobs must be run serially. Set `COMSOL_BIN` for your installation where supported. No installation, license, preferences or saved `.mph` models are redistributed. The Java files are source, not standalone executables. Archived continuation scripts need their original checkpoint/input tree; they are not independent model-building entry points. Never overwrite an existing run or launch a duplicate solver for a running manifest.

Several auxiliary modules and import-helper scripts are retained to preserve the original dependency chain. Their presence does not mean all historical main programs are part of the manuscript or suitable default entry points. In particular, `run_convergence.py` contains an older acoustic pilot; do not run its main program as the dry-plate reproduction.

The scientific plotting sources are provided, but the final manuscript's font-conversion and layout-audit pipeline is not redistributed. A plot generated here is not guaranteed to reproduce the final publication layout pixel for pixel.

## Verification and licensing

`PUBLIC_RELEASE_CHECKS.json` records the checks performed on this release. Full N=15/N=21 scans, optimization, COMSOL calculations and saved-data audits are **not rerun during packaging**. There is no continuous-error-bound claim from finite mesh differences.

No software license has been selected for this release. No third-party reference implementation or font files are included. Citation metadata will be updated when the associated paper has a final bibliographic record.
