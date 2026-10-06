# Public source release — 2026-10-06

The release preserves the relative source layout of the steel–epoxy MCST/PWE research. It does not publish the complete working directory or an executed-results archive.

## Adaptations

- The current PWE assembly, nodal shape/gradient algorithms, analytical thickness operators and the two frozen circular kernels are copied without numerical changes. The source manifest records byte-level hashes against the original project.
- A small `checks/smoke_check.py` and `examples/circular_point.py` provide software checks without starting a production scan, optimization or COMSOL job.
- Selected plotting scripts create missing output-parent directories. No spectra, band selection or plotting data are altered.
- Data-dependent prediction and physical-field-pairing scripts accept `MCST_DATA_ROOT`. Scientific saved-input and frozen-kernel hash checks are retained. Public reruns make new source/protocol manifests; they do not inherit a success status or source-hash claim from a previous execution.
- Checks that depended only on an unpublished manuscript/PDF or the original private Git ancestry are removed from the public experiment drivers. Historical protocols remain descriptions of the original experiments; new public execution is identified separately.
- A COMSOL runtime helper uses local `COMSOL_BIN` and `COMSOL_PREFS_FILE` configuration and reports missing external inputs. `COMSOL_BIN` is the installation's bin directory containing `comsolbatch.exe`. Java paths are made configurable where appropriate. Existing physical parameters, checkpoint relationships and scientific-input checks are preserved.

The manifest distinguishes identical copies, adapted files and files added for this release. Do not interpret an adapted file's new SHA256 as the original experiment's source hash.

## Not included / not rerun

Manuscript sources/PDFs, research-state logs, `.venv`, credentials, private preferences, third-party reference implementations, fonts, compiled Java classes, saved spectra, CSV/NPZ field data and `.mph` checkpoints are excluded. Source workflows requiring these inputs stop until the user supplies them.

Publication packaging does not rerun the full PWE parameter scans, N=15/N=21 production calculations, nodal optimization, 3D mesh studies or saved-data validation. Static parsing, small algebraic regression, a one-wavevector software example and portability checks are recorded in `PUBLIC_RELEASE_CHECKS.json`. These tests do not extend the paper's scientific validation.

The original research project is not modified by this release. No license is added without an author-selected license.
