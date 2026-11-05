# Data and code for size-dependent flexural band gaps and fixed-area shape design of periodic composite microplates

**Dataset release 1.1.0.**

This collection brings the figures, tables, optimized interfaces and supporting
computational records into one directory structure. Steel–epoxy cells provide the
numerical example for an inverse-factorized plane-wave model combining modified
couple stress theory (MCST) with Reddy third-order plate kinematics.

Authors: Xin Chen, Jun Cai, Alex Kalamkarov and Abdolhamid Akbarzadeh.

Release page:
[data-v1.1.0](https://github.com/Slight1211/mcst-tsdt-pwe/releases/tag/data-v1.1.0).
The source repository is
[Slight1211/mcst-tsdt-pwe](https://github.com/Slight1211/mcst-tsdt-pwe).

## Start with a figure, table or material length

Open `06_provenance/FIGURE_TABLE_MAP.csv` and find the display identifier
(`Figure 1`–`Figure 6`, `Figure S1`–`Figure S3`, `Table 1`–`Table 2`, or
`Table S1`–`Table S7`). The entry connects the display to its data files,
supporting records and plotting or extraction code. Paths are relative to the
unpacked collection unless an entry identifies an original source path.

| Directory | Contents | Typical starting point |
|---|---|---|
| `01_current_figures/` | Six main figures, three supplementary figures and their available source data | A spectrum, parameter scan, modal field, optimization trajectory or shape panel |
| `02_current_tables/` | CSV versions of two main tables and seven supplementary tables | Modal sensitivities, design gains, frequency checks and solid-model comparisons |
| `03_optimized_shapes/` | Selected boundaries and spectra at each of five material lengths | The shape and band gap for one value of `ell_um` |
| `04_reproduction/` | The recorded public code snapshot, current processing scripts and verification tools | Integrity checks, saved-data processing or a new calculation |
| `05_source_records/` | Original supporting records from Data 1–8 and Data 9, grouped into short run folders | Full-precision data, run settings and numerical provenance |
| `06_provenance/` | Display mappings, file hashes, source identities and historical-data index | Trace a convenient copy back to its original record |

Figure 1 is a geometry and wavevector schematic. Figures 2 and 3 show model
comparisons and parameter scans. Figure 4 and Table 1 concern modal energy and
parameter sensitivity. Figure 5 and Table 2 concern fixed-area design gains and
solid-model comparisons. Figure 6 compares the independently optimized
boundaries at the five material lengths.

Table CSVs reproduce the displayed table values, including their rounding.
Use the mapped source records for full-precision calculations. A displayed
`<` bound is a bound, and an empty cell is not zero. The mapping identifies
where a table is a display transcription rather than a new numerical result.

Four original full-precision solid–solid MAC records for the large-cell and
small-cell cases in Table S5(a) are not included in this collection. Their
display-rounded table values are supplied; the expected original filenames and
hashes are listed in `06_provenance/MISSING_SOURCE_RECORDS.csv`. The two
corresponding thick-plate records were recovered and hash-checked.

## The five optimized interfaces

All five current designs use cell size `a = 75 µm`, thickness `h = 6 µm`, steel
area fraction `0.09π`, Fourier order `N = 15`, a 48-vertex interface with six
shape coordinates, coordinate bounds `b = 0.24`, and 153 sampled wavevectors.
Each material length was optimized from two initial geometries; the reported
shape is the converged result with the larger relative gap.

| Epoxy material length `ell_um` | Relative gap on 153 points, % |
|---:|---:|
| 0 | 34.58 |
| 0.5 | 34.31 |
| 1 | 33.22 |
| 2 | 27.05 |
| 5 | 2.96 |

Find the corresponding entry in `03_optimized_shapes/` and its source mapping.
The selected vertex and spectrum files retain their numerical precision.
The table above is rounded for navigation. The initial boundary is an
equal-area regular 48-gon. The solid comparisons cover the initial and optimized
interfaces at `ell_um = 1` and `2` at Γ, X and M. Table 2 reports plate values
on its stated 45-point set, whereas Figure 6 and the shape summary use 153
points; preserve this point-set distinction when comparing values.

## Columns, units and quantities

CSV files contain a header row. JSON files carry calculation settings and
summary values; NPZ files contain NumPy arrays. Original column and array names
are retained in the source records.

| Name or notation | Meaning and units |
|---|---|
| `ell_um` | Epoxy material length, in micrometres; the steel material length is zero in these examples |
| `a`, `h`, `r` | Cell size, plate thickness and circular-inclusion radius; use the unit stated in the file or map |
| `x_over_a`, `y_over_a` | Interface vertex coordinates divided by cell size; dimensionless |
| `kx_a`, `ky_a` | Products of wavevector components with cell size; dimensionless, with X at `(π, 0)` and M at `(π, π)` |
| `point` | Saved wavevector-row identifier; it is not a frequency |
| `Omega1`–`Omega6` | Ordered dimensionless modal frequencies, `Ω = 2πaf / sqrt(E_B/ρ_B)`; `B` denotes the epoxy matrix |
| `L`, `U` | Lower and upper gap edges on the specified point set, in the file's frequency normalization |
| `J` or a relative-gap column | `200(U − L)/(U + L)`, reported as a percentage; `J_45` denotes the fourth-to-fifth-band gap, whereas `J153_percent` identifies evaluation on 153 points |
| Gain or gap difference in `pp` | Difference of two percentage-valued relative gaps, in percentage points |
| `R_mc`, `R_sh` | Couple-stress and classical shear energy fractions; dimensionless |
| `S_ell`, `S_h` | Logarithmic frequency sensitivities, `∂ ln f / ∂ ln p`; dimensionless |
| MAC | A dimensionless modal correlation; the source record specifies the weighting or physical-field projection |

Frequency columns labelled Hz, kHz or MHz are dimensional frequencies, not
`Omega` values. Use SI units in the normalization formula. In particular,
`kx_a = π` is different from a coordinate already divided by `π`.
Negative signed relative separation denotes overlap of the sampled bands.

## Verify and reproduce

After unpacking, run the integrity checker from the collection root:

```text
python 04_reproduction/verify_dataset.py --root .
```

The file inventory is `06_provenance/FILE_MANIFEST.csv`; file hashes are in
`06_provenance/SHA256SUMS.txt`. Verification checks saved files and does not
start a numerical solver. The display map identifies the input and processing
script for each figure or table.

`06_provenance/SOURCE_MAP.csv` maps every relocated source record to its original
archive and path. Numerical file contents are unchanged. On Windows, extract to
a short destination such as `C:\mcst_data` because the original code snapshot
retains some long module paths.

The processing stages have different requirements:

- **Read CSV/JSON files and verify hashes:** Python's standard library is sufficient
  for the supplied integrity checker.
- **Process saved spectra, fields or figures:** use the mapped Python script and
  its dependencies, including NumPy and Matplotlib where used. This does not
  require a COMSOL calculation.
- **Run new PWE spectra or shape optimization:** use the relevant numerical
  driver with NumPy, SciPy and its recorded settings in a new output directory.
- **Build or solve a three-dimensional model:** use the corresponding Java/model
  driver with a separately installed, licensed COMSOL environment. Displacement
  comparisons use the associated saved fields or newly exported fields.

The later design calculations used Python 3.14.0, NumPy 2.4.3, SciPy 1.17.1
and COMSOL 6.2.0.290. The plotting environment records Matplotlib 3.10.8.
Original computation records retain their actual software versions, including
earlier COMSOL scans. The public code snapshot keeps its original relative
layout; follow its workflow instructions rather than running a historical
controller over archived results. New runs should write to separate directories.

## Current data and historical records

The folders `01_current_figures/`–`03_optimized_shapes/` identify the results
used in the current article. The source records also retain earlier parameter
bounds, different point sets, truncation checks and evaluations of a fixed shape
at another material length. These records describe their original calculations.
They are indexed separately from the five current `b = 0.24` designs.

The earlier immutable data-and-code archive is
[Zenodo 1.0.0, DOI 10.5281/zenodo.23888401](https://doi.org/10.5281/zenodo.23888401).
It contains Data 1–8 and an earlier public code snapshot. That DOI identifies
the earlier archive; it is not a DOI for this release or a journal article.
Its original citation and attribution remain in the provenance records.

For this collection, use `CITATION.cff` and the release tag `data-v1.1.0`.
Original numerical data, figures and documentation are CC BY 4.0; original
executable code is MIT. See `LICENSES.md` for the scope of each licence.

