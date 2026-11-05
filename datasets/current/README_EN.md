# Composite microplate band gaps and fixed-area shape design: data and code

**Dataset 1.1.2** · [Download](https://github.com/Slight1211/mcst-tsdt-pwe/releases/tag/data-v1.1.2)

Authors: Xin Chen, Jun Cai, Alex Kalamkarov and Abdolhamid Akbarzadeh.

This collection contains nine figures, nine tables, five optimized interfaces and
their supporting calculations. Steel–epoxy cells illustrate a plane-wave model
combining modified couple stress theory (MCST) and Reddy third-order plate
kinematics. The data cover spectra, modal fields, sensitivity checks, fixed-area
interface optimization and three-dimensional model comparisons.

## Find the data

Use `06_provenance/FIGURE_TABLE_MAP.csv` to find each figure or table's data,
supporting records and processing code. Paths are relative to the unpacked
collection.

| Directory | Contents |
|---|---|
| `01_current_figures/` | Six main figures, three supplementary figures and source data |
| `02_current_tables/` | Two main tables and seven supplementary tables as CSVs |
| `03_optimized_shapes/` | Boundaries, spectra and settings, grouped by material length |
| `04_reproduction/` | Verification tools, processing scripts and numerical code |
| `05_source_records/` | Original calculation records, grouped by study and run |
| `06_provenance/` | Source mappings, file identities and checksums |

The five designs use `a = 75 µm`, `h = 6 µm`, steel area fraction `0.09π`,
`N = 15`, bounds `b = 0.24`, 48 interface vertices and six shape coordinates.
At `ell_um = 0, 0.5, 1, 2, 5`, their relative gaps on 153 wavevectors are
34.58%, 34.31%, 33.22%, 27.05% and 2.96%, respectively. Full-precision values
are in `03_optimized_shapes/summary.csv`. Table 2 uses its stated 45-point plate
evaluation and Γ, X, M solid comparison.

The six solid–solid MAC records for Table S5(a) are in
`02_current_tables/Table_S5/solid_solid_mac/`, grouped as `large_cell`,
`small_cell` and `thick_plate`. The adjacent `solid_solid_mac_recheck.csv`
compares source-derived values with the displayed table.
`06_provenance/MAC_RECORDS.csv` gives the file identities and hashes.

Table CSVs labelled `displayed_values` retain the article's rounding. Use the
mapped unrounded records for recalculation. Empty cells and `<` bounds retain
their meanings in the displayed tables.

## Verify and reproduce

Unpack to a short path and run from the collection root:

```text
python 04_reproduction/verify_dataset.py --root .
python 04_reproduction/verify_mac_records.py --root .
```

These standard-library Python tools check file integrity and saved MAC table
arithmetic. Use `FIGURE_TABLE_MAP.csv` to select processing scripts and
`SOURCE_MAP.csv` to locate their original inputs. Array and plotting scripts
use NumPy and Matplotlib; PWE calculations also use SciPy. Three-dimensional
solves require a licensed COMSOL installation. Follow each driver's recorded
settings and write new calculations to a separate output directory.

## Units and citation

`ell_um` is the epoxy material length in micrometres. Vertex coordinates are
divided by `a`; `kx_a`, `ky_a` and `Omega` are dimensionless, with
`Ω = 2πaf / sqrt(E_B/ρ_B)`. Here `B` denotes the epoxy matrix. Frequency columns
labelled Hz, kHz or MHz are dimensional. Relative gap
`J = 200(U−L)/(U+L)` is a percentage; gains in `pp` are percentage points.
Energy fractions, logarithmic sensitivities and MAC are dimensionless.

Use `CITATION.cff` to cite this dataset. Original data, figures and documentation
are CC BY 4.0; original executable code is MIT. Licence scope and attribution
are specified in `LICENSES.md`.
