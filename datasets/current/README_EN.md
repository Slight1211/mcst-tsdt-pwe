# Composite microplate band gaps and fixed-area shape design: data and code

**Dataset 1.1.1** · [Release and download](https://github.com/Slight1211/mcst-tsdt-pwe/releases/tag/data-v1.1.1)

Authors: Xin Chen, Jun Cai, Alex Kalamkarov and Abdolhamid Akbarzadeh.

This collection contains nine figures, nine tables, five optimized interfaces and
their supporting calculations. Steel–epoxy cells illustrate a plane-wave model
combining modified couple stress theory (MCST) and Reddy third-order plate
kinematics. The archive includes spectra, modal fields, sensitivity checks,
optimization records and three-dimensional model comparisons.

## Find the data

Start with `06_provenance/FIGURE_TABLE_MAP.csv`: each figure or table is linked to
its data, supporting records and processing code. Paths are relative to the
unpacked collection.

| Directory | Contents |
|---|---|
| `01_current_figures/` | Six main figures, three supplementary figures and source data |
| `02_current_tables/` | Two main tables and seven supplementary tables as CSVs |
| `03_optimized_shapes/` | Boundaries, spectra and settings, grouped by material length |
| `04_reproduction/` | Verification tools, processing scripts and numerical code |
| `05_source_records/` | Original calculation records, grouped by study and run |
| `06_provenance/` | Source mappings, file hashes and the historical-data index |

The five designs use `a = 75 µm`, `h = 6 µm`, steel area fraction `0.09π`,
`N = 15`, bounds `b = 0.24`, 48 interface vertices and six shape coordinates.
At `ell_um = 0, 0.5, 1, 2, 5`, their relative gaps on 153 wavevectors are
34.58%, 34.31%, 33.22%, 27.05% and 2.96%, respectively. Full-precision values
are in `03_optimized_shapes/summary.csv`. Table 2 uses its stated 45-point plate
evaluation and Γ, X, M solid comparison; retain these point-set distinctions.

Table S5(a)'s six original solid–solid MAC records are in
`02_current_tables/Table_S5/solid_solid_mac/`, grouped as `large_cell`,
`small_cell` and `thick_plate`. The adjacent `solid_solid_mac_recheck.csv`
compares the derived values with the displayed table.
`06_provenance/MAC_RECORDS.csv` records each file's identity and hash.

Table CSVs labelled `displayed_values` retain the article's rounding. Use the
mapped unrounded records for recalculation. An empty cell is not zero, and `<`
denotes an upper bound. Earlier settings and cross-applied designs remain in
the source records and historical index.

## Verify and reproduce

Unpack to a short path, then run from the collection root:

```text
python 04_reproduction/verify_dataset.py --root .
```

The checker uses Python's standard library. It verifies saved files without
starting a solver. Use `FIGURE_TABLE_MAP.csv` for processing scripts and
`SOURCE_MAP.csv` for original locations. Array and plotting scripts use NumPy
and Matplotlib; PWE calculations also use SciPy. New three-dimensional solves
require a separately licensed COMSOL installation. Follow each driver's
recorded settings and write new calculations to a separate output directory.

## Units and reuse

`ell_um` is the epoxy material length in micrometres. Vertex coordinates are
divided by `a`; `kx_a`, `ky_a` and `Omega` are dimensionless, with
`Ω = 2πaf / sqrt(E_B/ρ_B)`; `B` denotes the epoxy matrix in these examples.
Frequency columns explicitly labelled Hz, kHz or MHz
are dimensional. Relative gap `J = 200(U−L)/(U+L)` is a percentage; gains in
`pp` are percentage points. Energy fractions, logarithmic sensitivities and MAC
are dimensionless. File headers and individual folder notes give further detail.

Use `CITATION.cff` to cite this release. Original data, figures and documentation
are CC BY 4.0; original executable code is MIT, as detailed in `LICENSES.md`.
[Version 1.1.0](https://github.com/Slight1211/mcst-tsdt-pwe/releases/tag/data-v1.1.0)
remains available. [Zenodo DOI 10.5281/zenodo.23888401](https://doi.org/10.5281/zenodo.23888401)
identifies the earlier 1.0.0 archive, not this release.
