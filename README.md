# Composite microplate band gaps: data and code

[Dataset 1.1.2](https://github.com/Slight1211/mcst-tsdt-pwe/releases/tag/data-v1.1.2)
| [English guide](datasets/current/README_EN.md)
| [中文说明](datasets/current/README_ZH.md)

Data and computational sources for size-dependent flexural spectra, modal
sensitivities and fixed-area interface design in periodic composite microplates.
The plate model combines modified couple stress theory (MCST), Reddy third-order
kinematics and inverse-factorized plane-wave discretization. Steel-epoxy cells
provide the numerical examples, with selected three-dimensional comparisons.

## Dataset contents

The complete download contains six main figures, three supplementary figures,
two main tables, seven supplementary tables, five optimized interfaces and their
supporting records. Files are grouped by use:

| Directory | Contents |
|---|---|
| `01_current_figures/` | Figures and plotting data |
| `02_current_tables/` | Table CSVs and supporting records |
| `03_optimized_shapes/` | Interface vertices, spectra and optimization settings |
| `04_reproduction/` | Verification, processing and solver code |
| `05_source_records/` | Original calculation records |
| `06_provenance/` | Figure/table mappings, source identities and checksums |

The five designs use material lengths of 0, 0.5, 1, 2 and 5 micrometres,
with b=0.24, N=15 and a 153-wavevector evaluation. Table 2 reports its stated
45-point plate evaluation and Gamma, X, M solid comparison.

The [repository index](datasets/current/README.md) provides table CSVs,
optimized boundaries and spectra, six solid-solid MAC records, and processing
sources. Use the complete release ZIP for all figure data and calculation records.

## Verification and reproduction

Unpack the complete dataset to a short path. From its root, run:

```text
python 04_reproduction/verify_dataset.py --root .
python 04_reproduction/verify_mac_records.py --root .
```

The first command checks file integrity; the second also checks Table S5(a)
against the saved MAC records. Both use Python's standard library.
The [reproduction guide](datasets/current/04_reproduction/README.md) describes
processing scripts, numerical dependencies and solver settings.

## Citation and licences

Authors: Xin Chen, Jun Cai, Alex Kalamkarov and Abdolhamid Akbarzadeh.
Use [CITATION.cff](CITATION.cff) to cite this dataset. Original data, figures
and documentation use CC BY 4.0; original executable code uses MIT.
See [licence details](datasets/current/LICENSES.md).
