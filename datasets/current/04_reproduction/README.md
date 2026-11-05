# Reproduction

Run these commands from the complete, unpacked dataset:

```text
python 04_reproduction/verify_dataset.py --root .
python 04_reproduction/verify_mac_records.py --root .
```

Both tools use Python's standard library. The first checks file hashes; the
second also checks Table S5(a)'s values against the six saved MAC records.

`current_sources/` contains figure-processing, sensitivity and table-export
scripts. Use `06_provenance/FIGURE_TABLE_MAP.csv` to locate their inputs and
`06_provenance/SOURCE_MAP.csv` for original file locations. Assemble the mapped
inputs in a separate working directory, respecting each script's relative paths.
The table-export script reads manuscript TeX; exported CSVs are directly available
in `02_current_tables/`.

`original_code/` contains the PWE and COMSOL drivers and their requirements.
The `scientific_reports_20261104` directory provides the b=0.24 optimizer,
two initial coordinate sets per material length and saved-data checks.

Array processing uses NumPy and plotting uses Matplotlib; PWE solves also use
SciPy. New solid-model calculations require a separately licensed COMSOL
installation. Use the settings recorded with each calculation.
