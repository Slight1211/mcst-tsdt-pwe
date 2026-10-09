# Wave Motion supporting numerical results

This collection contains **58 saved numerical files**: 39 CSV tables, 8 JSON
datasets, 4 numeric NPZ files and 7 mesh PNGs. These are selected spectra,
figure/table results, sampled fields and boundary coordinates from completed
research runs. No new solve, optimization or interpolation was performed for
publication, and the included numerical files are byte-identical copies.

`catalog.json` identifies each asset by its label, scientific role and the
figure, table or appendix it supports. Assets use collection-local names;
the original project directory structure is not distributed. The catalog's
`supports` entries describe the actual included coverage, not complete
reproduction of every figure or experiment.

Original-source indices, source-file hashes, internal paths, execution logs
and runtime records are not included. A complete file containing such
information is excluded, rather than edited to remove individual columns.
The numerical tables and datasets that remain are unmodified. The full
original records and provenance are retained by the authors locally.

The representative scale and thickness spectra used for plotting are in the
figure-specific JSON datasets. Their final display Gamma point repeats the
stored first Gamma point to close the path; it is not an independently solved
point. The additional all-branch frequency export and its original-source
index are not part of this collection. Frequency data use the units indicated
by their field or column names; do not confuse MHz with the dimensionless
frequency used in the scale plot.

To check file sizes and CSV/JSON/NPZ/PNG container structure, with Python 3.10+
and NumPy available, run from this directory:

```sh
python verify_release.py
```

This check is read-only. It does not access the network, install software,
import research models or execute a solver. It is a collection-structure
check, not an original-source hash check or independent scientific
reproduction. The public collection intentionally contains no original-source
hash manifest. Publication upload integrity is checked separately by the
authors against the local originals.

The repository's research code is unchanged. These renamed assets are intended
for inspecting the reported results; they do not recreate the historical
input directory expected by some scripts. Native COMSOL models, all historical
eigenvector archives, manuscripts, commercial software and credentials are not
distributed. No software or data license is added without author selection.
