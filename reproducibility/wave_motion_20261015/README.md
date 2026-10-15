# Wave Motion supporting numerical results and reproducibility addendum

This additive collection supplies saved-data postprocessing for the analytic-sensitivity prediction results (Figure 9 and the local-prediction appendix table), together with four homogeneous-solid analytical checks (the homogeneous-solid table in Appendix B). It supplements the [selected numerical collection](../../data/wave_motion_20261009_v1/README.md); all files in that earlier collection remain unchanged. See also the [repository's core source code](../../README.md).

The Python commands below need only Python 3.10 or newer and its standard library. They read saved numerical results, do not solve an eigenproblem, and refuse to overwrite an existing output directory. Run from the repository root and choose unused output directory names:

```sh
python reproducibility/wave_motion_20261015/reproduce_prediction.py --data-root data/wave_motion_20261009_v1 --output results/prediction_new
python reproducibility/wave_motion_20261015/verify_uniform.py --output results/uniform_new
```

The checked reference outputs are in `prediction_recomputed/` and `uniform_recomputed/`. These are deterministic postprocessing results, not newly computed spectra. This addendum's Python scripts are newly authored portable entry points, not byte-identical copies of the historical private runners. They do not replace the original scientific kernels.

## Local prediction: inputs, units and interpretation

The inputs are already public at [commit 9988331](https://github.com/Slight1211/mcst-tsdt-pwe/tree/998833149606a0a9324d133f324eb3ceae6f78b2/data/wave_motion_20261009_v1). `--data-root` denotes that collection, with its `assets/` subdirectory. The script reads:

| Public asset | Fields used |
| --- | --- |
| `item-032.json` | Baseline fourth/fifth-band frequencies and couple-stress energy fractions |
| `item-039.json` | Analytic thickness sensitivities `S_h_Laurent` |
| `item-043.csv` | Actual frequencies at thickness/inner-scale perturbations of +/-1% and +/-3% |
| `item-015.csv` | Actual frequencies at thickness perturbations of +/-2% |

Historical finite-difference slopes and prediction columns in these inputs are not used. In particular, the historical prediction table `item-016.csv` is not the current analytic-sensitivity Figure 9 table. The addendum recomputes predictions from the actual saved frequencies and analytic first derivatives, without modifying any old asset.

For fractional perturbation `epsilon`, write `x=ln(1+epsilon)`. Each frequency uses `ln(f/f0) = S*x + T*x^2/2`, so the first- and second-order predicted frequencies are `f0*exp(S*x)` and `f0*exp(S*x+T*x^2/2)`. `S_h` is the analytic logarithmic thickness derivative, and `S_ell` is the saved couple-stress energy fraction. `T` is fitted using only the +/-1% frequencies. The +/-2% thickness and +/-3% thickness/inner-scale frequencies are held out from that fit. The relative interval is `J = 200*(f5-f4)/(f5+f4)` in percent. A difference in `J` is in percentage points; the relative error in that difference is a distinct percentage. The baseline is the circular-cell sensitivity case, not the initial nodal-optimization polygon. These predictions concern the fourth and fifth bands at fixed X, without re-searching perturbed full-Brillouin-zone extrema.

Frequencies are in MHz in the prediction inputs and outputs. Perturbations and frequency coefficients are dimensionless. Recomputing a prediction from saved spectra is distinct from rerunning its PWE solve. The +/-2% cases were originally obtained as separate held-out checks; this public postprocessing of those saved cases is retrospective.

## Homogeneous periodic solid: complete saved cases

`uniform/` contains five unchanged CSV exports for each prescribed pair `n=4,6` and `ell/a=0.05,0.10`, including all four cases rather than a filter on their `passed` status. Their names are `bulk_n{4,6}_ell{0p05,0p10}.{selected,checks,mesh,selection,discretization}.csv`.

The cube has side `a=100 micrometres`, Young's modulus `E=4.35 GPa`, Poisson's ratio `nu=4.35/(2*1.59)-1`, density `rho=1180 kg/m^3`, and wavevector `k*a/pi=(0.18,0.11,0.07)`. All three directions are Bloch-periodic. The measured meshes contain 64 or 216 hexahedra, respectively. Displacement and auxiliary rotation use order 2, and the multiplier uses order 1; the recorded COMSOL shape-property exports are retained.

Writing `mu=E/[2*(1+nu)]`, `lambda=E*nu/[(1+nu)*(1-2*nu)]`, and `k=norm(k)`, the reference squared angular frequencies are:

```text
omega_T^2 = (mu/rho) * (k^2 + ell^2*k^4/4)
omega_L^2 = ((lambda+2*mu)/rho) * k^2
```

| File suffix | Meaning |
| --- | --- |
| `selected.csv` | Two transverse waves and one longitudinal wave, with frequencies and physical diagnostics |
| `checks.csv` | Recorded numerical tolerances, transverse splitting and amplitude-Gram check, measured mesh counts and acceptance status |
| `mesh.csv` | Measured element counts by topology |
| `selection.csv` | Measured opposite-face and mesh-edge selection counts |
| `discretization.csv` | Recorded finite-element shape properties |

Frequencies here are **Hz**, unlike the prediction assets. All relative errors, polarization fractions, projection fractions, Gram quantities and splitting measures are dimensionless ratios, not percent. Empty longitudinal-wave `mcst_increment_relative_error` and `constraint_error_rotation_normalized` fields are not applicable, not zero. `raw_mode` is the local solution number within this independent cube; `solution_vector_length` is the saved solution-vector length, not the count after periodic elimination.

`verify_uniform.py` checks all four saved cases and computes the manuscript summary. Its maximum frequency error covers both length scales and all three waves; its maximum rotation-constraint error covers both scales and both transverse waves. The latter is displayed in percent only in the summary. The paired-length metric is:

```text
E_Deltaell = abs((mean(omega_T_num^2 at ell2) - mean(omega_T_num^2 at ell1))
                 / [(mu*k^4/(4*rho))*(ell2^2-ell1^2)] - 1)
```

Each mean averages the two transverse squared angular frequencies. This paired metric differs from the individual-mode classical-to-MCST increment diagnostic in `selected.csv`. The physical rotation constraint is `norm(psi/ell - curl(u)/2)/norm(curl(u)/2)`; it is not an algebraic eigensolver residual.

These data validate the homogeneous fully periodic solid. They do not supply the composite-plate checkpoint models or the separate saved-field exports used for the plate rotation-constraint and cross-method overlap checks. The earlier collection's catalog specifies its other selected supporting results.

## Optional COMSOL source

`comsol/UniformMcstPeriodic.java` is a portable publication adaptation of the independent cube builder. Only comments and output-directory handling are changed; the material, mesh, weak form, eigensolver settings, mode selection and numerical thresholds are retained. The historical four solves used COMSOL 6.0.0.318. This adapted source is supplied for inspection and potential reruns with a licensed COMSOL installation; it was **not** used for a new COMSOL solve during this addendum's packaging.

Set `MCST_BULK_N` to `4` or `6`, `MCST_BULK_RATIO` to `0.05` or `0.10`, `MCST_BULK_OUTPUT_DIR` to an existing directory of your choice, and `MCST_BULK_PREFIX` to a new filename prefix containing only letters, digits, `_` or `-`. Compile and run with your installation's supported COMSOL Java tools. A full rerun consists of the four combinations, run **serially** with distinct prefixes. The source creates new files, does not load or save historical `.mph` models, and retains diagnostics before reporting numerical acceptance failures. No installation or commercial software is redistributed.

Native model files, historical checkpoint hashes, original path indexes, execution logs and machine-specific settings are not part of this addendum. No software or data license is added; the authors have not selected one. A Git commit identifies the exact collection version; no DOI or independent archival deposit is claimed.
