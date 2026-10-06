# PWE versus three-dimensional physical-displacement MAC

## Public source port (2026-10-06)

The remainder is the historical calculation protocol. This public copy does
not claim that it performed that calculation or its original delivery.
MCST_DATA_ROOT selects a research data root containing the named runs/
directories, which are omitted from this source-only release. The frozen
circular-core hashes, physical reconstruction, field lineage checks and MAC
algorithm remain unchanged. The port removes manuscript/PDF input hashes and
private Git requirements, and accepts historical Windows provenance filenames
on other operating systems. Freeze records the current public implementation
and protocol bytes, not the original edited-source hashes. Fresh results and
audit JSON are written beside this source; do not reuse a historical frozen
input/audit JSON as proof for the port. Run compute_v2.py --stage freeze,
then --stage compute and --stage verify after supplying the saved inputs.

User-authorized bounded calculation, 2026-10-04 HK. This is a new analysis of
already completed spectra/vectors, not a new eigensolve or independent experiment.
Do not resume automation-3, launch COMSOL, overwrite raw files or edit the paper.

## Fixed comparisons (before computing overlaps)

At X = (pi/a,0), circular steel radius/a=.30, epoxy ell=1 um, steel ell=0:

| case | a (um) | h (um) | PWE order | reference |
|---|---:|---:|---:|---|
| baseline | 500 | 20 | 21 | saved M1L6 |
| small | 40 | 3.2 | 15 | saved M1L6 |
| thick | 75 | 24 | 15 | saved M1L6 |

Use both inverse-factorized MCST-FSDT and MCST-TSDT, with original archived
vectors, ordering, materials and phase convention. Compare PWE branches 1..6
against the six historically selected 3D flexural branches. The original 3D
solution numbers are baseline [1,2,3,4,5,6], small [1,2,3,5,8,9], thick
[1,2,6,7,8,10]. The two target branches are 4 and 5, fixed in advance. Do not
replace them by whichever branch produces the highest MAC.

## Reconstruction and inner product

The real representation stores interleaved (w,beta_x,beta_y) per reciprocal
vector. Restore complex Fourier coefficients using diag(1,i,i), as in the
archived assembler. In normalized coordinates X=x/a,Z=z/a,tau=h/a, use
exp(i(G+k).X). FSDT: u_alpha=Z beta_alpha, u_z=w. TSDT:
u_alpha=(Z-4Z^3/(3tau^2)) beta_alpha-4Z^3/(3tau^2) d_alpha w, u_z=w.
All displacement components share one arbitrary overall length/amplitude;
MAC eliminates that common factor. Never compare Fourier and FE coefficient
vectors directly, fit a field, shift the inclusion, or optimize phase conventions.

Evaluate on the exact existing 21x21x5 and 31x31x7 physical midpoints. Use
the actual exported density times volume for every component. Full 3-component
MAC is the primary metric. Retain all 6x6 entries, best counterparts, two edge
values, sampling differences, signed frequency errors and sampled norms.
Report mass-weighted norm of PWE/3D fields and numerical residuals separately.

## Checks and interpretation

- Freeze SHA256 of every input CSV/NPZ/config/manifest and implementation before
  analysis; verify field exports against their original completion hashes.
- Check a,h,r/a, material density, units, X coordinates, six solution numbers,
  original imaginary frequencies, saved PWE wavevector/order/source lineage.
- Compare closed reconstruction against the archived displacement operator,
  including complex coefficient conversion and through-thickness derivatives.
- Reassemble archived PWE K/M without solving: verify all eight saved vectors'
  residuals, M-orthogonality and frequency conversion. Do not change vectors.
- Phase invariance, permutation and invalid-weight algebra controls.
- Independently recompute overlaps by scalar complex sums rather than BLAS.
- MAC>0.95 and unique reciprocal maximum at the originally corresponding branch
  are descriptive identity screens, not a universal accuracy threshold.
- Absolute change <=1e-3 across the two existing sampling grids is a practical
  sampling screen, not a certified quadrature bound. Retain failures.
- Frequency accuracy and shape agreement are distinct: high MAC does not
  remove plate-model frequency discrepancy. No full-path shape, continuum or
  eigen-count/order convergence claim.

Outputs: reproducible script, frozen inputs, all MAC/frequency CSVs, detailed
JSON audit and a short report. Existing manuscript/PDF remain unchanged in this
calculation-only request. No new figure is needed to deliver these six comparisons.
