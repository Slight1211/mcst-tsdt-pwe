# Frozen thickness +/-2% numerical evaluation

## Public source port (2026-10-06)

The remainder records the historical experiment; this edited source has not
repeated its eigensolves or completed its manuscript/automation delivery.
The public port requires no private Git history. MCST_DATA_ROOT selects the
read-only research data root containing runs/ and joint_mesh_autoresearch/
experiments/. The original coefficient JSON, baseline NPZ, first-order JSON/CSV
and frozen circular-core hashes remain mandatory; regenerated public coefficient
JSON has different provenance bytes and is not a replacement for that archived
input. These numerical files are omitted from the source-only release.
Freeze creates new public protocol and implementation hashes; historical driver
hashes are not asserted for edited copies. Compute writes fresh raw results to
the public code root's runs/second_order_h2_validation_20261003, never the
external data root, and refuses existing raw results. Use audit_v2.py for the
independent audit and verify_saved.py for its subsequent read-only repetition.
No numerical algorithm, seed, physical constant or acceptance gate is changed.

2026-10-03, Asia/Hong_Kong. User request: recompute the previously uncomputed
2% thickness deviations. Commit this protocol and implementation before solving.

## Scope and frozen model

Two new real MCST-TSDT inverse-PWE spectra, N=15, fixed X=(pi/a,0),
a=75 um, radius/a=.30, epoxy ell=1 um, steel ell=0. Thickness only:
5.88 um (-2%) and 6.12 um (+2%). This is a prospective evaluation at new
points within the same numerical model, NOT external/3D/experimental validation.
The second-order model itself was chosen after seeing earlier +/-3% errors.
Retain that historical post-hoc designation and all old outputs unchanged.

Use existing coefficients_frozen.json in ../second_order_local_prediction_20261003:
SHA256 e633d4ef514b756f2c6e18bd82ef8e2c2fa6052f9c7d96ee070eef679376800a.
Baseline f4/f5=5.736427842197187/7.603394218073132 MHz.
S_h4/h5=.7434011573001114/.745877020231189;
T_h4/h5=-.10463177523306072/-.18382557251925813.
No recalibration, point selection, interpolation or criterion changes.
Freeze both first/second-order predictions before new eigensolves:
x=log(1+d); f1=f0*exp(S*x); f2=f0*exp(S*x+.5*T*x*x).
J=200*(f5-f4)/(f5+f4), in percent, delta J=J-J0, in percentage points.
Absolute error=abs(J_prediction-J_solve), in percentage points.
Increment relative error=100*absolute_error/abs(J_solve-J0), in percent.
This is not relative error in the total J nor a frequency error.

## Solver/version and quality criteria

Use exact archived solver snapshots in ../runs/scale_addendum_20260927/sources:
inverse_circle_bending.py SHA256
679a57197224239f2bd9e06603cb414ae2873fada1b5c47554d8f39d222745e2;
real_circle_bending.py SHA256
d49fd896b4bdbef91f2fec33a3cb633746b249c89f82669954c3634e70a484d4.
These match the historical original run_manifest.json, not later current sources.
Freeze all imported dependencies, driver, baseline NPZ, original slope JSON/CSV,
protocol and evaluator hashes. Verify baseline reconstructed Rayleigh frequencies
against archived mode4/5 (<1e-7 relative). Inspect actual module origins.

K=classical_inverse(prepare(15,h/a,.30,TSDT),X)+(ell/h)^2*KB;
M=assemble_real(15,X,h/a,.30,TSDT).M. Preserve physical constants, units,
MCST-only host contribution, mass, geometry and representation unchanged.
One process, one BLAS thread, two points serial, no time cutoff, no COMSOL.
ARPACK extracts16 eigenpairs with sigma=-.001, tol=1e-11, independent seeded
starts20260921/20260922, ncv40 then80 if needed, same historical extraction.
Both starts agree for first8 to relative1e-7; all positive/finite real eigenvalues.
Mass orthogonality <1e-7; first8 normalized algebraic residuals <1e-7.
Save complete16 eigenpairs/vectors, both starts, residuals and source provenance.
Match baseline modes4/5 to first8 with baseline-M mass MAC; require matching
sorted4/5 and MAC>=.99. If any criterion fails preserve raw evidence, report
FAIL and diagnose before interpreting; do not silently rerun/relabel.

## Evaluation, audit and completion

Both second-order absolute errors must decrease versus first-order, both predicted
delta-J directions must agree with actual and maximum increment relative error
must be <=10%. This is the unchanged previous practical usefulness screen,
not a certified error bound. Numerical-quality PASS and usefulness PASS separate.
Report both points regardless of result, including small-denominator effects.
An independent scalar evaluator recomputes predictions/errors directly, and
independently checks saved eigenvectors/residuals/MAC against reassembled K/M.
Raw files live in ../runs/second_order_h2_validation_20261003, never overwrite.
Deliver raw spectrum CSV/NPZ, frozen prediction JSON, audit CSV/JSON, commands,
source/input hashes, research record and compact same-source manuscript update.
Pause existing automation-3 when bounded delivery is complete. Full path/BZ,
higher N, mixed h-ell derivatives and new 3D/experiment validation NOT_RUN.
