# Second-order fixed-X local prediction (locked before new analysis)

## Public source port (2026-10-06)

The remainder is the historical study protocol, not a claim that this public
copy performed the original delivery. The port removes the private Git commit
lookup, manuscript delivery and automation requirements. It preserves the two
original numerical input hashes, calibration split, formulas and quality gates.
Set MCST_DATA_ROOT to the research data root containing runs/; these numerical
results are omitted from this source-only release. Fresh public freeze/evaluate
outputs are written beside this source and record source_variant,
protocol_sha256 and the current implementation hash. Original frozen source
hashes do not authenticate the edited public implementation. Run freeze,
evaluate, independent_audit.py, then plotting; do not reuse old audit JSON as a
new public audit. No Git checkout or manuscript/PDF is required by this port.

Date: 2026-10-03, Asia/Hong_Kong. User request: autoresearch, do second order.
Status: EXPLORATORY_POST_HOC. The +/-3% outcomes were already observed and the
87.62% first-order thickness error motivated this analysis. A calibration/evaluation
split here is not prospective independent validation. No new eigenproblem is solved.
Original H1/H2/H3 protocol and all raw spectra remain unchanged.

## Question and evidence

Can a quadratic log-frequency correction, calibrated only at +/-1%, reduce
the already-observed +/-3% fixed-X gap-increment errors without changing the
original frozen first derivatives? Case: inverse MCST-TSDT N15, a=75 um,
h=6 um, r/a=0.30, epoxy ell=1 um, X=(pi/a,0), modes 4 and 5.
Existing inputs under ../runs/manuscript_revision_20260925 (relative to research root):

- prediction_checks.csv: SHA256 aec5f6a0f7f591543d13e7a08ffdd448837a471b51b1c90192e4292676df4176.
- edge_prediction_frozen.json: SHA256 804841bfdda9ebd02cb21f9a4e6b1062f2f052d6dfaeb389b7be2d1685bc5f4e.

These record 16 matched modal responses (2 parameters x 4 changes x 2 modes),
not 16 independent geometries. +/-1% calibrates; +/-3% evaluates. Existing
mass MAC, residual and seed-agreement audits are checked, not newly recomputed
from unavailable perturbation vectors. Source hashes bind the original study.

## Model (sole primary method; no tuning)

x=ln(1+fractional_change), y_n=ln(f_n/f_n0).
Keep S_h,n and S_ell,n=R_mc,n from the original frozen JSON.
For each parameter and mode, fit only T_n to the two +/-1% responses:

T_n = 2 sum[x_i^2 (y_i-S_n*x_i)] / sum[x_i^4].

Second order: f_n^(2)=f_n0 exp(S_n*x + T_n*x^2/2).
First order: f_n^(1)=f_n0 exp(S_n*x) (the original comparator, NOT linear J).
Both use J=200(f5-f4)/(f5+f4), in percent; Delta J=J-J0 in percentage points.
T estimates local log-frequency curvature; it is not an analytic eigenvalue
second derivative. No refitting of S, mode swapping, smoothing or interpolation
of solved outcomes. No mixed h/ell cross derivative is inferred.

## Locked checks and decisions

1. Exact input SHA256, 16 rows, complete unique keys, finite values; matched bands
   equal original bands; MAC>=0.99, residual<=1e-7, seed agreement<=1e-7.
2. Reproduce original first-order frequencies within 1e-10 MHz and J within
   1e-10 percentage points; verify duplicate gap records and actual log shifts.
3. Save calibration-only rows and frozen coefficients in a separate stage before
   evaluation. Mutation of +/-3% outcomes must not change fitted coefficients.
4. Synthetic quadratic recovery, zero-curvature first-order equivalence, frequency
   unit-scale invariance, missing calibration-row rejection. Independently audit
   all coefficients/predictions using a different scalar algebra implementation.
5. Primary usefulness gate: for BOTH h +/-3% evaluation points, absolute Delta J
   error decreases, predicted direction agrees, and maximum relative increment
   error <=10%. Report FAIL unchanged if this threshold is missed.
6. Report ell +/-3% as a control and +/-1% calibration errors separately; do not
   pool them to advertise held-out accuracy. Report log-frequency errors too.
7. Preserve original first-order negative result (87.62%) in paper. Even PASS is
   local same-model post-hoc evidence, not 3D validation, full-path extremum
   prediction, continuum error bound or general second-order convergence proof.

## Delivery and stop

Commit this protocol before calculating new coefficients or predictions; commit
results separately. Export complete CSV, frozen JSON, independent audit, PNG/vector
plot, update findings/log/state, add a compact manuscript result with appendix
method and evaluation table. Existing first-order table stays. Compile existing
TEX, render affected pages and inspect. Native compiler preferred; existing
XeLaTeX allowed if unavailable. No installs, new tasks or COMSOL launches.
Pause automation-3 after this bounded delivery (including a reported FAIL if needed).
Human scientific/submission approval remains pending; no submission readiness claim.
