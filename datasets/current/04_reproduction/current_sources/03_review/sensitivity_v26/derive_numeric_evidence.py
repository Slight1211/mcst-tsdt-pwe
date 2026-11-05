"""Recalculate saved first-order checks; no model assembly or eigensolve."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
HIST=ROOT/'02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/manuscript_revision_20260925'
ARCH=ROOT/'06_release/github_v20/repository/data/wave_motion_20261009_v1/assets'
UNIFIED=ROOT/'02_evidence/scientific_data/output/review/analytic_sensitivity_unification_20261010_v1'
SHAPERUN=ROOT/'02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source(p):return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)}
def J(f4,f5):return 200*(f5-f4)/(f5+f4)
raw=rows(HIST/'prediction_checks.csv')
assert len(raw)==16
assert sha(HIST/'prediction_checks.csv')==sha(ARCH/'item-043.csv')
assert sha(HIST/'inverse_laurent_assessment.json')==sha(ARCH/'item-039.json')
p1=read(HIST/'p1_assessment.json')
energy={int(d['band']):d for d in p1['inverse_energy']['bands']}
laurent={int(d['band']):d for d in read(HIST/'inverse_laurent_assessment.json')['bands']}
S={('ell',n):energy[n]['R_mc_matrix'] for n in [4,5]}
S.update({('h',n):laurent[n]['kappa_3']-laurent[n]['kappa_minus1']-laurent[n]['eta_3'] for n in [4,5]})
f0={n:energy[n]['frequency_MHz'] for n in [4,5]}
baseJ=J(f0[4],f0[5])
plot=[]
for rownum,r in enumerate(raw,2):
    p=r['parameter']; n=int(r['baseline_band']);delta=float(r['fractional_change'])
    baseline=float(r['baseline_f_MHz']); actual=float(r['actual_f_MHz'])
    assert abs(baseline-f0[n])<1e-13 and int(r['matched_band'])==n
    x=math.log1p(delta); pred=S[p,n]*x;direct=math.log(actual/baseline)
    assert abs(direct-float(r['actual_delta_ln_f']))<1e-15
    plot.append(dict(parameter=p,fractional_change=delta,change_percent=100*delta,band=n,
        delta_ln_parameter=x,analytic_sensitivity=S[p,n],predicted_delta_ln_f=pred,
        actual_delta_ln_f=direct,signed_log_error_prediction_minus_solve=pred-direct,
        absolute_log_error=abs(pred-direct),baseline_f_MHz=baseline,
        predicted_f_MHz=baseline*math.exp(pred),actual_f_MHz=actual,
        mass_MAC=float(r['mass_MAC']),matched_band=int(r['matched_band']),
        max_residual=float(r['max_residual']),seed_agreement=float(r['seed_agreement']),
        mass_orthogonality=float(r['mass_orthogonality']),
        prediction_direction_matches_solve=(pred*direct>0),source_csv_row=rownum))
plot.sort(key=lambda r:({'ell':0,'h':1}[r['parameter']],r['fractional_change'],r['band']))
with (HERE/'plot_source.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(plot[0]));w.writeheader();w.writerows(plot)
groups=[]
for p in ['ell','h']:
    deriv=400*f0[4]*f0[5]/(f0[4]+f0[5])**2*(S[p,5]-S[p,4])
    for delta in [-.03,-.01,.01,.03]:
        pair={r['band']:r for r in plot if r['parameter']==p and r['fractional_change']==delta}
        predJ=J(pair[4]['predicted_f_MHz'],pair[5]['predicted_f_MHz'])
        actJ=J(pair[4]['actual_f_MHz'],pair[5]['actual_f_MHz'])
        predD=predJ-baseJ; actD=actJ-baseJ
        groups.append(dict(parameter=p,fractional_change=delta,baseline_J_percent=baseJ,
            predicted_J_percent=predJ,actual_J_percent=actJ,
            predicted_delta_J_pp=predD,actual_delta_J_pp=actD,
            first_linear_J_increment_pp=deriv*math.log1p(delta),
            signed_J_error_prediction_minus_solve_pp=predJ-actJ,
            absolute_J_error_pp=abs(predJ-actJ),
            relative_J_increment_error_percent=100*abs(predJ-actJ)/abs(actD),
            gap_direction_matches_solve=(predD*actD>0),
            linear_J_direction_matches_solve=(deriv*math.log1p(delta)*actD>0)))
# Check recomputed first-order values against the frozen unified file, not second-order columns.
unified={ (r['parameter'],float(r['fractional_change'])):r for r in rows(UNIFIED/'predictions_unified.csv') }
for r in plot:
    u=unified[r['parameter'],r['fractional_change']]
    assert abs(r['predicted_delta_ln_f']-float(u[f"first_delta_ln_f{r['band']}"]))<1e-14
    assert abs(r['actual_delta_ln_f']-float(u[f"actual_delta_ln_f{r['band']}"]))<1e-14
for g in groups:
    u=unified[g['parameter'],g['fractional_change']]
    assert abs(g['absolute_J_error_pp']-float(u['first_abs_J_error_pp']))<1e-12
response=[]
with np.load(ARCH/'item-010.npz') as saved_fields:
    for n in [4,5]:
        a=laurent[n];e=energy[n]
        Rsh=float(np.mean(saved_fields[f'shear{n-4}']))
        assert abs(Rsh-e['R_shear_spatial'])<1e-15
        assert abs(S['h',n]-a['S_h_Laurent'])<1e-15
        assert abs(a['kappa_minus1']+a['kappa_1']+a['kappa_3']-1)<1e-14
        assert abs(a['eta_1']+a['eta_3']-1)<1e-14
        response.append(dict(band=n,frequency_MHz=f0[n],R_mc_matrix=e['R_mc_matrix'],
            S_ell=S['ell',n],R_mc_spatial=e['R_mc_spatial'],R_sh_spatial=Rsh,
            R_sh_matrix=None,R_sh_matrix_status='NOT_STORED_IN_DISCOVERED_SAVED_RECORDS',
            R_sh_source='mean(shear0/shear1) of historical item-010.npz; inverse dual-field reconstruction on512x512 grid',
            kappa_minus1=a['kappa_minus1'],kappa_1=a['kappa_1'],kappa_3=a['kappa_3'],
            eta_1=a['eta_1'],eta_3=a['eta_3'],S_h_analytic=S['h',n],
            S_h_old_frozen_matrix_FD=a['S_h_frozen_matrix_FD'],
            classic_spatial_vs_matrix_relative_integration_error=e['classic_relative_integration_error'],
            mc_spatial_vs_matrix_relative_integration_error=e['mc_relative_integration_error']))
shape=read(SHAPERUN/'manifest.json'); shapechecks=[]
for gate in shape['gradient_gates']:
    for r in gate['records']:
        denom=math.sqrt(sum(x*x for x in r['central']))
        err=math.sqrt(sum((x-y)**2 for x,y in zip(r['analytic'],r['central'])))/denom
        assert abs(err-r['relative_error'])<1e-20
        shapechecks.append(dict(shape=gate['name'],kxa_kya=gate['k'],direction=gate['direction'],
            step=r['step'],analytic=r['analytic'],central=r['central'],relative_vector_error=err))
active=[]
for kp in read(SHAPERUN/'independent_validation/audit.json')['eigenpair_checks']:
    for r in kp['active_directional_derivatives']:
        err=math.sqrt(sum((x-y)**2 for x,y in zip(r['analytic'],r['central'])))/math.sqrt(sum(x*x for x in r['central']))
        active.append(dict(kxa_kya=kp['k'],direction=r['direction'],relative_vector_error=err))
maxerr=max(plot,key=lambda r:r['absolute_log_error'])
summaries={p:dict(
    maximum_abs_log_frequency_error=max(r['absolute_log_error'] for r in plot if r['parameter']==p),
    maximum_relative_frequency_error_percent=max(100*abs(r['predicted_f_MHz']/r['actual_f_MHz']-1) for r in plot if r['parameter']==p),
    minimum_mass_MAC=min(r['mass_MAC'] for r in plot if r['parameter']==p),
    maximum_absolute_J_error_pp=max(r['absolute_J_error_pp'] for r in groups if r['parameter']==p),
    maximum_relative_J_increment_error_percent=max(r['relative_J_increment_error_percent'] for r in groups if r['parameter']==p),
    all_frequency_directions_correct=all(r['prediction_direction_matches_solve'] for r in plot if r['parameter']==p),
    all_J_directions_correct=all(r['gap_direction_matches_solve'] for r in groups if r['parameter']==p)) for p in ['ell','h']}
evidence=dict(status='PASS_RECOMPUTED_SAVED_FIRST_ORDER_AND_SHAPE_CHECKS',human_verified=False,
    scope=dict(a_um=75,h_um=6,r_over_a=.3,ell_epoxy_um=1,ell_steel_um=0,N=15,k='fixed X',bands=[4,5],fractional_perturbations=[-.03,-.01,.01,.03],total_modal_comparisons=16),
    formulas=dict(x='ln(1+delta)',prediction='delta_ln_f=S_p,n*x; f_pred=f0*exp(S_p,n*x)',
        direct='delta_ln_f=ln(f_actual/f0)',
        log_error='delta_ln_f_pred-delta_ln_f_actual; dimensionless, not an eigenpair residual',
        relative_frequency_error_percent='100*abs(f_pred/f_actual-1)',
        J='200*(f5-f4)/(f5+f4)',
        relative_J_increment_error_percent='100*abs(J_pred-J_actual)/abs(J_actual-J0)',
        dJ_dlnp='400*f4*f5/(f4+f5)^2*(S_p5-S_p4)'),
    response_records=response,baseline_J_percent=baseJ,
    gap_derivatives={p:dict(dJ_dlnp_pp=400*f0[4]*f0[5]/(f0[4]+f0[5])**2*(S[p,5]-S[p,4]),
        d_absolute_width_dlnp_MHz=f0[5]*S[p,5]-f0[4]*S[p,4]) for p in ['ell','h']},
    global_summary=dict(maximum_abs_delta_ln_f_error=maxerr['absolute_log_error'],maximum_error_case=maxerr,
        minimum_mass_MAC=min(r['mass_MAC'] for r in plot),all16_frequency_directions_correct=all(r['prediction_direction_matches_solve'] for r in plot),
        all8_J_directions_correct=all(r['gap_direction_matches_solve'] for r in groups)),
    parameter_summaries=summaries,gap_checks=groups,
    first_order_only=True,second_order_coefficients_used=False,
    posthoc_record_note='The unified dataset also contains fitted second-order columns; those were not used. New first-order curves use the existing analytic modal slopes with saved direct frequencies, not a newly fitted slope.',
    old_prediction_column_warning='Original prediction_checks.csv h prediction columns used old frozen-matrix-FD slopes. We recompute from original actual frequencies using analytic Laurent slopes; this changes maximum absolute log error from8.314346883955825e-5 to8.33425110687e-5 and h maximum absolute gap error from0.00364613289369 to0.00364619807825pp.',
    R_sh_caution='Available saved Rsh values are spatial dual-field integrals, not a separately stored matrix ratio. A total classical-energy spatial/matrix check below0.025% is not a dedicated shear-ratio error bound. Caption must identify reconstructed Rsh or omit it; do not invent an exact matrix Rsh.',
    shape_gradient_checks=shapechecks,maximum_nonsymmetry_shape_gradient_relative_error=max(x['relative_vector_error'] for x in shapechecks),
    extra_terminal_symmetry_point_checks=active,maximum_extra_symmetry_point_relative_error=max(x['relative_vector_error'] for x in active),
    shape_gradient_error_definition='Euclidean norm of analytic minus central directional derivatives of frequency branches4/5, divided by norm of the central two-vector. Nonsymmetry checks cover two shapes, one direction and two step sizes; Gamma/X extra checks are separate.',
    sources=[source(p) for p in [HIST/'prediction_checks.csv',HIST/'inverse_laurent_assessment.json',HIST/'p1_assessment.json',UNIFIED/'predictions_unified.csv',UNIFIED/'assessment_unified.json',ARCH/'item-010.npz',ARCH/'item-031.csv',ARCH/'item-032.json',ARCH/'item-039.json',ARCH/'item-043.csv',SHAPERUN/'manifest.json',SHAPERUN/'independent_validation/audit.json']],
    plot_source=source(HERE/'plot_source.csv'),
    archive_mapping=dict(raw16modal_rows='Historical assets/item-043.csv (same bytes as prediction_checks.csv)',analytic_Sh_weights='Historical assets/item-039.json',
        modal_Sell_and_baseline='Historical assets/item-032.json baseline; p1 inverse_energy matrix participation',Rsh_fields='Historical assets/item-010.npz; item-031.csv R_sh_inverse_dual'),
    new_model_assemblies=0,new_eigensolves=0,new_COMSOL_runs=0)
(HERE/'NUMERIC_EVIDENCE.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=evidence['status'],rows=len(plot),summary=evidence['global_summary'],parameters=summaries,shape_max=evidence['maximum_nonsymmetry_shape_gradient_relative_error']),ensure_ascii=False,indent=2))
