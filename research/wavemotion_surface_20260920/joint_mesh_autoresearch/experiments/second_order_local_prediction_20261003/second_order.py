"""Bounded, post-hoc fixed-X prediction; no eigensolves or raw input writes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
RESEARCH = WORK.parent
DATA_ROOT = Path(os.environ.get('MCST_DATA_ROOT', str(RESEARCH))).expanduser().resolve()
ORIGINAL = DATA_ROOT / 'runs/manuscript_revision_20260925'
EXPECTED = {
    'prediction_checks.csv': 'aec5f6a0f7f591543d13e7a08ffdd448837a471b51b1c90192e4292676df4176',
    'edge_prediction_frozen.json': '804841bfdda9ebd02cb21f9a4e6b1062f2f052d6dfaeb389b7be2d1685bc5f4e',
}
SOURCE_VARIANT = 'public-source-port-20261006'


def sha(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required input missing: {path}. This source-only release omits results; set MCST_DATA_ROOT to a research data root containing runs/.')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = HERE / name
    if path.exists():
        raise FileExistsError(f'Refuse overwrite: {path}')
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def locked_inputs():
    sha(HERE / 'protocol.md')  # Freeze this public protocol, without private Git history.
    for name, expected in EXPECTED.items():
        assert sha(ORIGINAL / name) == expected, name
    base = json.loads((ORIGINAL / 'edge_prediction_frozen.json').read_text(encoding='utf-8'))
    with (ORIGINAL / 'prediction_checks.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    keys = {(r['parameter'], float(r['fractional_change']), int(r['baseline_band'])) for r in rows}
    assert keys == {(p, d, b) for p in ('h', 'ell') for d in (-.03, -.01, .01, .03) for b in (4, 5)}
    for r in rows:
        for k, value in r.items():
            if k not in ('parameter', 'scope'):
                assert np.isfinite(float(value)), k
        assert r['baseline_band'] == r['matched_band']
        assert float(r['mass_MAC']) >= .99
        assert float(r['max_residual']) <= 1e-7
        assert float(r['seed_agreement']) <= 1e-7
    return base, rows


def fit(base, rows):
    calibration = [r for r in rows if abs(float(r['fractional_change'])) == .01]
    assert len(calibration) == 8
    result = {}
    for p in ('h', 'ell'):
        slopes = base['S_h_f4_f5'] if p == 'h' else base['R_mc_f4_f5']
        result[p] = []
        for j, b in enumerate((4, 5)):
            rr = sorted((r for r in calibration if r['parameter'] == p and int(r['baseline_band']) == b),
                        key=lambda r: float(r['fractional_change']))
            assert len(rr) == 2 and {float(r['fractional_change']) for r in rr} == {-.01, .01}
            x = np.log1p([float(r['fractional_change']) for r in rr])
            y = np.log([float(r['actual_f_MHz']) / base['baseline_f4_f5_MHz'][j] for r in rr])
            t = float(2 * np.dot(x*x, y - slopes[j]*x) / np.dot(x*x, x*x))
            result[p].append({'band': b, 'S': slopes[j], 'T': t})
    return result, calibration


def selftests(base, rows):
    coefficients, _ = fit(base, rows)
    changed = [dict(r) for r in rows]
    for r in changed:
        if abs(float(r['fractional_change'])) == .03:
            r['actual_f_MHz'] = str(float(r['actual_f_MHz']) * 1.123)
    assert fit(base, changed)[0] == coefficients
    try:
        fit(base, [r for r in rows if not (r['parameter'] == 'h' and r['baseline_band'] == '4'
                                         and float(r['fractional_change']) == -.01)])
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing calibration negative control failed')
    scaled_base = dict(base, baseline_f4_f5_MHz=[v*1e6 for v in base['baseline_f4_f5_MHz']])
    scaled_rows = [dict(r, actual_f_MHz=str(float(r['actual_f_MHz'])*1e6)) for r in rows]
    scaled, _ = fit(scaled_base, scaled_rows)
    assert max(abs(scaled[p][j]['T']-coefficients[p][j]['T']) for p in coefficients for j in (0, 1)) < 1e-10
    x = np.log1p([-.01, .01])
    s, t = .74, -.18
    y = s*x + t*x*x/2
    assert abs(2*np.dot(x*x, y-s*x)/np.dot(x*x, x*x)-t) < 1e-12
    assert np.array_equal(np.exp(s*x), np.exp(s*x + 0*x*x/2))
    return dict(held_out_mutation='PASS', missing_calibration_rejected='PASS',
                unit_invariance='PASS', synthetic_quadratic='PASS', zero_curvature='PASS')


def freeze():
    base, rows = locked_inputs()
    coefficients, calibration = fit(base, rows)
    save('coefficients_frozen.json', {
        'status': 'FROZEN_BEFORE_NEW_SECOND_ORDER_EVALUATION',
        'analysis_intent': 'EXPLORATORY_POST_HOC_NOT_PROSPECTIVE',
        'source_variant': SOURCE_VARIANT, 'protocol_sha256': sha(HERE/'protocol.md'),
        'input_sha256': EXPECTED, 'code_sha256': sha(Path(__file__)),
        'baseline_f4_f5_MHz': base['baseline_f4_f5_MHz'],
        'baseline_X_J45_percent': base['baseline_X_J45_percent'],
        'calibration_changes': [-.01, .01], 'evaluation_changes': [-.03, .03],
        'coefficients': coefficients, 'calibration_rows': calibration,
        'selftests': selftests(base, rows),
    })
    print('PASS: input quality, protocol lock, calibration-only freeze and five selftests')


def evaluate():
    base, rows = locked_inputs()
    sha(HERE/'coefficients_frozen.json')
    frozen = json.loads((HERE/'coefficients_frozen.json').read_text(encoding='utf-8'))
    assert frozen.get('source_variant') == SOURCE_VARIANT, 'Run freeze with this public source; historical source hashes do not authenticate the port.'
    assert frozen['protocol_sha256'] == sha(HERE/'protocol.md'), 'Public protocol changed after freeze'
    assert frozen['code_sha256'] == sha(Path(__file__))
    assert frozen['coefficients'] == fit(base, rows)[0]
    j0 = base['baseline_X_J45_percent']
    result = []
    max_frequency_error, max_original_gap_error, max_actual_gap_error = 0., 0., 0.
    for p in ('ell', 'h'):
        for d in (-.03, -.01, .01, .03):
            rr = [next(r for r in rows if r['parameter'] == p and float(r['fractional_change']) == d
                       and int(r['baseline_band']) == b) for b in (4, 5)]
            x = np.log1p(d)
            c = frozen['coefficients'][p]
            f0 = np.array(base['baseline_f4_f5_MHz'])
            s, t = np.array([v['S'] for v in c]), np.array([v['T'] for v in c])
            f1, f2 = f0*np.exp(s*x), f0*np.exp(s*x + t*x*x/2)
            actual = np.array([float(r['actual_f_MHz']) for r in rr])
            gap = lambda f: float(200*(f[1]-f[0])/(f[1]+f[0]))
            ja, j1, j2 = gap(actual), gap(f1), gap(f2)
            max_frequency_error = max(max_frequency_error, max(abs(f1-np.array([float(r['predicted_f_MHz']) for r in rr]))))
            for r in rr:
                max_original_gap_error = max(max_original_gap_error, abs(j1-float(r['predicted_X_J45_percent'])))
                max_actual_gap_error = max(max_actual_gap_error, abs(ja-float(r['actual_X_J45_percent'])))
            assert max(abs(np.log(actual/f0)-np.array([float(r['actual_delta_ln_f']) for r in rr]))) < 1e-12
            row = {'parameter': p, 'fractional_change': d, 'split': 'calibration' if abs(d)==.01 else 'evaluation',
                   'baseline_J_percent': j0, 'actual_J_percent': ja, 'first_J_percent': j1, 'second_J_percent': j2,
                   'actual_delta_J_pp': ja-j0, 'first_delta_J_pp': j1-j0, 'second_delta_J_pp': j2-j0,
                   'first_abs_J_error_pp': abs(j1-ja), 'second_abs_J_error_pp': abs(j2-ja),
                   'first_relative_increment_error_percent': 100*abs(j1-ja)/abs(ja-j0),
                   'second_relative_increment_error_percent': 100*abs(j2-ja)/abs(ja-j0),
                   'first_direction_agrees': bool(np.sign(j1-j0)==np.sign(ja-j0)),
                   'second_direction_agrees': bool(np.sign(j2-j0)==np.sign(ja-j0)),
                   'min_original_mass_MAC': min(float(r['mass_MAC']) for r in rr),
                   'max_original_residual': max(float(r['max_residual']) for r in rr)}
            for j, b in enumerate((4, 5)):
                row.update({f'actual_f{b}_MHz': actual[j], f'first_f{b}_MHz': f1[j], f'second_f{b}_MHz': f2[j],
                            f'first_log_f{b}_error': float(np.log(f1[j]/actual[j])),
                            f'second_log_f{b}_error': float(np.log(f2[j]/actual[j]))})
            result.append(row)
    assert max_frequency_error < 1e-10 and max_original_gap_error < 1e-10 and max_actual_gap_error < 1e-10
    output = HERE/'predictions.csv'
    assert not output.exists()
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader(); writer.writerows(result)
    summaries = {}
    for p in ('ell', 'h'):
        summaries[p] = {}
        for split in ('calibration', 'evaluation'):
            rr = [r for r in result if r['parameter']==p and r['split']==split]
            summaries[p][split] = {'points': len(rr)}
            for model in ('first', 'second'):
                summaries[p][split][model] = {
                    'max_abs_J_error_pp': max(r[f'{model}_abs_J_error_pp'] for r in rr),
                    'max_relative_increment_error_percent': max(r[f'{model}_relative_increment_error_percent'] for r in rr),
                    'directions_agree': all(r[f'{model}_direction_agrees'] for r in rr),
                    'max_abs_log_frequency_error': max(abs(r[f'{model}_log_f{b}_error']) for r in rr for b in (4, 5)),
                }
    htest = [r for r in result if r['parameter']=='h' and r['split']=='evaluation']
    useful = all(r['second_abs_J_error_pp'] < r['first_abs_J_error_pp'] and r['second_direction_agrees'] for r in htest)
    useful &= summaries['h']['evaluation']['second']['max_relative_increment_error_percent'] <= 10
    report = {'status': 'PASS' if useful else 'FAIL', 'analysis_intent': frozen['analysis_intent'],
              'source_variant': SOURCE_VARIANT, 'input_sha256': EXPECTED,
              'frozen_sha256': sha(HERE/'coefficients_frozen.json'), 'predictions_sha256': sha(output),
              'first_order_reproduction_max_frequency_MHz': max_frequency_error,
              'first_order_reproduction_max_gap_pp': max_original_gap_error,
              'actual_gap_reproduction_max_pp': max_actual_gap_error,
              'summaries': summaries, 'new_eigensolves': 'NOT_RUN', 'mixed_derivative': 'NOT_RUN',
              'full_path_extrema': 'NOT_RUN', 'prospective_new_validation': 'NOT_RUN',
              'perturbation_vector_quality': 'Original recorded audits only; not independently re-evaluated',
              'scope': 'same-model fixed X N15 inverse PWE, post-hoc +/-1% calibration and +/-3% evaluation'}
    save('assessment.json', report)
    print(json.dumps(report, indent=2))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sha(HERE/'predictions.csv'); sha(HERE/'independent_audit.json')
    with (HERE/'predictions.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    audit = json.loads((HERE/'independent_audit.json').read_text(encoding='utf-8'))
    assert audit.get('source_variant') == SOURCE_VARIANT, 'Re-audit predictions with the public source before plotting.'
    assert audit['auditor_code_sha256'] == sha(HERE/'independent_audit.py'), 'Public auditor changed; repeat the audit.'
    assert audit['status']=='PASS' and audit['predictions_sha256']==sha(HERE/'predictions.csv')
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.bbox': 'tight', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    for ax, p, title in zip(axes, ('ell', 'h'), ('(a) Internal length', '(b) Thickness')):
        rr = sorted([r for r in rows if r['parameter']==p], key=lambda r: float(r['fractional_change']))
        xx = [100*float(r['fractional_change']) for r in rr]
        # Lines only connect observations/predictions for visual guidance, not new solved data.
        for name, style, color, label in [('first', '--o', '#0072B2', 'First order'),
                                         ('second', ':^', '#009E73', 'Second order'),
                                         ('actual', '-s', '#D55E00', 'PWE re-solved')]:
            ax.plot(xx, [float(r[f'{name}_delta_J_pp']) for r in rr], style, color=color,
                    label=label, linewidth=1.4, markersize=4)
        ax.axhline(0, color='.6', linewidth=.6)
        ax.set(title=title, xlabel='Parameter change (%)',
               ylabel=r'$\Delta J_{45}(X)$ (percentage points)', xticks=xx)
        ax.grid(axis='y', alpha=.18)
    axes[0].legend(frameon=False, fontsize=8)
    for suffix in ('pdf', 'png'):
        target = HERE/f'second_order_prediction.{suffix}'
        assert not target.exists()
        fig.savefig(target, dpi=300)
    plt.close(fig)
    print('PASS: two-panel chart from independently audited CSV, vector PDF + 300 dpi PNG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'evaluate', 'plot'])
    stage = parser.parse_args().stage
    globals()[stage]()
