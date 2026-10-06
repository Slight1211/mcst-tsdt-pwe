"""Scalar audit independent of the production NumPy fitting/prediction helpers."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

here = Path(__file__).resolve().parent
data_root = Path(os.environ.get('MCST_DATA_ROOT', str(here.parents[2]))).expanduser().resolve()
original = data_root/'runs/manuscript_revision_20260925'
SOURCE_VARIANT = 'public-source-port-20261006'
def sha(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required input missing: {path}. Set MCST_DATA_ROOT for omitted runs/ inputs and run second_order.py freeze/evaluate for public outputs.')
    return hashlib.sha256(path.read_bytes()).hexdigest()
for path in [original/'edge_prediction_frozen.json', original/'prediction_checks.csv',
             here/'coefficients_frozen.json', here/'predictions.csv', here/'assessment.json']:
    sha(path)
base = json.loads((original/'edge_prediction_frozen.json').read_text(encoding='utf-8'))
frozen = json.loads((here/'coefficients_frozen.json').read_text(encoding='utf-8'))
with (original/'prediction_checks.csv').open(newline='', encoding='utf-8') as stream:
    source = list(csv.DictReader(stream))
with (here/'predictions.csv').open(newline='', encoding='utf-8') as stream:
    results = list(csv.DictReader(stream))
report = json.loads((here/'assessment.json').read_text(encoding='utf-8'))
assert frozen.get('source_variant') == report.get('source_variant') == SOURCE_VARIANT, 'Re-freeze/evaluate using the public source; historical implementation hashes no longer apply.'
assert frozen['code_sha256'] == sha(here/'second_order.py')
assert frozen['protocol_sha256'] == sha(here/'protocol.md')
for name, value in frozen['input_sha256'].items():
    assert sha(original/name)==value
assert report['frozen_sha256']==sha(here/'coefficients_frozen.json')
assert report['predictions_sha256']==sha(here/'predictions.csv')
max_t, max_f, max_j, max_relative = 0., 0., 0., 0.
j0 = base['baseline_X_J45_percent']
curve = {}
for p in ('ell', 'h'):
    curve[p] = []
    slopes = base['R_mc_f4_f5'] if p=='ell' else base['S_h_f4_f5']
    for i, band in enumerate((4, 5)):
        rr = [r for r in source if r['parameter']==p and int(r['baseline_band'])==band
              and abs(float(r['fractional_change']))==.01]
        assert len(rr)==2
        # Compute half-curvature as least-squares projection then double.
        xx = [math.log(1+float(r['fractional_change'])) for r in rr]
        yy = [math.log(float(r['actual_f_MHz'])/base['baseline_f4_f5_MHz'][i]) for r in rr]
        half = math.fsum(v*v*(y-slopes[i]*v) for v, y in zip(xx, yy))/math.fsum(v**4 for v in xx)
        t = 2*half
        max_t = max(max_t, abs(t-frozen['coefficients'][p][i]['T']))
        curve[p].append((slopes[i], t))
for r in results:
    p, d = r['parameter'], float(r['fractional_change'])
    x = math.log(1+d)
    f0 = base['baseline_f4_f5_MHz']
    actual = [float(next(v for v in source if v['parameter']==p and float(v['fractional_change'])==d
                         and int(v['baseline_band'])==b)['actual_f_MHz']) for b in (4, 5)]
    models = {'actual': actual,
              'first': [f0[i]*math.exp(curve[p][i][0]*x) for i in (0, 1)],
              'second': [f0[i]*math.exp(curve[p][i][0]*x+curve[p][i][1]*x*x/2) for i in (0, 1)]}
    gap = lambda f: 200*(f[1]/f[0]-1)/(f[1]/f[0]+1)
    ja = gap(actual)
    for model, values in models.items():
        for i, b in enumerate((4, 5)):
            max_f = max(max_f, abs(values[i]-float(r[f'{model}_f{b}_MHz'])))
        jj = gap(values)
        max_j = max(max_j, abs(jj-float(r[f'{model}_J_percent'])),
                    abs(jj-j0-float(r[f'{model}_delta_J_pp'])))
        if model!='actual':
            err = abs(jj-ja)
            relative = 100*err/abs(ja-j0)
            max_j = max(max_j, abs(err-float(r[f'{model}_abs_J_error_pp'])))
            max_relative = max(max_relative, abs(relative-float(r[f'{model}_relative_increment_error_percent'])))
assert max_t<1e-10 and max_f<1e-10 and max_j<1e-10 and max_relative<1e-6
assert len(results)==8
output = {'status': 'PASS', 'source_variant': SOURCE_VARIANT, 'auditor': 'Independent scalar math implementation, automated not human approval',
          'coefficients_max_abs_difference': max_t, 'frequency_max_difference_MHz': max_f,
          'gap_max_difference_pp': max_j, 'relative_error_max_difference_percent': max_relative,
          'predictions_sha256': sha(here/'predictions.csv'), 'frozen_sha256': sha(here/'coefficients_frozen.json'),
          'auditor_code_sha256': sha(Path(__file__)), 'rows': len(results)}
assert not (here/'independent_audit.json').exists()
(here/'independent_audit.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
print(json.dumps(output, indent=2))
