"""Readable point-wise comparison; connectors indicate predictions only."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
for required in ('predictions.csv', 'independent_audit.json'):
    if not (here/required).is_file():
        raise FileNotFoundError(f'Missing public prediction output: {here/required}. Run second_order.py freeze/evaluate and independent_audit.py first; this release contains source only.')
with (here/'predictions.csv').open(encoding='utf-8', newline='') as stream:
    rows = list(csv.DictReader(stream))
audit = json.loads((here/'independent_audit.json').read_text(encoding='utf-8'))
assert audit.get('source_variant') == 'public-source-port-20261006', 'Re-audit with this public source before plotting.'
assert audit['auditor_code_sha256'] == hashlib.sha256((here/'independent_audit.py').read_bytes()).hexdigest(), 'Public auditor changed; repeat the audit.'
assert audit['status']=='PASS'
assert hashlib.sha256((here/'predictions.csv').read_bytes()).hexdigest()==audit['predictions_sha256']
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
for ax, p, title in zip(axes, ('ell', 'h'), ('(a) Internal length', '(b) Thickness')):
    subset = sorted([r for r in rows if r['parameter']==p], key=lambda r: float(r['fractional_change']))
    xx = [100*float(r['fractional_change']) for r in subset]
    ax.plot(xx, [float(r['first_delta_J_pp']) for r in subset], '--o',
            color='#0072B2', markersize=4, linewidth=1.4, label='First order')
    ax.plot(xx, [float(r['second_delta_J_pp']) for r in subset], ':^',
            color='#009E73', markersize=7, markerfacecolor='none', linewidth=1.5, label='Second order')
    ax.plot(xx, [float(r['actual_delta_J_pp']) for r in subset], 's',
            color='#D55E00', markersize=3.2, label='Existing PWE solves', zorder=4)
    ax.axhline(0, color='.6', linewidth=.6)
    ax.set(title=title, xlabel='Parameter change (%)',
           ylabel=r'$\Delta J_{45}(X)$ (percentage points)', xticks=xx)
    ax.grid(axis='y', alpha=.18)
axes[0].legend(frameon=False, fontsize=8)
for suffix in ('pdf', 'png'):
    output = here/f'second_order_prediction_v2.{suffix}'
    assert not output.exists()
    fig.savefig(output, dpi=300)
plt.close(fig)
print('PASS: v2 point-wise plot, solved data not connected or interpolated')
