"""Compact baseline spectrum and matched-constitutive band-edge discrepancies."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure_style_v31 import COL, style_context, tidy_axes, export_figure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'latex/figures'
REVIEW = ROOT / '03_review/figure2_labels_v43'
REVIEW.mkdir(parents=True, exist_ok=True)
SPECTRA = ROOT / '03_review/figures_v31/baseline/figure1_spectra_v31.csv'
EDGES = ROOT / '02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/edge_focus_20260926/section41_path_edges.csv'


def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


data, edges = rows(SPECTRA), rows(EDGES)
assert len(data) == 468 and len(edges) == 4
models = ['Classic-FSDT', 'Classic-TSDT', 'MCST-FSDT', 'MCST-TSDT', '3D classical', '3D MCST']
arrays = {}
for model in models:
    selected = [r for r in data if r['model'] == model]
    matrix = np.array([[float(next(r['frequency_kHz'] for r in selected
                                   if int(r['path_index']) == i and int(r['band']) == j))
                        for j in range(1, 7)] for i in range(13)])
    assert matrix.shape == (13, 6)
    arrays[model] = matrix
for r in edges:
    model = r['model']
    ref = '3D classical' if model.startswith('Classic') else '3D MCST'
    assert abs(arrays[model][:, 3].max() - float(r['PWE_L_kHz'])) < 1e-9
    assert abs(arrays[model][:, 4].min() - float(r['PWE_U_kHz'])) < 1e-9
    for band, edge, field in [(3, 'L', 'L_error_percent'), (4, 'U', 'U_error_percent')]:
        f = arrays[model][:, band].max() if edge == 'L' else arrays[model][:, band].min()
        g = arrays[ref][:, band].max() if edge == 'L' else arrays[ref][:, band].min()
        assert abs(100 * abs(f / g - 1) - float(r[field])) < 1e-10
lower = max(a[:, 3].max() for a in arrays.values())
upper = min(a[:, 4].min() for a in arrays.values())
gap = 200 * (upper - lower) / (upper + lower)
assert abs(lower - 446.5950873958621) < 1e-9
assert abs(upper - 597.6029509128907) < 1e-9
k = np.array([[float(next(r['kx_a'] for r in data if r['model'] == 'MCST-TSDT' and int(r['path_index']) == i)),
               float(next(r['ky_a'] for r in data if r['model'] == 'MCST-TSDT' and int(r['path_index']) == i))]
              for i in range(13)])
distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(k, axis=0), axis=1))] / np.pi
assert np.allclose(distance[[0, 4, 8, 12]], [0, 1, 2, 2 + np.sqrt(2)], atol=1e-12)
W, H = 164., 68.

with style_context():
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    a = fig.add_axes([14 / W, 13 / H, 84 / W, 43 / H])
    b = fig.add_axes([127 / W, 13 / H, 34 / W, 43 / H])
    for ax, letter in [(a, 'a'), (b, 'b')]:
        tidy_axes(ax)
        ax.text(-.15 if ax is a else -.26, 1.045, letter, transform=ax.transAxes,
                fontsize=9, fontweight='bold', ha='left', va='bottom', clip_on=False)
    for value in distance[[4, 8]]:
        a.axvline(value, color='#D9D9D9', lw=.6, zorder=0)
    a.axhspan(lower, upper, color=COL['mcst'], alpha=.12, ec='none', zorder=0)
    for j in range(6):
        a.plot(distance, arrays['Classic-TSDT'][:, j], color=COL['classic'], lw=1.15)
        a.plot(distance, arrays['MCST-TSDT'][:, j], color=COL['mcst'], lw=1.3)
        for model, marker, size in [('3D classical', 's', 3.5), ('3D MCST', 'o', 3.1)]:
            a.plot(distance, arrays[model][:, j], color='black', marker=marker,
                   ms=size, mfc='white', mec='black', mew=.65, ls='none', zorder=4)
    a.text(.055, (lower + upper) / 2, f'Common gap {gap:.1f}%',
           transform=a.get_yaxis_transform(), fontsize=7, color=COL['mcst'], va='center')
    a.set(xlim=(-.04, distance[-1] + .04), ylim=(-15, 850),
          xticks=distance[[0, 4, 8, 12]], xticklabels=[r'$\Gamma$', '$X$', '$M$', r'$\Gamma$'],
          yticks=[0, 200, 400, 600, 800], xlabel='Bloch path', ylabel='Frequency (kHz)')
    a.set_title('Six flexural branches', fontsize=8, pad=5)
    error_records = []
    for index, r in enumerate(edges):
        y = 3 - index
        color = COL['classic'] if r['model'].startswith('Classic') else COL['mcst']
        for key, marker, fill, name in [('L_error_percent', 'o', color, 'lower'),
                                       ('U_error_percent', '^', 'white', 'upper')]:
            value = float(r[key])
            b.plot(value, y, ls='none', marker=marker, ms=4., color=color, mfc=fill, mec=color, mew=.8)
            error_records.append({'model': r['model'], 'edge': name, 'absolute_discrepancy_percent': value})
    b.set(xlim=(.80, 1.12), ylim=(-.55, 3.55), xticks=[.8, .9, 1., 1.1],
          yticks=[3, 2, 1, 0],
          yticklabels=[r['model'].replace('Classic-', 'Classical ').replace('MCST-', 'MCST ') for r in edges],
          xlabel='Absolute discrepancy (%)')
    b.tick_params(axis='y', length=0, pad=3)
    b.set_title('Band-edge discrepancy', fontsize=8, pad=5)
    handles = [Line2D([], [], color=COL['classic'], lw=1.15, label='Classical TSDT'),
               Line2D([], [], color=COL['mcst'], lw=1.3, label='MCST TSDT'),
               Line2D([], [], color='black', ls='none', marker='s', mfc='white', ms=3., label='3D classical'),
               Line2D([], [], color='black', ls='none', marker='o', mfc='white', ms=3., label='3D MCST')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.365, .995), ncol=4,
               handlelength=1.5, columnspacing=.9, handletextpad=.45, borderaxespad=0, fontsize=7)
    edge_handles = [Line2D([], [], color='black', ls='none', marker='o', ms=3.5, label='Lower'),
                    Line2D([], [], color='black', ls='none', marker='^', mfc='white', ms=3.5, label='Upper')]
    fig.legend(handles=edge_handles, loc='upper center', bbox_to_anchor=(.87, .995), ncol=2,
               handlelength=.8, columnspacing=.7, handletextpad=.3, borderaxespad=0, fontsize=7)
    meta = export_figure(fig, OUT / 'figure2_baseline_v43')
    assert not meta['clipped_text'] and meta['minimum_visible_font_pt'] >= 7
    plt.close(fig)

with (REVIEW / 'edge_error_records.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(error_records[0]))
    writer.writeheader()
    writer.writerows(error_records)
meta.update({
    'status': 'GENERATED_PENDING_VISUAL_REVIEW', 'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [SPECTRA, EDGES]},
    'script_sha256': sha(Path(__file__)), 'all_source_frequency_values': 468,
    'displayed_full_branch_values': 312, 'band_edge_discrepancy_values': 8,
    'displayed_models_in_a': ['Classic-TSDT', 'MCST-TSDT', '3D classical', '3D MCST'],
    'models_in_b': [r['model'] for r in edges],
    'path_coordinate': 'Cumulative length in wavevector space divided by pi/a; stored13points, no additional samples.',
    'PWE_N': 21, 'solid_mesh': 'M2L4', 'common_gap_kHz': [lower, upper], 'common_gap_percent': gap,
    'discrepancy_reference': 'Each plate model uses the solid model with the same constitutive law.',
    'new_numerical_runs': False,
})
(REVIEW / 'PROVENANCE.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
captions = {
    'en': r'''Large-cell baseline spectrum and band-edge comparison for $(a,h,r)=(500,20,150)\,\mu\mathrm m$. (a) The first six flexural branches along $\Gamma$--$X$--$M$--$\Gamma$: classical and MCST--TSDT plate curves use $N=21$, and hollow markers show the corresponding M2L4 solid results. The horizontal coordinate is cumulative wavevector-path length; 12 distinct PWE samples reuse the closing $\Gamma$, while solids have 13 path samples. Lines join stored samples. Shading marks the interval common to all four plate and two solid models, 446.6--597.6 kHz (28.9\%). (b) Absolute lower- and upper-edge frequency discrepancies for all four plate models, each measured against the solid model with the same constitutive law. Circles and triangles denote lower and upper edges; grey/blue denote classical/MCST plate models. The point-plot axis is expanded to resolve the small differences. MCST prescribes epoxy $\ell=1\,\mu\mathrm m$ and steel $\ell=0$.''',
    'zh': r'''$(a,h,r)=(500,20,150)\,\mu\mathrm m$下的大单胞基准频谱与带边比较。(a) 沿$\Gamma$--$X$--$M$--$\Gamma$的前六条弯曲分支：经典与MCST--TSDT板模型曲线采用$N=21$，空心标记为对应M2L4实体结果。横坐标为累积波矢路径长度；PWE采用12个不同采样点并复用闭合$\Gamma$，实体模型采用13个路径点，连线连接已保存采样值。阴影为四种板模型与两种实体模型共有的区间446.6--597.6 kHz（28.9\%）。(b) 四种板模型相对相同本构实体模型的下、上带边绝对频率偏差；圆点与三角分别表示下、上带边，灰/蓝表示经典/MCST板模型。点图横轴采用局部范围以分辨小差异。MCST在环氧中取$\ell=1\,\mu\mathrm m$，钢中为零。''',
}
for lang, caption in captions.items():
    (REVIEW / f'caption_{lang}.tex').write_text(caption + '\n', encoding='utf-8')
print(json.dumps({'size_mm': [W, H], 'outputs': meta['outputs']}, ensure_ascii=False))
