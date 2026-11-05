"""Review-only 2x4 replacement layout; v36 manuscript and v37 preview are immutable."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, NullFormatter
from figure_style_v31 import COL, style_context, tidy_axes, export_figure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/prototypes_v38'
REVIEW = ROOT / '03_review/figure_2x4_preview_v38'
OUT.mkdir(parents=True, exist_ok=True)
REVIEW.mkdir(parents=True, exist_ok=True)
EDGE = ROOT / '02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/edge_focus_20260926'
SOURCES = [ROOT / '03_review/figures_v31/physical/physical_gap_records.csv',
           EDGE / 'scale_all_vs_3d_mcst_10scales.csv', EDGE / 'thickness_all_vs_3d_mcst.csv']
MODELS = ['Classic-FSDT', 'Classic-TSDT', 'MCST-FSDT', 'MCST-TSDT']
LABELS = {m: m.replace('Classic-', 'Classical ').replace('MCST-', 'MCST ') for m in MODELS}
LABELS['3D MCST'] = '3D MCST'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


protected = json.loads((ROOT / '03_review/figure_3x3_preview_v37/V36_PROTECTED_BEFORE.json').read_text(encoding='utf-8'))
old_preview = [ROOT / 'scripts/plot_3x3_comparison_preview_v37.py']
old_preview += list((ROOT / 'output/pdf/prototypes_v37').glob('figure_3x3_comparison_preview_v37.*'))
protected.update({p.relative_to(ROOT).as_posix(): sha(p) for p in old_preview})
for rel, expected in protected.items():
    assert sha(ROOT / rel) == expected, rel
before = REVIEW / 'PROTECTED_BEFORE.json'
if before.exists():
    assert json.loads(before.read_text(encoding='utf-8')) == protected
else:
    write_json(before, protected)

gaps = rows(SOURCES[0])
scans = [rows(SOURCES[1]), rows(SOURCES[2])]
assert len(gaps) == 54 and [len(x) for x in scans] == [40, 32]


def model_style(model):
    if model == '3D MCST':
        return dict(color='black', ls='none', marker='o', mfc='white', mec='black',
                    ms=3., mew=.7, lw=1., zorder=5)
    tsdt = model.endswith('TSDT')
    color = COL['classic'] if model.startswith('Classic') else COL['mcst']
    return dict(color=color, ls='-' if tsdt else (0, (3.1, 1.8)),
                marker='o' if tsdt else 's', mfc=color if tsdt else 'white', mec=color,
                ms=2.8 if tsdt else 3.1, mew=.65, lw=1.3 if tsdt else 1.05,
                zorder=4 if tsdt else 3)


def set_scan_axis(ax, kind):
    if kind == 'size':
        ticks, labels = [40, 100, 500, 2000], ['40', '100', '500', '2000']
        bounds = [40 * .91, 2000 * 1.10]
    else:
        ticks, labels = [.02, .08, .32], ['0.02', '0.08', '0.32']
        bounds = [.02 * .91, .32 * 1.10]
    ax.set_xscale('log')
    ax.set_xticks(ticks, labels)
    ax.set_xlim(bounds)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


W, H = 164., 90.
X = [14., 53.5, 93., 132.5]
Y = [51., 12.]
AXW, AXH = 28.5, 25.
records = []
with style_context():
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    axes = []
    for row, kind in enumerate(['size', 'thickness']):
        for col in range(4):
            ax = fig.add_axes([X[col] / W, Y[row] / H, AXW / W, AXH / H])
            axes.append(ax)
            tidy_axes(ax)
            ax.text(-.2, 1.055, 'abcdefgh'[row * 4 + col], transform=ax.transAxes,
                    fontsize=9, fontweight='bold', ha='left', va='bottom', clip_on=False)
            set_scan_axis(ax, kind)
            if row == 0:
                ax.set_title(['Relative gap', 'Lower edge', 'Upper edge', 'Gap difference'][col], pad=5, fontsize=8)
            if col == 0:
                for model in ['Classic-TSDT', 'MCST-TSDT', '3D-MCST']:
                    selected = sorted([r for r in gaps if r['scan'] == kind and r['model'] == model],
                                      key=lambda r: float(r['coordinate']))
                    xx = [float(r['coordinate']) for r in selected]
                    yy = [float(r['J_percent']) for r in selected]
                    ax.plot(xx, yy, **model_style('3D MCST' if model == '3D-MCST' else model))
                    records.extend(dict(panel='abcdefgh'[row * 4], scan=kind, series=model,
                                        coordinate=x, metric='J_percent', value=y) for x, y in zip(xx, yy))
                ax.set_ylabel(r'$J_{45}$ (%)', labelpad=2)
                if kind == 'size':
                    ax.set(ylim=(24.8, 30.2), yticks=[25, 27, 29])
                    setting = r'$h/a=0.08$'
                else:
                    ax.set(ylim=(0, 34), yticks=[0, 10, 20, 30])
                    setting = r'$a=75$ µm'
                ax.text(.97, .08, setting, transform=ax.transAxes, color=COL['muted'],
                        fontsize=7, ha='right', va='bottom')
            else:
                data = scans[row]
                key = 'a_um' if kind == 'size' else 'h_over_a'
                metric = ['L_to_3D_MCST_percent', 'U_to_3D_MCST_percent', 'J_difference_pp'][col - 1]
                for model in MODELS:
                    selected = sorted([r for r in data if r['PWE_model'] == model], key=lambda r: float(r[key]))
                    xx, yy = [float(r[key]) for r in selected], [float(r[metric]) for r in selected]
                    ax.plot(xx, yy, **model_style(model))
                    assert all(.06 <= y <= 50 if col < 3 else -2 <= y <= 25 for y in yy)
                    records.extend(dict(panel='abcdefgh'[row * 4 + col], scan=kind, series=model,
                                        coordinate=x, metric=metric, value=y) for x, y in zip(xx, yy))
                if col < 3:
                    ax.set_yscale('log')
                    ax.set_ylim(.06, 50)
                    ax.set_yticks([.1, 1, 10], ['0.1', '1', '10'])
                    ax.yaxis.set_minor_locator(NullLocator())
                    ax.set_ylabel(r'$e_L$ (%)' if col == 1 else r'$e_U$ (%)', labelpad=2)
                else:
                    ax.set_yscale('symlog', linthresh=1, linscale=1, base=10)
                    ax.set_ylim(-2, 25)
                    ax.set_yticks([-1, 0, 1, 10], ['−1', '0', '1', '10'])
                    ax.yaxis.set_minor_locator(NullLocator())
                    ax.axhline(0, color=COL['zero'], lw=.6, ls=':', zorder=0)
                    ax.set_ylabel(r'$\Delta J_{45}$ (pp)', labelpad=2)
    for y, text in zip(Y, ['Cell-size scan', 'Thickness scan']):
        fig.text(1.5 / W, (y + AXH / 2) / H, text, rotation=90, fontsize=8,
                 fontweight='bold', ha='center', va='center')
    fig.text(87 / W, 44.5 / H, 'Cell size $a$ (µm)', fontsize=8, ha='center', va='center')
    fig.text(87 / W, 2.2 / H, 'Thickness ratio $h/a$', fontsize=8, ha='center', va='center')
    legend_models = MODELS + ['3D MCST']
    handles = [Line2D([], [], label=LABELS[m], **{k: v for k, v in model_style(m).items() if k != 'zorder'})
               for m in legend_models]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.515, .998), ncol=5,
               handlelength=1.3, columnspacing=1.15, handletextpad=.4, borderaxespad=0, fontsize=7)
    assert len(fig.axes) == 8
    meta = export_figure(fig, OUT / 'figure_2x4_comparison_preview_v38')
    assert not meta['clipped_text']
    assert meta['minimum_visible_font_pt'] >= 7
    plt.close(fig)

assert len(records) == 54 + 216
with (REVIEW / 'plot_records.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['panel', 'scan', 'series', 'coordinate', 'metric', 'value'])
    writer.writeheader()
    writer.writerows(records)
assert all(sha(ROOT / rel) == expected for rel, expected in protected.items())
meta.update({
    'status': 'REVIEW_ONLY_NOT_INSERTED', 'manuscript_baseline': 'v36', 'previous_preview': 'v37',
    'axes_count': 8, 'layout': 'two rows, four columns: J, eL, eU, deltaJ',
    'old_to_new_panel_mapping': {'b': 'a', 'd': 'b', 'e': 'c', 'f': 'd', 'c': 'e', 'g': 'f', 'h': 'g', 'i': 'h'},
    'baseline_interval_panel_omitted': True, 'residual_strips': 0,
    'physical_gap_values': 54, 'discrepancy_values': 216, 'all_scan_values_preserved': True,
    'primary_N': 15, 'solid_mesh': 'M2L4',
    'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in SOURCES},
    'script_sha256': sha(Path(__file__)), 'manuscript_and_previous_artwork_unchanged': True,
    'frequency_error_axes': 'log10, limits0.06–50%',
    'signed_gap_error_axes': 'symlog10, limits−2–25pp, linear within±1pp',
    'thickness_ticks': [.02, .08, .32], 'all_eight_thickness_samples_retained': True,
    'new_eigensolves': 0, 'new_optimization_runs': 0, 'new_COMSOL_runs': 0,
})
write_json(REVIEW / 'PROVENANCE.json', meta)
captions = {
    'en': r'''Relative band gaps and model discrepancies in cell-size and thickness scans. (a--d) Ten cell sizes at fixed $h/a=0.08$; (e--h) eight thicknesses at fixed $a=75\,\mu\mathrm m$. All cases use $r/a=0.30$. Columns show the relative gap $J_{45}$, lower-edge discrepancy $e_L$, upper-edge discrepancy $e_U$, and signed gap difference $\Delta J_{45}=J_{45}^{\rm PWE}-J_{45}^{\rm 3D\,MCST}$. All plate spectra use $N=15$ and solid comparisons use the same-geometry MCST--M2L4 model. MCST has epoxy $\ell=1\,\mu\mathrm m$ and steel $\ell=0$. Grey and blue denote classical and MCST plate models; dashed hollow squares and solid filled circles denote FSDT and TSDT. Black hollow circles denote MCST solid results in the relative-gap panels. Parameter and frequency-discrepancy axes are logarithmic. Both signed-gap axes use the same symmetric-logarithmic scale, linear within $\pm1$ percentage point. Panels (a,e) use different vertical ranges. All saved scan points are shown; reduced thickness tick labels do not remove data. Lines guide the eye.''',
    'zh': r'''单胞尺寸与厚度扫描中的相对带隙及模型偏差。(a--d) 固定$h/a=0.08$的十种单胞尺寸；(e--h) 固定$a=75\,\mu\mathrm m$的八种厚度，各工况均取$r/a=0.30$。四列依次为相对带隙$J_{45}$、下带边频差$e_L$、上带边频差$e_U$及有符号带隙差$\Delta J_{45}=J_{45}^{\rm PWE}-J_{45}^{\rm 3D\,MCST}$。板模型均采用$N=15$，实体比较采用同几何MCST--M2L4。MCST环氧内尺度取$1\,\mu\mathrm m$，钢中为零。灰/蓝表示经典/MCST板模型，虚线空心方点及实线实心圆点分别表示FSDT/TSDT，黑色空心圆点表示相对带隙图中的MCST实体结果。参数轴与频差轴为对数刻度；两幅有符号带隙差图采用相同对称对数刻度，在$\pm1$个百分点内为线性区。(a,e)纵轴范围不同，厚度刻度仅标主要值，全部八个厚度样点保留，连线辅助观察。''',
}
for lang, caption in captions.items():
    (REVIEW / f'caption_draft_{lang}.tex').write_text(caption + '\n', encoding='utf-8')
print(json.dumps({'status': meta['status'], 'size_mm': [W, H], 'data_values': len(records),
                  'pdf': meta['outputs']['pdf']['path'], 'png': meta['outputs']['png']['path']}, ensure_ascii=False))
