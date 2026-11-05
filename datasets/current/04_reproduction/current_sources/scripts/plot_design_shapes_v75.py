"""Plot five independently optimized b=.24 shapes from preserved numeric records.

Only straight saved polygon edges are drawn. No eigenproblem or optimization is run.
The artwork replaces the earlier mixed shape/transfer/energy Figure 6.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from pypdf import PdfReader

from figure_style_v31 import COL, style_context, export_figure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/figures/v75'
REVIEW = ROOT / '03_review/design_focus_v75'
WIDTH_MM, HEIGHT_MM = 183.0, 55.0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def area(vertices):
    x, y = vertices.T
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def close(vertices):
    return np.vstack([vertices, vertices[0]])


def main():
    source = ROOT / 'output/uniform_b024_v47/current_results.json'
    result = json.loads(source.read_text(encoding='utf-8'))
    assert result['status'] == 'PASS_CURRENT_B024_RESULT_BINDING'
    cases = sorted(result['cases'], key=lambda c: c['ell_um'])
    assert [c['ell_um'] for c in cases] == [0, .5, 1, 2, 5]
    seed_path = ROOT / '05_validation/UNIFORM_B024_V47_20261103/ell_1_regular/seed_vertices.csv'
    initial = np.loadtxt(seed_path, delimiter=',', skiprows=1)
    assert initial.shape == (48, 2)
    expected_area = .09 * np.pi
    assert abs(area(initial) - expected_area) < 1e-12
    assert np.ptp(np.linalg.norm(initial, axis=1)) < 1e-12
    source_hashes = {source.relative_to(ROOT).as_posix(): sha(source),
                     seed_path.relative_to(ROOT).as_posix(): sha(seed_path)}
    checks = []
    with style_context():
        fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
        for i, case in enumerate(cases):
            vpath = ROOT / case['vertices']
            spath = ROOT / case['spectrum']
            assert sha(vpath) == case['vertices_sha256']
            assert sha(spath) == case['spectrum_sha256']
            source_hashes[case['vertices']] = sha(vpath)
            source_hashes[case['spectrum']] = sha(spath)
            vertices = np.loadtxt(vpath, delimiter=',', skiprows=1)
            assert vertices.shape == (48, 2)
            assert case['N'] == 15 and case['points'] == 153 and case['bound'] == .24
            with spath.open(encoding='utf-8-sig', newline='') as stream:
                spectrum = list(csv.DictReader(stream))
            lower = max(float(row['Omega4']) for row in spectrum)
            upper = min(float(row['Omega5']) for row in spectrum)
            gap = 200 * (upper - lower) / (upper + lower)
            assert len(spectrum) == 153
            assert abs(gap - case['J153_percent']) < 1e-10
            radii = np.linalg.norm(vertices, axis=1)
            ratio = float(radii.max() / radii.min())
            polygon_area = float(area(vertices))
            assert abs(polygon_area - expected_area) < 1e-12
            assert abs(ratio - case['radial_ratio']) < 1e-12
            assert np.max(np.abs(vertices)) < .49

            left = 4 + 36 * i
            ax = fig.add_axes([left / WIDTH_MM, 10 / HEIGHT_MM,
                               31 / WIDTH_MM, 31 / HEIGHT_MM])
            ax.add_patch(Rectangle((-.5, -.5), 1, 1, facecolor='white',
                                   edgecolor='#D9D9D9', linewidth=.6, zorder=0))
            ax.fill(*vertices.T, color=COL['opt'], alpha=.055, edgecolor='none', zorder=1)
            ax.plot(*close(initial).T, color=COL['init'], linestyle=(0, (3, 2.4)),
                    linewidth=1.2, zorder=2)
            ax.plot(*close(vertices).T, color=COL['opt'], linewidth=1.2, zorder=3)
            ax.set(xlim=(-.52, .52), ylim=(-.52, .52), aspect='equal')
            ax.set_axis_off()
            fig.text((left - 2) / WIDTH_MM, 52 / HEIGHT_MM, chr(97 + i),
                     fontsize=9, fontweight='bold', va='top', ha='left')
            fig.text((left + 15.5) / WIDTH_MM, 51.5 / HEIGHT_MM,
                     fr'$\ell={case["ell_um"]:g}$ µm' + '\n' + fr'$J_{{45}}={gap:.2f}\%$',
                     fontsize=8, linespacing=1.25, ha='center', va='top')
            if i == 0:
                # Exactly 0.5 a in the same normalized data coordinates as all outlines.
                ax.plot([-.45, .05], [-.63, -.63], color=COL['ink'], lw=.6, clip_on=False)
                for x in [-.45, .05]:
                    ax.plot([x, x], [-.66, -.60], color=COL['ink'], lw=.6, clip_on=False)
                ax.text(.14, -.63, '$0.5a$', fontsize=8, ha='left', va='center', clip_on=False)
            checks.append(dict(ell_um=case['ell_um'], vertices=case['vertices'],
                               spectrum=case['spectrum'], N=15, points=153, bound=.24,
                               vertex_count=len(vertices), polygon_area_fraction=polygon_area,
                               area_error=polygon_area - expected_area,
                               radial_ratio=ratio, J153_percent=gap,
                               L=lower, U=upper, display_gap=f'{gap:.2f}'))
        legend = [Line2D([], [], color=COL['init'], ls=(0, (3, 2.4)), lw=1.2, label='Initial'),
                  Line2D([], [], color=COL['opt'], lw=1.2, label='Optimized')]
        fig.legend(handles=legend, loc='lower right', bbox_to_anchor=(.98, .025), ncol=2,
                   fontsize=7, handlelength=2, columnspacing=1.6, borderaxespad=0)
        OUT.mkdir(parents=True, exist_ok=True)
        stem = OUT / 'figure6_shapes_v75'
        exports = export_figure(fig, stem)
        exports['native_width_note'] = '183 mm source artwork; common geometry scale; no tight bounding-box crop.'
        plt.close(fig)

    assert [c['display_gap'] for c in checks] == ['34.58', '34.31', '33.22', '27.05', '2.96']
    pdf = stem.with_suffix('.pdf')
    page = PdfReader(pdf).pages[0]
    pdf_mm = [float(page.mediabox.width) * 25.4 / 72, float(page.mediabox.height) * 25.4 / 72]
    assert abs(pdf_mm[0] - WIDTH_MM) < 1e-5 and abs(pdf_mm[1] - HEIGHT_MM) < 1e-5
    target = ROOT / 'latex/figures/figure6_shapes_v75.pdf'
    shutil.copy2(pdf, target)
    record = dict(status='GENERATED_PENDING_VISUAL_REVIEW', revision='v75',
                  purpose='Five independent fixed-area optimized shapes; transfer and endpoint comparisons removed from Figure 6.',
                  source_sha256=source_hashes, records=checks,
                  outputs={k: dict(path=Path(v['path']).relative_to(ROOT).as_posix(), sha256=v['sha256'])
                           for k, v in exports['outputs'].items()},
                  latex_asset=dict(path=target.relative_to(ROOT).as_posix(), sha256=sha(target)),
                  width_mm=WIDTH_MM, height_mm=HEIGHT_MM, pdf_page_mm=pdf_mm,
                  minimum_visible_font_pt=exports['minimum_visible_font_pt'],
                  clipped_text=exports['clipped_text'],
                  shared_style_sha256=exports['shared_style_sha256'],
                  style_source_sha256=exports['style_source_sha256'],
                  semantic_colors=COL,
                  geometry='Saved 48 vertices joined by straight segments; no smoothing; equal x/y scales and identical cell frames.',
                  baseline='Equal-area regular 48-gon from saved regular-start seed coordinates.',
                  scale_bar='0.5 a; shared normalized x/a and y/a coordinates.',
                  new_eigensolves=False, new_optimization_runs=False,
                  publisher_compliance_claim=False,
                  script_sha256=sha(Path(__file__)))
    REVIEW.mkdir(parents=True, exist_ok=True)
    (REVIEW / 'FIGURE_PROVENANCE.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=record['status'], output=target.relative_to(ROOT).as_posix(),
                         size_mm=pdf_mm, gaps=[c['display_gap'] for c in checks])))


if __name__ == '__main__':
    main()
