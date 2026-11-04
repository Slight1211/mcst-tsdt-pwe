"""Rebuild Figures 5/6 from current b=.24 numerical results with the approved style."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from figure_style_v31 import COL, style_context, tidy_axes, export_figure, panel_label
import run_lengthscale_optimization as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'latex/figures'
REVIEW = ROOT / '03_review/uniform_b024_v47'
load = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
closed = lambda v: np.vstack((v, v[0]))


def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def finish(fig, name):
    record = export_figure(fig, OUT / name)
    assert record['minimum_visible_font_pt'] >= 7
    plt.close(fig)
    return record


def main():
    global initial, shape, solid, mech, axmm, txt, panel, shape_cell, scale_bar
    result = load(ROOT / 'output/uniform_b024_v47/current_results.json')
    assert result['status'] == 'PASS_CURRENT_B024_RESULT_BINDING'
    cases = {c['ell_um']: c for c in result['cases']}
    initial = base.boundary(np.zeros(6))
    source = (ROOT / 'scripts/plot_compact_design_v40.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['axmm', 'txt', 'panel', 'shape_cell', 'scale_bar']]
    exec(compile(ast.Module(body=funcs, type_ignores=[]), 'v40_approved_layout_helpers', 'exec'), globals())
    shape = []
    for ell, c in sorted(cases.items()):
        v = np.loadtxt(ROOT / c['vertices'], delimiter=',', skiprows=1)
        shape += [dict(ell_um=ell, vertex=i, x_over_a=q[0], y_over_a=q[1], J153_percent=c['J153_percent']) for i, q in enumerate(v)]
    mech = []
    for r in result['transfer']:
        for label, key in [('initial', 'initial_J'), ('final', 'final_J')]:
            mech.append(dict(record_type='gap_transfer', ell_um=r['ell_um'], shape=label, J_percent=r[key]))
    redesign = next(r for r in result['primary_table'] if r['ell_um'] == 2 and r['model'] == 'PWE N15')
    mech.append(dict(record_type='gap_reoptimized', ell_um=2, J_percent=redesign['optimized']))
    mech.extend(dict(record_type='fixed_geometry_modal_energy', **r) for r in result['energies'])
    solid = []
    for r in result['primary_table']:
        if r['model'] == '3D M2L4':
            solid.extend([dict(ell_um=r['ell_um'], shape='initial', J_percent=r['initial']), dict(ell_um=r['ell_um'], shape='final', J_percent=r['optimized'])])
    start = source.index('with style_context():\n    H=116.')
    end = source.index('\nfor name,digest in hashes.items()', start)
    block = source[start:end]
    block = block.replace("label=fr'$\\ell={ell:g}$ µm'+(' †' if ell in [0,.5] else '')", "label=fr'$\\ell={ell:g}$ µm'")
    block = block.replace("    txt(fig,71,70,'† Bound-limited',H,fontsize=8,color=COL['muted'],va='center')\n", '')
    block = block.replace("'figure6_lengthscale_compact_v40'", "'figure6_lengthscale_b024_v47'")
    # Gain annotations follow the current calculations, including their units.
    p1 = next(r for r in result['primary_table'] if r['ell_um'] == 1 and r['model'] == 'PWE N15')
    transfer2 = next(r for r in result['transfer'] if r['ell_um'] == 2)
    block = block.replace("'+5.2 pp'", repr(f"+{p1['gain']:.1f} pp"))
    block = block.replace("'+0.8 pp'", repr(f"+{transfer2['gain_pp']:.1f} pp"))
    block = block.replace("'+2.1 pp'", repr(f"+{redesign['gain']:.1f} pp"))
    scope = dict(globals())
    exec(compile(block, str(Path(__file__)) + ':approved_figure6_layout', 'exec'), scope)
    fig6 = scope['f5']

    # Figure 5 shows the freshly computed regular-start trajectory; its band
    # comparison uses the selected two-start geometry, explicitly in the caption.
    cold = next(r for r in cases[1]['starts'] if r['seed'] == 'regular')
    folder = (ROOT / cold['manifest']).parent
    history = load(folder / 'trajectory.json')
    history = [r for i, r in enumerate(history) if i == 0 or r['iteration'] > 0]
    t = np.arange(len(history))
    regular_meta = load(folder / 'manifest.json')
    terminal_opt = regular_meta['rounds'][-1]['optimizer']
    # A rejected last callback can differ from SLSQP's accepted sol.x. Retain
    # every callback and add the accepted terminal state at the same iteration.
    if history[-1]['design'] != regular_meta['design'] or history[-1]['constraint_violation'] != terminal_opt['constraint_violation']:
        terminal = dict(history[-1], design=regular_meta['design'], J=regular_meta['final_ibz153']['J'],
                        epigraph_J=terminal_opt['objective'], constraint_violation=terminal_opt['constraint_violation'])
        history.append(terminal)
        t = np.r_[t, t[-1]]
    chosen = [0, max(1, (len(t) - 1) // 3), max(2, 2 * (len(t) - 1) // 3), len(t) - 1]
    with style_context():
        H = 112.
        fig = plt.figure(figsize=(164 / 25.4, H / 25.4))
        panel(fig, 1, 110, 'a', H)
        txt(fig, 10, 108, 'Regular-start interface evolution', H, fontsize=9, va='top')
        first = None
        for j, index in enumerate(chosen):
            row = history[index]
            ax = axmm(fig, 10 + 39 * j, 68, 27, 27, H)
            shape_cell(ax, base.boundary(np.asarray(row['design'])), reference=bool(index), color=COL['init'] if index == 0 else COL['opt'])
            first = first or ax
            title = 'Terminal' if index == len(history) - 1 else f'Step {int(t[index])}'
            ax.set_title(title + '\n' + fr'$J_{{45}}={row["J"]:.2f}\%$', fontsize=8, pad=4)
        scale_bar(first)
        fig.legend(handles=[Line2D([], [], color=COL['init'], ls='--', label='Initial'), Line2D([], [], color=COL['opt'], label='Updated')],
                   loc='upper right', bbox_to_anchor=(.978, .985), ncol=2, borderaxespad=0, fontsize=8, handlelength=1.8)
        b, c, d = axmm(fig, 13, 13, 40, 35, H), axmm(fig, 69, 13, 35, 35, H), axmm(fig, 120, 13, 40, 35, H)
        for x, letter in [(1, 'b'), (58, 'c'), (109, 'd')]:
            panel(fig, x, 57, letter, H)
        jj = np.array([r['J'] for r in history])
        aux = np.array([r['epigraph_J'] for r in history])
        violation = np.array([r['constraint_violation'] for r in history])
        feasible = violation <= 1e-6
        b.plot(t, aux, color=COL['ink'], ls='--', lw=1.2, label='Auxiliary')
        b.plot(t, jj, color=COL['init'], lw=1.2)
        for flag, label, color in [(False, 'Violated', COL['infeasible']), (True, 'Feasible', COL['feasible'])]:
            b.plot(t[feasible == flag], jj[feasible == flag], ls='none', marker='o', ms=3.1, color=color, label=label)
        b.set(xlabel='Iteration', ylabel='Relative gap (%)', xlim=(-.5, max(t) + .5))
        b.legend(loc='best', fontsize=7, handlelength=1.2, labelspacing=.2)
        c.semilogy(t, np.maximum(violation, 1e-12), color=COL['opt'], marker='o', ms=3, lw=1.2)
        c.axhline(1e-6, color=COL['muted'], lw=.6, ls='--')
        c.set(xlabel='Iteration', ylabel='Active-set violation', ylim=(1e-13, max(1, 2 * violation.max())), xlim=(-.5, max(t) + .5), yticks=[1e-12, 1e-6, 1])
        terminal = rows((ROOT / cases[1]['manifest']).parent / 'final_ibz45.csv')
        initpath = ROOT / '02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1/initial_ibz45.csv'
        factor = np.sqrt(4.35e9 / 1180) / (75e-6 * 2 * np.pi * 1e6)
        extrema = []
        for records, label, color, ls in [(rows(initpath), 'Initial', COL['init'], '--'), (terminal, 'Selected', COL['opt'], '-')]:
            curve = []
            for path_index, k in enumerate([*[np.array([np.pi * j / 8, 0.]) for j in range(9)], *[np.array([np.pi, np.pi * j / 8]) for j in range(1, 9)], *[np.array([np.pi * j / 8, np.pi * j / 8]) for j in range(7, -1, -1)]]):
                r = next(r for r in records if max(abs(np.array([float(r['kx_a']), float(r['ky_a'])]) - k)) < 1e-12)
                curve.append([float(r['Omega4']) * factor, float(r['Omega5']) * factor])
            curve = np.array(curve)
            for band in [0, 1]:
                d.plot(np.arange(25) / 8, curve[:, band], color=color, ls=ls, lw=1.2, label=label if band == 0 else None)
            extrema.append(curve)
        d.axhspan(max(extrema[1][:, 0]), min(extrema[1][:, 1]), color=COL['gap'], alpha=.15, zorder=-2)
        d.set(xlabel='Bloch path', ylabel='Frequency (MHz)', xticks=[0, 1, 2, 3], xticklabels=[r'$\Gamma$', '$X$', '$M$', r'$\Gamma$'], xlim=(-.04, 3.04))
        d.legend(loc='best', fontsize=7, handlelength=1.2)
        for ax in [b, c, d]:
            tidy_axes(ax)
        fig5 = finish(fig, 'figure5_optimization_b024_v47')
    record = dict(status='GENERATED_PENDING_VISUAL_REVIEW', revision='v47', bound=.24, figure5=fig5, figure6=fig6,
                  current_results_sha256=sha(ROOT / 'output/uniform_b024_v47/current_results.json'),
                  selected_regular_start_manifest=cold['manifest'], figure5_snapshot_indices=chosen,
                  figure5_regular_iterations=regular_meta['accepted_iterations'], figure5_accepted_terminal_included=True,
                  figure5_callback_records_preserved=True, figure5_selected_terminal_source=cases[1]['vertices'],
                  all_five_shape_states_replaced_by_b024=True, geometry='Saved 48 vertices, straight segments, identical cell frames and fixed area.',
                  script_sha256=sha(Path(__file__)))
    (REVIEW / 'FIGURE_PROVENANCE.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('GENERATED', 'figure5_optimization_b024_v47', 'figure6_lengthscale_b024_v47', flush=True)


if __name__ == '__main__':
    main()
