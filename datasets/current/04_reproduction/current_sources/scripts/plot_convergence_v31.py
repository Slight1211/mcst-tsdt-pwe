"""Restyle the saved circular-cell PWE convergence figure, without solves."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pypdf import PdfReader
from figure_style_v31 import COL, WIDTH_MM, style_context, export_figure, panel_label, tidy_axes

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'03_review/figures_v31/energy'
OUT=ROOT/'01_manuscript/figures'
SOURCE=ROOT/'06_release/github_v20/repository/data/wave_motion_20261009_v1/assets/item-053.csv'
STYLE=ROOT/'scripts/figure_style_v31.py'
STEM='figureS1_convergence_v31'
H=66.548
HERE.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source_hash=sha(SOURCE)
with SOURCE.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
assert [int(r['N']) for r in rows]==list(range(7,22))
assert all(float(r['a_um'])==75 and float(r['h_um'])==6 and float(r['ell_epoxy_um'])==1 and int(r['path_points'])==12 for r in rows)
ref_l,ref_u=float(rows[-1]['lower_MHz']),float(rows[-1]['upper_MHz'])
checks=[]
for r in rows:
    lo,up=float(r['lower_MHz']),float(r['upper_MHz'])
    recomputed=dict(N=int(r['N']),lower_error=100*abs(lo/ref_l-1),upper_error=100*abs(up/ref_u-1),J_percent=200*(up-lo)/(up+lo))
    for key,src in [('lower_error','lower_difference_to_N21_percent'),('upper_error','upper_difference_to_N21_percent'),('J_percent','J45_percent')]:
        assert math.isclose(recomputed[key],float(r[src]),abs_tol=1e-9)
    checks.append(recomputed)
n=np.asarray([int(r['N']) for r in rows])
lo=np.asarray([float(r['lower_difference_to_N21_percent']) for r in rows])
up=np.asarray([float(r['upper_difference_to_N21_percent']) for r in rows])
gap=np.asarray([float(r['J45_percent']) for r in rows])
with style_context():
    fig=plt.figure(figsize=(WIDTH_MM/25.4,H/25.4))
    a=fig.add_axes([15/WIDTH_MM,15/H,62/WIDTH_MM,39/H])
    b=fig.add_axes([100/WIDTH_MM,15/H,61/WIDTH_MM,39/H])
    a.plot(n,lo,color=COL['mode4'],marker='o',mfc=COL['mode4'],mew=.8,ms=4,lw=1.2,label=r'Lower edge $\delta_L$')
    a.plot(n,up,color=COL['mode5'],marker='s',mfc='white',mew=.8,ms=4,lw=1.2,ls='--',label=r'Upper edge $\delta_U$')
    a.set_ylabel(r'Difference from $N=21$ (%)')
    a.set_ylim(-.05*up.max(),1.12*up.max())
    b.plot(n,gap,color=COL['mcst'],marker='D',ms=3.8,mew=.8,lw=1.2)
    b.set_ylabel(r'Signed $J_{45}$ (%)')
    span=gap.max()-gap.min()
    b.set_ylim(gap.min()-.08*span,gap.max()+.08*span)
    for ax,letter in [(a,'a'),(b,'b')]:
        ax.set_xlim(6.6,21.4)
        ax.set_xticks(np.arange(7,22,2));ax.set_xticks(n,minor=True)
        ax.set_xlabel('Fourier truncation order $N$')
        tidy_axes(ax)
        panel_label(ax,letter,x=-.21,y=1.09)
    h,l=a.get_legend_handles_labels()
    fig.legend(h,l,loc='upper center',bbox_to_anchor=(.53,.975),ncol=2,handlelength=1.8,borderaxespad=0)
    exported=export_figure(fig,OUT/STEM)
    plt.close(fig)
for ext in ['pdf','svg']:
    shutil.copy2(OUT/f'{STEM}.{ext}',ROOT/f'latex/figures/{STEM}.{ext}')
assert sha(SOURCE)==source_hash
reader=PdfReader(OUT/f'{STEM}.pdf')
assert len(reader.pages)==1
assert abs(float(reader.pages[0].mediabox.width)*25.4/72-WIDTH_MM)<1e-8
assert abs(float(reader.pages[0].mediabox.height)*25.4/72-H)<1e-8
with (HERE/'convergence_plot_records.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
report=dict(status='GENERATED_PENDING_VISUAL_REVIEW',figure='S1',source_sha256={SOURCE.relative_to(ROOT).as_posix():source_hash,STYLE.relative_to(ROOT).as_posix():sha(STYLE)},
    primary_N=15,finite_order_reference_N=21,orders=15,points_per_path=12,all_saved_orders_retained=True,
    independently_recomputed_values=45,transformations='Original frequency-to-gap and N21-relative-frequency definitions; no filtering, smoothing or new sampling.',
    plot_limits=dict(N=[6.6,21.4],error=[float(-.05*up.max()),float(1.12*up.max())],J=[float(gap.min()-.08*span),float(gap.max()+.08*span)]),
    export_metadata=exported,new_eigensolves=0,new_optimizations=0,new_COMSOL_runs=0,
    caption_change='None; lower/upper edge colors now match modes4/5 roles; panel letters remain a,b.',
    alt_text='Two-panel PWE order check. Lower and upper band-edge discrepancies against N21 decrease as N rises from7 to21. The sampled relative gap increases toward the finite N21 value. All15 saved orders are shown.')
(HERE/'convergence_provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=report['status'],outputs=exported['outputs'],minimum_font_pt=exported['minimum_visible_font_pt']),ensure_ascii=False,indent=2))
