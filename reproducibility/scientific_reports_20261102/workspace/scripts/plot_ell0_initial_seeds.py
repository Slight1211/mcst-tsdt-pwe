"""Plot only the 16 immutable initial geometries; never read worker results."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_gap_closure_scan import polygon,sha,locate

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'05_validation/ELL0_MULTISTART_20261101'
DEFAULT_OUT=ROOT/'output/ell0_initial_seeds'
LABELS={'legacy_ell0p5':'Seed: ℓ=0.5 shape','legacy_ell2':'Seed: ℓ=2 shape','legacy_ell5':'Seed: ℓ=5 shape',
    'opposite_old':'Reversed contrast','rotated45_reduced_contrast':'45° reduced contrast',
    'smooth_cos4_positive':'Smooth fourfold (+)','smooth_cos4_negative':'Smooth fourfold (−)',
    'alternating_bounds':'Alternating bounds'}


def label(value):
    return 'Latin hypercube '+value[-2:] if value.startswith('lhs_') else LABELS[value]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--overwrite',action='store_true');args=p.parse_args()
    # The live controller is read only for its fixed seed definitions. Its run
    # fields and every worker manifest/spectrum are deliberately unused.
    raw=(BASE/'manifest.json').read_bytes();meta=json.loads(raw.decode('utf-8'))
    assert meta['new_starts']==16 and len(meta['seeds'])==16
    assert (meta['N'],meta['a_um'],meta['h_um'],meta['ell_epoxy_um'],meta['ell_steel_um'])==(15,75,6,0,0)
    inputs={};seeds=[]
    for definition in meta['seeds']:
        path=(ROOT/definition['path']).resolve();assert path.is_relative_to(BASE/'seeds')
        b=path.read_bytes();digest=hashlib.sha256(b).hexdigest();assert digest==definition['sha256']
        inputs[locate(path)]=digest;s=json.loads(b.decode('utf-8'))
        assert s['label']==definition['label'] and s['design']==definition['design']
        assert (s['N'],s['a_um'],s['h_um'],s['ell_epoxy_um'],s['ell_steel_um'])==(15,75,6,0,0)
        v=polygon(s['design']);area=.5*np.sum(v[:,0]*np.roll(v[:,1],-1)-v[:,1]*np.roll(v[:,0],-1))
        assert abs(area-.09*math.pi)<1e-13
        seeds.append(dict(label=s['label'],design=s['design'],vertices=v,path=locate(path),sha256=digest))
    limit=max(.4,math.ceil((max(np.max(abs(s['vertices'])) for s in seeds)+.02)/.05)*.05)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if not args.overwrite:assert not any(out.iterdir()),'Use --overwrite for these derived initial-seed assets.'
    snapshot=dict(status='INITIAL_SEEDS_ONLY_NOT_OPTIMIZATION_RESULTS',
        controller_manifest_sha256_at_read=hashlib.sha256(raw).hexdigest(),
        live_controller_manifest='05_validation/ELL0_MULTISTART_20261101/manifest.json',
        live_manifest_is_mutable=True,used_fields='Only fixed seed definitions and fixed model parameters; worker run/result fields were not inspected.',
        N=15,a_um=75,h_um=6,ell_epoxy_um=0,ell_steel_um=0,
        seeds=[{k:s[k] for k in ['label','design','path','sha256']} for s in seeds])
    snapshot_path=out/'seed_plan_snapshot.json';snapshot_path.write_text(json.dumps(snapshot,indent=2)+'\n',encoding='utf-8')
    inputs[locate(snapshot_path)]=sha(snapshot_path)
    initial=polygon(np.zeros(6));initial=np.vstack([initial,initial[0]])
    layout=[]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False}):
        fig=plt.figure(figsize=(12.5,13.2));width=.19;height=width*12.5/13.2
        for index,s in enumerate(seeds):
            col=index%4;row=index//4
            a=fig.add_axes([.065+.24*col,.72-.205*row,width,height])
            v=s['vertices'];closed=np.vstack([v,v[0]])
            a.fill(closed[:,0],closed[:,1],color='#0072B2',alpha=.06)
            a.plot(initial[:,0],initial[:,1],color='#777777',ls='--',lw=.9)
            a.plot(closed[:,0],closed[:,1],color='#0072B2',lw=1.3)
            a.scatter(v[:,0],v[:,1],color='#0072B2',s=4)
            a.set(xlim=(-limit,limit),ylim=(-limit,limit),xticks=[-limit,0,limit],yticks=[-limit,0,limit],
                xlabel='$x/a$',ylabel='$y/a$',title=label(s['label']))
            a.set_aspect('equal');a.tick_params(labelsize=8);a.grid(alpha=.15)
            if row<3:
                a.set_xlabel('');a.tick_params(labelbottom=False)
            a.title.set_fontsize(9.5)
        fig.suptitle('INITIAL SEEDS ONLY — BEFORE OPTIMIZATION',fontsize=17,y=.993)
        fig.text(.5,.956,r'16 additional starting shapes for $\ell=0\,\mu$m; $N=15$, $a=75\,\mu$m, $h=6\,\mu$m',ha='center',fontsize=11)
        handles=[Line2D([],[],color='#0072B2',lw=1.3,marker='o',ms=3,label='Initial seed (48 straight edges)'),
            Line2D([],[],color='#777777',lw=.9,ls='--',label='Initial equal-area regular 48-gon')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),ncol=2,frameon=False,fontsize=10)
        fig.text(.5,.011,'Identical scales and areas. Seed labels identify geometry sources. No optimization outcomes or gap values are shown.',ha='center',fontsize=9)
        fig.canvas.draw()
        for a in fig.axes:
            r=a.get_position();layout.append(dict(width_inches=r.width*12.5,height_inches=r.height*13.2))
        assert max(r['width_inches'] for r in layout)-min(r['width_inches'] for r in layout)<1e-10
        assert all(abs(r['width_inches']-r['height_inches'])<1e-10 for r in layout)
        bb=fig.get_tightbbox(fig.canvas.get_renderer())
        assert bb.x0>=0 and bb.y0>=0 and bb.x1<=12.5 and bb.y1<=13.2
        for ext in ['png','svg']:fig.savefig(out/f'ell0_initial_seeds_only.{ext}',dpi=220,facecolor='white')
        plt.close(fig)
    assert all(sha(ROOT/path)==digest for path,digest in inputs.items())
    record=dict(status='INITIAL_SEEDS_ONLY_NOT_OPTIMIZATION_RESULTS',seed_count=16,
        no_worker_results_read=True,no_gap_values_displayed=True,no_new_eigensolves=True,no_new_COMSOL=True,
        common_axis_limits=[-limit,limit],area_fraction=.09*math.pi,vertices=48,axes_physical_layout=layout,
        fixed_lattice_coordinates=True,no_individual_rescaling=True,no_smoothing=True,
        source_sha256=inputs,plot_script_sha256=sha(Path(__file__)),helper_script_sha256=sha(ROOT/'scripts/plot_gap_closure_scan.py'),
        output_sha256={name:sha(out/name) for name in ['ell0_initial_seeds_only.png','ell0_initial_seeds_only.svg','seed_plan_snapshot.json']},
        visual_review='PENDING_ACTUAL_OUTPUT_INSPECTION',human_verified=False)
    (out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(out=str(out),status=record['status'],axis_limits=record['common_axis_limits'],output_sha256=record['output_sha256']),indent=2))


if __name__=='__main__':main()
