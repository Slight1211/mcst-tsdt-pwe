"""Report only validated 3D M2L4 X/M thickness controls against frozen N15 PWE."""
from pathlib import Path
import csv
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/thickness_M2L4_20260924'
PWE=ROOT/'runs/inverse_scan_verified_N15_20260921T160730024316Z'
RATIO=ROOT/'runs/fixed_ratio_scale_20260922T175850279921Z'
ASSETS=ROOT.parents[1]/'output/pdf/modal_selectivity_assets'
STYLES=(('classic','FSDT','#0072BD','--'),
        ('classic','TSDT','#56B4E9','-'),
        ('mcst','FSDT','#D55E00','-.'),
        ('mcst','TSDT','#7E2F8E',':'))
VERTICAL_FRACTION_CUTOFF = 0.7

def controls(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest.get('exit_code')==0 and (folder/'model_Model.mph').is_file()
    assert manifest['settings']['COMSOL_MESH_LEVEL']=='2'
    assert manifest['settings']['COMSOL_DRY_SWEEP_LAYERS']=='4'
    rows=list(csv.DictReader((folder/'frequencies.csv').open(newline='')))
    result={}
    for point,ky in (('X',0),('M',1)):
        selected=sorted((r for r in rows
                         if abs(float(r['kx_pi_over_a'])-1)<1e-9
                         and abs(float(r['ky_pi_over_a'])-ky)<1e-9
                         and float(r['vertical_fraction'])>VERTICAL_FRACTION_CUTOFF
                         and float(r['frequency_mhz'])>0),
                        key=lambda r:float(r['frequency_mhz']))[:6]
        assert len(selected)==6
        assert all(abs(float(r['imag_frequency_mhz']))/float(r['frequency_mhz'])<1e-7
                   for r in selected)
        result[point]=[float(r['frequency_mhz']) for r in selected]
    return result

def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    assert json.loads((OUT/'completion.json').read_text())['status']=='ALL_THICKNESS_XM_VALIDATED'
    assert json.loads((PWE/'config.json').read_text())['fixed_a_um']==25
    allrows=[]
    for h in (0.5,1,1.5,2,3,5,8):
        for theory in ('classic','mcst'):
            folder=(RATIO/'mcst_a25') if h==2 and theory=='mcst' else OUT/f'{theory}_h{h:g}'
            if h==2 and theory=='mcst':
                settings=json.loads((folder/'manifest.json').read_text())['settings']
                assert float(settings['COMSOL_LATTICE_UM'])==25
                assert float(settings['COMSOL_THICKNESS_UM'])==2
            fem=controls(folder)
            for plate in ('FSDT','TSDT'):
                ell=0 if theory=='classic' else 1
                for point,index in (('X',4),('M',8)):
                    fn=PWE/f'N15_a25_h{h:g}_p{index}_{plate}_l{ell}.npz'
                    with np.load(fn) as z:
                        pwe=z['frequency_MHz'][:6]
                        assert np.all(np.isfinite(pwe)) and np.min(pwe)>0
                        assert np.max(z['residual'][:6])<1e-5
                        assert float(z['seed_agreement'])<1e-7
                    for band,(fp,ff) in enumerate(zip(pwe,fem[point]),1):
                        allrows.append(dict(h_um=h,three_dimensional=theory,
                                            plate=plate,point=point,band=band,
                                            pwe_MHz=float(fp),fem_MHz=ff,
                                            absolute_error_percent=100*abs(float(fp)/ff-1)))
    assert len(allrows)==7*2*2*2*6
    with (OUT/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)
    maxima=[]
    for h in (0.5,1,1.5,2,3,5,8):
        for theory,plate,*_ in STYLES:
            for point in ('X','M'):
                subset=[r for r in allrows if r['h_um']==h and
                        r['three_dimensional']==theory and
                        r['plate']==plate and r['point']==point]
                maxima.append(max(subset,key=lambda r:r['absolute_error_percent']))
    with (OUT/'maximum_errors.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=maxima[0]);w.writeheader();w.writerows(maxima)
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix',
                         'font.size':9,'axes.linewidth':.8,'pdf.fonttype':42,
                         'xtick.direction':'in','ytick.direction':'in'})
    fig,axes=plt.subplots(1,2,figsize=(7,3.15),sharey=True)
    for ax,point in zip(axes,('X','M')):
        for theory,plate,color,linestyle in STYLES:
            r=[v for v in maxima if v['three_dimensional']==theory and
               v['plate']==plate and v['point']==point]
            ax.plot([v['h_um'] for v in r],[v['absolute_error_percent'] for v in r],
                    color=color,ls=linestyle,marker='s' if plate=='FSDT' else 'o',
                    ms=3.5,lw=1.25,label=f'{theory.upper()}-{plate}')
        ax.set(xlabel=r'$h$ ($\mu$m)',title=point,xlim=(.4,8.4))
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Maximum error, branches 1-6 (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,frameon=False,loc='upper center',
               bbox_to_anchor=(.5,1.03),fontsize=8)
    fig.subplots_adjust(top=.81,bottom=.18,left=.11,right=.98,wspace=.12)
    fig.savefig(ASSETS/'thickness_M2L4_error.pdf',bbox_inches='tight')
    fig.savefig(OUT/'thickness_M2L4_error.png',dpi=250,bbox_inches='tight')
    plt.close(fig)
    (OUT/'assessment.json').write_text(json.dumps(dict(
        status='PASS_ACTUAL_M2L4_XM_COMPARISON',rows=len(allrows),
        max_percent=max(r['absolute_error_percent'] for r in allrows),
        vertical_fraction_cutoff=VERTICAL_FRACTION_CUTOFF,
        cutoff_note=('The original 0.8 cutoff dropped a genuine h=8 um X-point '
                     'bending-dominant mode with vertical fraction 0.765-0.766; '
                     '0.7 selects the identical first six modes at all other '
                     'thickness/point/theory combinations.'),
        NOT_RUN=['3D full-path band edges','joint in-plane/thickness mesh refinement']),indent=2))
    print((OUT/'assessment.json').read_text())

if __name__=='__main__':main()
