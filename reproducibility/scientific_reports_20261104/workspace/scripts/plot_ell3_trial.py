"""Plot a completed ell3 local design from saved spectra only.

The gap sign controls the shading: a positive sampled interval is shown as a
sampled gap; a negative interval is shown as frequency overlap. No eigensolver,
shape smoothing, or continuous-zone/global-optimum claim is made here.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
AREA=math.pi*.3**2
NORMALIZATION=math.sqrt(4.35e9/1180)/(2*math.pi*75e-6*1e6)


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locator(path):
    try:return path.relative_to(ROOT).as_posix()
    except ValueError:return str(path)


def xy(path):
    data=rows(path)
    values=np.asarray([[float(r['x_over_a']),float(r['y_over_a'])] for r in data])
    assert values.shape==(48,2) and np.isfinite(values).all(),path
    area=.5*np.sum(values[:,0]*np.roll(values[:,1],-1)-values[:,1]*np.roll(values[:,0],-1))
    assert abs(area-AREA)<1e-12,(path,area)
    return values


def spectrum(path,grid=16):
    data=rows(path)
    assert len(data)==(grid+1)*(grid+2)//2,(path,len(data))
    coordinates=np.asarray([[float(r['kx_a']),float(r['ky_a'])] for r in data])
    expected={(i,j) for i in range(grid+1) for j in range(i+1)}
    found={(round(k[0]*grid/math.pi),round(k[1]*grid/math.pi)) for k in coordinates}
    assert found==expected and len(found)==len(data),(path,'wrong/duplicate grid')
    frequencies=np.asarray([[float(r[f'Omega{b}']) for b in range(1,7)] for r in data])
    assert np.isfinite(frequencies).all() and np.all(np.diff(frequencies,axis=1)>=-1e-9)
    lower=int(np.argmax(frequencies[:,3]));upper=int(np.argmin(frequencies[:,4]))
    L,U=frequencies[lower,3],frequencies[upper,4]
    info=dict(points=len(data),L=float(L),U=float(U),J_percent=float(200*(U-L)/(U+L)),
        L_MHz=float(L*NORMALIZATION),U_MHz=float(U*NORMALIZATION),
        lower_k_a=coordinates[lower].tolist(),upper_k_a=coordinates[upper].tolist(),
        positive_sampled_gap=bool(U>L))
    return data,info


def path_data(data,grid=16):
    gx=sorted([r for r in data if abs(float(r['ky_a']))<1e-12],key=lambda r:float(r['kx_a']))
    xm=sorted([r for r in data if abs(float(r['kx_a'])-math.pi)<1e-12 and float(r['ky_a'])>1e-12],key=lambda r:float(r['ky_a']))
    mg=sorted([r for r in data if abs(float(r['kx_a'])-float(r['ky_a']))<1e-12 and float(r['kx_a'])<math.pi-1e-12],key=lambda r:-float(r['kx_a']))
    path=gx+xm+mg
    assert len(path)==3*grid+1
    horizontal=np.r_[np.linspace(0,1,grid+1),np.linspace(1,2,grid+1)[1:],np.linspace(2,2+math.sqrt(2),grid+1)[1:]]
    values=np.asarray([[float(r[f'Omega{i}'])*NORMALIZATION for i in range(1,7)] for r in path])
    return path,horizontal,values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=ROOT/'output/ell3_trial')
    parser.add_argument('--overwrite',action='store_true',help='Regenerate these derived plots explicitly; never changes raw data.')
    args=parser.parse_args();run=args.run_dir.resolve();out=args.out.resolve()
    meta_path=run/'manifest.json';meta=load(meta_path)
    assert meta['status'].startswith('PASS'),('Incomplete/failed run must not be presented as a final design',meta['status'])
    assert all(meta[k]==v for k,v in [('ell_epoxy_um',3.),('ell_steel_um',0.),('N',15),('a_um',75.),('h_um',6.)])
    final_path=run/'final_ibz153.csv';data,info=spectrum(final_path)
    assert all(abs(info[k]-meta['final_ibz153'][k])<1e-9 for k in ['L','U'])
    assert abs(info['J_percent']-meta['final_ibz153']['J'])<1e-8
    path,horizontal,freq=path_data(data)
    out.mkdir(parents=True,exist_ok=True)
    generated=['ell3_local_shape.png','ell3_local_shape.svg','ell3_band_diagram.png','ell3_band_diagram.svg','ell3_band_path.csv']
    if not args.overwrite:
        assert not any((out/name).exists() for name in generated),'Output exists; use --overwrite for explicit derived-plot regeneration.'

    initial=ROOT/'02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1/initial_vertices.csv'
    previous=ROOT/'05_validation/V07_ell2_reoptimization_20261030/restart/final_vertices.csv'
    final_vertices=run/'final_vertices.csv'
    polygons=[(initial,'Initial 48-gon','#777777','--'),
              (previous,r'$\ell=2$ design','#009E73','-.'),
              (final_vertices,r'$\ell=3$ local design','#D55E00','-')]
    polygon_values=[(p,label,color,ls,xy(p)) for p,label,color,ls in polygons]
    extent=max(.45,math.ceil(max(np.max(abs(v)) for *_,v in polygon_values)*20)/20+.025)
    style={'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelsize':11,'axes.grid':True,'grid.alpha':.18,
        'svg.fonttype':'none'}
    with plt.rc_context(style):
        fig,ax=plt.subplots(figsize=(6,6),layout='constrained')
        for p,label,color,ls,v in polygon_values:
            closed=np.vstack([v,v[0]])
            ax.plot(closed[:,0],closed[:,1],color=color,ls=ls,lw=2 if p==final_vertices else 1.4,label=label)
            if p==final_vertices:ax.scatter(v[:,0],v[:,1],color=color,s=9,zorder=3)
        ax.set(xlabel='$x/a$',ylabel='$y/a$',xlim=(-extent,extent),ylim=(-extent,extent),
               title=r'Local design at $\ell=3\,\mu$m ($N=15$)')
        ax.set_aspect('equal');ax.legend(loc='upper right',frameon=False,fontsize=9)
        for extension in ['png','svg']:
            fig.savefig(out/f'ell3_local_shape.{extension}',dpi=220,facecolor='white')
        plt.close(fig)

        # Legends sit above the axes so no legend rectangle can erase a branch.
        fig,ax=plt.subplots(1,2,figsize=(11.5,5.6),layout='constrained')
        fig.suptitle(r'$\ell=3\,\mu$m local design: $a=75\,\mu$m, $h=6\,\mu$m, $N=15$',fontsize=13)
        colors=['#777777','#009E73','#CC79A7','#0072B2','#D55E00','#555555']
        for b in range(6):
            ax[0].plot(horizontal,freq[:,b],color=colors[b],lw=1.6 if b in [3,4] else 1,
                ls='-' if b in [0,2,3] else '--',label=f'Band {b+1}')
        for b,marker in [(3,'o'),(4,'s')]:
            ax[1].plot(horizontal,freq[:,b],color=colors[b],lw=1.5,marker=marker,ms=3,
                ls='-' if b==3 else '--',label=f'Band {b+1}')
        L,U=info['L_MHz'],info['U_MHz']
        lo,hi=sorted([L,U])
        label='Sampled gap (153)' if U>L else 'Frequency overlap' if U<L else 'Touching bands'
        shade='#009E73' if U>L else '#CC6677' if U<L else '#999999'
        for a in ax:
            if hi>lo:a.axhspan(lo,hi,color=shade,alpha=.16,label=label if a is ax[1] else None)
            a.axhline(L,color=colors[3],lw=.8,ls=':')
            a.axhline(U,color=colors[4],lw=.8,ls=':')
            a.set(xticks=[0,1,2,2+math.sqrt(2)],xticklabels=[r'$\Gamma$','$X$','$M$',r'$\Gamma$'],
                xlabel='Bloch path',ylabel='Frequency (MHz)',xlim=(0,2+math.sqrt(2)))
            for location in [1,2]:a.axvline(location,color='#999999',lw=.7,alpha=.4)
        ax[0].set(title='First six plate branches',ylim=(0,float(np.max(freq))*1.07))
        ax[0].legend(loc='lower center',bbox_to_anchor=(.5,1.10),ncol=3,fontsize=8,frameon=False)
        margin=max(.5,.65*(hi-lo))
        ax[1].set(title='Bands 4 and 5 (zoom)',ylim=(max(0,lo-margin),hi+margin))
        ax[1].legend(loc='lower center',bbox_to_anchor=(.5,1.10),ncol=3,fontsize=8,frameon=False)
        fig.supxlabel(f'$L_{{153}}={L:.3f}$ MHz;  $U_{{153}}={U:.3f}$ MHz;  $J_{{153}}={info["J_percent"]:.2f}\\%$\n'
            'Bounds use all 153 wavevectors; curves show the sampled boundary path.',fontsize=9.5)
        for extension in ['png','svg']:
            fig.savefig(out/f'ell3_band_diagram.{extension}',dpi=220,facecolor='white')
        plt.close(fig)

    fields=['path_coordinate','kx_a','ky_a']+[f'f{i}_MHz' for i in range(1,7)]
    with (out/'ell3_band_path.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for x,r,f in zip(horizontal,path,freq):
            writer.writerow(dict(path_coordinate=float(x),kx_a=float(r['kx_a']),ky_a=float(r['ky_a']),
                **{f'f{i}_MHz':float(f[i-1]) for i in range(1,7)}))
    sources=[meta_path,final_path]+[p for p,*_ in polygons]
    manifest=dict(status='DERIVED_FROM_COMPLETED_ELL3_TRIAL',selected_run=locator(run),
        destination='General local-research review figures; no publisher compliance claim.',
        parameters=dict(a_um=75,h_um=6,ell_epoxy_um=3,ell_steel_um=0,N=15),
        normalization=dict(MHz_per_Omega=NORMALIZATION,E_ref_Pa=4.35e9,rho_ref_kg_m3=1180,a_m=75e-6,
            formula='f_MHz=Omega*sqrt(E_ref/rho_ref)/(2*pi*a_m*1e6)'),
        sampled_gap=info,optimization_final45=meta.get('final_ibz45'),
        dense_gap_drift_pp=meta.get('dense_gap_drift_pp'),
        stationarity_on153_not_inferred=True,source_sha256={locator(p):sha(p) for p in sources},
        transformations=['49 saved boundary-path samples from the153point triangular set; lines join samples without smoothing.',
            'Full153point set defines L=maxOmega4 andU=minOmega5, regardless of where the path extrema lie.',
            'Shading is a positive sampled interval only if U>L; otherwise it labels frequency overlap.',
            'Actual48straight polygon edges andvertices shown; steel area unchanged.',
            'Left frequency panel shows allsix retained plate branches, right panel is an explicitly labelled zoom.'],
        uncertainty='Prescribed deterministic calculation; no statistical confidence intervals inferred.',
        output_sha256={name:sha(out/name) for name in generated},visual_review='PENDING_INSPECTION',
        new_eigensolves=False,new_COMSOL_runs=False,human_verified=False)
    (out/'plot_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(outputs=generated,selected_run=locator(run),sampled_gap=info),indent=2))


if __name__=='__main__':main()
