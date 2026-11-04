"""Plot completed, classified gap-closure cases without using RUNNING values.

All interval bounds use the full 153-point set. Curves join the 49 saved boundary
samples. A vetted candidate may be nonstationary and is distinguished from the
best converged terminal result; connecting ell points is not a monotonicity proof.
"""
from __future__ import annotations
from pathlib import Path
import argparse,csv,hashlib,io,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
AREA=math.pi*.09
FACTOR=math.sqrt(4.35e9/1180)/(2*math.pi*75e-6*1e6)
STYLE={'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
       'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'svg.fonttype':'none'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def locate(p):return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)
def source_path(value):
    p=Path(value)
    p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve()
    assert p.is_relative_to(ROOT),(value,'Source outside project')
    return p


class Sources:
    def __init__(self):self.hashes={}
    def read(self,p):
        b=p.read_bytes();self.hashes[locate(p)]=hashlib.sha256(b).hexdigest();return b
    def json(self,p):return json.loads(self.read(p).decode('utf-8-sig'))
    def rows(self,p):return list(csv.DictReader(io.StringIO(self.read(p).decode('utf-8-sig'))))
    def unchanged(self):
        assert all(sha(source_path(p))==h for p,h in self.hashes.items()),'An immutable completed source changed during plotting'


def polygon(design):
    """Independent explicit Helmert reconstruction of the 48-edge polygon."""
    y=np.asarray(design,dtype=float);assert y.shape==(6,) and np.isfinite(y).all() and max(abs(y))<=.16+1e-10
    H=np.zeros((7,6))
    for j in range(6):
        i=j+1;H[:i,j]=1/math.sqrt(i*(i+1));H[i,j]=-i/math.sqrt(i*(i+1))
    radial=np.exp(H@y);idx=np.minimum(np.arange(48)%12,12-np.arange(48)%12)
    angles=np.arange(48)*math.pi/24
    v=radial[idx,None]*np.column_stack([np.cos(angles),np.sin(angles)])
    area=.5*np.sum(v[:,0]*np.roll(v[:,1],-1)-v[:,1]*np.roll(v[:,0],-1))
    return v*math.sqrt(AREA/area)


def validate_vertices(v):
    assert v.shape==(48,2) and np.isfinite(v).all() and np.max(abs(v))<.49
    area=.5*np.sum(v[:,0]*np.roll(v[:,1],-1)-v[:,1]*np.roll(v[:,0],-1))
    assert abs(area-AREA)<1e-12


def spectrum(path,sources):
    data=sources.rows(path);assert len(data)==153,(path,len(data))
    k=np.array([[float(r['kx_a']),float(r['ky_a'])] for r in data])
    expected={(i,j) for i in range(17) for j in range(i+1)}
    assert {(round(x*16/math.pi),round(y*16/math.pi)) for x,y in k}==expected
    assert len({tuple(np.round(q,12)) for q in k})==153
    f=np.array([[float(r[f'Omega{i}']) for i in range(1,7)] for r in data])
    assert np.isfinite(f).all() and np.all(np.diff(f,axis=1)>=-1e-9)
    il,iu=int(np.argmax(f[:,3])),int(np.argmin(f[:,4]));L,U=f[il,3],f[iu,4]
    info=dict(L=float(L),U=float(U),J=float(200*(U-L)/(U+L)),L_MHz=float(L*FACTOR),U_MHz=float(U*FACTOR),
        lower_k_a=k[il].tolist(),upper_k_a=k[iu].tolist(),points=153)
    gx=sorted([r for r in data if abs(float(r['ky_a']))<1e-12],key=lambda r:float(r['kx_a']))
    xm=sorted([r for r in data if abs(float(r['kx_a'])-math.pi)<1e-12 and float(r['ky_a'])>1e-12],key=lambda r:float(r['ky_a']))
    mg=sorted([r for r in data if abs(float(r['kx_a'])-float(r['ky_a']))<1e-12 and float(r['kx_a'])<math.pi-1e-12],key=lambda r:-float(r['kx_a']))
    path_data=gx+xm+mg;assert len(path_data)==49
    x=np.r_[np.linspace(0,1,17),np.linspace(1,2,17)[1:],np.linspace(2,2+math.sqrt(2),17)[1:]]
    pf=np.array([[float(r[f'Omega{i}'])*FACTOR for i in range(1,7)] for r in path_data])
    return dict(gap=info,path_rows=path_data,x=x,f=pf,spectrum=locate(path))


def completed_geometry(folder,ell,sources,allow_anchor=False):
    mpath=folder/'manifest.json';m=sources.json(mpath)
    assert m['status'].startswith('PASS'),(folder,'Only complete converged workers may be displayed')
    assert all(m[k]==v for k,v in [('N',15),('a_um',75.),('h_um',6.),('ell_steel_um',0.)])
    assert abs(m['ell_epoxy_um']-ell)<1e-12
    if not allow_anchor:assert m['optimization_points']==153
    p=folder/'final_ibz153.csv';assert sha(p)==m['output_sha256'][p.name]
    result=spectrum(p,sources)
    assert all(abs(result['gap'][k]-m['final_ibz153'][k])<1e-9 for k in ['L','U','J'])
    vp=folder/'final_vertices.csv';assert sha(vp)==m['output_sha256'][vp.name]
    v=np.array([[float(r['x_over_a']),float(r['y_over_a'])] for r in sources.rows(vp)])
    validate_vertices(v);assert np.max(abs(v-polygon(m['design'])))<1e-12
    result.update(ell=ell,vertices=v,kind='converged terminal',run=locate(folder),
        optimization_points=m.get('optimization_points',m.get('ibz_search_points')),manifest=locate(mpath))
    return result


def vetted_geometry(record,ell,sources):
    assert record is not None
    p=source_path(record['spectrum']);assert sha(p)==record['spectrum_sha256']
    result=spectrum(p,sources)
    assert all(abs(result['gap'][k]-record[k])<1e-9 for k in ['L','U','J'])
    v=polygon(record['design']);validate_vertices(v)
    result.update(ell=ell,vertices=v,kind='vetted candidate (not necessarily stationary)',
        run=record['run'],round=record['round'])
    return result


def load_cases(meta,base,sources):
    result=[]
    for record in meta.get('cases',[]):
        assert record['classification'] in ['POSITIVE_SAMPLED_GAP_FOUND','NO_POSITIVE_SAMPLED_GAP_FOUND','NEAR_ZERO_UNRESOLVED']
        ell=float(record['ell_um']);summary_path=base/f'ell_{ell:g}/case_summary.json'
        assert sources.json(summary_path)==record,'Controller and completed case summary differ'
        selected=completed_geometry(source_path(record['selected_run']),ell,sources)
        witness=vetted_geometry(record['best_vetted_candidate'],ell,sources)
        assert abs(selected['gap']['J']-record['best_converged_J153'])<1e-9
        assert abs(witness['gap']['J']-record['best_vetted_J153'])<1e-9
        result.append(dict(record=record,ell=ell,selected=selected,vetted=witness,anchor=False))
    assert len({c['ell'] for c in result})==len(result)
    return sorted(result,key=lambda c:c['ell'])


def initial_anchor(meta,sources):
    state=meta['initial_state'];mp=source_path(state['seed_manifest']);assert sha(mp)==state['seed_sha256']
    selected=completed_geometry(mp.parent,float(state['ell_um']),sources,allow_anchor=True)
    assert abs(selected['gap']['J']-state['J153'])<1e-9
    rec=dict(ell_um=state['ell_um'],classification='POSITIVE_SAMPLED_GAP_INITIAL_ANCHOR',
        best_converged_J153=state['J153'],best_vetted_J153=state['J153'],selected_run=locate(mp.parent))
    return dict(record=rec,ell=float(state['ell_um']),selected=selected,vetted=selected,anchor=True)


def savefig(fig,stem,generated,out,overwrite):
    for extension in ['png','svg']:
        path=stem.with_suffix('.'+extension)
        if path.exists() and not overwrite:raise FileExistsError(f'Derived output exists; use --overwrite: {path}')
        path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=220,facecolor='white');generated.append(path.relative_to(out).as_posix())
    plt.close(fig)


def shade(a,info,label=True):
    L,U=info['L_MHz'],info['U_MHz'];positive=U>L
    name='Sampled gap (153)' if positive else 'Frequency overlap' if U<L else 'Touching bands'
    if U!=L:a.axhspan(min(L,U),max(L,U),color='#009E73' if positive else '#CC6677',alpha=.16,label=name if label else None)
    a.axhline(L,color='#0072B2',lw=.8,ls=':');a.axhline(U,color='#D55E00',lw=.8,ls=':')


def band_axes(a):
    a.set(xticks=[0,1,2,2+math.sqrt(2)],xticklabels=[r'$\Gamma$','$X$','$M$',r'$\Gamma$'],
        xlabel='Bloch path',ylabel='Frequency (MHz)',xlim=(0,2+math.sqrt(2)))
    for location in [1,2]:a.axvline(location,color='#999999',lw=.7,alpha=.4)


def draw_case(case,out,generated,overwrite):
    g=case['selected'];ell=g['ell'];info=g['gap'];folder=out/f'ell_{ell:g}'
    fig,a=plt.subplots(figsize=(6,6),layout='constrained')
    initial=polygon(np.zeros(6));v=g['vertices']
    for points,label,color,ls in [(initial,'Initial 48-gon','#777777','--'),(v,'Local design','#D55E00','-')]:
        closed=np.vstack([points,points[0]]);a.plot(closed[:,0],closed[:,1],color=color,ls=ls,lw=1.8,label=label)
    a.scatter(v[:,0],v[:,1],color='#D55E00',s=9,zorder=3)
    extent=max(.45,math.ceil(np.max(abs(v))*20)/20+.025)
    a.set(xlabel='$x/a$',ylabel='$y/a$',xlim=(-extent,extent),ylim=(-extent,extent),
        title=fr'Local design at $\ell={ell:g}\,\mu$m ($N=15$)')
    a.set_aspect('equal');a.legend(loc='upper right',frameon=False,fontsize=9)
    savefig(fig,folder/'local_shape',generated,out,overwrite)

    fig,ax=plt.subplots(1,2,figsize=(11.5,5.6),layout='constrained')
    fig.suptitle(fr'$\ell={ell:g}\,\mu$m local design: $a=75\,\mu$m, $h=6\,\mu$m, $N=15$',fontsize=13)
    colors=['#777777','#009E73','#CC79A7','#0072B2','#D55E00','#555555']
    for b in range(6):ax[0].plot(g['x'],g['f'][:,b],color=colors[b],lw=1.6 if b in [3,4] else 1,
        ls='-' if b in [0,2,3] else '--',label=f'Band {b+1}')
    for b,marker in [(3,'o'),(4,'s')]:ax[1].plot(g['x'],g['f'][:,b],color=colors[b],lw=1.5,marker=marker,
        ms=3,ls='-' if b==3 else '--',label=f'Band {b+1}')
    for a in ax:band_axes(a);shade(a,info,a is ax[1])
    ax[0].set(title='First six plate branches',ylim=(0,float(np.max(g['f']))*1.07))
    ax[0].legend(loc='lower center',bbox_to_anchor=(.5,1.10),ncol=3,fontsize=8,frameon=False)
    lo,hi=sorted([info['L_MHz'],info['U_MHz']]);margin=max(.5,.65*(hi-lo))
    ax[1].set(title='Bands 4 and 5 (zoom)',ylim=(max(0,lo-margin),hi+margin))
    ax[1].legend(loc='lower center',bbox_to_anchor=(.5,1.10),ncol=3,fontsize=8,frameon=False)
    fig.supxlabel(f'$L_{{153}}={info["L_MHz"]:.3f}$ MHz;  $U_{{153}}={info["U_MHz"]:.3f}$ MHz;  $J_{{153}}={info["J"]:.3f}\\%$\n'
        'Bounds use all 153 wavevectors; curves join the 49 boundary samples.',fontsize=9.5)
    savefig(fig,folder/'band_diagram',generated,out,overwrite)
    path=folder/'band_path.csv'
    if path.exists() and not overwrite:raise FileExistsError(path)
    with path.open('w',encoding='utf-8',newline='') as h:
        fields=['path_coordinate','kx_a','ky_a']+[f'f{i}_MHz' for i in range(1,7)]
        writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader()
        for x,r,f in zip(g['x'],g['path_rows'],g['f']):writer.writerow(dict(path_coordinate=float(x),kx_a=float(r['kx_a']),ky_a=float(r['ky_a']),**{f'f{i}_MHz':float(f[i-1]) for i in range(1,7)}))
    generated.append(path.relative_to(out).as_posix())


def draw_overview(cases,meta,out,generated,overwrite):
    x=[c['ell'] for c in cases];conv=[c['selected']['gap']['J'] for c in cases];vetted=[c['vetted']['gap']['J'] for c in cases]
    margin=float(meta['gap_sign_buffer_pp']);different=any(abs(a-b)>margin for a,b in zip(conv,vetted))
    fig,a=plt.subplots(figsize=(8,5.4),layout='constrained')
    a.plot(x,conv,ls='--',lw=1,color='#666666',marker='o',mfc='white',ms=4,label='Best converged design (153 points)')
    if different:a.plot(x,vetted,ls=':',lw=1,color='#CC79A7',
        label='Best evaluated shape (may be nonstationary)')
    seen=set()
    for c,j,evaluated in zip(cases,conv,vetted):
        classification=c['record']['classification']
        if c['anchor']:name,color,marker=fr'$\ell={c["ell"]:g}$ anchor (45-point search)','#0072B2','^'
        elif classification.startswith('POSITIVE'):name,color,marker='Positive sampled gap','#0072B2','o'
        elif classification=='NO_POSITIVE_SAMPLED_GAP_FOUND':name,color,marker='No positive sampled gap found','#D55E00','X'
        else:name,color,marker='Near zero (unresolved)','#666666','D'
        classified_value=evaluated if different else j
        a.plot(c['ell'],classified_value,ls='none',marker=marker,color=color,ms=6,label=name if name not in seen else None);seen.add(name)
        a.annotate(f'{j:.3f}',(c['ell'],j),xytext=(0,7 if j>=0 else -12),textcoords='offset points',ha='center',fontsize=8)
        if abs(evaluated-j)>margin:
            a.annotate(f'evaluated {evaluated:.3f}',(c['ell'],evaluated),xytext=(7,-12 if evaluated>=j else 7),
                textcoords='offset points',ha='left',fontsize=8,color='#8B3A75')
    a.axhline(0,color='black',lw=1)
    a.set(xlabel=r'Epoxy length scale $\ell$ ($\mu$m)',ylabel='Signed sampled interval $J_{153}$ (%)',xticks=x,
        title='Sampled gap after local shape searches ($N=15$)')
    a.margins(x=.09,y=.15);a.legend(loc='best',fontsize=8,frameon=False)
    footer='Line segments connect tested states.'
    if any(c['anchor'] for c in cases):footer+='\nThe ell=3 anchor used a 45-point search and a 153-point check.'
    fig.supxlabel(footer,fontsize=8.5)
    savefig(fig,out/'best_gap_vs_ell',generated,out,overwrite)


def draw_endpoints(positive,negative,out,generated,overwrite):
    # Endpoint classification uses vetted full-grid candidates, which may differ
    # from terminal stationary candidates; this distinction remains on the image.
    states=[positive['vetted'],negative['vetted']]
    fig,ax=plt.subplots(2,2,figsize=(11.5,8.6),layout='constrained')
    initial=polygon(np.zeros(6));ext=max(.45,max(math.ceil(np.max(abs(g['vertices']))*20)/20+.025 for g in states))
    minf=min(min(np.min(g['f'][:,3:5]),g['gap']['L_MHz'],g['gap']['U_MHz']) for g in states)
    maxf=max(max(np.max(g['f'][:,3:5]),g['gap']['L_MHz'],g['gap']['U_MHz']) for g in states);pad=(maxf-minf)*.06
    for col,(g,which) in enumerate(zip(states,['Last tested positive sampled gap','First tested state without a positive gap'])):
        v=g['vertices'];closed=np.vstack([v,v[0]]);circle=np.vstack([initial,initial[0]])
        ax[0,col].plot(circle[:,0],circle[:,1],color='#777777',ls='--',lw=1,label='Initial 48-gon')
        ax[0,col].plot(closed[:,0],closed[:,1],color='#D55E00',lw=1.8,label='Best evaluated shape')
        ax[0,col].set(xlabel='$x/a$',ylabel='$y/a$',xlim=(-ext,ext),ylim=(-ext,ext),
            title=f'{which}: $\\ell={g["ell"]:g}\\,\\mu$m')
        ax[0,col].set_aspect('equal');ax[0,col].legend(loc='upper right',fontsize=8,frameon=False)
        for b,color,marker in [(3,'#0072B2','o'),(4,'#D55E00','s')]:ax[1,col].plot(g['x'],g['f'][:,b],color=color,
            ls='-' if b==3 else '--',lw=1.3,marker=marker,ms=2.3,label=f'Band {b+1}')
        band_axes(ax[1,col]);shade(ax[1,col],g['gap'])
        ax[1,col].set(ylim=(max(0,minf-pad),maxf+pad),title=f'$J_{{153}}={g["gap"]["J"]:.3f}\\%$')
        ax[1,col].legend(loc='lower center',bbox_to_anchor=(.5,1.08),ncol=3,fontsize=8,frameon=False)
    fig.suptitle('Comparison of the two tested endpoint states',fontsize=13)
    fig.supxlabel('Fixed-area searches in the same 48-node design space; all gap bounds use 153 wavevectors.',fontsize=8.5)
    savefig(fig,out/'endpoint_comparison',generated,out,overwrite)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'05_validation/GAP_CLOSURE_SCAN_20261031/manifest.json')
    parser.add_argument('--out',type=Path,default=ROOT/'output/gap_closure_scan')
    parser.add_argument('--ell',type=float,help='Plot this completed case instead of the latest one.')
    parser.add_argument('--all-cases',action='store_true',help='Also produce shape and band figures for every completed case.')
    parser.add_argument('--include-anchor',action='store_true',help='Include the separately validated initial ell anchor; its distinct search grid is labelled.')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args();mp=args.manifest.resolve();out=args.out.resolve()
    raw=mp.read_bytes();meta=json.loads(raw.decode('utf-8-sig'));scan_sha=hashlib.sha256(raw).hexdigest()
    assert (meta['N'],meta['a_um'],meta['h_um'],meta['ell_steel_um'])==(15,75,6,0)
    sources=Sources();cases=load_cases(meta,mp.parent,sources)
    if not cases:
        print('No completed classified cases in manifest.cases; RUNNING worker values are deliberately not plotted.')
        return
    chosen=next((c for c in cases if abs(c['ell']-args.ell)<1e-12),None) if args.ell is not None else cases[-1]
    assert chosen is not None,'Requested ell is not a completed case'
    displayed=cases.copy()
    if args.include_anchor:displayed=[initial_anchor(meta,sources)]+displayed
    out.mkdir(parents=True,exist_ok=True);generated=[]
    with plt.rc_context(STYLE):
        draw_overview(displayed,meta,out,generated,args.overwrite)
        for c in cases if args.all_cases else [chosen]:draw_case(c,out,generated,args.overwrite)
        endpoint_ells=[meta.get('last_positive_tested_ell_um'),meta.get('first_no_positive_gap_tested_ell_um')]
        if all(v is not None for v in endpoint_ells):
            endpoints=[]
            for ell in endpoint_ells:
                c=next((c for c in cases if abs(c['ell']-ell)<1e-12),None)
                if c is None and abs(meta['initial_state']['ell_um']-ell)<1e-12:c=initial_anchor(meta,sources)
                assert c is not None,'Endpoint is absent from completed evidence'
                endpoints.append(c)
            draw_endpoints(*endpoints,out,generated,args.overwrite)
    with (out/'scan_plot_data.csv').open('w',encoding='utf-8',newline='') as h:
        fields=['ell_um','classification','best_converged_J153','best_vetted_J153','selected_run','vetted_spectrum','initial_anchor']
        writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader()
        for c in displayed:writer.writerow(dict(ell_um=c['ell'],classification=c['record']['classification'],
            best_converged_J153=c['selected']['gap']['J'],best_vetted_J153=c['vetted']['gap']['J'],
            selected_run=c['selected']['run'],vetted_spectrum=c['vetted']['spectrum'],initial_anchor=c['anchor']))
    generated.append('scan_plot_data.csv')
    (out/'scan_manifest_snapshot.json').write_bytes(raw);generated.append('scan_manifest_snapshot.json')
    sources.unchanged()
    record=dict(status='DERIVED_FROM_COMPLETED_CLASSIFIED_CASES',scan_status_at_read=meta['status'],
        source_manifest=locate(mp),source_manifest_sha256_at_read=scan_sha,
        live_manifest_changed_during_plot=sha(mp)!=scan_sha,completed_ells=[c['ell'] for c in cases],
        only_completed_classified_cases_plotted=True,
        excluded_running_ell=meta.get('current_ell_um') if meta['status']=='RUNNING' and meta.get('current_ell_um') not in [c['ell'] for c in cases] else None,
        selected_ell=chosen['ell'],selected_run=chosen['selected']['run'],selected_gap=chosen['selected']['gap'],
        plotted_source_sha256=sources.hashes,plot_script_sha256=sha(Path(__file__)),
        normalization_MHz_per_Omega=FACTOR,matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,
        vetted_vs_converged_distinction='Both raw values retained; alternative vetted-candidate curve shown if any difference exceeds the reporting buffer.',
        decision_buffer_pp=meta['gap_sign_buffer_pp'],buffer_is_not_a_certified_error_bound=True,
        transformations=['All bounds use 153 wavevectors. Each band curve joins 49 saved boundary samples; actual path length includes the sqrt2 diagonal leg.',
            'Only completed, classified manifest.cases are shown; the optional initial anchor is explicitly distinct.',
            'Positive gap and frequency overlap shading follow the measured sign, without clipping negative intervals.',
            'Endpoint panels use best vetted geometry, which may be nonstationary; per-case panels use selected converged terminals.',
            'Geometry shows 48 straight edges; no spline or raster approximation.'],
        no_monotonicity_proof=True,no_global_absence_proof=True,no_continuous_zone_certification=True,
        output_sha256={name:sha(out/name) for name in generated},visual_review='PENDING_INSPECTION',
        human_verified=False,new_eigensolves=False,new_COMSOL_runs=False)
    (out/'plot_provenance.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(completed_ells=record['completed_ells'],selected_ell=record['selected_ell'],selected_gap=record['selected_gap'],outputs=generated),indent=2))


if __name__=='__main__':main()
