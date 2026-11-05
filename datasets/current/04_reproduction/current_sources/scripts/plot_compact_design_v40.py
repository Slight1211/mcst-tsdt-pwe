"""Compact v40 layouts, preserving all saved v31 design records and old artwork."""
from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pypdf import PdfReader
from figure_style_v31 import RC,COL,WIDTH_MM,style_context,tidy_axes,export_figure

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'03_review/figures_v27/design'
HERE=ROOT/'03_review/revision_v40/compact/design'
OUT=ROOT/'latex/figures'
HERE.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected_art={p:sha(p) for p in OUT.glob('*.pdf') if '_v40' not in p.name}
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def closed(v):return np.vstack([v,v[0]])
def axmm(fig,x,y,w,h,H):return fig.add_axes([x/164,y/H,w/164,h/H])
def txt(fig,x,y,s,H,**kw):return fig.text(x/164,y/H,s,**kw)
def panel(fig,x,y,s,H):txt(fig,x,y,s,H,fontsize=9,fontweight='bold',va='top')

sources=[OLD/n for n in ['figure4_snapshot_vertices.csv','figure4_iteration_records.csv','figure4_boundary_spectra.csv','figure5_shape_vertices.csv','figure5_solid_gap_records.csv','design_provenance.json']]
sources += [ROOT/'output/data/Figure6_design_story_v19.csv',ROOT/'scripts/figure_style_v31.py',ROOT/'FABLE_REVIEW/style/mcst_paper.mplstyle']
hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
oldprov=json.loads((OLD/'design_provenance.json').read_text(encoding='utf8'))
for name,digest in oldprov['source_sha256'].items():assert sha(ROOT/name)==digest,name
snap=rows(sources[0]);it=rows(sources[1]);spec=rows(sources[2]);shape=rows(sources[3]);solid=rows(sources[4]);mech=rows(ROOT/'output/data/Figure6_design_story_v19.csv')
assert (len(snap),len(it),len(spec),len(shape),len(solid))==(192,20,100,240,4)
initial=np.array([[float(r['x_over_a']),float(r['y_over_a'])] for r in snap if int(r['iteration'])==0])

def shape_cell(ax,v,reference=True,color=None):
    # Thin frame denotes the actual a-by-a unit cell, not a decorative card.
    ax.add_patch(Rectangle((-.5,-.5),1,1,fc='white',ec='#D9D9D9',lw=.6,zorder=0))
    if reference:ax.plot(*closed(initial).T,color=COL['init'],ls=(0,(3,2.4)),lw=1.2,zorder=2)
    color=color or COL['opt']
    ax.fill(*v.T,color=color,alpha=.055,ec='none',zorder=1)
    ax.plot(*closed(v).T,color=color,lw=1.2,ls='--' if color==COL['init'] else '-',zorder=3)
    ax.set(xlim=(-.52,.52),ylim=(-.52,.52),aspect='equal')
    ax.set_axis_off()

def scale_bar(ax,side=False):
    y=-.85 if side else -.66
    ax.plot([-.45,.05],[y,y],color=COL['ink'],lw=.6,clip_on=False)
    for x in [-.45,.05]:ax.plot([x,x],[y-.03,y+.03],color=COL['ink'],lw=.6,clip_on=False)
    if side:ax.text(.13,y,r'$0.5a$',ha='left',va='center',fontsize=8,clip_on=False)
    else:ax.text(-.20,-.79,r'$0.5a$',ha='center',va='top',fontsize=8,clip_on=False)

def finish(fig,name):
    fig.canvas.draw();rend=fig.canvas.get_renderer();bad=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip():
            b=t.get_window_extent(rend)
            if b.x0<-.6 or b.y0<-.6 or b.x1>fig.bbox.width+.6 or b.y1>fig.bbox.height+.6:bad.append(t.get_text())
    assert not bad,bad
    record=export_figure(fig,OUT/name)
    assert record['minimum_visible_font_pt']>=7
    p=PdfReader(OUT/f'{name}.pdf').pages[0]
    record['fonts']=[str(f.get_object()['/BaseFont']) for f in p['/Resources']['/Font'].get_object().values()]
    record['clipped_text']=bad
    plt.close(fig)
    return record

with style_context():
    H=112.
    fig=plt.figure(figsize=(164/25.4,H/25.4))
    panel(fig,1,110,'a',H)
    txt(fig,10,108,'Evolution of the interface',H,fontsize=9,va='top')
    first=None
    for j,step in enumerate([0,2,6,19]):
        rec=[r for r in snap if int(r['iteration'])==step]
        v=np.array([[float(r['x_over_a']),float(r['y_over_a'])] for r in rec])
        ax=axmm(fig,10+39*j,68,27,27,H)
        shape_cell(ax,v,reference=bool(step),color=COL['init'] if step==0 else COL['opt'])
        first=first or ax
        ax.set_title(f'Step {step}\n'+fr'$J_{{45}}={float(rec[0]["active_set_gap_percent"]):.2f}\%$',fontsize=8,pad=4)
    scale_bar(first)
    handles=[Line2D([],[],color=COL['init'],ls=(0,(3,2.4)),lw=1.2,label='Initial'),Line2D([],[],color=COL['opt'],lw=1.2,label='Updated')]
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.978,.985),ncol=2,borderaxespad=0,fontsize=8,handlelength=1.8)
    b=axmm(fig,14,12,39,35,H);c=axmm(fig,68,12,38,35,H);d=axmm(fig,121,12,39,35,H)
    for x,letter in [(1,'b'),(55,'c'),(109,'d')]:panel(fig,x,58,letter,H)
    tt=np.array([int(r['iteration']) for r in it]);jj=np.array([float(r['J_percent']) for r in it]);aux=np.array([float(r['auxiliary_J_percent']) for r in it]);vv=np.array([float(r['constraint_violation']) for r in it]);feas=vv<=1e-6
    assert feas[0] and np.all(feas[14:]) and not np.any(feas[1:14])
    b.axvspan(14,19,color=COL['gap'],alpha=.08,zorder=-2)
    b.plot(tt,aux,color=COL['ink'],ls=(0,(4,2.4)),lw=1.2,label='Auxiliary')
    b.plot(tt,jj,color=COL['init'],lw=1.2)
    for flag,name,color in [(False,'Violated',COL['infeasible']),(True,'Feasible',COL['feasible'])]:
        b.plot(tt[feas==flag],jj[feas==flag],ls='none',marker='o',ms=3.2,mec=color,mfc=color,label=name)
    for step in [0,2,6,19]:b.plot(step,jj[step],marker='D',ms=4.8,mfc='white',mec=COL['feasible'] if feas[step] else COL['infeasible'],mew=1.,ls='none')
    b.set(xlim=(-.5,19.5),ylim=(27,38),xticks=[0,6,14,19],yticks=[28,32,36],xlabel='Iteration',ylabel='Relative gap (%)')
    b.text(.98,.08,'14–19',transform=b.transAxes,ha='right',va='bottom',fontsize=8,color=COL['muted'])
    b.legend(loc='upper right',ncol=1,borderaxespad=.15,labelspacing=.20,handlelength=1.3,handletextpad=.35)
    c.axvspan(14,19,color=COL['gap'],alpha=.08,zorder=-2)
    c.semilogy(tt,np.maximum(vv,1e-12),color=COL['opt'],marker='o',ms=3.,lw=1.2)
    c.axhline(1e-6,color=COL['muted'],ls=(0,(4,2.4)),lw=.6)
    c.text(.98,.57,r'$10^{-6}$',transform=c.transAxes,ha='right',va='bottom',fontsize=8,color=COL['muted'])
    c.set(xlim=(-.5,19.5),ylim=(1e-13,1),xticks=[0,6,14,19],yticks=[1e-12,1e-6,1],xlabel='Iteration',ylabel='Maximum violation')
    c.set_title('Constraint satisfaction',pad=16)
    bands={}
    for name,color,ls in [('Initial',COL['init'],(0,(4,2.4))),('Terminal',COL['opt'],'-')]:
        values=[]
        for band in [4,5]:
            rr=sorted([r for r in spec if r['shape']==name and int(r['band'])==band],key=lambda r:int(r['path_index']))
            xx=np.array([int(r['path_index'])/8 for r in rr]);ff=np.array([float(r['frequency_MHz']) for r in rr]);values.append(ff)
            d.plot(xx,ff,color=color,ls=ls,lw=1.2,label=name if band==4 else None)
            k=np.argmax(ff) if band==4 else np.argmin(ff)
            d.plot(xx[k],ff[k],marker='^' if band==4 else 'v',ms=3.8,color=color,ls='none')
        bands[name]=values
    assert all(2.35<float(f.min()) and float(f.max())<11.75 for pair in bands.values() for f in pair)
    d.axhspan(max(bands['Terminal'][0]),min(bands['Terminal'][1]),color=COL['gap'],alpha=.15,zorder=-3)
    d.set(xlim=(-.04,3.04),ylim=(2.35,11.75),xticks=[0,1,2,3],xticklabels=[r'$\Gamma$','$X$','$M$',r'$\Gamma$'],yticks=[3,7,11],xlabel='Bloch path',ylabel='Frequency (MHz)')
    d.legend(loc='lower left',bbox_to_anchor=(.03,1.07),borderaxespad=0,labelspacing=.35,handlelength=1.8)
    for ax in [b,c,d]:tidy_axes(ax)
    f4=finish(fig,'figure5_optimization_compact_v40')

with style_context():
    H=116.
    fig=plt.figure(figsize=(164/25.4,H/25.4))
    panel(fig,1,114,'a',H)
    first=None
    for j,ell in enumerate([0,.5,1,2,5]):
        rr=[r for r in shape if float(r['ell_um'])==ell]
        vv=np.array([[float(r['x_over_a']),float(r['y_over_a'])] for r in rr])
        ax=axmm(fig,9+30*j,77,27,27,H);shape_cell(ax,vv);first=first or ax
        label=fr'$\ell={ell:g}$ µm'+(' †' if ell in [0,.5] else '')
        ax.set_title(label+'\n'+fr'$J_{{45}}={float(rr[0]["J153_percent"]):.2f}\%$',fontsize=8,pad=4)
    scale_bar(first,side=True)
    txt(fig,71,70,'† Bound-limited',H,fontsize=8,color=COL['muted'],va='center')
    b=axmm(fig,14,12,62,40,H);c=axmm(fig,102,12,57,40,H)
    panel(fig,1,57,'b',H);panel(fig,89,57,'c',H)
    txt(fig,14,56,'Gap retention and redesign',H,fontsize=8,va='top')
    txt(fig,102,56,'Modal response',H,fontsize=8,va='top')
    transfer=[r for r in mech if r['record_type']=='gap_transfer'];redesign=[r for r in mech if r['record_type']=='gap_reoptimized'];energy=[r for r in mech if r['record_type']=='fixed_geometry_modal_energy']
    for name,color,marker,label in [('initial',COL['init'],'o','Initial, plate'),('final',COL['opt'],'s','1 µm design, plate')]:
        rr=sorted([r for r in transfer if r['shape']==name],key=lambda r:float(r['ell_um']))
        b.plot([float(r['ell_um']) for r in rr],[float(r['J_percent']) for r in rr],color=color,marker=marker,ms=4,lw=1.2,label=label)
    b.plot(2,float(redesign[0]['J_percent']),color=COL['reopt'],marker='*',ms=8,ls='none',label='2 µm design, plate')
    for name,marker,label in [('initial','o','Initial, solid'),('final','s','Redesigned, solid')]:
        rr=[r for r in solid if r['shape']==name]
        b.plot([float(r['ell_um']) for r in rr],[float(r['J_percent']) for r in rr],color=COL['solid3d'],marker=marker,ms=5.1,mfc='white',mew=1.15,ls='none',label=label)
    b.set(xlim=(.4,2.15),ylim=(23.8,35.3),xticks=[.5,1,1.5,2],yticks=[24,28,32,35],xlabel='Epoxy length scale ℓ (µm)',ylabel=r'Relative gap $J_{45}$ (%)')
    # Direct labels identify the three modal series; the shared top row keys gap roles.
    shared_handles=[Line2D([],[],color=COL['init'],marker='o',lw=1.2,label='Initial, plate'),
                    Line2D([],[],color=COL['opt'],marker='s',lw=1.2,label='1 µm design, plate'),
                    Line2D([],[],color=COL['reopt'],marker='*',ms=7,lw=0,label='2 µm design, plate'),
                    Line2D([],[],color='black',marker='o',mfc='white',ms=5,lw=0,label='3D, Γ/X/M')]
    fig.legend(handles=shared_handles,loc='center',bbox_to_anchor=(.52,63/H),ncol=4,
               handlelength=1.4,columnspacing=1.,handletextpad=.5,borderaxespad=0,fontsize=7)
    gap_initial={float(r['ell_um']):float(r['J_percent']) for r in transfer if r['shape']=='initial'}
    gap_design={float(r['ell_um']):float(r['J_percent']) for r in transfer if r['shape']=='final'}
    b.plot([1,1],[gap_initial[1]+.20,gap_design[1]-.20],color=COL['muted'],lw=.6,zorder=1)
    b.text(.94,.5*(gap_initial[1]+gap_design[1]),'+5.2 pp',ha='right',va='center',fontsize=7)
    b.plot([2,2],[gap_initial[2]+.08,gap_design[2]-.08],color=COL['muted'],lw=.6,zorder=1)
    b.text(1.92,24.60,'+0.8 pp',ha='right',va='center',fontsize=7)
    b.text(2.035,27.15,'+2.1 pp',ha='left',va='bottom',fontsize=7,color=COL['reopt'],clip_on=False)
    b.axvline(1,color=COL['zero'],lw=.6,ls=':',zorder=0)
    for point,band,marker,ls,label in [('X',4,'o','-',r'$X_4$'),('X',5,'s',(0,(4,2.4)),r'$X_5$'),('Gamma',5,'^',(0,(1.5,2)),r'$\Gamma_5$')]:
        rr=sorted([r for r in energy if r['point']==point and int(r['band'])==band],key=lambda r:float(r['ell_um']))
        color=COL['mode5G'] if point=='Gamma' else COL[f'mode{band}']
        c.plot([float(r['ell_um']) for r in rr],[float(r['R_mc']) for r in rr],color=color,marker=marker,ms=4,ls=ls,lw=1.2,label=label)
        c.text(1.95,float(rr[-1]['R_mc'])+.012,label,ha='right',va='bottom',fontsize=7,color=color)
    c.axvline(1,color=COL['zero'],lw=.6,ls=':')
    c.text(1.035,.96,'Design state',transform=c.get_xaxis_transform(),fontsize=7,color=COL['muted'],va='top')
    c.set(xlim=(.4,2.1),ylim=(0,.30),xticks=[.5,1,1.5,2],yticks=[0,.1,.2,.3],xlabel='Epoxy length scale ℓ (µm)',ylabel=r'Couple-stress fraction $R_{\mathrm{mc}}$')
    for ax in [b,c]:tidy_axes(ax)
    f5=finish(fig,'figure6_lengthscale_compact_v40')

for name,digest in hashes.items():assert sha(ROOT/name)==digest
record=dict(status='GENERATED_PENDING_VISUAL',source_sha256=hashes,raw_source_sha256=oldprov['source_sha256'],figure5=f4,figure6=f5,shared_style_sha256=sha(ROOT/'scripts/figure_style_v31.py'),new_solves=0,new_optimizations=0,
 data_unchanged_from_v27=True,geometry='Saved48 vertices, straight segments; samecellframe[-.5,.5]^2; scale bar0.5a',
 initial_feasible=True,continuous_feasible_iterations=list(range(14,20)),zero_violation_display=1e-12,
 solid_scope='Four Gamma/X/M points; optimizedgeometries differ atell1/ell2; no connectinglines',
 note_J45='Subscript45 denotes branches4and5, notpointcount',script_sha256=sha(Path(__file__)))
for source in sources[:5]:
    dest=HERE/source.name
    shutil.copy2(source,dest)
    assert dest.read_bytes()==source.read_bytes()
(HERE/'plot_source_records.json').write_text(json.dumps(dict(snapshots=snap,iterations=it,spectra=spec,
    lengthscale_vertices=shape,solid_gaps=solid,gap_and_modal_records=mech),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
assert f'{gap_design[1]-gap_initial[1]:.1f}'=='5.2'
assert f'{gap_design[2]-gap_initial[2]:.1f}'=='0.8'
assert f'{float(redesign[0]["J_percent"])-gap_initial[2]:.1f}'=='2.1'
record.update(data_record_counts=dict(snapshots=len(snap),iterations=len(it),spectra=len(spec),shape_vertices=len(shape),solid_gaps=len(solid),mechanism_records=len(mech)),
 mode_colors={'X4':COL['mode4'],'X5':COL['mode5'],'Gamma5':COL['mode5G']},
 requested_gain_annotations_pp=[5.2,.8,2.1],solid_marker_connection='none',
 caption_changes=['Teal/copper -> blue/vermilion; Gamma band5 is green.',
 'Panel(a) gap widths are in the titles above each shape, not below the frames.',
 'Panel(b) direct gain labels +5.2,+0.8,+2.1pp and common 3D hollow-marker key.',
 'Panel(c) direct mode labels replace the legend.'])
(HERE/'design_provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'status':record['status'],'figure5':f4['outputs'],'figure6':f5['outputs']},ensure_ascii=False,indent=2))

assert all(sha(p)==v for p,v in protected_art.items())
