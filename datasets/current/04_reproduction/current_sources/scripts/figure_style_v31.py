"""Author-selected open-frame style; native164mm manuscript artwork.

Derived from FABLE_REVIEW/style/mcst_paper.mplstyle. Publisher rules and
author choices are recorded separately; this preset is not certification.
"""
from pathlib import Path
import hashlib
import matplotlib as mpl
from matplotlib import pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
STYLE_FILE=ROOT/'FABLE_REVIEW/style/mcst_paper.mplstyle'
WIDTH_MM=164.
MM=1/25.4
COL={
 'classic':'#6E6E6E','mcst':'#0072B2','mode4':'#0072B2','mode5':'#D55E00',
 'mode5G':'#009E73','mode5_gamma':'#009E73','init':'#6E6E6E','initial':'#6E6E6E',
 'opt':'#0072B2','design':'#0072B2','reopt':'#D55E00','solid3d':'#000000',
 'infeasible':'#6E6E6E','feasible':'#0072B2','steel_mask':'#D9D9D9',
 'zero':'#B5B5B5','ink':'#1A1A1A','muted':'#6E6E6E','axis':'#404040',
 'grid':'#E5E5E5','pale':'#F4F4F4','gap':'#0072B2',
}
RC=dict(mpl.rc_params_from_file(STYLE_FILE,use_default_template=False))
RC.update({'savefig.bbox':None,'savefig.pad_inches':0.,'savefig.transparent':False,
 'savefig.facecolor':'white','figure.facecolor':'white','axes.facecolor':'white',
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600,
 'text.color':COL['ink'],'axes.labelcolor':COL['ink'],
 'axes.unicode_minus':True})
RCPARAMS=RC
STYLE={'FSDT':dict(ls=(0,(4,2.4)),marker='s',mfc='white',lw=1.2),
       'TSDT':dict(ls='-',marker='o',lw=1.56)}
ENERGY_CMAP=mpl.colormaps['cividis']

def style_context():return plt.rc_context(RC)

def panel_label(ax,letter,x=-.14,y=1.055,**kwargs):
    opts=dict(transform=ax.transAxes,fontweight='bold',fontsize=9.,color=COL['ink'],ha='left',va='bottom',clip_on=False)
    opts.update(kwargs)
    return ax.text(x,y,letter,**opts)
add_panel_label=panel_label

def tidy_axes(ax,grid=None):
    ax.grid(False,which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():spine.set_linewidth(.6);spine.set_color(COL['axis'])
    ax.tick_params(axis='both',which='both',direction='out',width=.6,colors=COL['axis'])
    return ax

def finalize_style(fig):
    """Standardize residual explicit typography without changing data geometry."""
    for ax in fig.axes:
        if not ax.axison:
            ax.set_xticks([]);ax.set_yticks([])
            continue
        tidy_axes(ax)
        ax.xaxis.label.set_fontsize(8);ax.yaxis.label.set_fontsize(8)
        ax.title.set_fontsize(8)
        for tick in ax.get_xticklabels()+ax.get_yticklabels():tick.set_fontsize(7)
        for t in [ax.xaxis.get_offset_text(),ax.yaxis.get_offset_text()]:t.set_fontsize(7)
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text().strip():continue
        s=t.get_text().strip()
        panel=len(s)==1 and s.islower() and t.get_fontweight() in ('bold',700)
        t.set_fontsize(9 if panel else max(6.5,min(8,t.get_fontsize())))
        t.set_fontfamily('sans-serif')
    for legend in fig.legends+[a.get_legend() for a in fig.axes if a.get_legend() is not None]:
        for t in legend.get_texts():t.set_fontsize(7)

def export_figure(fig,stem,dpi=600):
    stem=Path(stem);stem.parent.mkdir(parents=True,exist_ok=True)
    finalize_style(fig);fig.canvas.draw();renderer=fig.canvas.get_renderer()
    clipped=[]
    for t in fig.findobj(mpl.text.Text):
        if t.get_visible() and t.get_text().strip():
            b=t.get_window_extent(renderer)
            if b.x0<-.7 or b.y0<-.7 or b.x1>fig.bbox.width+.7 or b.y1>fig.bbox.height+.7:clipped.append(t.get_text())
    assert not clipped,clipped
    outputs={}
    for suffix in ['pdf','png','svg']:
        p=stem.with_suffix('.'+suffix)
        fig.savefig(p,dpi=dpi,facecolor='white',transparent=False,bbox_inches=None)
        outputs[suffix]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    fonts=[float(t.get_fontsize()) for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
    return dict(outputs=outputs,width_mm=fig.get_figwidth()*25.4,height_mm=fig.get_figheight()*25.4,
        minimum_visible_font_pt=min(fonts) if fonts else None,rcparams={k:str(v) for k,v in RC.items()},
        semantic_colors=COL,shared_style_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        style_source_sha256=hashlib.sha256(STYLE_FILE.read_bytes()).hexdigest(),clipped_text=clipped,
        native_width_note='164mm matches embedded manuscript width; no tight-bbox dimension changes')
