"""Four-panel vector figure from real, fully completed SQP evidence."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    run=args.run; m=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
    if m["status"]!="PASS_SAMPLED_N15_LOCAL_SQP": raise ValueError("Incomplete optimization")
    rows=json.loads((run/"trajectory.json").read_text(encoding="utf-8"))
    last=m["rounds"][-1]["optimizer"]
    # The terminal OptimizeResult may differ slightly from the last callback.
    # Add it as a separately labelled stored state; never invent a callback.
    if not np.array_equal(rows[-1]["design"],last["design"]):
        finalrow=dict(rows[-1]); gap=last["sampled"]
        finalrow.update(event=len(rows),iteration=last["iterations"],initial=False,
                        state_kind="terminal_result",J=gap["J"],L=gap["L"],U=gap["U"],
                        epigraph_J=last["objective"],constraint_violation=last["constraint_violation"],
                        epigraph_L=last["x"][-2],epigraph_U=last["x"][-1],
                        elapsed_seconds=None,design=last["design"],evaluations=last["evaluations"])
        rows.append(finalrow)
    for r in rows: r.setdefault("state_kind","initial" if r["initial"] else "callback")
    # Keep all actual callbacks; no smoothing and no discarded infeasible steps.
    x=np.arange(len(rows)); actual=np.array([r["J"] for r in rows])
    auxiliary=np.array([r["epigraph_J"] for r in rows]); vio=np.array([r["constraint_violation"] for r in rows])
    with (run/"iteration_table.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.writer(h); keys=["event","round","iteration","initial","state_kind","J","epigraph_J","constraint_violation","L","U","area","elapsed_seconds"]
        w.writerow(keys); w.writerows([[r[k] for k in keys] for r in rows])
    initial=np.loadtxt(run/"initial_vertices.csv",delimiter=",",skiprows=1)
    final=np.loadtxt(run/"final_vertices.csv",delimiter=",",skiprows=1)
    f0=np.loadtxt(run/"initial_ibz45.csv",delimiter=",",skiprows=1)
    f1=np.loadtxt(run/f"round{len(m['rounds'])}_ibz45.csv",delimiter=",",skiprows=1)
    idx=lambda i,j:i*(i+1)//2+j
    path=[idx(i,0) for i in range(9)]+[idx(8,j) for j in range(1,9)]+[idx(i,i) for i in range(7,-1,-1)]
    length=np.r_[np.arange(9)/8,1+np.arange(1,9)/8,2+np.sqrt(2)*np.arange(1,9)/8]
    c=np.sqrt(4.35e9/1180)/(75e-6*2*np.pi*1e6)
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
                         "font.size":9,"axes.labelsize":9,"legend.fontsize":8,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"savefig.dpi":300})
    blue="#0072B2"; orange="#D55E00"; gray="#888888"
    fig,axs=plt.subplots(2,2,figsize=(6.7,4.7),layout="constrained")
    a,b,d,e=axs.flat
    for polygon,label,color,style in [(final,"Optimized",orange,"-"),(initial,"Initial",blue,"--")]:
        closed=np.vstack([polygon,polygon[0]])
        a.plot(*closed.T,color=color,ls=style,lw=1.4,label=label)
    a.set_aspect("equal"); a.set(xlim=(-.5,.5),ylim=(-.5,.5),xlabel=r"$x/a$",ylabel=r"$y/a$")
    a.set_xticks([-.5,0,.5]); a.set_yticks([-.5,0,.5]); a.legend(loc="upper right",frameon=False)
    a.text(.03,.03,r"$\phi=0.09\pi$ (fixed)",transform=a.transAxes,fontsize=8)
    b.plot(x,auxiliary,color=gray,ls="--",lw=1.1,label="Auxiliary objective")
    b.plot(x,actual,color=blue,lw=1.4,marker="o",ms=3,label="Active-set solved gap")
    axislabel="SLSQP iteration" if len(m["rounds"])==1 else "Stored SQP states"
    b.set(xlabel=axislabel,ylabel=r"$J_{45}^{\mathcal{A}}$ (%)"); b.legend(frameon=False,loc="lower right")
    b.set_xticks([0,5,10,15,len(rows)-1])
    d.semilogy(x,np.maximum(vio,1e-12),color=orange,lw=1.4,marker="s",ms=3)
    d.axhline(1e-6,color=gray,ls="--",lw=1)
    d.text(.97,.86,r"Acceptance: $10^{-6}$",ha="right",transform=d.transAxes,fontsize=8)
    d.set(xlabel=axislabel,ylabel="Max. constraint violation")
    d.set_xticks([0,5,10,15,len(rows)-1])
    for data,color,ls,label in [(f0,blue,"--","Initial"),(f1,orange,"-","Optimized")]:
        for band in (3,4):
            e.plot(length,data[path,3+band]*c,color=color,ls=ls,
                   alpha=1 if band in (3,4) else .3,lw=1.3 if band in (3,4) else .7,
                   label=label if band==3 else None)
    e.set(xlim=(0,2+np.sqrt(2)),ylim=(3.5,10.8),xlabel="Bloch path",ylabel="Frequency (MHz)")
    e.set_xticks([0,1,2,2+np.sqrt(2)],[r"$\Gamma$","X","M",r"$\Gamma$"])
    e.axhspan(m["final_ibz45"]["L"]*c,m["final_ibz45"]["U"]*c,color=orange,alpha=.09)
    e.legend(frameon=False,loc="lower right")
    for i,ax in enumerate(axs.flat):
        ax.set_title(f"({chr(97+i)})",loc="left",fontsize=10,fontweight="bold")
        if ax is not a: ax.grid(alpha=.15,lw=.5)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"),bbox_inches="tight",dpi=300)
    print("FIGURE",args.output,flush=True)

if __name__=="__main__":main()
