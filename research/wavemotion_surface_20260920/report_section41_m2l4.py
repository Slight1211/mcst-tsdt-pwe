"""Report only measured h=20 um PWE/3D spectra; no interpolation or invented bands."""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parents[1] / "output/pdf/modal_selectivity_assets"
PWE = ROOT / "runs/large_scale_3d_20260921T145045670173Z"
MESH = ROOT / "runs/section41_mesh_20260923T125509Z"
PATHS = ROOT / "runs/section41_fullpath_M2L4_20260924"
EXTRA = ROOT / "runs/section41_extra_M3L2_20260925"
STYLES = [
    ("FSDT", 0., "Classic-FSDT", "#0072BD", "--"),
    ("TSDT", 0., "Classic-TSDT", "#56B4E9", "-"),
    ("FSDT", 1., "MCST-FSDT", "#D55E00", "-."),
    ("TSDT", 1., "MCST-TSDT", "#7E2F8E", ":"),
]

def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

def pwe_spectrum(rows, theory, ell, method, point):
    hit = sorted((r for r in rows if int(r["N"]) == 21
                  and r["theory"] == theory and float(r["ell_um"]) == ell
                  and r["method"] == method and int(r["point"]) == point),
                 key=lambda r: int(r["band"]))
    assert len(hit) >= 6, (theory,ell,method,point,len(hit))
    frequencies = np.array([float(r["frequency_MHz"])*1000 for r in hit[:6]])
    assert np.all(np.isfinite(frequencies)) and np.all(frequencies >= 0)
    return frequencies

def bending(rows, point, zero=False):
    selected = sorted((r for r in rows if int(r["point_index"]) == point
                       and float(r["frequency_mhz"]) > 1e-5
                       and float(r["vertical_fraction"]) > .8),
                      key=lambda r: float(r["frequency_mhz"]))
    n = 5 if zero else 6
    assert len(selected) >= n, (point,len(selected))
    for r in selected[:n]:
        assert abs(float(r["imag_frequency_mhz"]))/float(r["frequency_mhz"]) < 1e-7
    f = np.array([float(r["frequency_mhz"])*1000 for r in selected[:n]])
    return np.r_[0.,f] if zero else f

def valid_path(theory, mesh, layers):
    folder = path_folder(theory, mesh, layers)
    try:
        record = json.loads((folder/"manifest.json").read_text())
        if record.get("status") != "PASS_EXPORT_AND_PATH_SPECTRUM":
            return False
        return (folder/"frequencies.csv").is_file()
    except (OSError, ValueError):
        return False

def path_spectrum(theory, mesh, layers):
    assert valid_path(theory,mesh,layers)
    rows = csv_rows(path_folder(theory,mesh,layers)/"frequencies.csv")
    result = np.array([bending(rows,p,zero=(p in (0,12))) for p in range(13)])
    assert result.shape == (13,6)
    return result

def path_folder(theory, mesh, layers):
    parent = EXTRA if (mesh, layers) == (3, 2) else PATHS
    return parent / f"{theory}_M{mesh}_L{layers}"

def write_rows(path, rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)

def main():
    assert json.loads((PWE/"config.json").read_text())["h_um"] == 20
    assert json.loads((MESH/"config.json").read_text())["h_um"] == 20
    ASSETS.mkdir(parents=True,exist_ok=True)
    PATHS.mkdir(parents=True,exist_ok=True)
    pwe_rows=csv_rows(PWE/"pwe.csv")
    comparison=[]
    controls={}
    for label,ell in (("classic",0.),("mcst",1.)):
        rows=csv_rows(MESH/f"{label}_M2_L4"/"frequencies.csv")
        for index,name in ((0,"X"),(1,"M")):
            controls[label,name]=bending(rows,index)
    for theory,ell,_,_,_ in STYLES:
        label="classic" if ell == 0 else "mcst"
        for index,name in ((4,"X"),(8,"M")):
            for method in ("direct","inverse"):
                pwe=pwe_spectrum(pwe_rows,theory,ell,method,index)
                fem=controls[label,name]
                for band,(fp,ff) in enumerate(zip(pwe,fem),1):
                    comparison.append(dict(theory=theory,ell_um=ell,method=method,
                        point=name,band=band,pwe_kHz=fp,fem_M2L4_kHz=ff,
                        error_percent=100*abs(fp/ff-1)))
    write_rows(PATHS/"comparison_M2L4.csv",comparison)
    table=[]
    for theory,ell,_,_,_ in STYLES:
        values=[]
        for point in ("X","M"):
            for method in ("direct","inverse"):
                values.append(max(r["error_percent"] for r in comparison
                    if r["theory"]==theory and r["ell_um"]==ell
                    and r["point"]==point and r["method"]==method))
        label="经典" if ell == 0 else "MCST"
        table.append(label+" & "+theory+" & "+" & ".join(f"{v:.3f}" for v in values)+r" \\")
    tex=[r"\begin{tabular}{llrrrr}\toprule",
         r"本构 & 板理论 & \multicolumn{2}{c}{$X$点最大频差/\%} & \multicolumn{2}{c}{$M$点最大频差/\%}\\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r" & & 直接法 & 逆法 & 直接法 & 逆法\\\midrule",
         *table,r"\bottomrule\end{tabular}"]
    (ASSETS/"section41_M2L4_method_error_table.tex").write_text(
        "\n".join(tex),encoding="utf-8")
    summary=dict(status="M2L4_XM_CONTROLS_VALIDATED",
                 max_direct_percent=max(r["error_percent"] for r in comparison if r["method"]=="direct"),
                 max_inverse_percent=max(r["error_percent"] for r in comparison if r["method"]=="inverse"),
                 full_path_cases=[])
    plt.rcParams.update({"font.family":"Times New Roman","font.size":9,
        "mathtext.fontset":"stix","axes.linewidth":.8,
        "xtick.direction":"in","ytick.direction":"in","pdf.fonttype":42})
    if valid_path("classic",2,4) and valid_path("mcst",2,4):
        refs={0.:path_spectrum("classic",2,4),1.:path_spectrum("mcst",2,4)}
        for label,ell in (("classic",0.),("mcst",1.)):
            for index,name in ((4,"X"),(8,"M")):
                delta=np.max(100*np.abs(refs[ell][index]/controls[label,name]-1))
                assert delta < .5,(label,name,delta)
        spectra={}
        for theory,ell,_,_,_ in STYLES:
            arr=np.array([pwe_spectrum(pwe_rows,theory,ell,"inverse",p)
                          for p in range(12)])
            spectra[theory,ell]=np.vstack((arr,arr[0]))
        all_spectra=list(spectra.values())+list(refs.values())
        lower=max(s[:,3].max() for s in all_spectra)
        upper=min(s[:,4].min() for s in all_spectra)
        path=np.loadtxt(PWE/"path.csv",delimiter=",",skiprows=1)
        path=np.vstack((path,path[0]))
        x=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]/np.pi
        fig,ax=plt.subplots(figsize=(7,4.2))
        fig.subplots_adjust(left=.105,right=.98,bottom=.13,top=.80)
        handles=[]
        for theory,ell,label,color,style in STYLES:
            for band in range(6):
                ax.plot(x,spectra[theory,ell][:,band],color=color,ls=style,lw=1.15)
            handles.append(Line2D([0],[0],color=color,ls=style,label=label))
        for ell,label,marker,color in ((0.,"3D classic M2L4","s",".30"),
                                       (1.,"3D MCST M2L4","o","black")):
            for band in range(6):
                ax.plot(x,refs[ell][:,band],ls="none",marker=marker,ms=3.6,
                        mfc="none",mec=color,mew=.7)
            handles.append(Line2D([0],[0],ls="none",marker=marker,ms=4,
                                  mfc="none",mec=color,label=label))
        gap=200*(upper-lower)/(upper+lower)
        if upper>lower:
            ax.axhspan(lower,upper,color="#B7CDDD",alpha=.35,lw=0,zorder=0)
        if upper > lower:
            for edge in (lower, upper):
                ax.axhline(edge, color="#3A6382", ls="--", lw=.65, alpha=.8)
            x_label = x[-1] * .53
            label_box = dict(facecolor="white", edgecolor="none",
                             alpha=.78, pad=.8)
            ax.text(x_label, (lower+upper)/2,
                    rf"$L_{{45}}$-$U_{{45}}$: {lower:.1f}-{upper:.1f} kHz"
                    + "\n" + rf"$J_{{45}}={gap:.2f}\%$",
                    ha="center", va="center", fontsize=8.5,
                    linespacing=1.15, bbox=label_box)
        for xx in x[[4,8]]: ax.axvline(xx,color=".82",lw=.6)
        ax.set(xlim=(x[0],x[-1]),ylim=(0,None),
               xticks=x[[0,4,8,12]],
               xticklabels=[r"$\Gamma$","X","M",r"$\Gamma$"],
               xlabel="Bloch wave vector",ylabel="Frequency (kHz)")
        ax.tick_params(top=True,right=True)
        fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.54,.99),
                   ncol=3,frameon=False,fontsize=8.5)
        fig.savefig(ASSETS/"section41_M2L4_bands.pdf")
        plt.close(fig)
        summary.update(status="M2L4_FULL_PATH_VALIDATED",
                       common_path_lower_kHz=float(lower),
                       common_path_upper_kHz=float(upper),
                       common_path_J45_percent=float(gap))
    meshes=[(3,2),(3,4),(2,2),(2,4),(2,6),(1,4)]
    summary["coarse_M3L2_to_M3L4"] = {}
    for theory in ("classic","mcst"):
        if all(valid_path(theory,m,l) for m,l in meshes):
            coarse=path_spectrum(theory,3,2)
            finer=path_spectrum(theory,3,4)
            control_delta=100*np.abs(coarse[[4,8]]/finer[[4,8]]-1)
            summary["coarse_M3L2_to_M3L4"][theory]=dict(
                max_XM_first6_percent=float(control_delta.max()),
                max_full_path_first6_percent=float(
                    (100*np.abs(coarse[1:12]/finer[1:12]-1)).max()))
            fig,axes=plt.subplots(2,3,figsize=(7.2,4.15),sharey=True)
            fig.subplots_adjust(left=.085,right=.99,bottom=.11,top=.92,
                                hspace=.42,wspace=.24)
            for ax,(m,l) in zip(axes.flat,meshes):
                arr=path_spectrum(theory,m,l)
                for band in range(6):
                    ax.plot(range(13),arr[:,band],color="#0072BD" if band<4
                            else "#D55E00",lw=.9)
                low=arr[:,3].max();high=arr[:,4].min()
                j=200*(high-low)/(high+low)
                if high>low:ax.axhspan(low,high,color="#B7CDDD",alpha=.35,lw=0)
                for value in (low,high):
                    ax.axhline(value,color="#30753B",ls="--",lw=.65,alpha=.8)
                if high>low:
                    ax.text(6,(low+high)/2,f"{low:.1f}-{high:.1f} kHz",
                            ha="center",va="center",color="#20592C",fontsize=8,
                            bbox=dict(facecolor="white",edgecolor="none",
                                      alpha=.82,pad=.6))
                ax.set_title(f"M{m}L{l}: J45={j:.2f}%",fontsize=9)
                ax.set(xlim=(0,12),xticks=(0,4,8,12),
                       xticklabels=(r"$\Gamma$","X","M",r"$\Gamma$"))
                ax.grid(axis="y",alpha=.18)
                ax.tick_params(labelsize=7)
            fig.supylabel("Frequency (kHz)",x=.005,fontsize=9)
            fig.savefig(ASSETS/f"section41_{theory}_mesh_bands.pdf")
            plt.close(fig)
            summary["full_path_cases"] += [f"{theory}_M{m}_L{l}" for m,l in meshes]
    (PATHS/"assessment.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
