"""Fixed-area N15 analytic SQP with durable, unfiltered iteration evidence.

This is a new a75/h6 run. It neither launches COMSOL nor reuses old optima.
Only sampled-band local optimization is asserted. No global/BZ/FEM claim.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "msse_20260919/python")]
from run_node_shape_pilot import Evaluator, ibz
from node_shape_bending import AREA, boundary, fourier, signed_area
from node_shape_sensitivity import sampled_gap, shape_point
from inverse_circle_bending import prepare

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)

class LoggedEvaluator(Evaluator):
    def evaluate(self, y):
        fresh = self.last_y is None or not np.array_equal(self.last_y, y)
        if fresh:
            print("EVALUATE", self.calls+1, "kpoints", len(self.k), flush=True)
        result = super().evaluate(y)
        if fresh:
            print("EVALUATED", self.calls, "J", sampled_gap(result[0])["J"], flush=True)
        return result

def spectrum(y, kpoints, out, stem, N=15):
    """Persist each completed point, never interpolate missing points."""
    F = fourier(y, N)
    cache = prepare(N, .08, theory="TSDT", indicator_matrix=F)
    values = []
    with (out / f"{stem}.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point", "kx_a", "ky_a"] + [f"Omega{i}" for i in range(1,7)])
        for i, k in enumerate(kpoints):
            f = shape_point(y, N, k, h_over_a=.08, ell_over_h=1/6,
                            gradient=False, geometry=(F,None), prepared=cache)
            values.append(f)
            writer.writerow([i,*k,*f]); handle.flush()
            print(stem, i+1, "/", len(kpoints), flush=True)
    return np.asarray(values)

def gradient_gate(y, name):
    """Directional central differences at N15, same full material model."""
    k = np.array([2.7,.63])
    direction = np.arange(1.,7.); direction /= np.linalg.norm(direction)
    f,g,_ = shape_point(y,15,k,h_over_a=.08,ell_over_h=1/6)
    records = []
    for step in (1e-4,5e-5):
        fp = shape_point(y+step*direction,15,k,h_over_a=.08,ell_over_h=1/6,gradient=False)
        fm = shape_point(y-step*direction,15,k,h_over_a=.08,ell_over_h=1/6,gradient=False)
        fd = (fp-fm)/(2*step)
        exact = (g@direction)[3:5]
        error = np.linalg.norm(exact-fd[3:5])/max(np.linalg.norm(fd[3:5]),1e-12)
        records.append(dict(step=step,analytic=exact.tolist(),central=fd[3:5].tolist(),relative_error=float(error)))
    passed = max(r["relative_error"] for r in records) < 1e-4
    print("GRADIENT_GATE", name, "PASS" if passed else "FAIL", records, flush=True)
    if not passed:
        raise ArithmeticError(f"N15 analytic gradient gate failed: {name}")
    return dict(name=name,k=k.tolist(),direction=direction.tolist(),records=records,status="PASS")

def optimize_round(ev, start, rid, out, history, clock, maxiter):
    freq,_ = ev.evaluate(start)
    edges = sampled_gap(freq)
    x0 = np.r_[start,edges["L"],edges["U"]]
    def obj(x):
        L,U=x[-2:]; return -200*(U-L)/(U+L)
    def jac(x):
        L,U=x[-2:]; return np.r_[np.zeros(6),400*U/(U+L)**2,-400*L/(U+L)**2]
    def con(x):
        f,_=ev.evaluate(x[:6])
        return np.r_[x[-2]-f[:,3],f[:,4]-x[-1]]
    def cjac(x):
        _,g=ev.evaluate(x[:6]); nk=len(ev.k)
        return np.vstack((np.column_stack((-g[:,3],np.ones(nk),np.zeros(nk))),
                          np.column_stack((g[:,4],np.zeros(nk),-np.ones(nk)))))
    local_it=0
    def record(x,initial=False):
        nonlocal local_it
        if not initial: local_it+=1
        f,g=ev.evaluate(x[:6]); gap=sampled_gap(f)
        violation=float(max(0.,-np.min(con(x))))
        p=boundary(x[:6])
        row=dict(event=len(history),round=rid,iteration=local_it,initial=initial,
                 J=gap["J"],L=gap["L"],U=gap["U"],epigraph_J=-obj(x),
                 epigraph_L=float(x[-2]),epigraph_U=float(x[-1]),
                 constraint_violation=violation,active_points=len(ev.k),
                 area=float(signed_area(p)),max_vertex=float(np.max(abs(p))),
                 elapsed_seconds=time.perf_counter()-clock,evaluations=ev.calls,
                 design=x[:6].tolist())
        history.append(row); save_json(out/"trajectory.json",history)
        save_json(out/"latest_iterate.json",row)
        print("ITER",rid,local_it,"J",row["J"],"violation",violation,flush=True)
    record(x0,initial=True)
    sol=minimize(obj,x0,jac=jac,method="SLSQP",
                 bounds=[(-.16,.16)]*6+[(.001,100.)]*2,
                 constraints=[dict(type="ineq",fun=con,jac=cjac)],
                 callback=record,options=dict(maxiter=maxiter,ftol=1e-8,disp=False))
    f,g=ev.evaluate(sol.x[:6]); gap=sampled_gap(f)
    multipliers=np.asarray(sol.multipliers)
    lag=jac(sol.x)-cjac(sol.x).T@multipliers
    bounds=np.asarray([[-.16,.16]]*6+[[.001,100.]]*2)
    projected=lag.copy()
    for j in range(8):
        if sol.x[j] <= bounds[j,0]+1e-7: projected[j]=min(lag[j],0.)
        elif sol.x[j] >= bounds[j,1]-1e-7: projected[j]=max(lag[j],0.)
    record_data=dict(round=rid,design=sol.x[:6].tolist(),x=sol.x.tolist(),
                     success=bool(sol.success),message=str(sol.message),iterations=int(sol.nit),
                     evaluations=ev.calls,objective=-float(sol.fun),sampled=gap,
                     constraint_violation=float(max(0.,-np.min(con(sol.x)))),
                     qp_multipliers=multipliers.tolist(),
                     projected_lagrangian_gradient_inf=float(np.max(abs(projected))),
                     complementarity_inf=float(np.max(abs(multipliers*con(sol.x)))))
    save_json(out/f"round{rid}_optimizer.json",record_data)
    return record_data

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--maxiter",type=int,default=100)
    parser.add_argument("--max-rounds",type=int,default=8)
    args=parser.parse_args()
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/"run_node_shape_pilot.py"]+list((ROOT/"src").glob("*.py"))
    sources += list((ROOT.parent/"msse_20260919/python").glob("*.py"))
    snapshot=out/"source_snapshot"; snapshot.mkdir()
    hashes={str(p):sha(p) for p in sources}
    for i,p in enumerate(sources): shutil.copy2(p,snapshot/f"{i:03d}_{p.name}")
    meta=dict(status="RUNNING",pid=os.getpid(),started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
              N=15,a_um=75.,h_um=6.,h_over_a=.08,ell_epoxy_um=1.,ell_steel_um=0.,
              target_steel_area=AREA,vertices=48,shape_variables=6,fixed_area=True,
              shape_bounds=[-.16,.16],allow_concave=True,threads=2,
              optimizer="SLSQP with analytic objective and eigenfrequency Jacobians",
              ftol=1e-8,feasibility_tolerance=1e-6,maxiter=args.maxiter,max_rounds=args.max_rounds,
              ibz_search_points=45,ibz_validation_points=153,
              analysis="single-start local sampled-band optimization",source_sha256=hashes,
              no_global_optimality_claim=True,no_continuous_BZ_claim=True,no_3d_FEM_claim=True,
              environment=dict(python=sys.version,platform=platform.platform()))
    save_json(out/"manifest.json",meta)
    clock=time.perf_counter(); history=[]; rounds=[]; start=np.zeros(6)
    k=ibz(8); active={0,36,44}
    try:
        with threadpool_limits(limits=2):
            meta["gradient_gates"]=[gradient_gate(start,"initial")]
            save_json(out/"manifest.json",meta)
            for rid in range(1,args.max_rounds+1):
                idx=sorted(active)
                ev=LoggedEvaluator(15,k[idx],1/6,h_over_a=.08)
                local=optimize_round(ev,start,rid,out,history,clock,args.maxiter)
                y=np.asarray(local["design"])
                values=spectrum(y,k,out,f"round{rid}_ibz45")
                full=sampled_gap(values)
                L,U=local["x"][-2:]
                vio=float(max(0.,full["L"]-L,U-full["U"]))
                new=set(map(int,np.where((values[:,3]>L+1e-6)|(values[:,4]<U-1e-6))[0]))-active
                rounds.append(dict(optimizer=local,active_indices=idx,new_indices=sorted(map(int,new)),
                                   full=full,full_constraint_violation=vio))
                meta.update(rounds=rounds,elapsed_seconds=time.perf_counter()-clock)
                save_json(out/"manifest.json",meta)
                if local["success"] and local["constraint_violation"]<1e-6 and vio<1e-6:
                    break
                if not new: raise ArithmeticError("Optimizer failed or active constraints are inconsistent")
                active.update(new); start=y
            else: raise ArithmeticError("Active-wavevector rounds did not converge")
            baseline=spectrum(np.zeros(6),k,out,"initial_ibz45")
            meta["initial_ibz45"]=sampled_gap(baseline); meta["final_ibz45"]=full
            # Independent denser wavevector sampling is validation, not a new search.
            dense=spectrum(y,ibz(16),out,"final_ibz153")
            meta["final_ibz153"]=sampled_gap(dense)
            meta["gradient_gates"].append(gradient_gate(y,"final"))
            np.savetxt(out/"final_vertices.csv",boundary(y),delimiter=",",header="x_over_a,y_over_a",comments="")
            np.savetxt(out/"initial_vertices.csv",boundary(np.zeros(6)),delimiter=",",header="x_over_a,y_over_a",comments="")
            meta.update(status="PASS_SAMPLED_N15_LOCAL_SQP",design=y.tolist(),
                        area_fraction=float(signed_area(boundary(y))),
                        dense_gap_drift_pp=meta["final_ibz153"]["J"]-full["J"],
                        accepted_iterations=sum(r["optimizer"]["iterations"] for r in rounds),
                        elapsed_seconds=time.perf_counter()-clock,N21_check="NOT_RUN",FEM_check="NOT_RUN")
            if any(sha(p)!=v for p,v in hashes.items()):
                raise ArithmeticError("A production source changed during the run")
            meta["source_hash_check"]="PASS"
            save_json(out/"manifest.json",meta)
            print("STATUS",meta["status"],"OUTPUT",out,flush=True)
    except BaseException as exc:
        meta.update(status="FAIL",error=repr(exc),traceback=traceback.format_exc(),
                    elapsed_seconds=time.perf_counter()-clock)
        save_json(out/"manifest.json",meta); raise

if __name__=="__main__": main()
