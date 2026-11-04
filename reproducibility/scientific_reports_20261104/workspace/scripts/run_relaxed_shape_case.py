"""Local N15 fixed-area shape search with a configurable box and geometry guards.

Reuse the archived eigenproblem, derivatives and 153-point constraint-exchange
protocol. Original drivers and numerical evidence are never modified.
"""
from pathlib import Path
import argparse
import inspect
import json
import sys
import numpy as np
from scipy.optimize import minimize
import run_lengthscale_optimization as base
import run_ell0_multistart_case as legacy
from feasible_sqp import solve as fsqp_solve

ROOT = Path(__file__).resolve().parents[1]
BOUND = None
SOLVER = None


def geometry_constraints(y):
    p, dp = base.boundary(y, derivative=True)
    # C4v makes these seven x coordinates and six sector edges sufficient.
    edges = p[1:7]-p[:6]
    de = dp[1:7]-dp[:6]
    lengths = np.linalg.norm(edges, axis=1)
    c = np.r_[.49-p[:7,0], lengths-1e-5]
    jac = np.vstack((-dp[:7,0,:],np.einsum('ij,ijk->ik',edges,de)/lengths[:,None]))
    return c, jac


def optimize_round(ev, start, rid, out, history, clock, maxiter):
    import time
    f,_=ev.evaluate(start)
    gap=base.sampled_gap(f)
    # FSQP requires strict feasibility at entry; SLSQP uses the original edges.
    margin=1e-7 if SOLVER=='fsqp' else 0.
    x0=np.r_[start,gap['L']+margin,gap['U']-margin]
    def fun(x):
        L,U=x[-2:]
        return -200*(U-L)/(U+L),np.r_[np.zeros(6),400*U/(U+L)**2,-400*L/(U+L)**2]
    def oracle(x, gradient=True):
        # Domain failures are rejected by feasible line search; no fake spectra.
        if np.max(np.abs(x[:6]))>BOUND+1e-9:
            raise ValueError('Outside the configured shape-coordinate box')
        geom,gj=geometry_constraints(x[:6])
        freq,grad=ev.evaluate(x[:6]);nk=len(freq)
        c=np.r_[x[-2]-freq[:,3],freq[:,4]-x[-1],BOUND-x[:6],BOUND+x[:6],geom,x[-2:]-.001,100.-x[-2:]]
        if not gradient:return c,None
        jac=np.vstack((np.column_stack((-grad[:,3],np.ones(nk),np.zeros(nk))),
            np.column_stack((grad[:,4],np.zeros(nk),-np.ones(nk))),
            np.column_stack((-np.eye(6),np.zeros((6,2)))),
            np.column_stack((np.eye(6),np.zeros((6,2)))),
            np.column_stack((gj,np.zeros((len(geom),2)))),
            np.column_stack((np.zeros((2,6)),np.eye(2))),
            np.column_stack((np.zeros((2,6)),-np.eye(2)))))
        return c,jac
    iteration=0
    def record(x,info=None):
        nonlocal iteration
        if info is None:iteration+=1
        else:iteration=int(info.get('iteration',iteration))
        f,_=ev.evaluate(x[:6]);g=base.sampled_gap(f);c,_=oracle(x,False)
        row=dict(event=len(history),round=rid,iteration=iteration,design=x[:6].tolist(),
            J=g['J'],L=g['L'],U=g['U'],epigraph_J=-fun(x)[0],
            constraint_violation=float(max(0.,-c.min())),active_points=len(ev.k),
            area=float(base.signed_area(base.boundary(x[:6]))),max_vertex=float(np.max(abs(base.boundary(x[:6])))),
            elapsed_seconds=time.perf_counter()-clock,evaluations=ev.calls)
        history.append(row);base.save_json(out/'trajectory.json',history)
        base.save_json(out/'latest_iterate.json',row)
        print('ITER',rid,iteration,'J',g['J'],'violation',row['constraint_violation'],flush=True)
    if SOLVER=='slsqp':
        record(x0,{'iteration':0})
        sol=minimize(lambda x:fun(x)[0],x0,jac=lambda x:fun(x)[1],method='SLSQP',
            bounds=[(-BOUND,BOUND)]*6+[(.001,100.)]*2,
            constraints=[dict(type='ineq',fun=lambda x:oracle(x,False)[0],jac=lambda x:oracle(x,True)[1])],
            callback=record,options=dict(maxiter=maxiter,ftol=1e-7,disp=False))
        x=sol.x;success=bool(sol.success);message=str(sol.message);nit=int(sol.nit)
        c,jac=oracle(x,True);lam=np.asarray(sol.multipliers)
        lag=fun(x)[1]-jac.T@lam
        bounds=np.asarray([[-BOUND,BOUND]]*6+[[.001,100.]]*2)
        for j in range(8):
            if x[j]<=bounds[j,0]+1e-7:lag[j]=min(lag[j],0.)
            elif x[j]>=bounds[j,1]-1e-7:lag[j]=max(lag[j],0.)
        stationarity=float(np.max(abs(lag)));comp=float(np.max(abs(lam*c)))
    else:
        sol=fsqp_solve(fun,oracle,x0,callback=record,maxiter=maxiter,kkt_tol=1e-3,comp_tol=1e-4)
        x=sol['x'];success=bool(sol['success']);message=sol['message'];nit=int(sol['nit'])
        c,jac=oracle(x,True);lam=np.asarray(sol['kkt']['multipliers'])
        stationarity=float(sol['kkt']['stationarity']);comp=float(sol['kkt']['complementarity'])
    spectra,_=ev.evaluate(x[:6]);gap=base.sampled_gap(spectra)
    result=dict(round=rid,design=x[:6].tolist(),x=x.tolist(),success=success,message=message,
        iterations=nit,evaluations=ev.calls,objective=-fun(x)[0],sampled=gap,
        constraint_violation=float(max(0.,-c.min())),qp_multipliers=lam.tolist(),
        projected_lagrangian_gradient_inf=stationarity,complementarity_inf=comp,
        shape_bound=BOUND,solver=SOLVER,explicit_geometry_constraints=True)
    base.save_json(out/f'round{rid}_optimizer.json',result)
    return result


def main():
    global BOUND,SOLVER
    p=argparse.ArgumentParser();p.add_argument('--bound',type=float,required=True)
    p.add_argument('--solver',choices=['slsqp','fsqp'],default='slsqp')
    args,rest=p.parse_known_args()
    BOUND=float(args.bound);SOLVER=args.solver
    assert .16<BOUND<=.32
    # The inherited protocol checks bounds in three places. Bind its numerical
    # constant to this case, retaining the remaining control/acceptance logic.
    source=inspect.getsource(legacy.main)
    source=source.replace('def main():','def inherited_main():',1)
    source=source.replace('<=.16+','<=SHAPE_BOUND+').replace('shape_bounds=[-.16,.16]','shape_bounds=[-SHAPE_BOUND,SHAPE_BOUND]')
    source=source.replace('sources=[Path(__file__),',
        "sources=[Path(__file__),ROOT/'scripts/run_relaxed_shape_case.py',ROOT/'04_code/research/wavemotion_surface_20260920/src/feasible_sqp.py',")
    source=source.replace("vertices=48,shape_variables=6,", "solver=RELAXED_SOLVER,explicit_geometry_constraints=True,vertices=48,shape_variables=6,")
    scope=dict(legacy.__dict__,SHAPE_BOUND=BOUND,RELAXED_SOLVER=SOLVER)
    # base is shared by the inherited evaluator/gradient/spectrum functions.
    base.optimize_round=optimize_round
    exec(compile(source,str(Path(__file__).resolve())+':inherited_protocol','exec'),scope)
    sys.argv=[sys.argv[0]]+rest
    scope['inherited_main']()


if __name__=='__main__':main()
