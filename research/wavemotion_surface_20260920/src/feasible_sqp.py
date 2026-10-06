"""Feasible SQP with direction tilting, curved line search and damped BFGS.

Independent Python implementation of the Panier--Tits FSQP construction,
not the proprietary CFSQP package. Reference: Mathematical Programming 59
(1993), 261--276, doi:10.1007/BF01581247. The three subproblems are convex
QPs with LINEAR constraints; SciPy SLSQP is used only to solve those small
algebraic QPs, never to run the outer nonlinear/physical optimization.

Convention: c(x)>=0. Every accepted point passes the actual constraints and
an optional expensive full-constraint guard. No constraint-penalty tradeoff.
"""
import numpy as np
from scipy.optimize import minimize, nnls


def qp(H, g, A, b, start=None):
    """min .5*d'Hd+g'd subject to b+A*d>=0."""
    n=len(g)
    sol=minimize(lambda d:.5*d@H@d+g@d,
                 np.zeros(n) if start is None else start,
                 jac=lambda d:H@d+g,method='SLSQP',
                 constraints=[dict(type='ineq',fun=lambda d:b+A@d,jac=lambda d:A)],
                 options=dict(ftol=1e-14,maxiter=500))
    slack=b+A@sol.x
    if not sol.success or slack.min() < -1e-9:
        raise ArithmeticError(f'QP failed: {sol.message}; slack={slack.min()}')
    lam=np.asarray(sol.multipliers)
    stat=np.linalg.norm(H@sol.x+g-A.T@lam,np.inf)
    if stat>1e-4:raise ArithmeticError(f'QP stationarity residual {stat}')
    return sol.x,lam,dict(iterations=int(sol.nit),stationarity=float(stat),min_slack=float(slack.min()))


def kkt(g,c,J,active_tol=1e-6):
    active=np.flatnonzero(c<=active_tol);lam=np.zeros(len(c))
    if len(active):lam[active]=nnls(J[active].T,g,maxiter=5000)[0]
    return dict(stationarity=float(np.linalg.norm(g-J.T@lam,np.inf)),
                complementarity=float(np.max(abs(lam*c))),
                min_constraint=float(c.min()),multipliers=lam.tolist())


def solve(fun,oracle,x0,guard=None,callback=None,trial_callback=None,
          maxiter=100,kkt_tol=2e-5,comp_tol=1e-7):
    """oracle(x,gradient)->(c,J); guard verifies omitted constraints.

    All acceptances use c>=0 (no negative feasibility allowance). Trial and
    auxiliary points may be infeasible and are separately recorded. Geometry
    domain failures must be raised as ValueError, not hidden as fake spectra.
    """
    x=np.array(x0,float);n=len(x);H=np.eye(n)
    f,g=fun(x);c,J=oracle(x,True)
    if c.min()<0:raise ValueError('A feasible starting point is required')
    if guard is not None and not guard(x):raise ValueError('Initial full guard failed')
    if callback:callback(x,dict(iteration=0,objective=float(f),**kkt(g,c,J)))
    for iteration in range(1,maxiter+1):
        residual=kkt(g,c,J)
        if residual['stationarity']<=kkt_tol and residual['complementarity']<=comp_tol:
            return dict(x=x,success=True,message='KKT tolerances satisfied',nit=iteration-1,kkt=residual)
        d0,lam,q0=qp(H,g,J,c)
        # QP for feasible descent: min .5||v||^2+z,
        # g'v<=z and -c-Jv<=z.
        H1=np.zeros((n+1,n+1));H1[:n,:n]=np.eye(n)
        g1=np.r_[np.zeros(n),1.]
        A1=np.vstack((np.r_[-g,1.],np.column_stack((J,np.ones(len(c))))))
        v,_,q1=qp(H1,g1,A1,np.r_[0.,c]);d1=v[:n]
        tilt=float((d0@d0)/(1+d0@d0));d=(1-tilt)*d0+tilt*d1
        slope=float(g@d)
        if slope>=0:
            return dict(x=x,success=False,message='No descent direction',nit=iteration-1,kkt=residual)
        # Nonlinear curvature correction; vanishing O(||d||^2.5) inward
        # shift is an algorithmic device, not a changed physical target.
        corr=np.zeros(n);soc_info=dict(used=False)
        try:
            caux,_=oracle(x+d,False)
            cc,_,qa=qp(H,g+H@d,J,caux-np.linalg.norm(d)**2.5)
            if np.linalg.norm(cc)<=min(np.linalg.norm(d),1.):
                corr=cc;soc_info=dict(used=True,norm=float(np.linalg.norm(cc)),qp=qa)
        except (ValueError,ArithmeticError) as exc:
            soc_info=dict(used=False,reason=str(exc))
        accepted=False
        for backtrack in range(35):
            alpha=.5**backtrack;xt=x+alpha*d+alpha**2*corr;ft,_=fun(xt)
            row=dict(iteration=iteration,backtrack=backtrack,alpha=alpha,
                     x=xt.tolist(),objective=float(ft),soc=soc_info)
            if ft>f+1e-4*alpha*slope:
                row['reason']='insufficient objective decrease'
            else:
                try:
                    ct,_=oracle(xt,False);row['min_constraint']=float(ct.min())
                    if ct.min()<0:row['reason']='nonlinear infeasibility'
                    elif guard is not None and not guard(xt):row['reason']='full sampling guard rejected'
                    else:row['reason']='accepted';accepted=True
                except ValueError as exc:row['reason']='outside geometry domain: '+str(exc)
            if trial_callback:trial_callback(row)
            if accepted:break
        if not accepted:
            return dict(x=x,success=False,message='Line search stalled',nit=iteration-1,kkt=residual)
        fn,gn=fun(xt);cn,Jn=oracle(xt,True)
        if cn.min()<0:raise ArithmeticError('Gradient re-evaluation lost feasibility')
        step=xt-x;z=gn-g-(Jn-J).T@lam;Hs=H@step;sHs=float(step@Hs);sz=float(step@z)
        if sHs>1e-20:
            theta=1. if sz>=.2*sHs else .8*sHs/(sHs-sz)
            r=theta*z+(1-theta)*Hs
            H=H-np.outer(Hs,Hs)/sHs+np.outer(r,r)/(step@r)
        if np.linalg.eigvalsh(H).min()<=0:raise ArithmeticError('BFGS lost positive definiteness')
        x,f,g,c,J=xt,fn,gn,cn,Jn
        info=dict(iteration=iteration,objective=float(f),step_norm=float(np.linalg.norm(step)),
                  alpha=alpha,backtracks=backtrack,tilt=tilt,soc=soc_info,qp=q0,
                  **kkt(g,c,J))
        if callback:callback(x,info)
    return dict(x=x,success=False,message='Iteration limit reached',nit=maxiter,kkt=kkt(g,c,J))


def self_test():
    results=[]
    # Smooth nonlinear disk: known optimum (1,0), including boundary start.
    for start in ([0.,.5],[0.,1.]):
        def f(x):return float(np.sum((x-[2.,0.])**2)),2*(x-[2.,0.])
        def c(x,gradient):return np.array([1-x@x]),np.array([-2*x]) if gradient else None
        hist=[]
        r=solve(f,c,start,callback=lambda x,i:hist.append((x.copy(),i)),kkt_tol=2e-6)
        feasible=all(1-x@x>=0 for x,_ in hist)
        monotone=all(hist[i][1]['objective']<=hist[i-1][1]['objective'] for i in range(1,len(hist)))
        assert r['success'] and np.linalg.norm(r['x']-[1.,0.])<1e-4 and feasible and monotone,r
        results.append(dict(start=start,nit=r['nit'],x=r['x'].tolist(),kkt=r['kkt'],feasible=feasible,monotone=monotone))
    # Intersection of curved inequalities, independent SciPy solution check.
    def f(x):return float((x[0]-2)**2+(x[1]-1)**2),2*(x-[2.,1.])
    def c(x,gradient):
        return np.array([1-x@x,x[1]-.2*x[0]**2]),np.array([-2*x,[-.4*x[0],1.]]) if gradient else None
    hist=[];r=solve(f,c,[.1,.3],callback=lambda x,i:hist.append((x.copy(),i)),kkt_tol=2e-6)
    ref=minimize(lambda x:f(x)[0],[.1,.3],jac=lambda x:f(x)[1],method='SLSQP',
                 constraints=[dict(type='ineq',fun=lambda x:c(x,False)[0],jac=lambda x:c(x,True)[1])],options=dict(ftol=1e-12))
    assert r['success'] and ref.success and np.linalg.norm(r['x']-ref.x)<1e-4
    assert all(c(x,False)[0].min()>=0 for x,_ in hist)
    results.append(dict(name='two nonlinear constraints',nit=r['nit'],x=r['x'].tolist(),reference_x=ref.x.tolist(),kkt=r['kkt']))
    return dict(status='PASS',tests=results)
