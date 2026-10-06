"""Independent isotropic 3D P-SV layer at real frequency, exp(-i omega t).

Only classical bulk elasticity. Fluid tangential traction zero; no plate theory.
Potential convention u_x=phi_x-psi_z, u_z=phi_z+psi_x.
"""
import numpy as np
from scipy.linalg import solve

def layer_scattering(omega,k,h,E,nu,rho,minus,plus,a=1.,b=0.):
    mu=E/(2*(1+nu));la=E*nu/((1+nu)*(1-2*nu))
    p=np.sqrt(complex(rho*omega**2/(la+2*mu)-k*k))
    s=np.sqrt(complex(rho*omega**2/mu-k*k))
    def fields(z):
        uz=[];xz=[];zz=[]
        for sign in [1,-1]:
            phase=np.exp(1j*sign*p*z)
            uz.append(1j*sign*p*phase)
            xz.append(-2*mu*sign*k*p*phase)
            zz.append((-la*k*k-(la+2*mu)*p*p)*phase)
        for sign in [1,-1]:
            phase=np.exp(1j*sign*s*z)
            uz.append(1j*k*phase)
            xz.append(mu*(s*s-k*k)*phase)
            zz.append(-2*mu*sign*k*s*phase)
        return np.array(uz),np.array(xz),np.array(zz)
    ub,sxb,szb=fields(-h/2);ut,sxt,szt=fields(h/2)
    qf=[np.sqrt(complex((omega/f['c'])**2-k*k)) for f in [minus,plus]]
    if any(q.real<=0 or q.imag!=0 for q in qf):raise ValueError('Benchmark requires open fluid ports')
    Ym,Yp=[q/(f['rho']*omega) for q,f in zip(qf,[minus,plus])];Zm,Zp=1/Ym,1/Yp
    L=np.array([sxb,sxt,szb+1j*omega*Zm*ub,szt-1j*omega*Zp*ut]);rhs=np.array([0,0,-2*a,-2*b],complex)
    scale=np.linalg.norm(L,axis=1);x=solve(L/scale[:,None],rhs/scale)
    bottom=ub@x;top=ut@x;r=a+1j*omega*Zm*bottom;t=b-1j*omega*Zp*top
    Pin=.5*(Ym.real*abs(a)**2+Yp.real*abs(b)**2)
    R=.5*Ym.real*abs(r)**2/Pin;T=.5*Yp.real*abs(t)**2/Pin
    return dict(r=r,t=t,R=float(R),T=float(T),u_bottom=bottom,u_top=top,
        power_error=float(R+T-1),residual=float(np.linalg.norm(L@x-rhs)/(np.linalg.norm(L)*np.linalg.norm(x)+np.linalg.norm(rhs))),
        scaled_condition=float(np.linalg.cond(L/scale[:,None])))
