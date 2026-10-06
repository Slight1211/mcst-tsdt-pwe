"""Independent through-thickness Hermite FE of a homogeneous 3D P-SV layer.

Four DOFs per node: U, U_z, W, W_z. Analytical polynomial element integrals.
MCST uses omega_y=(U_z-i*k*W)/2 and m=2*mu*ell^2*chi.
No plate displacement ansatz; no artificial rotations or penalty constraints.
"""
import numpy as np
from scipy.linalg import solve

def matrices(ne,k,h,E,nu,rho,ell=0.,surface=(0.,0.)):
    if ne<1 or h<=0 or E<=0 or rho<=0 or ell<0:raise ValueError('Invalid layer parameters')
    L=h/ne;nd=4*(ne+1);mu=E/(2*(1+nu));la=E*nu/((1+nu)*(1-2*nu))
    shapes=np.array([[1,0,-3,2],[0,L,-2*L,L],[0,0,3,-2],[0,0,-L,L]],float).T
    U=np.zeros((4,8),complex);W=U.copy();U[:,[0,1,4,5]]=shapes;W[:,[2,3,6,7]]=shapes
    def dz(poly):
        result=np.zeros_like(poly);result[:-1]=np.arange(1,4)[:,None]*poly[1:]/L;return result
    Uz=dz(U);Wz=dz(W);Uzz=dz(Uz)
    moments=L/(np.arange(4)[:,None]+np.arange(4)[None,:]+1)
    def gram(a,b):return np.einsum('ri,rs,sj->ij',a.conj(),moments,b)
    ex=1j*k*U;ez=Wz;gxz=Uz+1j*k*W
    kc=(la+2*mu)*(gram(ex,ex)+gram(ez,ez))+la*(gram(ex,ez)+gram(ez,ex))+mu*gram(gxz,gxz)
    rotation2=Uz-1j*k*W;rotation2z=Uzz-1j*k*Wz
    km=mu*ell*ell/4*(gram(rotation2z,rotation2z)+k*k*gram(rotation2,rotation2))
    me=rho*(gram(U,U)+gram(W,W))
    Kc=np.zeros((nd,nd),complex);Km=Kc.copy();M=Kc.copy();Ks=Kc.copy()
    for element in range(ne):
        inds=np.arange(4*element,4*element+8);ix=np.ix_(inds,inds)
        Kc[ix]+=kc;Km[ix]+=km;M[ix]+=me
    ls,ms=surface
    for node in [0,ne]:Ks[4*node,4*node]+=(ls+2*ms)*k*k
    Sb=np.zeros(nd);St=Sb.copy();Sb[2]=1;St[4*ne+2]=1
    return dict(Kcl=Kc,Kmc=Km,Ks=Ks,M=M,Sb=Sb,St=St)

def response(ne,k,h,E,nu,rho,omega,minus,plus,ell=0.,surface=(0.,0.)):
    s=matrices(ne,k,h,E,nu,rho,ell,surface);Sb=s['Sb'];St=s['St']
    qm=np.sqrt(complex((omega/minus['c'])**2-k*k));qp=np.sqrt(complex((omega/plus['c'])**2-k*k))
    if qm.real<=0 or qp.real<=0:raise ValueError('Only open incident/transmitted ports in this benchmark')
    Ym=qm/(minus['rho']*omega);Yp=qp/(plus['rho']*omega);Zm=1/Ym;Zp=1/Yp
    D=s['Kcl']+s['Kmc']+s['Ks']-omega**2*s['M']
    L=D-1j*omega*(Zm*np.outer(Sb,Sb)+Zp*np.outer(St,St));rhs=2*Sb
    # Congruent coordinate scaling by positive mass diagonal, no matrix alteration.
    scale=1/np.sqrt(np.diag(s['M']).real);A=scale[:,None]*L*scale[None,:]
    x=scale*solve(A,scale*rhs)
    r=1+1j*omega*Zm*(Sb@x);t=-1j*omega*Zp*(St@x)
    R=abs(r)**2;T=Yp.real/Ym.real*abs(t)**2
    return dict(r=r,t=t,R=float(R),T=float(T),u=x,
        power_error=float(R+T-1),residual=float(np.linalg.norm(L@x-rhs)/(np.linalg.norm(L)*np.linalg.norm(x)+np.linalg.norm(rhs))),
        hermitian_error=float(np.linalg.norm(D-D.conj().T)/np.linalg.norm(D)))
