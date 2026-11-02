"""Exact bending block for thickness-symmetric isotropic five-variable plates.

Not a new kinematics approximation: use only after full/block regression.
No residual stress, asymmetric face constants, or through-thickness asymmetry.
"""
import numpy as np
from msse_core import operators
from two_phase import reciprocal,disk
from surface import surface_strain

def assemble_bending(N,k,theory='TSDT',h=.1,indicator=None,materials=None,surface_constants=None):
    if materials is None:materials=[(10.,.3,2.),(1.,.3,1.)]
    ij,G=reciprocal(N);q=G+np.asarray(k);n=len(q)
    if indicator is None:indicator=lambda g:disk(g,np.sqrt(.3/np.pi))
    F=indicator(G[:,None,:]-G[None,:,:]);FB=np.eye(n)-F
    t=np.linspace(-.5,.5,4);V=t[:,None]**np.arange(4)
    sample=[[operators(theory,w,x*h,h) for w in q] for x in t];coeff=[]
    for j in range(3):
        data=np.array([[s[j][:,[2,3,4]] for s in row] for row in sample])
        coeff.append(np.linalg.solve(V,data.reshape(4,-1)).reshape(data.shape))
    H=np.array([[h/((i+j+1)*2**(i+j)) if (i+j)%2==0 else 0 for j in range(4)] for i in range(4)])
    L=np.linalg.cholesky(H);P,E,C=[np.einsum('rs,rnid->snid',L,x) for x in coeff]
    def gram(x,weight,mask):
        a=np.einsum('snid,i,smie->ndme',x.conj(),weight,x,optimize=True)
        return (a*mask[:,None,:,None]).reshape(3*n,3*n)
    Kcl=np.zeros((3*n,3*n),complex);M=Kcl.copy();phases=[]
    for (young,nu,rho),mask in zip(materials,[F,FB]):
        if young<=0 or rho<=0 or not -1<nu<.5:raise ValueError('Invalid bulk material')
        mu=young/(2*(1+nu));Q=young/(1-nu*nu);ks=5/6 if theory=='FSDT' else 1
        cp=np.diag([Q,Q,mu,ks*mu,ks*mu]);cp[0,1]=cp[1,0]=nu*Q
        ec=np.einsum('ij,snjd->snid',np.linalg.cholesky(cp).T,E)
        Kcl+=gram(ec,np.ones(5),mask);M+=gram(P,np.full(3,rho),mask)
        phases.append(gram(C,np.full(6,2*mu*h*h),mask))
    if surface_constants is None:surface_constants={p:(.0001,.0001) for p in ['A','B']}
    surface={}
    for side in [-1,1]:
        b=np.array([surface_strain(w,h,side,theory)[:,[2,3,4]] for w in q])
        for phase,mask in [('A',F),('B',FB)]:
            la,mu=surface_constants[phase]
            if not ((la==0 and mu==0) or (mu>0 and la+mu>0)):raise ValueError('Invalid surface material')
            cs=np.array([[la+2*mu,la,0],[la,la+2*mu,0],[0,0,mu]])
            mat=np.einsum('nid,ij,mje->ndme',b.conj(),cs,b,optimize=True)
            surface[f'Ks_{phase}_{side}']=(mat*mask[:,None,:,None]).reshape(3*n,3*n)
    S=np.zeros((n,3*n));S[np.arange(n),3*np.arange(n)]=1
    return dict(Kcl=Kcl,KA=phases[0],KB=phases[1],M=M,Ks_parts=surface,S=S,G=G,ij=ij,F=F)
