"""Classical-only exact symmetric bending block, without unused high-order arrays."""
import numpy as np
from msse_core import operators
from two_phase import reciprocal,disk

def assemble_classical(N,k,theory='TSDT',h=.1,materials=None):
    if materials is None:raise ValueError('Explicit materials required')
    ij,G=reciprocal(N);q=G+np.asarray(k);n=len(q);F=disk(G[:,None,:]-G[None,:,:],np.sqrt(.3/np.pi));FB=np.eye(n)-F
    t=np.linspace(-.5,.5,4);V=t[:,None]**np.arange(4);sample=[[operators(theory,w,x*h,h) for w in q] for x in t]
    coeff=[]
    for j in [0,1]:
        data=np.array([[s[j][:,[2,3,4]] for s in row] for row in sample]);coeff.append(np.linalg.solve(V,data.reshape(4,-1)).reshape(data.shape))
    H=np.array([[h/((i+j+1)*2**(i+j)) if (i+j)%2==0 else 0 for j in range(4)] for i in range(4)]);L=np.linalg.cholesky(H)
    P,E=[np.einsum('rs,rnid->snid',L,x) for x in coeff]
    def gram(x,weight,mask):
        a=np.einsum('snid,i,smie->ndme',x.conj(),weight,x,optimize=True)
        return (a*mask[:,None,:,None]).reshape(3*n,3*n)
    K=np.zeros((3*n,3*n),complex);M=K.copy()
    for (young,nu,rho),mask in zip(materials,[F,FB]):
        if young<=0 or rho<=0 or not -1<nu<.5:raise ValueError('Invalid material')
        mu=young/(2*(1+nu));Q=young/(1-nu*nu);ks=5/6 if theory=='FSDT' else 1
        cp=np.diag([Q,Q,mu,ks*mu,ks*mu]);cp[0,1]=cp[1,0]=nu*Q
        ec=np.einsum('ij,snjd->snid',np.linalg.cholesky(cp).T,E);K+=gram(ec,np.ones(5),mask);M+=gram(P,np.full(3,rho),mask)
    S=np.zeros((n,3*n));S[np.arange(n),3*np.arange(n)]=1
    return dict(Kcl=K,M=M,S=S,G=G,ij=ij)
