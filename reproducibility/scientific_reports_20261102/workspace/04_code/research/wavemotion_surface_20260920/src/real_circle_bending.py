"""Unitary real representation for centered circular, isotropic bending only."""
import numpy as np
from msse_core import operators
from two_phase import reciprocal,disk
def assemble_real(N,k,h=.1,radius=None,components=False,theory='TSDT',indicator_matrix=None,shape_dF=None,return_shape_operators=False):
    ij,G=reciprocal(N);q=G+np.asarray(k);n=len(q)
    if radius is None:radius=np.sqrt(.3/np.pi)
    F=(np.zeros((n,n),complex) if radius==0 else disk(G[:,None,:]-G[None,:,:],radius)) if indicator_matrix is None else np.asarray(indicator_matrix)
    if F.shape!=(n,n) or np.max(abs(F.imag))>1e-10 or np.linalg.norm(F-F.conj().T)>1e-10*max(1,np.linalg.norm(F)):
        raise ValueError('Expected a real Hermitian indicator matrix')
    F=F.real
    t=np.linspace(-.5,.5,4);V=t[:,None]**np.arange(4)
    sample=[[operators(theory,w,x*h,h) for w in q] for x in t]
    rowphases=[[1j,1j,1],[1,1,1,1j,1j],[1,1,1,1j,1j,1]];col=np.array([1,1j,1j]);coeff=[]
    for j in range(3):
        a=np.array([[s[j][:,[2,3,4]] for s in row] for row in sample])*col
        a=a*np.conj(np.array(rowphases[j]))[None,None,:,None]
        assert np.linalg.norm(a.imag)<=1e-13*max(1,np.linalg.norm(a.real))
        coeff.append(np.linalg.solve(V,a.real.reshape(4,-1)).reshape(a.shape))
    H=np.array([[h/((i+j+1)*2**(i+j)) if (i+j)%2==0 else 0 for j in range(4)] for i in range(4)])
    L=np.linalg.cholesky(H);P,E,C=[np.einsum('rs,rnid->snid',L,x) for x in coeff]
    def gram(x,weights,mask):
        a=np.einsum('snid,i,smie->ndme',x,weights,x,optimize=True)
        a*=mask[:,None,:,None];return a.reshape(3*n,3*n)
    K=np.zeros((3*n,3*n));M=K.copy()
    if components: Ksh=K.copy()
    if shape_dF is not None:
        shape_dF=np.asarray(shape_dF,float)
        if shape_dF.ndim!=3 or shape_dF.shape[1:]!=(n,n):
            raise ValueError('shape_dF must have dimensions (parameters, waves, waves)')
        dM=np.zeros((len(shape_dF),3*n,3*n));dKB=dM.copy()
    for (young,nu,rho),mask in zip([(210.6/4.35,.3,7780/1180),(1.,4.35/(2*1.59)-1,1.)],[F,np.eye(n)-F]):
        mu=young/(2*(1+nu));Q=young/(1-nu*nu);ks=5/6 if theory=='FSDT' else 1
        cp=np.diag([Q,Q,mu,ks*mu,ks*mu]);cp[0,1]=cp[1,0]=nu*Q
        ec=np.einsum('ij,snjd->snid',np.linalg.cholesky(cp).T,E)
        K+=gram(ec,np.ones(5),mask);M+=gram(P,np.full(3,rho),mask)
        if components: Ksh+=gram(E,np.array([0.,0.,0.,ks*mu,ks*mu]),mask)
    nu=4.35/(2*1.59)-1;KB=gram(C,np.full(6,h*h/(1+nu)),np.eye(n)-F)
    result=dict(Kcl=K,M=M,KB=KB,ij=ij,G=G)
    if return_shape_operators:
        result['P_operator']=P
        result['C_operator']=C
    if shape_dF is not None:
        for j,dF in enumerate(shape_dF):
            dM[j]=(7780/1180-1)*gram(P,np.ones(3),dF)
            dKB[j]=-gram(C,np.full(6,h*h/(1+nu)),dF)
        result['dM']=dM;result['dKB']=dKB
    if components: result['Ksh']=Ksh
    return result
