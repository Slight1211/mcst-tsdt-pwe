"""Generalized classical compliance inverse; MCST remains direct (zero-l steel)."""
import numpy as np
from scipy.linalg import eigh,cho_factor,cho_solve
from two_phase import reciprocal,disk
from real_circle_bending import assemble_real
def generalized_D(E,nu,h=.1,theory='TSDT'):
    mu=E/(2*(1+nu));Q=E/(1-nu*nu);C=np.diag([Q,Q,mu,mu,mu]);C[0,1]=C[1,0]=nu*Q
    if theory=='FSDT':
        D=np.zeros((5,5));D[:3,:3]=h**3/12*C[:3,:3];D[3:,3:]=5/6*mu*h*np.eye(2);return D
    if theory!='TSDT':raise ValueError(theory)
    T=np.zeros((4,5,8));T[1,:3,:3]=h*np.eye(3);T[3,:3,:3]=-4*h/3*np.eye(3);T[3,:3,3:6]=-4*h/3*np.eye(3);T[0,3:,6:]=np.eye(2);T[2,3:,6:]=-4*np.eye(2)
    H=np.array([[h/((i+j+1)*2**(i+j)) if (i+j)%2==0 else 0 for j in range(4)] for i in range(4)])
    return np.einsum('rs,ria,ij,sjb->ab',H,T,C,T,optimize=True)
def strain(q,theory='TSDT'):
    x,y=q.T;n=len(q);B=np.zeros((n,8,3))
    B[:,0,1]=-x;B[:,1,2]=-y;B[:,2,1]=-y;B[:,2,2]=-x
    B[:,3,0]=-x*x;B[:,4,0]=-y*y;B[:,5,0]=-2*x*y
    B[:,6,0]=x;B[:,6,1]=1;B[:,7,0]=y;B[:,7,2]=1
    if theory=='FSDT':return B[:,[0,1,2,6,7],:]
    if theory!='TSDT':raise ValueError(theory)
    return B
def prepare(N,h=.1,radius=None,theory='TSDT',indicator_matrix=None):
    if radius is None:radius=np.sqrt(.3/np.pi)
    ij,G=reciprocal(N);n=len(G)
    F=(disk(G[:,None,:]-G[None,:,:],radius).real
       if indicator_matrix is None else np.asarray(indicator_matrix,float))
    if F.shape!=(n,n) or np.linalg.norm(F-F.T)>1e-10*max(1,np.linalg.norm(F)):
        raise ValueError('Indicator matrix must be real symmetric on the same Fourier grid')
    DA=generalized_D(210.6/4.35,.3,h,theory);DB=generalized_D(1.,4.35/(2*1.59)-1,h,theory);dim=len(DA)
    # eigh returns V^T DB V=I, V^T DA V=diag(d), W=V^-T.
    d,V=eigh(DA,DB);W=np.linalg.solve(V.T,np.eye(dim))
    assert np.linalg.norm(W@W.T-DB)/np.linalg.norm(DB)<1e-11
    assert np.linalg.norm((W*d)@W.T-DA)/np.linalg.norm(DA)<1e-11
    groups=[];alphas=[];remaining=list(range(dim))
    while remaining:
        j=remaining[0];ids=[i for i in remaining if abs(d[i]/d[j]-1)<1e-10];remaining=[i for i in remaining if i not in ids]
        S=np.eye(n)+(1/d[j]-1)*F
        C=cho_solve(cho_factor(S,lower=True),np.eye(n))
        assert np.linalg.norm(C-C.T)/np.linalg.norm(C)<1e-11
        groups.append((ids,C));alphas.append(1/d[j]-1)
    return dict(N=N,h=h,radius=radius,theory=theory,ij=ij,G=G,F=F,DA=DA,DB=DB,W=W,groups=groups,alphas=alphas)
def classical_inverse(cache,k):
    B=strain(cache['G']+np.asarray(k),cache.get('theory','TSDT'));Z=np.einsum('ab,nbd->nad',cache['W'].T,B);n=len(B);K=np.zeros((3*n,3*n))
    for ids,C in cache['groups']:
        block=np.einsum('nid,mie->ndme',Z[:,ids,:],Z[:,ids,:],optimize=True)
        block*=C[:,None,:,None];K+=block.reshape(K.shape)
    return K
def assemble_inverse(cache,k):
    s=assemble_real(cache['N'],k,h=cache['h'],radius=cache['radius'],
                    theory=cache.get('theory','TSDT'),indicator_matrix=cache['F'])
    del s['Kcl'];s['Kcl']=classical_inverse(cache,k);return s
def regression():
    p=prepare(2);k=np.array([.71,.39]);B=strain(p['G']+k);n=len(B)
    S=np.einsum('ij,ab->iajb',p['F'],np.linalg.inv(p['DA']))+np.einsum('ij,ab->iajb',np.eye(n)-p['F'],np.linalg.inv(p['DB']))
    A=np.zeros((8*n,3*n))
    for i in range(n):A[8*i:8*i+8,3*i:3*i+3]=B[i]
    target=A.T@np.linalg.solve(S.reshape(8*n,8*n),A);K=classical_inverse(p,k)
    err=np.linalg.norm(K-target)/np.linalg.norm(target);assert err<1e-10
    D=np.einsum('ij,ab->iajb',p['F'],p['DA'])+np.einsum('ij,ab->iajb',np.eye(n)-p['F'],p['DB'])
    direct=A.T@D.reshape(8*n,8*n)@A;s=assemble_real(2,k)
    derr=np.linalg.norm(direct-s['Kcl'])/np.linalg.norm(s['Kcl']);assert derr<1e-11
    # A homogeneous cell must reduce to the same local stiffness, independent of Fourier rule.
    uniform=A.T@np.kron(np.eye(n),p['DB'])@A
    compliance=A.T@np.linalg.solve(np.kron(np.eye(n),np.linalg.inv(p['DB'])),A)
    uerr=np.linalg.norm(uniform-compliance)/np.linalg.norm(uniform);assert uerr<1e-11
    return dict(full_inverse_equivalence=err,direct_generalized_equivalence=derr,homogeneous_limit=uerr,channels=len(p['groups']))
