"""Complex Fourier geometry and full5 direct energy Galerkin assembly."""
import numpy as np
from scipy.special import j1
from scipy.linalg import eigh
from msse_core import operators

def reciprocal(N):
    if not isinstance(N,int) or N<0:raise ValueError('N must be nonnegative integer')
    ij=np.array([(i,j) for i in range(-N,N+1) for j in range(-N,N+1)])
    return ij,2*np.pi*ij

def disk(g,r,center=(0,0)):
    g=np.asarray(g);c=np.asarray(center)
    if r<=0 or np.any(abs(c)+r>=.5):raise ValueError('Disk crosses cell boundary')
    x=np.linalg.norm(g,axis=-1)*r;v=np.ones_like(x,float);np.divide(2*j1(x),x,out=v,where=x!=0)
    return np.pi*r*r*v*np.exp(-1j*(g@c))

def rectangle(g,width,height,center=(0,0)):
    g=np.asarray(g);c=np.asarray(center)
    if min(width,height)<=0 or np.any(abs(c)+np.array([width,height])/2>=.5):raise ValueError('Rectangle outside cell')
    return width*height*np.sinc(g[...,0]*width/(2*np.pi))*np.sinc(g[...,1]*height/(2*np.pi))*np.exp(-1j*(g@c))

def polygon(g,nodes):
    p=np.asarray(nodes,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all():raise ValueError('Invalid vertices')
    if np.any(abs(p)>=.5):raise ValueError('Polygon outside cell')
    e=np.roll(p,-1,axis=0)-p
    if np.min(np.linalg.norm(e,axis=1))<1e-12:raise ValueError('Repeated adjacent vertex')
    cross=lambda a,b:a[0]*b[1]-a[1]*b[0]
    def intersect(a,b,c,d):
        u,v=cross(b-a,c-a),cross(b-a,d-a);w,z=cross(d-c,a-c),cross(d-c,b-c)
        return u*v<=0 and w*z<=0 and np.all(np.maximum(np.minimum(a,b),np.minimum(c,d))<=np.minimum(np.maximum(a,b),np.maximum(c,d))+1e-14)
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            if j==i+1 or (i==0 and j==len(p)-1):continue
            if intersect(p[i],p[(i+1)%len(p)],p[j],p[(j+1)%len(p)]):raise ValueError('Self intersection')
    area=.5*np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))
    if area<=1e-14:raise ValueError('Require positive CCW area')
    g=np.asarray(g);shape=g.shape[:-1];flat=g.reshape(-1,2);g2=np.sum(flat**2,axis=1)
    phase=flat@e.T
    integral=np.exp(-.5j*phase)*np.sinc(phase/(2*np.pi))
    normal=flat[:,0,None]*e[None,:,1]-flat[:,1,None]*e[None,:,0]
    value=np.full(len(flat),area,complex);nz=g2>0
    value[nz]=1j*np.sum(normal[nz]*np.exp(-1j*flat[nz]@p.T)*integral[nz],axis=1)/g2[nz]
    return value.reshape(shape)

def assemble(N,k,theory='TSDT',h=.1,indicator=None,materials=None):
    if materials is None:materials=[(10.,.3,2.),(1.,.3,1.)]
    ij,G=reciprocal(N);q=G+np.asarray(k);n=len(q)
    if indicator is None:indicator=lambda g:disk(g,np.sqrt(.3/np.pi))
    F=indicator(G[:,None,:]-G[None,:,:]);FB=np.eye(n)-F
    # Polynomial coefficients in t=z/h and exact moment Gram factorization.
    t=np.linspace(-.5,.5,4);V=t[:,None]**np.arange(4)
    sample=[[operators(theory,w,x*h,h) for w in q] for x in t]
    coeff=[]
    for j in range(3):
        a=np.array([[s[j] for s in row] for row in sample]);coeff.append(np.linalg.solve(V,a.reshape(4,-1)).reshape(a.shape))
    H=np.array([[h/((i+j+1)*2**(i+j)) if (i+j)%2==0 else 0 for j in range(4)] for i in range(4)])
    L=np.linalg.cholesky(H)
    P,E,C=[np.einsum('rs,rnid->snid',L,x) for x in coeff]
    def gram(x,weight,F):
        a=np.einsum('snid,i,smie->ndme',x.conj(),weight,x,optimize=True)
        return (a*F[:,None,:,None]).reshape(5*n,5*n)
    Kcl=np.zeros((5*n,5*n),complex);M=Kcl.copy();Kshear=Kcl.copy();phases=[]
    for (young,nu,rho),f in zip(materials,[F,FB]):
        if young<=0 or rho<=0 or not -1<nu<.5:raise ValueError('Nonpositive phase')
        mu=young/(2*(1+nu));Q=young/(1-nu**2);ks=5/6 if theory=='FSDT' else 1
        cp=np.diag([Q,Q,mu,ks*mu,ks*mu]);cp[0,1]=cp[1,0]=nu*Q
        ec=np.einsum('ij,snjd->snid',np.linalg.cholesky(cp).T,E)
        Kcl+=gram(ec,np.ones(5),f);M+=gram(P,np.full(3,rho),f)
        Kshear+=gram(E,np.array([0,0,0,ks*mu,ks*mu]),f)
        phases.append(gram(C,np.full(6,2*mu*h*h),f))
    return dict(Kcl=Kcl,KA=phases[0],KB=phases[1],Kshear=Kshear,M=M,G=G,ij=ij,F=F)

def solve(s,ell=(.25,.25),count=16):
    K=s['Kcl']+ell[0]**2*s['KA']+ell[1]**2*s['KB'];M=s['M']
    herm=max(np.linalg.norm(K-K.conj().T)/np.linalg.norm(K),np.linalg.norm(M-M.conj().T)/np.linalg.norm(M))
    if herm>1e-10:raise ValueError(f'Non-Hermitian {herm}')
    np.linalg.cholesky(M)
    lam,v=eigh(K,M,subset_by_index=[0,min(count,len(K))-1],driver='gvx')
    if np.min(lam)<-1e-9:raise ValueError(f'Negative eigenvalue {lam.min()}')
    energy=lambda A:np.einsum('ij,ij->j',v.conj(),A@v).real
    total=energy(K);RA=ell[0]**2*energy(s['KA'])/total;RB=ell[1]**2*energy(s['KB'])/total
    residual=np.linalg.norm(K@v-M@v*lam)/(np.linalg.norm(K)*np.linalg.norm(v)+np.linalg.norm(M)*np.linalg.norm(v*lam))
    return dict(lam=lam,v=v,RA=RA,RB=RB,hermitian=herm,residual=residual,closure=np.max(abs(energy(s['Kcl'])+ell[0]**2*energy(s['KA'])+ell[1]**2*energy(s['KB'])-total))/max(total))
