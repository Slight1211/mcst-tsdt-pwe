"""Primary Python full-five-variable plate operators; analytic thickness moments."""
import numpy as np
from numpy.polynomial.legendre import leggauss

def operators(theory,q,z,h):
    if theory not in ('FSDT','TSDT'): raise ValueError('Unknown theory')
    if not np.isfinite(h) or h<=0: raise ValueError('Invalid thickness')
    if theory=='FSDT': A,Az,Azz,B,Bz,Bzz=z,1,0,0,0,0
    else: A,Az,Azz,B,Bz,Bzz=z-4*z**3/(3*h*h),1-4*z*z/(h*h),-8*z/(h*h),-4*z**3/(3*h*h),-4*z*z/(h*h),-8*z/(h*h)
    x,y=1j*np.asarray(q)
    p=np.array([[1,0,x*B,A,0],[0,1,y*B,0,A],[0,0,1,0,0]],complex)
    pz=np.array([[0,0,x*Bz,Az,0],[0,0,y*Bz,0,Az],[0]*5],complex)
    pzz=np.array([[0,0,x*Bzz,Azz,0],[0,0,y*Bzz,0,Azz],[0]*5],complex)
    e=np.array([x*p[0],y*p[1],y*p[0]+x*p[1],pz[0]+x*p[2],pz[1]+y*p[2]])
    r=np.array([y*p[2]-pz[1],pz[0]-x*p[2],x*p[1]-y*p[0]])/2
    rz=np.array([y*pz[2]-pzz[1],pzz[0]-x*pz[2],x*pz[1]-y*pz[0]])/2
    b=np.array([x*r[0],y*r[1],rz[2],(rz[1]+y*r[2])/np.sqrt(2),(rz[0]+x*r[2])/np.sqrt(2),(y*r[0]+x*r[1])/np.sqrt(2)])
    return p,e,b

def blocks(theory,q,h,method):
    E,nu,rho=10.,.3,2.;mu=E/(2*(1+nu));Q=E/(1-nu*nu)
    ks=5/6 if theory=='FSDT' else 1
    C=np.diag([Q,Q,mu,ks*mu,ks*mu]);C[0,1]=C[1,0]=nu*Q
    def product(a,b):
        p,e,c=a;pj,ej,cj=b
        return np.array([e.conj().T@C@ej,2*mu*h*h*c.conj().T@cj,rho*p.conj().T@pj])
    out=np.zeros((3,5,5),complex)
    if method=='analytic':
        t=np.linspace(-.5,.5,4);v=t[:,None]**np.arange(4)
        sample=[operators(theory,q,x*h,h) for x in t]
        coeff=[np.linalg.solve(v,np.array([s[j] for s in sample]).reshape(4,-1)).reshape(4,*sample[0][j].shape) for j in range(3)]
        for i in range(4):
            for j in range(4):
                if (i+j)%2==0:out+=h/((i+j+1)*2**(i+j))*product([c[i] for c in coeff],[c[j] for c in coeff])
    else:
        t,w=leggauss(method)
        for z,weight in zip(t*h/2,w*h/2):
            a=operators(theory,q,z,h);out+=weight*product(a,a)
    return out


