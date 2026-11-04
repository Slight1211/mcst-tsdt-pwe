"""Area-average surface energy; input Lamé constants are Cs/(E_B*a)."""
import numpy as np

def surface_strain(Q,h,side,theory='TSDT'):
    if side not in (-1,1) or h<=0:raise ValueError('Invalid surface or thickness')
    if theory not in ('TSDT','FSDT'):raise ValueError('Unknown theory')
    x,y=1j*np.asarray(Q)
    A=side*h/(3 if theory=='TSDT' else 2)
    B=-side*h/6 if theory=='TSDT' else 0
    ux=np.array([1,0,x*B,A,0]);uy=np.array([0,1,y*B,0,A])
    return np.array([x*ux,y*uy,y*ux+x*uy])

def surface_blocks(Q,h,F,constants,theory='TSDT'):
    """constants[(side,phase)] = (lambda_s/(EB*a),mu_s/(EB*a))."""
    Q=np.asarray(Q);n=len(Q);out={}
    for side in (-1,1):
        b=np.array([surface_strain(q,h,side,theory) for q in Q])
        for phase,mask in [('A',F),('B',np.eye(n)-F)]:
            la,mu=constants[(side,phase)]
            if not ((la==0 and mu==0) or (mu>0 and la+mu>0)):raise ValueError('Surface elasticity not positive')
            C=np.array([[la+2*mu,la,0],[la,la+2*mu,0],[0,0,mu]])
            # NO thickness integration or extra h multiplier here.
            a=np.einsum('nid,ij,mje->ndme',b.conj(),C,b,optimize=True)
            out[f'K_s_{phase}_{"top" if side==1 else "bottom"}']=(a*mask[:,None,:,None]).reshape(5*n,5*n)
    return out

def normal_selector(n):
    S=np.zeros((n,5*n));S[np.arange(n),5*np.arange(n)+2]=1
    return S
