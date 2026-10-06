"""Classical inverse TSDT with dense low-spectrum solve; no MCST stiffness.

Only bands 4/5 derivatives are requested by the constrained mass optimization.
Old mechanics and old sparse eigensolver sources remain unmodified.
"""
import numpy as np
from scipy.linalg import eigh
from inverse_circle_bending import classical_inverse, strain
from real_circle_bending import assemble_real


def point(F, dF, N, k, cache, gradient=True):
    parts=assemble_real(N,k,h=.8,theory='TSDT',indicator_matrix=F,
                        return_shape_operators=gradient)
    K=classical_inverse(cache,k)  # Exactly ell=0; KB is never added.
    M=parts['M']
    val,q=eigh(K,M,subset_by_index=(0,7),driver='gvx',check_finite=False)
    gamma=np.linalg.norm(k)<1e-12
    first=1 if gamma else 0
    if val[first:].min()<=0 or (gamma and abs(val[0])>1e-7):
        raise ArithmeticError('Nonpositive elastic mode or invalid rigid mode')
    residual=np.linalg.norm(K@q-M@q*val,axis=0)/(np.linalg.norm(K@q,axis=0)+abs(val)*np.linalg.norm(M@q,axis=0)+1e-30)
    ortho=float(np.max(abs(q.T@M@q-np.eye(8))))
    res=float(residual[first:].max())
    if res>1e-5 or ortho>1e-8:raise ArithmeticError('Eigenpair verification failed')
    f=np.sqrt(np.maximum(val[:6],0))
    audit=dict(raw_eigenvalues=val.tolist(),residual_nonrigid=res,mass_orthogonality=ortho,
               solver='scipy.linalg.eigh(gvx), indices 0:7')
    if not gradient:return f,None,audit
    G=cache['G'];B=strain(G+np.asarray(k),'TSDT')
    Z=np.einsum('ab,nbd->nad',cache['W'].T,B)
    df=np.full((6,len(dF)),np.nan)
    for b in (3,4):
        qb=q[:,b].reshape(len(G),3);y=np.einsum('nid,nd->ni',Z,qb)
        dc=np.zeros(len(dF))
        for (ids,C),alpha in zip(cache['groups'],cache['alphas']):
            v=C@y[:,ids];dc-=alpha*np.einsum('ni,jnp,pi->j',v,dF,v,optimize=True)
        p=np.einsum('snid,nd->sni',parts['P_operator'],qb)
        dm=(7780/1180-1)*np.einsum('sni,jnp,spi->j',p,dF,p,optimize=True)
        df[b]=(dc-val[b]*dm)/(2*f[b])
    return f,df,audit
