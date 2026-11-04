"""Audited dense MCST inverse-PWE spectrum and band-4/5 derivatives."""
import numpy as np
from scipy.linalg import eigh
from inverse_circle_bending import classical_inverse,strain
from real_circle_bending import assemble_real

H_OVER_A=.8
ELL_OVER_H=1/60

def point(F,dF,N,k,cache,gradient=True):
    parts=assemble_real(N,k,h=H_OVER_A,theory='TSDT',indicator_matrix=F,return_shape_operators=gradient)
    K=classical_inverse(cache,k)+ELL_OVER_H**2*parts['KB'];M=parts['M']
    val,q=eigh(K,M,subset_by_index=(0,7),driver='gvx',check_finite=False)
    gamma=np.linalg.norm(k)<1e-12;first=1 if gamma else 0
    if val[first:].min()<=0 or (gamma and abs(val[0])>1e-7):raise ArithmeticError('Invalid elastic eigenvalues')
    res=np.linalg.norm(K@q-M@q*val,axis=0)/(np.linalg.norm(K@q,axis=0)+abs(val)*np.linalg.norm(M@q,axis=0)+1e-30)
    ortho=float(np.max(abs(q.T@M@q-np.eye(8))));maxres=float(res[first:].max())
    if maxres>1e-5 or ortho>1e-8:raise ArithmeticError('Eigenpair audit failed')
    f=np.sqrt(np.maximum(val[:6],0));audit=dict(raw_eigenvalues=val.tolist(),residual_nonrigid=maxres,mass_orthogonality=ortho)
    # Adjacent gaps are recorded; analytic simple-eigenvalue derivatives need
    # checking when either optimized band belongs to a repeated eigenvalue.
    audit['target_relative_eigen_separation']=float(min(np.min(abs(val[b]-np.delete(val,b)))/abs(val[b]) for b in [3,4]))
    if not gradient:return f,None,audit
    G=cache['G'];B=strain(G+np.asarray(k),'TSDT');Z=np.einsum('ab,nbd->nad',cache['W'].T,B)
    df=np.full((6,len(dF)),np.nan)
    for b in [3,4]:
        qb=q[:,b].reshape(len(G),3);y=np.einsum('nid,nd->ni',Z,qb);dc=np.zeros(len(dF))
        for (ids,C),alpha in zip(cache['groups'],cache['alphas']):
            v=C@y[:,ids];dc-=alpha*np.einsum('ni,jnp,pi->j',v,dF,v,optimize=True)
        p=np.einsum('snid,nd->sni',parts['P_operator'],qb)
        cc=np.einsum('snid,nd->sni',parts['C_operator'],qb)
        dm=(7780/1180-1)*np.einsum('sni,jnp,spi->j',p,dF,p,optimize=True)
        nu=4.35/(2*1.59)-1
        dmc=-ELL_OVER_H**2*H_OVER_A**2/(1+nu)*np.einsum('sni,jnp,spi->j',cc,dF,cc,optimize=True)
        df[b]=(dc+dmc-val[b]*dm)/(2*f[b])
    return f,df,audit
