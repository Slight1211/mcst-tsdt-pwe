"""Seven free radii and seven free angles, C4v completion to 56 vertices.

All physical polar variables are free. Internally r=s*0.49/cos(theta)
enforces cell clearance without an extra radial/area cap. No area normalization
is applied except when constructing the circular initial design.
"""
import numpy as np
from node_shape_bending import _line_moments, _polygon_trial_fourier, AREA
from two_phase import reciprocal

SECTOR=np.pi/4
ANGLE_GAP=1e-4  # radians: numerical separation, not a fixed node angle
MIN_EDGE=1e-5

def initial():
    theta=(np.arange(7)+.5)*SECTOR/7
    r=np.sqrt(AREA/(28*np.sin(np.pi/28)))
    return np.r_[np.full(7,r)*np.cos(theta)/.49,theta]

def physical(x):
    return np.r_[.49*np.asarray(x[:7])/np.cos(x[7:]),x[7:]]

def boundary(x,derivative=False):
    x=np.asarray(x,float);s=x[:7];theta=x[7:]
    r=.49*s/np.cos(theta)
    ids0=np.r_[np.arange(7),np.arange(6,-1,-1)]
    signs0=np.r_[np.ones(7),-np.ones(7)]
    angles0=np.r_[theta,np.pi/2-theta[::-1]]
    ids=np.tile(ids0,4);signs=np.tile(signs0,4)
    angles=np.concatenate([angles0+q*np.pi/2 for q in range(4)])
    unit=np.column_stack((np.cos(angles),np.sin(angles)))
    tangent=np.column_stack((-np.sin(angles),np.cos(angles)))
    p=r[ids,None]*unit
    if not derivative:return p
    dp=np.zeros((56,2,14))
    for j in range(7):
        mask=ids==j
        dp[mask,:,j]=.49/np.cos(theta[j])*unit[mask]
        dp[mask,:,j+7]=r[j]*(np.tan(theta[j])*unit[mask]+signs[mask,None]*tangent[mask])
    return p,dp

def area(x,derivative=False):
    p,dp=boundary(x,True);pn=np.roll(p,-1,axis=0);dn=np.roll(dp,-1,axis=0)
    a=.5*np.sum(p[:,0]*pn[:,1]-p[:,1]*pn[:,0])
    da=.5*np.sum(dp[:,0,:]*pn[:,1,None]+p[:,0,None]*dn[:,1,:]
                 -dp[:,1,:]*pn[:,0,None]-p[:,1,None]*dn[:,0,:],axis=0)
    return (a,da) if derivative else a

def angular_constraints(x):
    theta=np.asarray(x[7:]);return np.r_[theta[0]-ANGLE_GAP/2,
        np.diff(theta)-ANGLE_GAP,SECTOR-theta[-1]-ANGLE_GAP/2]

def angular_jacobian(x):
    a=np.zeros((8,14));a[0,7]=1;a[-1,13]=-1
    for i in range(6):a[i+1,7+i]=-1;a[i+1,8+i]=1
    return a

def edge_constraints(x,derivative=False):
    p,dp=boundary(x,True);e=np.roll(p,-1,axis=0)-p;de=np.roll(dp,-1,axis=0)-dp
    lengths=np.linalg.norm(e,axis=1)
    c=lengths-MIN_EDGE
    jac=np.einsum('ni,nij->nj',e,de)/np.maximum(lengths[:,None],1e-30)
    return (c,jac) if derivative else c

def fourier(x,N,derivative=False):
    p,dp=boundary(x,True)
    _,G=reciprocal(N);ij=np.rint(G/(2*np.pi)).astype(int);size=4*N+1
    grid=np.array([(i,j) for i in range(-2*N,2*N+1) for j in range(-2*N,2*N+1)])
    wave=2*np.pi*grid
    value=_polygon_trial_fourier(wave,p)
    assert np.max(abs(value.imag))<1e-10
    delta=ij[:,None,:]-ij[None,:,:];rows=delta[...,0]+2*N;cols=delta[...,1]+2*N
    F=value.real.reshape(size,size)[rows,cols]
    if not derivative:return F
    e=np.roll(p,-1,axis=0)-p;de=np.roll(dp,-1,axis=0)-dp
    normals=np.column_stack((e[:,1],-e[:,0]))
    i0,i1=_line_moments(wave@e.T);phase=np.exp(-1j*wave@p.T)
    base=np.einsum('ed,edj->ej',normals,dp);slope=np.einsum('ed,edj->ej',normals,de)
    dv=np.einsum('ge,ge,ej->gj',phase,i0,base,optimize=True)+np.einsum('ge,ge,ej->gj',phase,i1,slope,optimize=True)
    assert np.max(abs(dv.imag))<1e-9
    dF=dv.real.T.reshape(14,size,size)[:,rows,cols]
    a,da=area(x,True)
    assert np.max(abs(np.diag(F)-a))<1e-12
    assert np.max(abs(dF[:,0,0]-da))<1e-10
    return F,dF
