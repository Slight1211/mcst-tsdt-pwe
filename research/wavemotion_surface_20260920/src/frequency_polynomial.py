"""Finite-N dry dynamic polynomial on one fixed-angle Bloch folding segment.

Kinematic derivatives are polynomial in in-plane wavevector: stiffness degree
at most four, mass degree at most two. Acoustic impedance is NOT interpolated.
Floating-point interpolation must be validated on withheld frequencies before use.
"""
import numpy as np
from numpy.polynomial.chebyshev import chebfit,chebval
from symmetric_bending import assemble_bending
from acoustics import fold_incidence

def build(N,interval,c=.25,theta=np.pi/6,phi=0,h=.1,ell_over_h=.25):
    lo,hi=interval
    if not 0<lo<hi:raise ValueError('Invalid interval')
    center=(lo+hi)/2;half=(hi-lo)/2
    x=np.cos(np.pi*(np.arange(5)+.5)/5);samples=[];reference=None
    for xi in x:
        w=center+half*xi;_,k,inc=fold_incidence(w,c,theta,phi)
        if reference is not None and not np.array_equal(inc,reference):raise ValueError('Interval crosses folding boundary')
        reference=inc.copy();s=assemble_bending(N,k,h=h)
        samples.append(s['Kcl']+ell_over_h**2*(s['KA']+s['KB'])+sum(s['Ks_parts'].values())-w*w*s['M'])
    for w in [lo,hi]:
        if not np.array_equal(fold_incidence(w,c,theta,phi)[2],reference):raise ValueError('Interval crosses folding boundary')
    shape=samples[0].shape
    coeff=chebfit(x,np.asarray(samples).reshape(5,-1),4).reshape((5,)+shape)
    return dict(coeff=coeff,interval=[lo,hi],center=center,half=half,S=s['S'],ij=s['ij'],G=s['G'],incident_order=reference)

def evaluate(model,omega):
    if not model['interval'][0]<=omega<=model['interval'][1]:raise ValueError('Polynomial extrapolation prohibited')
    return chebval((omega-model['center'])/model['half'],model['coeff'])
