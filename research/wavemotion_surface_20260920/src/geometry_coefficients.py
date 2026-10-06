"""Evaluate repeated Fourier wavevectors once without rounding their values."""
import numpy as np

def unique_coefficients(indicator,g):
    g=np.asarray(g)
    if g.shape[-1]!=2 or not np.all(np.isfinite(g)):raise ValueError('Finite two-vector wavevectors required')
    unique,inverse=np.unique(g.reshape(-1,2),axis=0,return_inverse=True)
    values=indicator(unique)
    return np.asarray(values)[inverse].reshape(g.shape[:-1])
