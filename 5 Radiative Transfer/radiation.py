import numpy as np
from scaling import CGS, MKS, scaling

def Planck (T, lamb, units=None, method='', verbose=0):
    """ The Planck function, in dimensionless units, per log nu or lambda """
    if units== None:
        x = (1.0/lamb)/T
    else:
        c = (units.h_P/units.k_B)*units.c
        x = (c/(lamb/units.l))/(T)
        if verbose>0:
            m = T.shape[0]//2
            print('Planck: c =',c,' x =', x[0,0,0], x[m,m,m])

    if method=='simple':
        f = T**4 * x**4 * 1/ (np.exp(x)-1.0)
    else:
        e = np.exp(-x)
        f = ((T*x)**2)**2 * e/(1.0-e)

    if units==None:
        return f
    else:
        # Cf. Eq. (2-3) in Radiation.ipynb
        return f * (units.Stefan/np.pi)
