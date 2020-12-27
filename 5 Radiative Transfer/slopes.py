def left_slope(f,axis=0):
    return (f-np.roll(f,1,axis))
""" HD and MUCSL classes """

import numpy as np

def deriv(f):
    """ Central slope (e.g. derivative*dx) """
    if f.ndim==1:
        return 0.5*(np.roll(f,-1) - np.roll(f,1))
    else:
        shape=np.insert(f.shape,0,f.ndim)
        slopes=np.zeros(shape)
        for i in range(f.ndim):
            slopes[i]=0.5*(np.roll(f,-1,axis=i) - np.roll(f,1,axis=i))
        return slopes

def MonCen(f):
    """ Monotonized central slope limiter """
    if f.ndim==1:
        ls=left_slope(f)
        rs=np.roll(ls,-1)
        cs=np.zeros(ls.shape)
        w=np.where(ls*rs>0.0)
        cs[w]=2.0*ls[w]*rs[w]/(ls[w]+rs[w])
        return cs
    else:
        shape=np.insert(f.shape,0,f.ndim)
        slopes=np.zeros(shape)
        for i in range(f.ndim):
            ls=left_slope(f,axis=i)
            rs=np.roll(ls,-1,axis=i)
            cs=np.zeros(f.shape)
            w=np.where(ls*rs>0.0)
            cs[w]=2.0*ls[w]*rs[w]/(ls[w]+rs[w])
            slopes[i]=cs
        return slopes
