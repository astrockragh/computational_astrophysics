import numpy as np
from scaling import scaling, CGS

# define a void class to make it easy to collect variables
class void(object):
    pass

# main class containing the variables relevant to an HD experiment
class HD(object):
    def __init__(u,n=(32,32,32), nv=5, gamma=1.4, cs=1., L=1.0, d0=1., trace=0, units=None):
        # make n a 3-tuple if given as an integer
        if isinstance(n, int):
            m = (n,n,n)
        else:
            m = n
        
        u.trace = trace                     # option to trace calls
        u.t = 0.0                           # time
        u.dt = 0.0                          # time step
        u.n = m                             # resolution of the grid is (n[1],n[2],n[3])
        u.nv = nv                           # number of variables. We use nv = 8 = (density, total energy, momentum[xyz], magnetic field[xyz])
        u.ndim = 3                          # dimensionality of the experiment
        u.gamma = gamma                     # adiabatic index, currently an isothermal equation of state is not supported
        u.cs    = cs                        # sound speed only relevant if we support gamma=1.
        u.var   = np.zeros((nv,)+m)         # conserved variables. "(nv,)+m gives a 4-tuple with (nv,n[0],n[1],n[2])"
        u.D     = u.var[0]                  # mass density pointer
        u.E     = u.var[1]                  # total energy density pointer
        u.M     = u.var[2:5]                # momentum density components (2,3,4)
        u.D[:,:,:] = d0                     # constand density
        u.E[:,:,:] = d0/u.gamma/(u.gamma-1) # sound speed = 1.0
        u.coordinates(n=m,L=L)              # define coorindate system
        u.units = units                     # code units
        u.varnames=np.array(('D','E','vx','vy','vz'))
        
    def coordinates(u,n,L=1.0):             # function to define coordinates
        # make cell centered coordinates with domain boundaries [-L:L]^ndim
        u.ds = 2.0*L / n[0]                 # cell size
        LL = 0.5*u.ds*n[0], 0.5*u.ds*n[1], 0.5*u.ds*n[2]
        u.x = np.linspace(-LL[0],LL[0],n[0], endpoint=False) + 0.5 * u.ds # cell centered coordinates
        u.y = np.linspace(-LL[1],LL[1],n[1], endpoint=False) + 0.5 * u.ds
        u.z = np.linspace(-LL[2],LL[2],n[2], endpoint=False) + 0.5 * u.ds
        u.coords = np.array(np.meshgrid(u.x,u.y,u.z,indexing='ij')) # expand to 3D grid
        u.p = np.sqrt(np.sum(u.coords[0:2]**2,axis=0)) # cylinder radius
        u.r = np.sqrt(np.sum(u.coords[0:3]**2,axis=0)) # spherical radius
        u.L = np.array(LL)                  # box size
        u.n = n                             # number of grid points

    def velocity(u):                        # velocity from momentum and density
        iD = 1./u.D
        return np.array([u.M[0]*iD, u.M[1]*iD, u.M[2]*iD])

    def pgas(u):                            # gas pressure
        u.v = u.velocity()
        Eint = u.E - 0.5*(u.D*np.sum(u.v**2,0))
        return Eint*(u.gamma-1.0)

    def v_sound(u):                         # sound speed
        return np.sqrt(u.gamma*u.pgas()/u.D)

    def temperature(u):                     # temperature of gas per mu/kB
        u.P = u.pgas()
        return u.P/u.D

    def Courant(u,C):                       # Courant condition
        u.dtds = C/u.v_sound().max()
        u.dt = u.dtds * u.ds
        return u.dtds
