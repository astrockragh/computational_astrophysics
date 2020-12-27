import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
from scipy import integrate, interpolate

from scaling import scaling, CGS

def pr(s,v,v2=None):
    """ Pretty print """
    if v2==None:
        print('{:>20} : {:.2e}'.format(s,v))
    else:
        print('{:>20} : {:.2e}, {:.2e}'.format(s,v,v2))

def BE_sphere(u, gamma=1.0001, m_solar=1., T_K=10., x_ratio=6.5, y_ratio=14.1, verbose=0):
    """ Solve Lane-Emden equation to obtain correct BE-sphere initial condition """
    print('BE_sphere: m_solar, gamma, T_K=',m_solar,gamma,T_K)

    def lane_emden(y,xi):
        dy0 = y[1]
        dy1 = -2./xi*y[1] + np.exp(-y[0])
        return [dy0,dy1]

    # Parameters from HD
    G = u.units.G
    u.G = G
    T_BE = T_K/u.units.T
    cs = np.sqrt(u.gamma*CGS.k_B*T_K/(CGS.mu*CGS.m_u))/u.units.v
    M_BE = m_solar*CGS.m_Sun/u.units.m
    rho_0 = (1.18*cs**3)**2/(M_BE**2*G**3)
    rho_c = rho_0*y_ratio

    # Solve Lane-Emden
    y_init=[0, 0]                       # initial conditions
    xi = sp.linspace(0.0001, 50, 5000)  # equidistant integration steps.
    solution = integrate.odeint(lane_emden, y_init, xi)

    psi = solution[:,0]                 # psi
    dpsi = solution[:,1]                # dpsi
    rho_frac = np.exp(-psi)             # rho(r)/rho_central = exp(-psi)

    xi0 = np.sqrt(4.*np.pi*G*rho_c)/cs  # inverese BE-radius in code units
    rr = xi / xi0                       # radius variable in code units

    if verbose>1:
        print('========= In code units ========')
        #pr('BE radius', R_BE)
        pr('BE mass', M_BE)
        pr('central density', rho_c)
        pr('xi0', xi0)
        pr('G/G0', G/u.units.G)
        pr('cs', cs*u.units.v/CGS.kms)
        pr('xi'      , xi.min()      , xi.max())
        pr('u.r'     , u.r.min()     , u.r.max())
        pr('rho_frac', rho_frac.min(), rho_frac.max())
        pr('rr'      , rr.min()      , rr.max())

    rho_r = interpolate.interp1d(rr, rho_frac)
    
    u.D[:,:,:] = rho_r(u.r)*rho_c*1.05
    ii = np.where(u.D < (rho_0))
    u.D[ii] = rho_0
    u.E = cs**2/(u.gamma*(u.gamma-1.)) * u.D
    u.D[ii] = rho_0 / 500.
    u.T = u.E*(u.gamma-1.0)/u.D
