import numpy as np

class CGS():
    name = 'CGS'
    m_Earth = 5.972e27
    m_Sun = 1.989e33
    r_Earth = 6.371e8
    grav = 6.673e-8         # 1 / (density time^2)
    G = grav
    yr = 3.156e+7
    au = 1.498e+13
    AU = au
    mu = 2.4
    m_u = 1.6726e-24
    m_p = m_u
    k_B = 1.3807e-16        # energy per K
    h_P = 6.6260e-27        # energy per Hertz
    e = 4.8032e-10
    c = 2.9979e10
    Stefan = 5.6704e-8      # energy pre unit area and time
    pc = 180./np.pi*3600.*au
    kyr = 1e3*yr
    Myr = 1e6*yr
    kms = 1e5
    Angstrom = 1e-8
    micron = 1e-4
    l = 1.

class MKS():
    name = 'MKS'
    m_Earth = 5.972e24
    m_Sun = 1.989e30
    r_Earth = 6.371e6
    grav = 6.673e-11
    G = grav
    yr = 3.156e+7
    au = 1.498e+11
    AU = au
    mu = 2.4
    m_u = 1.6726e-27
    m_p = m_u
    k_B = 1.3807e-23
    h_P = 6.62606e-34
    e = 1.6022e-19
    c = 2.9979e8
    Stefan = 5.6704e-5
    pc = 180./np.pi*3600.*au
    kyr = 1e3*yr
    Myr = 1e6*yr
    kms = 1e3
    Angstrom = 1e-10
    micron = 1e-6

class SI(MKS):
    name = 'SI'

class scaling():
    '''
    Return a structure with scaling constants.  Use e.g.

        scgs=scaling(cgs)
        sSI=scaling(SI)
        print (cgs.k_b, SI.k_b)
    '''
    def __init__(self,system=CGS,l=0,m=0,t=0,D=0,v=0,mu=2.4,verbose=0):
        """ Object holding scaling information for code units """

        if type(system)==type('str'):
            if   system=='cgs' or system=='CGS':
                system = CGS
            elif system=='mks' or system=='MKS':
                system = MKS
            elif system=='si'  or system=='SI':
                system = SI
        self.system = system

        if verbose>0:
            print("using "+system.name+" units")

        # Interpret string values
        if l == 'pc'    : l = system.pc
        if m == 'Solar' : m = system.m_Sun
        if m == 'm_Sun' : m = system.m_Sun
        if t == 'yr'    : t = system.yr
        if t == 'kyr'   : t = system.kyr
        if t == 'Myr'   : t = system.Myr
        if v == 'kms'   : v = system.kms

        # Count input scaling constants
        inputs = [l,m,t,D,v]
        nonzero = [i>0 for i in inputs].count(True)
        if verbose > 2:
            print('there','is' if nonzero==1 else 'are',nonzero,'input value'+('s' if nonzero>1 else ''))
        if nonzero != 3:
            print('exactly 3 input values must be specified (there {} {})'.\
                  format('was' if nonzero==1 else 'were',nonzero))
            return None
        
        self.l = l
        self.m = m
        self.t = t
        self.D = D
        self.v = v

        # If velocity units were given, either length or time units were not
        if v > 0:
            if l == 0:
                self.l = v*t
            else:
                self.t = l/v
        else:
            self.v = l/t

        # If density units were given, either length or mass unis were not
        if D > 0:
            if m == 0:
                self.m = D*l**3
            else:
                self.l = (m/D)**(1/3)
        else:
            self.D = m/l**3

        # Compute derived scaling constants for other quantities
        self.m = self.D*self.l**3                           # mass
        self.P = self.D*self.v**2                           # pressure
        self.e = self.m*self.v**2                           # energy
        self.E = self.D*self.v**2                           # energy density
        self.mu = mu                                        # mean molecular weight
        self.T = mu*(system.m_u)/(system.k_B)*self.v**2     # temperature

        # Scaling of constants of nature into code units
        self.G = system.G*self.D*self.t**2                  # constant of gravity
        self.Stefan = system.Stefan*self.T**4/(self.E*self.v)  # Stefan's constant, rescaled
        self.h_P = system.h_P/(self.t*self.e)               # Planck's constant, rescaled
        self.k_B = system.k_B*self.T/self.e                 # Boltzmann's constant, rescaled
        self.c = system.c/self.v                            # speed of light, rescaled

        if verbose>1:
            print(vars(self))
