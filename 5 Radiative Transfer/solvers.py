import numpy as np

from HD          import HD
from slopes      import MonCen, deriv
from Riemann     import HLL
from selfgravity import FFT

class MUSCL(HD):
    """ The Monotonic Upwind Scheme for Conservation Laws has these steps:
        1. Compute size of timestep
        2. Compute primitive variables
        3. Compute slopes of primitive variables
        4. Compute predicted states at time t+dt/2 
        5. Compute 1/2 source term for gravity @t
        Repeat 6+7 for all coordinate directions:
        6. Compute left and right face values
        7. Compute fluxes using Riemann solver
        8. Update the conserved variables
        9. Compute the gravitational Potential
        10. Compute 1/2 source term for gravity @t+dt
    """
    def stat(self,x,name=""):
        print('{:6s}  Min:{:11.4e}   Mean:{:11.4e}   Max:{:11.4e}   stddev:{:11.4e}'.\
              format(name,np.amin(x),np.mean(x),np.max(x),np.std(x)))
        
    def __init__(self, u, C=0.2, Solver=HLL, Slope=MonCen, Poisson=FFT):
        """ Initialize a MUSCL solver """
        self.Solver   = Solver
        self.Slope    = Slope
        self.Poisson  = Poisson
        u.it= 0                             # number of updates (integer)
        u.t = 0.0                           # Time
        n  = u.n; nv = u.nv
        # extra HD arrays neeeded for MUSCL method
        u.Phi  = np.zeros(n)          # potential
        u.dPhi = np.zeros((3,)+n)     # gradien of potential
        u.prim  = np.zeros((nv,)+n)   # primitive variables (density, internal energy, velocity)
        u.predict = np.zeros((nv,)+n) # centered primitive variables half time step forward.
        u.slope = np.zeros((nv,3,)+n) # slope of primitive variables
        u.face  = np.zeros((nv,2)+n)  # Left and right face values
        u.flux  = np.zeros((nv,3)+n)  # fluxes

    def Prim(self,u):
        """ Compute 5 primitive variables D,Eint,v -- all centered """
        D = u.D
        v = u.velocity()
        E = u.E - 0.5*(D*np.sum(v**2,0))
        u.prim[:,:,:,:] = np.array([D,E,v[0],v[1],v[2]])

    def Slopes(self,u):
        """ Compute slopes for primitive variables shape (nv,ndim,n,n,n) """
        for i in range(u.nv):
            u.slope[i] = self.Slope(u.prim[i])
                                    
    def Predict(self,u):
        """ Compute predicted solution at time t+dt/2 shape (nv,n,n,n) """        
        # shorthand for variables and slopes
        D  = u.prim[0];  E  = u.prim[1];  v  = u.prim[2:5]
        dD = u.slope[0]; dE = u.slope[1]; dv = u.slope[2:5]
        dPhi = u.dPhi
        
        # div(velocity) = dU / dx + dV / dy + dW / dz
        div_v = dv[0,0] + dv[1,1] + dv[2,2]

        # Time  evolution by +dt/2 for density, Energy and velocity:
        # We use np.sum(x*y,axis=0) to make dot products
        
        # drhodt = - v.grad(rho) - rho (div.v)
        u.predict[0] = D + 0.5*u.dtds* ( - np.sum(v*dD,axis=0) - D*div_v )

        # dEdt = - v.grad(E) - gamma E (div.v) - rho v.grad(Phi) + Qrad
        u.predict[1] = E + 0.5*u.dtds* ( \
                       - np.sum(v*dE,axis=0) - u.gamma*E*div_v - D * np.sum(v*dPhi,axis=0) )

        # dv_i/dt = - v.grad(v_i) - 1/rho grad(P) - grad(Phi)
        dpgas = self.Slope(u.pgas())
        dir=['x','y','z']
        for i in range(3):
            u.predict[2+i] = v[i] + 0.5*u.dtds* ( \
                             - v[0]*dv[i,0] - v[1]*dv[i,1] - v[2]*dv[i,2] - dpgas[i]/D - dPhi[i] )
            #self.stat(u.predict[2+i],name='v'+dir[i])
            #self.stat(dpgas[i],name='dP'+dir[i])
            #self.stat(dPhi[i],name='dPhi'+dir[i])
        
    def Faces(self,u,idim):
        """ Compute left and right face values _at_cell_interface_ with shape (nv,2,n,n,n) """
        for i in range(u.nv):
            uc = u.predict[i]                   # predicted state at t + dt/2 (n,n,n)
            ul = uc + 0.5*u.slope[i,idim]       # left face
            ur = uc - 0.5*u.slope[i,idim]       # right face
            ur = np.roll(ur,-1,idim)            # align right faces
            u.face[i] = np.array((ul,ur))

    def Riemann(self,u,idim,Solver=HLL):
        """ Reorder variables and call the 1-D Riemmann solver.
            Before the call to the solver, the variables are
            reordered, with the perpedicular components always
            in the same place.  After the call, the oreder is
            restored again, when copying the fluxes back in place.
        """
        if  idim==0:
            faces = u.face[[0,1,2,3,4]]
            flux = Solver(faces,u)
            u.flux[:,idim] = flux[[0,1,2,3,4]]
        elif idim==1:
            faces = u.face[[0,1,3,4,2]] 
            #               0 1 2 3 4
            flux = Solver(faces,u)
            u.flux[:,idim] = flux[[0,1,4,2,3]]
        elif idim==2:
            faces = u.face[[0,1,4,2,3]]                
            #               0 1 2 3 4
            flux = Solver(faces,u)
            u.flux[:,idim] = flux[[0,1,3,4,2]]

    def Update(self,u):
        """ Flux updates with shape (n,n,n) """
        for iv in range(u.nv):
            for idim in range(u.ndim):
                # Divergence update
                u.var[iv] -= u.dtds*(u.flux[iv,idim]-np.roll(u.flux[iv,idim],1,axis=idim))
    
    def Source(self,u):
        """ Add contribution from gravitation with 1/2*dt
            S(E_tot) = - (rho v).grad(Phi)
            S(rho v) = - rho grad(Phi)
        """
        u.E += - 0.5*u.dtds*np.sum(u.M*u.dPhi,axis=0)
        for i in range(3):
            u.M[i] += - 0.5*u.dtds*u.D*u.dPhi[i]
            
    def Step(self,u,BCs=None,C=0.2,tend=None):
        """ Full time update of the MUSCL method """
        if u.it == 0:                      # Gravitational potential
            u.Phi[:,:,:] = self.Poisson(u) # has to be defined if it=0
            u.dPhi[:,:,:,:] = deriv(u.Phi)
            
        u.Courant(C=C)                     # Courant condition
        if not tend==None:
            if u.t >= tend:                # do nothing if we have already passed tend
                return
            if u.t+u.dt > tend:            # adjust dt if we will pass tend
                u.dt = tend - u.t
                u.dtds = u.dt / u.ds
        self.Prim(u)                       # Primitive vars       shape (nv+1,n,n,n)
        self.Predict(u)                    # Primitive vars @dt/2 shape (nv,n,n,n)
        self.Source(u)                     # Add 1/2 source term @t. OBS! changing u.var is OK here
        for idim in range(u.ndim):
            self.Faces(u,idim)             # face values          shape (nv,2,n,n,n)
            self.Riemann(u,idim)           # flux                 shape (nv,3,n,n,n)
        self.Update(u)                     # upd w flux           shape (nv,n,n,n)
        u.Phi  = self.Poisson(u)           # Gravitational potential for next timestep
        u.dPhi = deriv(u.Phi)              # - Gravitational acceleration for next timestep
        self.Source(u)                     # Add 1/2 source term @t+dt
        u.t  += u.dt                       # Update current time and nr of iterations
        u.it += 1
        if not BCs==None:                  # Apply boundary conditions
            u.var[:,:,:, 0]=BCs[0]
            u.var[:,:,:,-1]=BCs[1]