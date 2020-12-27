import numpy as np

# Solve for the HLL flux give to HD state vectors (left and right of the interface) stored in face[vars, 2, n,n,n], where 2=left, right
# face[:,] = [Density, Eint, vO, v1, v2], where vO is normal to the interface between left and right state, and v[1,2] are parallel 
#                0       1    2   3   4   5   6   7
def HLL(face,u):
    # shorthand values
    Dl = face[0,0]; Eintl = face[1,0]; ul = face[2:5,0]
    Dr = face[0,1]; Eintr = face[1,1]; ur = face[2:5,1]
    # Get pressure
    Pl = (u.gamma-1)*Eintl; Pr = (u.gamma-1)*Eintr    

    # Compute conserved variables [density, total energy, momentum]
    def conserved_vars(D,Eint,u):
        Etot = Eint + 0.5*D*np.sum(u**2,axis=0)
        return np.array([D,Etot,D*u[0],D*u[1],D*u[2]])

    Cl = conserved_vars(Dl,Eintl,ul)
    Cr = conserved_vars(Dr,Eintr,ur)
    
    # sound speed for each side of interface (l==left, r==right)
    c2_left  = u.gamma*Pl/Dl; c2_right = u.gamma*Pr/Dr
    c_max = np.sqrt(np.maximum(c2_left,c2_right))

    # maximum wave speeds to the left and right (guaranteed to have right sign)
    SL = np.minimum(np.minimum(ul[0],ur[0])-c_max,0) # <= 0.
    SR = np.maximum(np.maximum(ul[0],ur[0])+c_max,0) # >= 0.

    # 1D HD fluxes for density, total energy, and momentum with advection velocity "u"
    def hydro_flux(u,D,Etot,U,P):
        return np.array([u*D, u*(Etot+P), u*U[0]+P, u*U[1], u*U[2]])

    #                   Dens   Etot   Momentum Pressure
    Fl = hydro_flux(ul[0], Cl[0], Cl[1], Cl[2:5], Pl)
    Fr = hydro_flux(ur[0], Cr[0], Cr[1], Cr[2:5], Pr)

    # HLL flux based on wavespeeds. If SL < 0 and SR > 0 (sub-sonic) then mix appropriately. 
    Flux=np.empty_like(Fl) # shape (nv,n[0],n[1],n[2])
    for iv in range(u.nv):
        Flux[iv] = (SR * Fl[iv] - SL*Fr[iv] + SL*SR*(Cr[iv] - Cl[iv])) / (SR - SL)

    return Flux