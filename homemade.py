def deriv1(f,ds,axis=0):
    """ 
    Compute derivative in the axis-direction, given f[i,j,k] 
    and ds = mesh size in the axis-direction
    """
    return (np.roll(f,-1,axis)-np.roll(f,+1,axis))/(2.0*ds)

def deriv4(f,ds,axis=0):
    """ 
    Compute 4th order linear derivative in the axis-direction, given f[i,j,k] 
    and ds = mesh size in the axis-direction
    """
    return (8*(np.roll(f,-1,axis)-np.roll(f,+1,axis))-(np.roll(f,-2,axis)-np.roll(f,+2,axis)))/(12.0*ds)