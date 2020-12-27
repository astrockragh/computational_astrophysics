import numpy as np
from numpy import fft
 
def FFT(u,verbose=0):
    """ FFT Solve he gravitational potential with an FFT method 
    """

    def k2f(n):
        """ Wavenumber squared in 3-D box with ds=1 """
        k    = np.fft.fftfreq(n,1./(2.*np.pi))
        k_3d = np.meshgrid(k,k,k)
        k2   = np.sum(np.array(k_3d)**2,0)
        return k2 #*np.complex(1.,0)

    def fft_3d(f):
        """ Forward 3-D FFT"""
        return np.fft.fft(fft.fft(fft.fft(f,axis=0),axis=1),axis=2)

    def ifft_3d(f):
        """ Inverse 3-D FFT """
        return np.fft.ifft(fft.ifft(fft.ifft(f,axis=0),axis=1),axis=2)

    # Scaling of the source term
    source = -(4.*np.pi*u.G)*u.D*u.ds**2
    sourcet = fft_3d(source)
    n = source.shape[0]
    k2 = k2f(n)

    # Avoid divide fault -- remove the mean potential 
    sourcet[0,0,0] = 0.0
    m = n//2
    k2[0,0,0]   = 1.0
    phi = -np.real(ifft_3d(sourcet/k2))

    if verbose:
        for i in range(m-2,m+2):
            print(source[m-2:m+2,m-2:m+2,i])
        for i in range(m-2,m+2):
            print(phi[m-2:m+2,m-2:m+2,i])
    return phi
