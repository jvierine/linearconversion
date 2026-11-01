"""Conditional coherent beam-filling radar response; no transmitter inference."""
import numpy as np
from scipy.constants import c,k as kb,physical_constants
PT=1e6;TSYS=150.;RADAR=430e6;TAU=44e-6;GAIN_DB=61.1
RE=physical_constants['classical electron radius'][0];LAM=c/RADAR;KR=2*np.pi/LAM

def beam(z,gain_db=GAIN_DB):
    # Normalized paraxial Gaussian mode: integral |u|^2 dA = 1.
    gain=10**(gain_db/10);w0=np.sqrt(gain)*LAM/(np.sqrt(8)*np.pi)
    zr=np.pi*w0*w0/LAM;w=w0*np.sqrt(1+(z/zr)**2)
    curvature=z*(1+(zr/z)**2)
    return w,2/w**2,KR/curvature,np.arctan(z/zr)
