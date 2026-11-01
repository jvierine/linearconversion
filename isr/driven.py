"""Conditional linear forced response; no assumed transmitter-to-plasma coupling.

For an imposed longitudinal external potential phi_ext(k,omega),
phi_total=phi_ext/epsilon and delta_ne=(epsilon0*k^2/e)*chi_e*phi_total.
The sign uses electron charge -e. Squared amplitude is insensitive to it.
This function cannot turn a far-field HF transmitter power into a radar PSD:
inhomogeneous EM propagation/mode conversion must supply phi_ext at radar k.
"""
import numpy as np
from scipy.constants import epsilon_0,e,m_e,k as kb

def forced_density(k,chi_e,epsilon,phi_external_volts):
    return epsilon_0*k*k/e*np.asarray(chi_e)/np.asarray(epsilon)*phi_external_volts

def matching_density(pump_hz,k,Te,B=0.,theta_deg=0.):
    """Warm-fluid high-frequency estimate for matching a fixed radar wavenumber.
    omega^2=wp^2+3*k^2*kb*Te/me+Omega^2 sin(theta)^2.
    Returns NaN if no positive density matches. Not a conversion efficiency.
    """
    w=2*np.pi*np.asarray(pump_hz); om=e*B/m_e
    wp2=w*w-3*k*k*kb*Te/m_e-om*om*np.sin(np.deg2rad(theta_deg))**2
    return np.where(wp2>0,epsilon_0*m_e/e**2*wp2,np.nan)
