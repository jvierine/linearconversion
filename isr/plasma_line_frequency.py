"""Weakly damped Maxwellian electron plasma-line resonance at radar Bragg k.

Nicolls et al. (2006), doi:10.1029/2006GL027222, equations 1, 2 and 4.
Solve Re[1+chi_e]=0 on the high-frequency electron branch, neglecting ion
motion, collisions and bulk Doppler shift. This is a frequency diagnostic,
not a calculation of line power or of collisionless damping in the solver.
"""
import numpy as np
from scipy.constants import e, m_e, epsilon_0, k as kb, c
from scipy.optimize import brentq
from scipy.special import ive, wofz


def dielectric(f_hz, ne, te, magnetic_t, theta_deg, radar_hz=430e6, sideband=1):
    """Electron dielectric at k=2*pi*(2*f_r+sideband*f)/c; theta from B."""
    w=2*np.pi*f_hz
    k=2*np.pi*(2*radar_hz+sideband*f_hz)/c
    v=np.sqrt(kb*te/m_e)
    if magnetic_t == 0:
        z=w/(np.sqrt(2)*k*v)
        response=1+z*1j*np.sqrt(np.pi)*wofz(z)
    else:
        om=e*magnetic_t/m_e
        angle=np.deg2rad(theta_deg)
        kp=k*abs(np.cos(angle));kt=k*np.sin(angle)
        if kp < 1e-8*k:
            raise ValueError('Exactly perpendicular propagation requires a Bernstein-mode treatment')
        b=(kt*v/om)**2
        nmax=max(6,int(np.ceil(5*np.sqrt(b)+5)))
        n=np.arange(-nmax,nmax+1);weights=ive(n,b)
        while abs(weights.sum()-1)>1e-13:
            nmax+=2;n=np.arange(-nmax,nmax+1);weights=ive(n,b)
        a=np.sqrt(2)*kp*v
        response=1+1j*w*np.sqrt(np.pi)/a*np.sum(weights*wofz((w-n*om)/a))
    return 1+ne*e**2/(epsilon_0*kb*te*k*k)*response


def frequency_hz(ne,te,magnetic_t,theta_deg,radar_hz=430e6,sideband=1):
    """High-frequency real root; scalar input. Raises if root is not bracketed."""
    if not (ne>0 and te>0 and magnetic_t>=0):
        raise ValueError('Require positive density/temperature and nonnegative B')
    fp=np.sqrt(ne*e*e/(epsilon_0*m_e))/(2*np.pi)
    fc=e*magnetic_t/(2*np.pi*m_e)
    k=4*np.pi*radar_hz/c
    guess=np.sqrt(fp*fp+fc*fc*np.sin(np.deg2rad(theta_deg))**2+3*k*k*kb*te/(4*np.pi**2*m_e))
    fun=lambda f:dielectric(f,ne,te,magnetic_t,theta_deg,radar_hz,sideband).real
    return brentq(fun,.9*guess,1.1*guess,xtol=.01,rtol=1e-12)


def profile_hz(ne,te,magnetic_t,theta_deg,radar_hz=430e6,sideband=1):
    arrays=np.broadcast_arrays(ne,te,magnetic_t,theta_deg)
    return np.array([frequency_hz(*v,radar_hz,sideband) for v in zip(*(a.flat for a in arrays))]).reshape(arrays[0].shape)
