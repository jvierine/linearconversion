"""Coherent Born radar operator from Kudeki (2026), ECE 458 notes,
Eqs. (5.49), (5.51), and section 5.3.4 (pp. 97-100).

Signal uses exp(-i omega_R t), the conjugate of the notes' carrier convention.
Returns a power-normalized complex envelope: |s|**2 is available power in W.
Keep the density real in the general retarded operator; phasor sidebands carry 1/2.
"""
import numpy as np
from scipy.constants import c, k as kb, physical_constants
from scipy.integrate import trapezoid
RE = physical_constants['classical electron radius'][0]
SIGMA_E = 4*np.pi*RE**2


def electron_kernel(range_m, transmit_gain, receive_area_m2, phase_rad=0.):
    """Single-electron amplitude coefficient (Kudeki 5.51, divided by sqrt power).

    Antenna pattern phase belongs in phase_rad; round-trip propagation is separate.
    For a reciprocal far-field antenna A_r=lambda**2*G_r/(4*pi).
    """
    r=np.asarray(range_m)
    if np.any(r<=0):raise ValueError('Radar range must be positive')
    return np.sqrt(np.asarray(transmit_gain)*np.asarray(receive_area_m2)*SIGMA_E)/(4*np.pi*r**2)*np.exp(1j*np.asarray(phase_rad))


def retarded_echo(times_s, positions_m, volume_weights_m3, density, envelope,
                  kernel, radar_hz=430e6, transmit_power_w=1e6):
    """Quadrature of Kudeki 5.49, before squaring; no independence assumption.

    density(position[N,3], scattering_time[N]) returns real perturbations [m^-3].
    envelope(transmit_time[N]) is the dimensionless complex transmitted envelope.
    kernel[N] contains antenna amplitude/phase, excluding exp(2 i k_R r).
    An arbitrary receive filter can subsequently act on the returned voltage.
    """
    xyz=np.asarray(positions_m);r=np.linalg.norm(xyz,axis=1)
    if np.any(r<=0):raise ValueError('Radar range must be positive')
    weights=np.asarray(volume_weights_m3)*np.asarray(kernel)*np.exp(4j*np.pi*radar_hz*r/c)
    result=[]
    for t in np.asarray(times_s):
        n=np.asarray(density(xyz,t-r/c))
        if np.iscomplexobj(n):raise ValueError('Use real density, not a peak phasor')
        result.append(np.sqrt(transmit_power_w)*np.sum(weights*n*envelope(t-2*r/c)))
    return np.asarray(result)


def gaussian_parameters(z_m, radar_hz=430e6, gain_db=61.1):
    z=np.asarray(z_m);lam=c/radar_hz;kr=2*np.pi/lam
    waist=np.sqrt(10**(gain_db/10))*lam/(np.sqrt(8)*np.pi)
    rayleigh=np.pi*waist**2/lam
    width=waist*np.sqrt(1+(z/rayleigh)**2)
    return width, z+rayleigh**2/z, np.arctan(z/rayleigh)


def gaussian_overlap(z_m, kx_rad_m, plasma_hz, branch=1, radar_hz=430e6,
                     gain_db=61.1, transverse_retardation=True):
    """Analytic transverse integral of the normalized Gaussian radar mode squared.

    Includes Fresnel phase and density retardation r/c through second order in
    transverse distance. Uses the same antenna mode at both sidebands (narrowband).
    branch=+1 is upshifted; -1 is downshifted. Full transverse beam filling assumed.
    """
    if branch not in (-1,1):raise ValueError('branch must be +1 or -1')
    z=np.asarray(z_m);w,R,g=gaussian_parameters(z,radar_hz,gain_db)
    a=2/w**2;b=2*np.pi*radar_hz/(c*R)
    if transverse_retardation:b=b+branch*np.pi*plasma_hz/(c*z)
    return a/(a-1j*b)*np.exp(-kx_rad_m**2/(4*(a-1j*b))-2j*g)


def sideband_amplitude(z_m, density_phasor_m3, plasma_hz, kx_rad_m=0.,
                       window=1., branch=1, reference_range_m=0.,
                       radar_hz=430e6, transmit_power_w=1e6, gain_db=61.1,
                       transverse_retardation=True):
    """Return sqrt(W) sideband coefficient for Re[n_tilde exp(i kx x-i omega t)].

    The carrier phase reference is reference_range_m. Apart from an irrelevant
    constant phase, s(t)=A_plus*exp(-i omega t)+A_minus*exp(+i omega t).
    window is an amplitude weight, not a power weight.
    """
    z=np.asarray(z_m);n=np.asarray(density_phasor_m3)
    if np.any(np.diff(z)<=0) or np.any(z<=0):raise ValueError('Positive increasing altitude required')
    if branch==-1:n=n.conj()
    overlap=gaussian_overlap(z,kx_rad_m,plasma_hz,branch,radar_hz,gain_db,transverse_retardation)
    k=(4*np.pi*radar_hz+branch*2*np.pi*plasma_hz)/c
    integral=trapezoid(.5*n*np.asarray(window)*overlap*np.exp(1j*k*(z-reference_range_m)),z)
    return np.sqrt(transmit_power_w)*RE*(c/radar_hz)*integral


def spectral_snr(signal_sqrt_w, sample_rate_hz, nfft, temperature_k=150., calibration=1.):
    """Rectangular-record spectrum; positive frequency means RF upshift.

    Uses a positive-exponent transform because the analytic carrier is exp(-i wt).
    Noise variance per complex sample is k_B*T*fs. Zero padding does not add energy.
    """
    s=np.asarray(signal_sqrt_w);n=len(s)
    if nfft<n:raise ValueError('nfft must retain all input samples')
    transform=np.fft.fftshift(np.fft.ifft(s,nfft)*nfft)
    freq=np.fft.fftshift(np.fft.fftfreq(nfft,1/sample_rate_hz))
    snr=calibration*abs(transform)**2/(n*kb*temperature_k*sample_rate_hz)
    return freq,snr
