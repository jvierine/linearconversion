"""Independent point-target, phase, sideband and spectral normalizations."""
import numpy as np
from scipy.constants import c,k as kb
from scipy.integrate import quad
from scipy.special import j0
from isr.coherent import (RE,SIGMA_E,electron_kernel,retarded_echo,
 gaussian_parameters,gaussian_overlap,sideband_amplitude,spectral_snr)

def test_single_electron_recovers_radar_equation():
 r=118e3;g=10**6.11;lam=c/430e6;area=lam**2*g/(4*np.pi);pt=1e6
 v=retarded_echo([0],[[0,0,r]],[1],lambda x,t:np.ones(len(t)),lambda t:np.ones(len(t)),electron_kernel(r,g,area),transmit_power_w=pt)[0]
 assert np.isclose(abs(v)**2,pt*g*area*SIGMA_E/(4*np.pi)**2/r**4,rtol=1e-13,atol=0)

def test_coherent_cancellation_and_density_squared_scaling():
 r=118e3;lam=c/430e6;xyz=np.array([[0,0,r],[0,0,r+lam/4]])
 f=lambda x,t:np.ones(len(t));a=lambda t:np.ones(len(t))
 v=retarded_echo([0],xyz,[1,1],f,a,[1,1],transmit_power_w=1)[0]
 assert abs(v)<1e-8
 v1=retarded_echo([0],xyz[:1],[1],f,a,[1],transmit_power_w=1)[0]
 v2=retarded_echo([0],xyz[:1],[2],f,a,[1],transmit_power_w=1)[0]
 assert np.isclose(abs(v2)**2,4*abs(v1)**2)

def test_retarded_density_selects_upshift_for_downward_wave():
 fr=430e6;fp=5.34e6;kp=(4*np.pi*fr+2*np.pi*fp)/c
 z=118e3+np.linspace(0,10,4001);dz=z[1]-z[0];weights=np.full(len(z),dz);weights[[0,-1]]*=.5
 xyz=np.column_stack([z*0,z*0,z]);t=np.arange(128)/(fp*16)
 s=retarded_echo(t,xyz,weights,lambda x,t:np.cos(-kp*(x[:,2]-118e3)-2*np.pi*fp*t),lambda t:np.ones(len(t)),np.ones(len(z)),radar_hz=fr,transmit_power_w=1)
 up=np.mean(s*np.exp(2j*np.pi*fp*t));down=np.mean(s*np.exp(-2j*np.pi*fp*t))
 assert np.isclose(abs(up),5.,rtol=1e-10)
 assert abs(down/up)<.01

def test_gaussian_overlap_matches_independent_radial_quadrature():
 z=118e3;kx=.032;fp=5.34e6;fr=430e6
 w,R,g=gaussian_parameters(z,fr);a=2/w**2
 for branch in [-1,1]:
  b=2*np.pi*fr/(c*R)+branch*np.pi*fp/(c*z)
  integrand=lambda rho:2*a*rho*np.exp(-(a-1j*b)*rho**2)*j0(kx*rho)*np.exp(-2j*g)
  num=quad(lambda r:integrand(r).real,0,6*w,epsabs=1e-10,limit=400)[0]+1j*quad(lambda r:integrand(r).imag,0,6*w,epsabs=1e-10,limit=400)[0]
  assert np.allclose(num,gaussian_overlap(z,kx,fp,branch),rtol=1e-8,atol=1e-10)

def test_rectangular_record_snr_and_parseval():
 fs=25e6;n=1100;nfft=2048;power=1e-13;freq=400*fs/nfft;t=np.arange(n)/fs
 s=np.sqrt(power)*np.exp(-2j*np.pi*freq*t);f,snr=spectral_snr(s,fs,nfft)
 assert np.isclose(snr.max(),power*(n/fs)/(kb*150),rtol=1e-12)
 assert np.isclose(f[np.argmax(snr)],freq)
 assert np.isclose(np.sum(snr)*fs/nfft,power/(kb*150),rtol=1e-12)

def test_phasor_sideband_has_half_amplitude():
 z=118e3+np.linspace(0,2,2001);fp=5.34e6;fr=430e6;kr=(4*np.pi*fr+2*np.pi*fp)/c
 n=np.exp(-1j*kr*(z-z[0]))/gaussian_overlap(z,0,fp)
 amplitude=sideband_amplitude(z,n,fp,reference_range_m=z[0],transmit_power_w=1)
 assert np.allclose(amplitude,RE*c/fr,rtol=1e-10,atol=0)
