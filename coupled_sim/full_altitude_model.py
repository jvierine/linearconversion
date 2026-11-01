"""ACF observation model for the native Arecibo pulse integrations.

The line calculation retains magnetized electron response and a mixture of
unmagnetized ions. Number-conserving neutral BGK is not a Coulomb operator.
All lags retain the archived integer-bin averages. Receiver response is the
power response of the 25-sample boxcar before 25 MHz -> 1 MHz decimation.
"""
import numpy as np
from scipy.constants import k as kb,e,epsilon_0,m_e,atomic_mass
from fit_eregion import K,response,neutral_rates,LAGS

MASSES=np.array([16.,1.,4.,30.,32.,14.])
def receiver_power(f):
 return (np.sinc(f/1e6)/np.sinc(f/25e6))**2

def spectrum(f,ti,te,ne,B,theta,ns,fractions,v=0.,electron=None,remove_background=False):
 w=2*np.pi*f-K*v;nue,nui=neutral_rates(te,*ns)
 ce,qe=response(w,ne,te,m_e,nue,B,theta) if electron is None else electron
 ci=np.zeros_like(ce);qi=np.zeros_like(qe)
 for mass,frac in zip(MASSES,fractions):
  if frac<1e-6:continue
  a,b=response(w,ne*frac,ti,mass*atomic_mass,nui)
  ci+=a;qi+=frac*b
 den=1+ce+ci
 s=abs(ce/den)**2*qi+abs((1+ci)/den)**2*qe
 return s-qe/abs(1+ce)**2 if remove_background else s

class FullModel:
 def __init__(self,mode,csp_lags=None,df=None,fmax=None,tail_step=500.,tail_max=2000000.):
  self.mode=mode
  self.lags=LAGS if mode=='clp' else [np.arange(4,40)*1e-6 if l==0 else (-l+np.arange(40))*1e-6 for l in csp_lags]
  df=df or (250. if mode=='clp' else 31.25);fmax=fmax or (128000. if mode=='clp' else 64000.)
  self.f=np.arange(-fmax,fmax+df*.1,df)
  self.kernel=np.array([np.exp(2j*np.pi*t[:,None]*self.f).mean(axis=0)*df for t in self.lags])
  # Exact trapezoid endpoints on the central band.
  self.kernel[:,[0,-1]]*=.5
  self.kernel*=receiver_power(self.f)
  if mode=='clp':
   # Electron susceptibility is smooth across the ion-line band. Interpolate
   # the response, not the sharply resonant total spectrum.
   self.nodes=np.cos(np.pi*(np.arange(16)+.5)/16)
   self.interp=np.polynomial.chebyshev.chebvander(self.f/fmax,15)@np.linalg.inv(np.polynomial.chebyshev.chebvander(self.nodes,15))
   self.node_f=self.nodes*fmax
   self.wide_f=np.arange(-4000000.,4000000.1,500.)
   self.wide_kernel=np.array([np.exp(2j*np.pi*t[:,None]*self.wide_f).mean(axis=0)*500 for t in self.lags])*receiver_power(self.wide_f)
   self.wide_kernel[:,[0,-1]]*=.5
  if mode=='csp':
   positive=np.arange(fmax,tail_max+tail_step*.1,tail_step)
   self.tail=np.r_[-positive[::-1],positive]
   weights=np.full(len(self.tail),tail_step,dtype=float);weights[[0,len(positive)-1,len(positive),-1]]*=.5
   self.tail_kernel=np.exp(2j*np.pi*self.lags[0][:,None]*self.tail).mean(axis=0)*weights*receiver_power(self.tail)
 def __call__(self,ti,te,v,ne,B,theta,ns,fractions):
  electron=None
  if self.mode=='clp':
   nue,_=neutral_rates(te,*ns)
   ce,qe=response(2*np.pi*self.node_f-K*v,ne,te,m_e,nue,B,theta)
   electron=(self.interp@ce,self.interp@qe)
  s=spectrum(self.f,ti,te,ne,B,theta,ns,fractions,v,electron,self.mode=='clp')
  _,nui=neutral_rates(te,*ns)
  exact=0.
  # Only the single-molecular-ion, strongly collisional D-region branch
  # needs removal of a sub-grid central Lorentzian.
  if self.mode=='csp' and nui>8*K*np.sqrt(kb*ti/(30*atomic_mass)):
   ae=ne*e**2/(epsilon_0*kb*te*K*K);ai=ne*e**2/(epsilon_0*kb*ti*K*K)
   d=K*K*kb*ti/(30*atomic_mass*nui);gamma=d*(1+ai/(1+ae));weight=(ae/(1+ae))**2*d/gamma
   w=2*np.pi*self.f-K*v
   s=s-weight*2*gamma/(w*w+gamma*gamma)
   exact=np.array([np.mean(weight*np.exp(-gamma*abs(t)+1j*K*v*t)) for t in self.lags])
  out=self.kernel@s+exact
  if self.mode=='clp':
   # Subtract the broad pure-electron background before the finite ion-band
   # transform, preventing a spurious truncated-continuum ACF. Its long-lag
   # value is negligible for warm electrons; retain it for cold tenuous plasma.
   blend=np.clip((600.-te)/200.,0,1)*np.clip((2e10-ne)/1e10,0,1)
   if blend>0:
    ce,qe=response(2*np.pi*self.wide_f-K*v,ne,te,m_e,nue,B,theta)
    out+=blend*(self.wide_kernel@(qe/abs(1+ce)**2))
  if self.mode=='csp':
   out[0]+=self.tail_kernel@spectrum(self.tail,ti,te,ne,B,theta,ns,fractions,v)
  return out
