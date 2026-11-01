"""Fit measured ACFs with MSIS temperature priors and neutral BGK collisions.

Outputs are conditional on molecular (NO+, 30 u) composition. A separate
metal-ion sensitivity fit is saved. Exact transmit ambiguity weights were not
archived: use the actual integer lag bins with equal weights and flag this
approximation. Source paths and assumptions remain in scientific metadata.
"""
from pathlib import Path
import argparse
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import h5py
from scipy.constants import e,m_e,atomic_mass,epsilon_0,k as kb,c
from scipy.special import wofz,ive
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).parent/'data/eregion_fit'
K=4*np.pi*430e6/c
F=np.arange(-64000.,64000.1,250.)
DF=F[1]-F[0]
LAGS=[np.arange(4,10)*1e-6]+[(np.arange(10)+10*i)*1e-6 for i in range(1,43)]
KERNEL=np.array([np.exp(2j*np.pi*t[:,None]*F[None,:]).mean(axis=0)*DF for t in LAGS])

def neutral_rates(te,n2,o2,o):
 return (2.33e-17*n2*(1-1.21e-4*te)*te+1.82e-16*o2*(1+.036*np.sqrt(te))*np.sqrt(te)
         +8.9e-17*o*(1+5.7e-4*te)*np.sqrt(te)),4.36e-16*n2+4.28e-16*o2+2.45e-16*o

def response(w,ne,t,m,nu,B=0.,theta=0.):
 sig=np.sqrt(kb*t/m);z=w+1j*nu
 if B:
  om=e*B/m;kp=K*np.cos(np.deg2rad(theta));kt=K*np.sin(np.deg2rad(theta))
  b=(kt*sig/om)**2;nmax=max(6,int(np.ceil(5*np.sqrt(b)+5)))
  n=np.arange(-nmax,nmax+1);weights=ive(n,b)
  while abs(weights.sum()-1)>1e-12:
   nmax+=2;n=np.arange(-nmax,nmax+1);weights=ive(n,b)
  a=np.sqrt(2)*kp*sig
  j=(weights[:,None]*np.sqrt(np.pi)/a*wofz((z[None,:]-n[:,None]*om)/a)).sum(axis=0)
 else:j=np.sqrt(np.pi)/(np.sqrt(2)*K*sig)*wofz(z/(np.sqrt(2)*K*sig))
 j=j/(1-nu*j)
 chi=ne*e**2/(epsilon_0*kb*t*K**2)*(1+1j*w*j)
 return chi,2*j.real

def model(ti,te,v,ne,B,theta,n2,o2,o,mass=30.):
 nu_e,nu_i=neutral_rates(te,n2,o2,o)
 w=2*np.pi*F-K*v
 ce,qe=response(w,ne,te,m_e,nu_e,B,theta)
 ci,qi=response(w,ne,ti,mass*atomic_mass,nu_i)
 eps=1+ce+ci
 s=abs(ce/eps)**2*qi+abs((1+ci)/eps)**2*qe
 return KERNEL@s

def fit_one(y,var,prior,ne,mass=30.):
 tn,B,theta,n2,o2,o=prior
 scale=max(float(y[0].real),1e-6)
 yn=y/scale;err=np.sqrt(var)/scale
 # Avoid treating an empirically underestimated single-bin variance as exact.
 err=np.maximum(err,.02)
 def physical(p):return np.exp(p[0]),tn*np.exp(p[1]),tn*np.exp(p[2]),p[3]
 def residual(p):
  amp,ti,te,v=physical(p)
  m=amp*model(ti,te,v,ne,B,theta,n2,o2,o,mass)
  d=(m-yn)/err
  # MSIS constrains neutral temperature, not electron or ion temperature exactly.
  return np.r_[d.real,d.imag,p[1]/.25,p[2]/np.log(3.)]
 guess=[np.log(2),0.,.1,0.]
 fit=least_squares(residual,guess,bounds=([-4.,np.log(.5),np.log(.5),-300.],[5.,np.log(3.),np.log(15.),300.]),max_nfev=70)
 amp,ti,te,v=physical(fit.x);pred=amp*model(ti,te,v,ne,B,theta,n2,o2,o,mass)*scale
 cov=np.linalg.pinv(fit.jac.T@fit.jac)
 stderr=np.sqrt(np.maximum(np.diag(cov),0))*np.array([amp*scale,ti,te,1])
 return np.array([amp*scale,ti,te,v]),stderr,pred,float(np.sum(residual(fit.x)[:-2]**2)/(2*len(y)-4)),fit.success

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args()
 with h5py.File(ROOT/'compact_inputs.h5') as h:
  a=h['acf/mean'][:];var=h['acf/component_variance'][:];t=h['acf/time_unix_s'][:];r=h['acf/range_km'][:]
 with h5py.File(ROOT/'priors.h5') as h:
  tn=h['Tn_K'][:];B=h['B_T'][:];theta=h['angle_k_B_deg'][:];n2=h['N2_m3'][:];o2=h['O2_m3'][:];o=h['O_m3'][:]
 shape=a.shape[:2];params=np.full(shape+(4,),np.nan);errors=params.copy();pred=np.full_like(a,np.nan);chi=np.full(shape,np.nan);success=np.zeros(shape,bool)
 ne=np.clip(a[:,:,0].real*r[None,:]**2*4.285964912280702e10,1e9,2e12)
 indices=[(i,j) for i in range(len(t)) for j in range(len(r)) if a[i,j,0].real>5*np.sqrt(var[i,j,0])]
 if args.preview:indices=[(i,int(np.argmax(a[i,:,0].real))) for i in [5,15,22,30,38]]
 with threadpool_limits(limits=1):
  for num,(i,j) in enumerate(indices):
   p=(tn[i,j],B[j],theta[j],n2[i,j],o2[i,j],o[i,j])
   params[i,j],errors[i,j],pred[i,j],chi[i,j],success[i,j]=fit_one(a[i,j],var[i,j],p,ne[i,j])
   if args.preview or num%100==0:print(num,len(indices),i,r[j],params[i,j],chi[i,j],flush=True)
 out=ROOT/('preview_fit.h5' if args.preview else 'plasma_fits.h5')
 with h5py.File(out,'w') as h:
  h['time_unix_s']=t;h['range_km']=r;h['parameters']=params;h['formal_sigma']=errors;h['model_acf']=pred;h['reduced_chi2']=chi;h['fit_success']=success;h['density_seed_m3']=ne
  h.attrs['parameter_order']='amplitude, Ti_K, Te_K, apparent_velocity_m_s'
  h.attrs['composition']='NO+ 30 u; conditional fit, not a measurement of composition'
  h.attrs['density_status']='amplitude fitted; legacy density only seeds dielectric response; plasma-line calibration pending'
  h.attrs['prior']='ln(Ti/Tn) sigma 0.25, ln(Te/Tn) sigma ln(3); MSIS 2.1'
  h.attrs['collision_model']='number-conserving neutral BGK, MSIS composition; no Coulomb collisions'
  h.attrs['ambiguity']='equal average over archived integer lag bins; original transmit spectral weights unavailable'
  h.attrs['velocity_sign']='apparent ACF phase velocity; absolute receiver Doppler sign not independently verified'
  h.attrs['ion_magnetic_approximation']='unmagnetized ion susceptibility; electron magnetization retained; validate against full model'
 print('saved',out,flush=True)

if __name__=='__main__':main()
