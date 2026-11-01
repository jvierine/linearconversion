"""Independent numerical checks against recovered legacy and published formulas."""
from pathlib import Path
import importlib.util,json
import numpy as np
from scipy.integrate import simpson
import scipy.integrate as integrate
from scipy.constants import e,m_e,epsilon_0
from scipy.interpolate import PchipInterpolator
from scipy.special import jv
from isr.kinetic import *
from isr.suprathermal import *
# Compatibility shim only in this validation process; archive stays unchanged.
integrate.simps=lambda y,x:simpson(y,x=x)
spec=importlib.util.spec_from_file_location('legacy',Path('evidence/legacy_isr_spec/isr_spec.py'))
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
f=np.r_[np.linspace(-10000,-10,101),np.linspace(10,10000,101)]
par=dict(n_e=1e11,freq=430e6,t_i=[1000.],t_e=2000.,m_i=[16.],ion_fractions=[1.],alpha=90.,B=35e-6)
a=legacy.isr_spectrum(2*np.pi*f,plpar=par,n_points=6000,ni_points=12000)/par['n_e']
es,ions=oxygen_plasma();b=density_spectrum(f,radar_k(),es,ions)['total']
err=np.max(np.abs(a-b))/np.max(b)
assert err<1e-5,err
# Direct translation of the archive's harmonic susceptibility expression.
def archived_tail_im(w,k,B,th,E,p,eta,nmax=30):
 kp=k*np.cos(np.deg2rad(th));kt=k*np.sin(np.deg2rad(th));om=e*B/m_e
 EJ=E*e;F=p/(4*np.pi*e); interp=PchipInterpolator(EJ,F,extrapolate=False)
 val=0.; mh=0.
 for n in range(-nmax,nmax+1) if th else [0]:
  En=m_e/2*((w-n*om)/kp)**2
  aa=kt/om*np.sqrt(2*np.maximum(EJ-En,0)/m_e); mask=EJ>En
  if not mask.any(): continue
  mh+=2*np.pi**2/kp*eta*np.trapezoid(np.sqrt(m_e/(2*EJ))*jv(n,aa)**2*F*mask,EJ)
  if n:
   term=kt*kt/(2*n*om*om)*np.trapezoid((jv(n+1,aa)**2-jv(n-1,aa)**2)*F/np.sqrt(2*EJ)*mask,EJ)
  else:
   en0=m_e/2*(w/kp)**2
   ff=float(interp(en0)) if EJ[0]<=en0<=EJ[-1] else 0.
   ratio=np.divide(jv(1,aa),aa,out=np.full_like(aa,.5),where=aa!=0)
   term=-m_e*ff/np.sqrt(2*en0)+np.sqrt(2)*kt*kt/(om*om)*np.trapezoid(jv(0,aa)*ratio*F/np.sqrt(EJ)*mask,EJ)
  val+=term
 # conjugate reference's exp(+iwt) susceptibility to our convention
 return mh,-2*np.pi**2/(k*k*kp)*eta*np.sqrt(m_e)*w*val
E=np.arange(1,100.00001,.005);p=energy_distribution(E)
checks=[]
for angle in [0.,30.,60.,85.]:
 w=2*np.pi*6e6;k=radar_k()
 mh,im,_=tail_terms(w,k,35e-6,angle,E,p,1e-4)
 mh0,im0=archived_tail_im(w,k,35e-6,angle,E,p,1e-4)
 checks.append({'angle':angle,'noise_rel_error':abs(mh-mh0)/abs(mh),'susceptibility_rel_error':abs(im-im0)/abs(im)})
 # Original snaps the resonance lower bound to its energy grid; ours inserts it.
 assert checks[-1]['noise_rel_error']<.005,checks[-1]
 assert checks[-1]['susceptibility_rel_error']<.005,checks[-1]
report={'legacy_thermal_max_error_over_peak':float(err),'published_tail_formula_checks':checks,
        'note':'Checks formulas and numerical limits; not an event fit or exact reproduction of Figure 6.'}
Path('data/validation.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
