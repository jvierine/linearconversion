"""Independent ion-line comparisons for the full-altitude fit composition.

This checks spectrum normalization and mixture handling separately from the
lag operator. Existing magnetic/BGK checks are retained in isr_spec_validation.
"""
import sys,hashlib
from pathlib import Path
import numpy as np,h5py
base=Path(__file__).resolve().parent/'isr_spec_validation';sys.path.insert(0,str(base))
from compare import load_juha,vierinen,swoboda,error,JUHA_REV,JOHN_REV
sys.path.insert(0,str(base/'.cache/swoboda'))
from ISRSpectrum.ISRSpectrum import Specinit
from full_altitude_model import spectrum
juha=load_juha(base/'.cache/vierinen/isr_spec.py')
original=juha.gordoyev2
def longer(*args,**kwargs):
 kwargs['inf']=2*kwargs.get('inf',1e-3)
 return original(*args,**kwargs)
juha.gordoyev2=longer
f=np.arange(-40000.,40000.1,500.)
cases=[('molecular',1e11,300.,400.,[('NO+',30.,1.)],[0,0,0,1,0,0]),('f_region',4e11,800.,2000.,[('O+',16.,.95),('NO+',30.,.05)],[.95,0,0,.05,0,0]),('topside',1e10,2500.,3000.,[('O+',16.,.5),('H+',1.,.5)],[.5,.5,0,0,0,0])]
with h5py.File('data/eregion_fit/independent_fit_spectrum_checks.h5','w') as h:
 for name,ne,ti,te,ions,frac in cases:
  case=dict(ne=ne,ti=ti,te=te,radar_hz=430e6,B=0.,theta=47.,ions=ions,frequency_hz=f)
  a=spectrum(f,ti,te,ne,0,47,[0,0,0],frac);low=vierinen(juha,case,80000)[-1];b=vierinen(juha,case,160000)[-1];c=swoboda(Specinit,case)[-1]
  matrix=np.array([[error(x,y) for y in [a,b,c]] for x in [a,b,c]])
  g=h.create_group(name);g['frequency_Hz']=f;g['fit_spectrum']=a;g['Vierinen']=b;g['Swoboda']=c;g['pairwise_peak_relative_error']=matrix
  for k,v in dict(ne_m3=ne,Te_K=te,Ti_K=ti,vi_m_s=0.,B_T=0.,nu_en_s_1=0.,nu_ei_s_1=0.,angle_k_B_deg=47.,f_r_MHz=430.).items():g.attrs[k]=v
  g.attrs['vierinen_git']='https://github.com/jvierine/isr_analysis';g.attrs['vierinen_checkout']=JUHA_REV;g.attrs['swoboda_git']='https://github.com/jswoboda/ISRSpectrum';g.attrs['swoboda_checkout']=JOHN_REV
  g.attrs['fit_model_sha256']=hashlib.sha256(Path('full_altitude_model.py').read_bytes()).hexdigest()
  g.attrs['vierinen_time_window_multiplier']=2;g.attrs['vierinen_points']=160000;g.attrs['vierinen_quadrature_refinement_error']=error(low,b)
  print(name,matrix,flush=True)
  assert np.max(matrix)<1e-6
