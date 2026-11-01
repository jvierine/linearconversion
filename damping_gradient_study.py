"""Separated loss diagnostics and fixed-illumination gradient counterfactual.

Vertical stretching n_g(z)=n_ref(z_res+g*z) changes dn/dz by g, preserves
all density values, and anchors the cold plasma-resonance level at z=0.
The same incident flux, modal weights, kx, B, Te and receiver gate are used.
Reference normalization to the measured 5.34 MHz enhancement removes an
unknown constant power calibration, not uncertainty in incident illumination.
"""
import gc
import h5py
import numpy as np
from scipy.constants import c,e,m_e,epsilon_0
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid
from isr.conversion import Parameters,matrix,scatter,physical_density
from isr.observed_layer import fit_observed_layer,scattering_profile,profile
from conversion_lab import coherent_gate
from deconvolve_plasma_line import load

MODES=('collisional','collisionless','combined')

def local_losses():
 with h5py.File('data/landau_bragg_study.h5','a') as s:
  for epoch in s:
   for freq in s[epoch]:
    g=s[epoch+'/'+freq]
    with h5py.File('data/conversion_runs/'+g.attrs['full_wave_solution_id']+'.h5') as h:
     par=Parameters(frequency_hz=float(freq)*1e6,temperature_k=300,magnetic_t=35.025e-6,magnetic_tilt_deg=46.815-h.attrs['layer_tilt_deg'],q=np.sin(np.deg2rad(-20)))
     pr=h['kinetic_branch/density_ratio'][:];xx=h['kinetic_branch/pressure_response'][:]
     p=g['density_ratio'][:];z=g['height_relative_resonance_m'][:]
     xi=PchipInterpolator(pr,xx.real)(np.clip(p,pr[0],pr[-1]))+1j*PchipInterpolator(pr,xx.imag)(np.clip(p,pr[0],pr[-1]))
     for mode in MODES:
      nu=0 if mode=='collisionless' else 1000.
      x=xi.real if mode=='collisional' else xi
      mu=np.linalg.eigvals(matrix(p,np.zeros_like(p),np.full_like(p,nu/par.omega),par,x))
      j=np.argmin(mu.imag,axis=1);kd=1j*par.k0*mu[np.arange(len(p)),j]
      alpha=np.maximum(kd.imag,0)
      tau=-cumulative_trapezoid(alpha[::-1],z[::-1],initial=0)[::-1]
      for name,value in dict(alpha_m1=alpha,optical_depth=tau,amplitude_survival=np.exp(-tau),power_survival=np.exp(-2*tau)).items():
       key=f'damping/{mode}/{name}'
       if key in g:del g[key]
       g[key]=value
      g.attrs[mode+'_power_survival_to_bragg']=float(np.exp(-2*np.interp(-g.attrs['resonance_to_bragg_m'],z,tau)))
      print('LOSS',epoch,freq,mode,g.attrs[mode+'_power_survival_to_bragg'],flush=True)
    g['damping'].attrs.update(collision_rate_s=1000.,interpretation='Local full-wave eigenvalue attenuation integrated along the profile; excludes conversion efficiency, focusing and interference. Common real kinetic dispersion, loss mechanisms switched separately. Near conversion this is a diagnostic; the driven boundary solution is authoritative.')

def gradient_scan():
 epoch=1525453430;fit=fit_observed_layer(epoch);res=117941.5654266378
 with h5py.File('data/conversion_runs/89e5c8fa00707a48.h5') as h:
  par=Parameters(frequency_hz=5.33e6,temperature_k=300,magnetic_t=35.025e-6,magnetic_tilt_deg=46.815-h.attrs['layer_tilt_deg'],q=np.sin(np.deg2rad(-20)))
  pr=h['kinetic_branch/density_ratio'][:];xx=h['kinetic_branch/pressure_response'][:];weights=h['incoming_weights'][:]
 with h5py.File('data/landau_bragg_study.h5') as h:zb=float(h['1525453430/5.33'].attrs['bragg_height_m'])
 lo=fit['es_center_m']-5*fit['es_sigma_m'];hi=fit['es_center_m']+5*fit['es_sigma_m']
 nB,dnB,*_=profile(zb,fit['coefficients'],fit['knots_km'])
 f,zz,power,bg,sigma,_=load(epoch);j=np.argmin(abs(f-5.34));region=(zz>117)&(zz<119)
 observed=float(np.max((power-bg)[region,j]));noise=float(sigma[j]);threshold=5*noise/observed
 out='data/gradient_detectability.h5'
 with h5py.File(out,'w') as h:
  h.attrs.update(reference_density_gradient_m4=float(dnB),reference_density_m3=float(nB),
   reference_scale_height_m=float(nB/dnB),reference_bragg_height_m=zb,reference_resonance_height_m=res,
   observed_peak_excess_power=observed,noise_sigma_power=noise,detection_threshold_fraction=threshold,
   detection_definition='5 times robust off-source per-frequency power scatter, relative to the measured enhancement peak; assumed same integration and receiver response',
   normalization='same unknown incident HF flux for every case, fixed reference optimized modal weights; received power normalized to combined-loss g=1 prediction and measured enhancement',
   counterfactual='vertical stretch of the full fitted layer about its plasma resonance, n_g(z)=n_ref(z_res+g*z); derivative reduced by g',
   limitation='conditional coherent vertical Gaussian gate and fixed illumination; not a universal Es-only criterion or absolute transmitter-power estimate',
   provenance='damping_gradient_study.py; isr/conversion.py fourth-order constant-cell Pade propagation')
  h['incident_weights']=weights;h['gate_sigma_m']=[75.,95.,150.]
 for factor in [1.,.5,.25,.125,.08]:
  dz=.02;z=np.linspace((lo-res)/factor,(hi-res)/factor,round((hi-lo)/(factor*dz))+1);base=res+factor*z
  ne,dn=scattering_profile(base,fit['coefficients'],fit['knots_km']);ratio=ne*e*e/(epsilon_0*m_e*par.omega**2);grad=factor*dn/ne
  u=np.clip(np.minimum(base-base[0],base[-1]-base)/400,0,1);taper=10*u**3-15*u**4+6*u**5;nu=1000*taper;nu[[0,-1]]=0
  x=PchipInterpolator(pr,xx.real)(np.clip(ratio,pr[0],pr[-1]))+1j*PchipInterpolator(pr,xx.imag)(np.clip(ratio,pr[0],pr[-1]))
  x=1+taper*(x-1);x[[0,-1]]=1.;center=(zb-res)/factor
  for mode in MODES:
   with h5py.File(out,'r') as h:
    if f'{factor:.3f}/{mode}' in h:continue
   r=scatter(z,ratio,grad,nu*int(mode!='collisionless'),par,pressure_response=x.real if mode=='collisional' else x,cell_order=4,incident_weights=weights)
   density=physical_density(r['h'],par,1e-6);amps=[]
   for sig in [75.,95.,150.]:amps.append(coherent_gate(z,ne,density,center,sig,4*np.pi*430e6/c)[0])
   win=np.linspace(center-10,center+10,8001)
   with h5py.File(out,'a') as h:
    g=h.create_group(f'{factor:.3f}/{mode}');g['radar_amplitude']=amps;g['height_relative_bragg_m']=win-center;g['density_fraction']=np.interp(win,z,density)
    g.attrs.update(gradient_factor=factor,density_gradient_m4=float(factor*dnB),scale_height_m=float(nB/dnB/factor),dz_m=dz,cells=len(z),bragg_z_m=center,
       **{k:r[k] for k in ['reflection','conversion','absorption_collision','absorption_landau','energy_error']})
   print('SCAN',factor,mode,abs(amps[1])**2,'energy',r['energy_error'],flush=True)
   del r,density;gc.collect()
  del z,ne,ratio,grad,x;gc.collect()
 with h5py.File(out,'a') as h:
  reference=abs(h['1.000/combined/radar_amplitude'][:])**2
  for fac in [1.,.5,.25,.125,.08]:
   for mode in MODES:
    g=h[f'{fac:.3f}/{mode}'];g['relative_received_power']=abs(g['radar_amplitude'][:])**2/reference
    print('REL',fac,mode,g['relative_received_power'][:],flush=True)
if __name__=='__main__':
 local_losses();gradient_scan()
