"""Publication arrays from the optimized sustained coupled solution.
Time-height panels evaluate Re[n_tilde(z) exp(-i omega t)]; not startup growth.
"""
from pathlib import Path
import sys,hashlib
import h5py,numpy as np
from scipy.constants import c,e,epsilon_0,m_e,k as kb
from scipy.optimize import brentq
from scipy.interpolate import CubicHermiteSpline
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
src=ROOT/'output/coupled/minimum_incident/best_model.h5'
with h5py.File(src) as h,h5py.File(ROOT/'data/radar_calibration.h5') as cal,h5py.File(ROOT/'data/smooth_observed_layer_1525453430.h5') as profile:
 z=h['z'][:];rel=z-h.attrs['center_m'];scale=np.sqrt(h.attrs['incident_vertical_flux_W_m2']);freq=h.attrs['frequency_hz'];w=2*np.pi*freq
 dn=h['density_phasor_m3'][:]*scale;E=h['E_phasor_V_m'][:]*scale;n=h['ne'][:];gyro=h['gyro'][:]*c;kx=float(h['kx_rad_m'][()]);kB=(4*np.pi*430e6+w)/c
 p=CubicHermiteSpline(profile['model_height_m'][:],profile['model_ne_m3'][:],profile['model_dne_dz_m4'][:])
 kv=np.array([kx,0,-kB]);D=w*w*(w*w-gyro@gyro)/(w*w-(gyro@kv)**2/(kv@kv))
 match=brentq(lambda zz:p(zz)*e*e/(epsilon_0*m_e)+3*kb*300/m_e*(kv@kv)-D,h.attrs['center_m']-15,h.attrs['center_m'])
 # Phase rotation simply chooses time zero; it leaves all amplitudes unchanged.
 anchor=np.argmax(abs(dn)*(abs(rel)<15));phase=np.exp(-1j*np.angle(dn[anchor]));dn*=phase;E*=phase
 ze=np.linspace(-15,3,1801);zc=(ze[:-1]+ze[1:])/2
 te=np.linspace(0,3/freq,301);tc=(te[:-1]+te[1:])/2
 interp=lambda a:np.interp(zc,rel,a)
 density=interp(dn);timeheight=np.real(density[:,None]*np.exp(-1j*w*tc[None,:]))/1e6
 phasek=np.gradient(np.unwrap(np.angle(dn)),z)
 use=(rel>-12)&(rel<-.8);phase_match=rel[use][np.argmin(abs(phasek[use]+kB))]
 dest=REPO/'data/minimum_incident_plots.h5'
 with h5py.File(dest,'w') as out:
  out.attrs.update(dict(h.attrs));out.attrs.update(bragg_height_relative_m=match-h.attrs['center_m'],solved_phase_bragg_m=phase_match,bragg_wavelength_m=2*np.pi/kB,
     density_at_bragg_m3=float(abs(np.interp(match,z,dn))),density_fraction_at_bragg=float(abs(np.interp(match,z,dn))/p(match)),
     time_height_definition='Real part of sustained harmonic density at x=0; arbitrary time origin; no growth or startup implied',model_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
  out['plot/z_edges_m']=ze;out['plot/time_edges_us']=te*1e6;out.create_dataset('plot/density_timeheight_1e6_m3',data=timeheight,compression='gzip')
  out['plot/height_m']=zc;out['plot/Ez_mV_m']=np.interp(zc,rel+h['dz_cells'][:]/2,abs(E[2]))*1e3
  out['plot/density_real_1e6_m3']=density.real/1e6;out['plot/density_amplitude_1e6_m3']=abs(density)/1e6
  out['plot/wavelength_m']=2*np.pi/(-np.interp(zc,rel,phasek))
  of=cal['line_frequency_hz'][:];out['plot/observed_frequency_mhz']=of/1e6;out['plot/observed_snr']=cal['line_measured_snr'][:]
  ff=np.linspace(5.27e6,5.41e6,1201);out['plot/model_frequency_mhz']=ff/1e6
  out['plot/model_snr']=cal.attrs['fitted_on_frequency_snr']*np.sinc((ff-freq)*cal.attrs['pulse_duration_s'])**2+cal['line_fit_fixed'][2]
  with h5py.File(ROOT/'output/coupled/minimum_incident/scan.h5') as scan:
   for key in ['angle_deg','incident_vertical_flux_W_m2','incident_reference_E_rms_V_m','dz_m']:out['scan/'+key]=scan[key][:]
  out['provenance/generator']=Path(__file__).read_text()
  print(dict(out.attrs));print('max density /1e6',abs(timeheight).max())
