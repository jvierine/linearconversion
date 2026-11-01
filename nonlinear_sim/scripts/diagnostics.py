"""Compact HDF5 diagnostics of actual time-domain output (not a kinetic validation)."""
from pathlib import Path
import hashlib
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'
DEST=ROOT/'reference';DEST.mkdir(exist_ok=True)
def load(name):
 with h5py.File(OUT/(name+'.h5')) as h:
  s=h['state'][:];n=h['background_density_normalized'][:]
  harmonics=np.fft.rfft(s[:,:,:,0],axis=1)/s.shape[1]
  t=h['time_s'][:];phase=np.exp(-1j*t*h.attrs['omega_si'])
  exact=phase[:,None]*h['initial_phasor_normalized'][0][None,:]
  return dict(z=h['z_m'][:]-h['resonance_m'][()],n=n,time=t,cycles=t*h.attrs['omega_si']/(2*np.pi),
   density_m3=n*h.attrs['density_si'],real=s[-1,0,:,0]/n,
   fundamental=2*harmonics[-1,1]/n,
   initial=h['initial_phasor_normalized'][0]/n,
   fundamental_relative_error=np.linalg.norm(2*harmonics[:,1]-exact,axis=1)/np.linalg.norm(exact,axis=1),
   dc=np.max(abs(harmonics[:,0])/n,axis=1),second=np.max(2*abs(harmonics[:,2])/n,axis=1),
   audit=h['audit_gauss_divB_max_dn'][:],initial_peak=h.attrs['requested_initial_peak_dn_over_n'])
names=['weak_lifted','weak_linear','medium_full','strong_full','weak_refined_lifted','weak_long']
with h5py.File(DEST/'nonlinear_validation.h5','w') as h:
 h.attrs['electron_fluid_git']='https://github.com/jvierine/electron-fluid.git@379f4072919a30f866eada41e487a7a411663ec6'
 h.attrs['fdtd_git']='https://github.com/jvierine/fdtd.git@82877cec9a95b4151479703ef1f7357d520fce31'
 h.attrs['diagnostics_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 h.attrs['scope']='Nonlinear ten-moment Maxwell; harmonic initialization; zero heat flux and BGK bath. Not a kinetic or steady-state validation.'
 for name in names:
  if not (OUT/(name+'.h5')).exists():continue
  d=load(name);g=h.create_group(name)
  for key,v in d.items():g[key]=v
  g.attrs['source_sha256']=hashlib.sha256((OUT/(name+'.h5')).read_bytes()).hexdigest()
  print(name,'cycles',d['cycles'][-1], 'Gauss',d['audit'][:,0].max(), 'fundamental error',d['fundamental_relative_error'][-1], 'DC',d['dc'][-1], 'second',d['second'][-1])
 if 'weak_refined_lifted' in h:
  d=load('weak_refined_lifted');k=np.gradient(np.unwrap(np.angle(d['fundamental'])),d['z'])
  g=h['weak_refined_lifted'];g['phase_kz_rad_m']=k
  mask=(d['z']<-1)&(d['z']>-10);g['wavelength_m']=np.where(mask & (k<0),2*np.pi/abs(k),np.nan)
 # This compares normalized shapes; independent peak normalization does not
 # establish convergence of absolute power, absorbed fraction, or radar RCS.
 if all(n in h for n in ['weak_lifted','weak_refined_lifted']):
  a=load('weak_lifted');b=load('weak_refined_lifted');use=(b['z']>-5)&(b['z']<1)
  x=np.interp(b['z'][use],a['z'],a['fundamental'].real)+1j*np.interp(b['z'][use],a['z'],a['fundamental'].imag)
  h.attrs['normalized_complex_shape_relative_difference_near_conversion']=np.linalg.norm(x-b['fundamental'][use])/np.linalg.norm(b['fundamental'][use])
