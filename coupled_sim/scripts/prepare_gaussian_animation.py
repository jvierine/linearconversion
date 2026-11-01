"""Reconstruct a transient and save physical observables for MP4 rendering."""
from pathlib import Path
import numpy as np,h5py,argparse
from scipy.fft import fft
from scipy.interpolate import CubicSpline
from scipy.constants import epsilon_0,m_e,e,k as kb,c
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/gaussian_pulse'

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--dz",type=float,default=.002);dz=parser.parse_args().dz
 with h5py.File(OUT/f'response_dz{dz:g}_z60.h5') as h:
  assert h.attrs['complete']
  meta=dict(h.attrs);df=float(meta['frequency_spacing_hz']);f0=float(meta['carrier_hz']);period=1/df
  z=h['height_m'][:];ne=h['ne_m3'][:]
  ov=slice(*h['views/overview'][:]);bo=slice(*h['views/bottomside'][:])
  indices=np.r_[np.arange(len(z))[ov],np.arange(len(z))[bo][(z[bo]-meta['bottom_resonance_m']>=-50)&(z[bo]-meta['bottom_resonance_m']<=50)],np.arange(len(z))[bo][::20]]
  selected_z=z[indices];selected_ne=ne[indices]
  n_overview=ov.stop-ov.start;n_close=np.sum((z[bo]-meta['bottom_resonance_m']>=-50)&(z[bo]-meta['bottom_resonance_m']<=50))
  offsets=np.arange(-160000.,160000.+df/2,df)
  spectrum=np.stack([h[f'response/{freq:+.3f}'][:][:,indices] for freq in offsets])
 # Each step advances the actual carrier phase by 1/24 cycle modulo whole
 # periods. The time display reports those actual times; stroboscopic sampling
 # makes the carrier phase progression visible without plotting every cycle.
 times=[30e-6]
 while times[-1]<.003:
  cycles=1 if times[-1]<140e-6 else 10 if times[-1]<700e-6 else 100
  times.append(times[-1]+(cycles+1/24)/f0)
 times=np.asarray(times)
 nf=16384;fft_time=np.arange(nf)*period/nf
 sigma=meta['field_envelope_FWHM_s']/np.sqrt(8*np.log(2));t0=meta['source_peak_time_s']
 weights=df*sigma*np.sqrt(2*np.pi)*np.exp(-.5*(2*np.pi*offsets*sigma)**2)*np.exp(2j*np.pi*offsets*t0)
 modes=np.rint(offsets/df).astype(int)%nf
 path=OUT/'movie.h5'
 with h5py.File(path,'w') as out:
  out.attrs.update(meta);out.attrs['time_sampling']='Stroboscopic carrier phase, actual physical time in labels; carrier advances 1/24 cycle per frame modulo whole periods'
  out['time_s']=times;out['height_m']=selected_z;out['ne_m3']=selected_ne
  out['views/overview']=[0,n_overview];out['views/closeup']=[n_overview,n_overview+n_close];out['views/wave_path']=[n_overview+n_close,len(indices)]
  fields={name:out.create_dataset(name,shape=(len(times),len(indices)),dtype='complex64',compression='lzf') for name in ['Ex','Ey','Ez','Jx','Jy','Jz','density']}
  fluxes={name:out.create_dataset(name,shape=(len(times),len(indices)),dtype='f4',compression='lzf') for name in ['EM_flux','pressure_flux']}
  peaks=np.zeros(nf);pre=0.;maximum=0.
  for start in range(0,len(indices),256):
   stop=min(start+256,len(indices));n=selected_ne[start:stop]
   fs=np.zeros((nf,10,stop-start),np.complex64);fs[modes]=spectrum[:,:,start:stop]*weights[:,None,None]
   envelopes=fft(fs,axis=0,workers=1);del fs
   # Monitor electric energy amplitude and density amplitude over the entire
   # period; all outputs use the same transform and absolute normalization.
   norm=np.max(abs(envelopes[:,2]),axis=1);peaks=np.maximum(peaks,norm)
   signal=CubicSpline(fft_time,envelopes,axis=0)(times)
   for j,name in enumerate(['Ex','Ey','Ez']):fields[name][:,start:stop]=signal[:,j]
   for j,name in enumerate(['Jx','Jy','Jz']):fields[name][:,start:stop]=signal[:,6+j]*e*np.sqrt(epsilon_0*n/m_e)
   dn=signal[:,9]*np.sqrt(epsilon_0*n/(3*kb*300.))
   fields['density'][:,start:stop]=dn
   fluxes['EM_flux'][:,start:stop]=.5*epsilon_0*c*np.real(signal[:,0]*signal[:,4].conj()-signal[:,1]*signal[:,3].conj())
   uz=-signal[:,8]*np.sqrt(epsilon_0/(m_e*n))
   fluxes['pressure_flux'][:,start:stop]=.5*np.real(3*kb*300*dn*uz.conj())
   print('reconstructed',stop,'/',len(indices),flush=True)
  out['validation/time_s']=fft_time;out['validation/peak_Ez_V_m']=peaks
  pre=np.max(peaks[fft_time<10e-6])/peaks.max();tail=np.max(peaks[fft_time>.003])/peaks.max()
  out.attrs.update(prepulse_relative_Ez=pre,late_relative_Ez=tail)
  first=times<75e-6
  amplitude=np.sqrt(sum(abs(fields[name][:,0])**2 for name in ['Ex','Ey','Ez']))
  tt=times[first];aa=amplitude[first];half=aa.max()/2
  crossing=np.where((aa[:-1]-half)*(aa[1:]-half)<0)[0]
  assert len(crossing)==2
  crossings=[tt[i]+(tt[i+1]-tt[i])*(half-aa[i])/(aa[i+1]-aa[i]) for i in crossing]
  width=crossings[1]-crossings[0]
  out.attrs['incident_field_envelope_FWHM_s']=width
  assert abs(width/meta['field_envelope_FWHM_s']-1)<.03,width
  out['provenance/prepare']=Path(__file__).read_text()
  print('TRANSIENT CHECK',pre,tail,'frames',len(times),flush=True)
  assert pre<.002,('periodic wraparound too large',pre)
  assert tail<.002,('extend decay time',tail)
if __name__=='__main__':main()
