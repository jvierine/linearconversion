"""Synthesize a radar record from the saved coupled density using Kudeki 5.49.
Source is a sustained harmonic solution, not extrapolation of a transient run.
"""
from pathlib import Path
import argparse,hashlib,sys
import h5py
import numpy as np
from scipy.constants import k as kb
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from isr.coherent import sideband_amplitude,spectral_snr

def window(z,center,lo=-100,hi=30):
 d=z-center
 return np.sin(.5*np.pi*np.clip(np.minimum(d-lo,hi-d)/10,0,1))**2

def amplitudes(h,transverse_retardation=True):
 z=h['z'][:];center=h.attrs['center_m'];n=h['density_phasor_m3'][:]
 return np.array([sideband_amplitude(z,n,h.attrs['frequency_hz'],float(h['kx_rad_m'][()]),window(z,center),branch,center,transverse_retardation=transverse_retardation) for branch in [1,-1]])

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'reference/coupled/coherent_echo.h5');args=p.parse_args()
 with h5py.File(args.model) as model_file,h5py.File(ROOT/'coupled_sim/data/radar_calibration.h5') as cal:
  src=model_file['model_input'] if 'model_input' in model_file else model_file
  amp=amplitudes(src);legacy=amplitudes(src,False)
  target=cal.attrs['fitted_on_frequency_snr'];C=cal.attrs['ion_line_measured_over_predicted'];tau=cal.attrs['pulse_duration_s'];T=cal.attrs['system_temperature_K'];fp=src.attrs['frequency_hz']
  flux=target*kb*T/(C*tau*abs(amp[0])**2);amp_scaled=amp*np.sqrt(flux)
  fs=25e6;time=np.arange(round(tau*fs))/fs
  signal=amp_scaled[0]*np.exp(-2j*np.pi*fp*time)+amp_scaled[1]*np.exp(2j*np.pi*fp*time)
  freq,snr=spectral_snr(signal,fs,2048,T,C);dense_f,dense_snr=spectral_snr(signal,fs,32768,T,C)
  args.output.parent.mkdir(parents=True,exist_ok=True)
  with h5py.File(args.output,'w') as h:
   h.attrs.update(reference='Kudeki (2026), ECE 458 Lecture Notes on Applications of Radio Wave Propagation, Eqs. 5.49 and 5.51, section 5.3.4',
    notes_version='458sp26draft15.pdf; received 2026-10-27',notes_sha256='30a1fb450e398ceb5a1902445a181c79e1f8cf0db71cae76896296ac45ce432f',model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    incident_flux_W_m2=flux,legacy_power_up_W_per_W_m2=abs(legacy[0])**2,
    power_up_W_per_W_m2=abs(amp[0])**2,power_down_W_per_W_m2=abs(amp[1])**2,
    transverse_retardation_power_ratio=abs(amp[0]/legacy[0])**2,
    down_over_up_db=20*np.log10(abs(amp[1]/amp[0])),radar_hz=430e6,plasma_hz=fp,
    sample_rate_hz=fs,record_duration_s=tau,ion_line_power_calibration=C,temperature_k=T,
    fitted_peak_snr=target,native_bin_peak_snr=snr.max(),
    scope='Coherent beam-filling single horizontal mode; harmonic continuation; effective rectangular record; no thermal plasma-line background or measured code/receiver response; amplitude fitted to data; narrowband antenna mode shared by sidebands')
   h['sideband_amplitudes_sqrt_W']=amp_scaled;h['sideband_offsets_hz']=[fp,-fp]
   h['time_s']=time;h['echo_sqrt_W']=signal;h['frequency_hz']=freq;h['snr']=snr
   h['dense_frequency_hz']=dense_f;h['dense_snr']=dense_snr
   h['plot/time_us']=time*1e6;h['plot/I_uV']=np.sqrt(100)*signal.real*1e6;h['plot/Q_uV']=np.sqrt(100)*signal.imag*1e6
   h['plot/dense_frequency_mhz']=dense_f/1e6;h['plot/model_snr']=dense_snr+cal['line_fit_fixed'][2]
   h['plot/observed_frequency_mhz']=cal['line_frequency_hz'][:]/1e6
   h['observed_frequency_hz']=cal['line_frequency_hz'][:];h['observed_snr']=cal['line_measured_snr'][:];h['fitted_background_snr']=cal['line_fit_fixed'][2]
   # Compact original simulation input makes the radar calculation reproducible.
   g=h.create_group('model_input');z=src['z'][:];mask=(z-src.attrs['center_m']>=-110)&(z-src.attrs['center_m']<=40)
   for key in ['z','density_phasor_m3']:g[key]=src[key][:][mask]
   g['kx_rad_m']=src['kx_rad_m'][()]
   g.attrs['center_m']=src.attrs['center_m'];g.attrs['frequency_hz']=fp
   print({k:h.attrs[k] for k in ['incident_flux_W_m2','transverse_retardation_power_ratio','down_over_up_db','native_bin_peak_snr']})
if __name__=='__main__':main()
