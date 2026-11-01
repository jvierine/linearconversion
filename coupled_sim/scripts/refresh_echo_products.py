"""Refresh paper amplitudes from the coherent observation operator, not plasma dynamics."""
import argparse,sys
from pathlib import Path
import numpy as np,h5py
from scipy.constants import k as kb
from coherent_echo import amplitudes,window,ROOT
from isr.coherent import sideband_amplitude
p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);a=p.parse_args()
echo=ROOT/'reference/coupled/coherent_echo.h5';summary=ROOT/'reference/coupled/summary.h5'
with h5py.File(echo) as e,h5py.File(summary,'r+') as h:
 newflux=e.attrs['incident_flux_W_m2'];oldflux=h.attrs['incident_flux_to_match_observed_W_m2'];scale=np.sqrt(newflux/oldflux)
 for key in ['density_real_ppm','density_amplitude_ppm','Ez_mV_m','cold_Ez_mV_m']:
  h['plot/'+key][:]*=scale
 for key in ['peak_Ez_mV_m','required_density_at_bragg_ppm']:h.attrs[key]*=scale
 h.attrs['incident_flux_to_match_observed_W_m2']=newflux
 h.attrs['snr_at_previous_incident_flux']=e.attrs['power_up_W_per_W_m2']*h.attrs['reference_incident_flux_W_m2']*e.attrs['ion_line_power_calibration']*e.attrs['record_duration_s']/(kb*e.attrs['temperature_k'])
 h['plot/spectrum_snr'][:]=np.interp(h['plot/spectrum_frequency_mhz'][:]*1e6,e['dense_frequency_hz'][:],e['dense_snr'][:])+e['fitted_background_snr'][()]
 h.attrs['radar_reference']='Kudeki 2026 ECE 458, Eqs. 5.49 and 5.51; Gaussian coherent reduction with longitudinal and transverse density retardation'
 h.attrs['echo_generator']='coupled_sim/scripts/coherent_echo.py'
 rows=['baseline','fine','finest','convergence','outer_mesh','larger_buffer','coupled','recorded']
 checks=h['checks'][:]
 for i,name in enumerate(rows):
  with h5py.File(a.model_dir/(name+'.h5')) as model:checks[i,3]=abs(amplitudes(model)[0])**2
 h['checks'][:]=checks;h.attrs['mesh_radar_power_relative_change']=abs(checks[2,3]/checks[3,3]-1)
 with h5py.File(a.model_dir/'recorded.h5') as model:
  z=model['z'][:];center=model.attrs['center_m']
  powers=np.array([abs(sideband_amplitude(z,model['density_phasor_m3'][:],model.attrs['frequency_hz'],float(model['kx_rad_m'][()]),window(z,center,lo,hi),reference_range_m=center))**2 for lo,hi in [(-80,20),(-100,30),(-110,40)]])
  h.attrs['radar_window_power_spread']=np.ptp(powers)/powers[1]
 print('Updated summary',newflux,h.attrs['required_density_at_bragg_ppm'],h.attrs['peak_Ez_mV_m'])
with h5py.File(a.model_dir/'density_csp_sensitivity.h5') as h,h5py.File(echo,'r+') as e:
 amp=amplitudes(h);flux=e.attrs['fitted_peak_snr']*kb*e.attrs['temperature_k']/(e.attrs['ion_line_power_calibration']*e.attrs['record_duration_s']*abs(amp[0])**2)
 e.attrs['csp_incident_flux_W_m2']=flux;print('CSP flux',flux)
