"""Check saved optimal-model radar normalization, Bragg phase and mesh result."""
from pathlib import Path
import sys
import h5py,numpy as np
from scipy.constants import c,k
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from isr.coherent import sideband_amplitude
with h5py.File(ROOT/'reference/coupled/minimum_incident.h5') as h,h5py.File(ROOT/'coupled_sim/data/radar_calibration.h5') as cal:
 z=h['z'][:];rel=z-h.attrs['center_m'];window=np.sin(.5*np.pi*np.clip(np.minimum(rel+100,30-rel)/10,0,1))**2
 amp=sideband_amplitude(z,h['density_phasor_m3'][:],h.attrs['frequency_hz'],float(h['kx_rad_m'][()]),window,reference_range_m=h.attrs['center_m'])
 snr=abs(amp)**2*h.attrs['incident_vertical_flux_W_m2']*cal.attrs['ion_line_measured_over_predicted']*cal.attrs['pulse_duration_s']/(k*cal.attrs['system_temperature_K'])
 np.testing.assert_allclose(snr,cal.attrs['fitted_on_frequency_snr'],rtol=1e-10)
 print('Saved-density calibrated SNR:',snr)
with h5py.File(ROOT/'data/minimum_incident_plots.h5') as h:
 z=h['plot/height_m'][:];wave=h['plot/wavelength_m'][:];idx=np.argmin(abs(z-h.attrs['bragg_height_relative_m']))
 assert abs(wave[idx]/h.attrs['bragg_wavelength_m']-1)<.01
 assert wave[np.argmin(abs(z+12))]<wave[np.argmin(abs(z+8))]<wave[np.argmin(abs(z+4))]<wave[np.argmin(abs(z+2))]
 assert h.attrs['flux_mesh_relative_change']<.001
 assert np.isfinite(h['plot/density_timeheight_1e6_m3'][:]).all()
 assert np.all(h['scan/incident_vertical_flux_W_m2'][:]>0)
 assert h.attrs['incident_vertical_flux_W_m2']<=h['scan/incident_vertical_flux_W_m2'][:].min()*(1+1e-10)
 print('Bragg match, downward wavelength contraction, finite map and mesh check passed')
