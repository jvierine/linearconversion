"""Observed plasma-line map with calibrated ion-line and smooth-model fpe.

Provenance: archived lp_plot_compare.py motivated the display-only background
subtraction. Signed bins are mapped to |f| explicitly, avoiding the legacy
one-bin reversal offset. Thermal/magnetic frequency corrections are NOT added
to fpe: both overlay curves are sqrt(ne e^2 / epsilon0 me)/(2 pi).
"""
from pathlib import Path
import argparse
import h5py
import numpy as np
from scipy.constants import e,m_e,epsilon_0
from scipy.ndimage import uniform_filter,gaussian_filter
from isr.observed_layer import fit_observed_layer,save_profile,scattering_profile

ROOT=Path(__file__).resolve().parent


def prepare(epoch):
    f=fit_observed_layer(epoch)
    save_profile(f,ROOT/f'data/smooth_observed_layer_{epoch}.h5')
    with h5py.File(ROOT/f'evidence/seid_event/results4/spl-{epoch}.h5') as h:
        power=h['PL'][:];frequency=h['freqs'][:];height=h['range'][:];time=float(h['t0'][()])
    excess=power-power[:10].mean(axis=0)
    # Display-only high pass exposes narrow lines, preserving raw power in HDF5.
    highpass=excess-uniform_filter(excess,size=(21,21),mode='wrap')
    displayed=gaussian_filter(highpass,sigma=(.5,.7))
    fi=np.where((frequency<=-2)&(frequency>=-8))[0][::-1]
    zi=np.where((height>=99.9)&(height<=130.1))[0]
    center,width=f['es_center_m'],f['es_sigma_m']
    simulation_z=np.linspace(center-5*width,center+5*width,10001)
    simulation_n,_=scattering_profile(simulation_z,f['coefficients'],f['knots_km'])
    model_height=f['model_height_m'].copy();model_ne=f['model_ne_m3'].copy()
    use=(model_height>=simulation_z[0])&(model_height<=simulation_z[-1])
    model_ne[use]=np.interp(model_height[use],simulation_z,simulation_n)
    fpe=lambda n: np.sqrt(n*e*e/(epsilon_0*m_e))/(2*np.pi*1e6)
    path=ROOT/f'data/conversion_overlay_{epoch}.h5'
    with h5py.File(path,'w') as h:
        for k,v in dict(frequency_mhz=-frequency[fi],height_km=height[zi],
            displayed_power=displayed[np.ix_(zi,fi)]/1e14,unfiltered_excess=highpass[np.ix_(zi,fi)],raw_power=power[np.ix_(zi,fi)],
            ion_height_km=f['source_height_m']/1000,ion_fpe_mhz=fpe(f['source_ne_m3']),
            model_height_km=model_height/1000,model_fpe_mhz=fpe(model_ne),
            model_ne_m3=model_ne,simulation_height_m=simulation_z,simulation_ne_m3=simulation_n,
            spline_knots_km=f['knots_km'],spline_coefficients=f['coefficients']).items():h[k]=v
        h.attrs.update(plasma_line_time_unix_s=time,ion_line_time_unix_s=f['time_unix_s'],
            ion_line_time_offset_s=f['time_unix_s']-time,
            plasma_line_source=f'evidence/seid_event/results4/spl-{epoch}.h5',
            ion_line_source=f['source_path'],density_source_sha256=f['source_sha256'],
            calibration_gain=f['calibration_gain'],calibration_log_scatter=f['calibration_log_scatter'],
            map_processing='subtract first ten range-bin means per frequency, then 21x21 bin boxcar high pass; display Gaussian sigma 0.5 range bins (75 m), 0.7 frequency bins (8.54 kHz); unsigned display clips negative power; raw and unsmoothed excess retained',
            color_transform='10 log10(1 + max(displayed_power,0)*100); reference power 1e12 archive units; 0..15 dB for 19:03, 0..40 dB for 17:04 to preserve enhancement dynamic range',
            frequency_axis='absolute negative-frequency sideband; physical receiver Doppler sign unverified',
            model=f['model'],model_boundaries='same C2 reservoir continuation as conversion_lab.py; full underlying density model shown outside scattering domain',
            provenance='prepare_conversion_overlay.py; isr/observed_layer.py; plotter/src/bin/conversion_overlay.rs',
            caveat='fpe overlays are plasma frequency, not predicted warm magnetized thermal spectral peak; time pairing uses nearest ~80.4 s cycle')
    print(path, 'time offset',f['time_unix_s']-time,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=int,default=1525460596)
    prepare(p.parse_args().epoch)
