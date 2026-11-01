"""Observed plasma-line map with magnetized, thermal resonance overlays.

Provenance: archived lp_plot_compare.py motivated the display-only background
subtraction. Signed bins are mapped to |f| explicitly, avoiding the legacy
one-bin reversal offset. The displayed curves solve the real Maxwellian electron dielectric root
with the IGRF14 field and the upshifted 430 MHz Bragg wavenumber.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import ppigrf
import h5py
import numpy as np
from scipy.constants import e,m_e,epsilon_0
from scipy.ndimage import uniform_filter,gaussian_filter
from isr.plasma_line_frequency import profile_hz,dielectric
from isr.observed_layer import fit_observed_layer,save_profile,scattering_profile

ROOT=Path(__file__).resolve().parent


def add_snr(epoch):
    """Use the same off-range noise estimator as the paired sideband spectra."""
    source=ROOT/f'evidence/seid_event/results4/spl-{epoch}.h5'
    with h5py.File(source) as src:
        power=src['PL'][:];frequency=src['freqs'][:];height=src['range'][:]
    noise=np.median(power[(height<105)|(height>130)],axis=0)
    if not np.all(np.isfinite(noise)&(noise>0)):
        raise ValueError('Invalid spectral noise reference')
    fi=np.where((frequency<=-2)&(frequency>=-8))[0][::-1]
    zi=np.where((height>=99.9)&(height<=130.1))[0]
    snr=power[np.ix_(zi,fi)]/noise[fi][None,:]-1
    with h5py.File(ROOT/f'data/conversion_overlay_{epoch}.h5','a') as dst:
        for key,value in [('spectral_snr',snr),('noise_power',noise[fi])]:
            if key in dst:del dst[key]
            dst[key]=value
        dst['spectral_snr'].attrs['definition']='(P-N)/N; signed spectral excess; no high-pass filter or smoothing'
        dst['noise_power'].attrs['estimator']='median received power over ranges below 105 km or above 130 km, independently at each frequency'
        dst.attrs['article_color_transform']='10 log10(max(spectral_snr,10**(-1.5))); limits -15 to 20 dB; nonpositive and sub-floor values use lowest color'
        dst.attrs['snr_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()


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
    # Use the same acquisition cycle as the density fit, retaining measured Te.
    with h5py.File(f['source_path']) as src:
        zfit=src['height_km'][:]
        tefit=src['results'][f['source_time_index'],:,1]
    ion_z=f['source_height_m']/1000
    ion_te=np.interp(ion_z,zfit,tefit)
    if not np.isfinite(ion_te).all() or np.any(ion_te<=0):
        raise ValueError('Invalid fitted electron temperature in overlay')
    # Same vertical range-as-altitude convention as the density and simulation.
    date=datetime.datetime.fromtimestamp(time,datetime.timezone.utc).replace(tzinfo=None)
    coeff=Path(ppigrf.__file__).parent/'IGRF14.shc'
    magnetic={}
    for prefix,z,ne,te in [('ion',ion_z,f['source_ne_m3'],ion_te),
                          ('model',model_height/1000,model_ne,np.full_like(model_ne,300.))]:
        east,north,up=[v[0]*1e-9 for v in ppigrf.igrf(-66.7528,18.3442,z,date,coeff_fn=str(coeff))]
        B=np.sqrt(east**2+north**2+up**2)
        theta=np.rad2deg(np.arccos(np.clip(abs(up)/B,0,1)))
        resonance=profile_hz(ne,te,B,theta)/1e6
        residual=np.array([dielectric(fr*1e6,n,t,b,a).real for fr,n,t,b,a in zip(resonance,ne,te,B,theta)])
        if np.max(abs(residual))>1e-7:raise RuntimeError('Plasma resonance root residual too large')
        magnetic[prefix]=dict(plasma_line_mhz=resonance,Te_K=te,B_T=B,angle_k_B_deg=theta,
                              B_east_T=east,B_north_T=north,B_up_T=up,dielectric_real_residual=residual)
    path=ROOT/f'data/conversion_overlay_{epoch}.h5'
    with h5py.File(path,'w') as h:
        for k,v in dict(frequency_mhz=-frequency[fi],height_km=height[zi],
            displayed_power=displayed[np.ix_(zi,fi)]/1e14,unfiltered_excess=highpass[np.ix_(zi,fi)],raw_power=power[np.ix_(zi,fi)],
            ion_height_km=f['source_height_m']/1000,ion_fpe_mhz=fpe(f['source_ne_m3']),
            model_height_km=model_height/1000,model_fpe_mhz=fpe(model_ne),
            model_ne_m3=model_ne,simulation_height_m=simulation_z,simulation_ne_m3=simulation_n,
            spline_knots_km=f['knots_km'],spline_coefficients=f['coefficients']).items():h[k]=v
        for prefix,fields in magnetic.items():
            for key,value in fields.items():h[f'{prefix}_{key}']=value
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
            magnetic_model='IGRF14',igrf_coefficients_sha256=hashlib.sha256(coeff.read_bytes()).hexdigest(),
            lat_deg=18.3442,lon_deg=-66.7528,radar_frequency_hz=430e6,
            resonance='Re(1+chi_e)=0; Maxwellian magnetized electrons; k=2*pi*(2*430e6+f)/c; Nicolls et al. 2006',
            temperatures='ion: fitted Te at density cycle; smooth model: 300 K as in propagation simulation',
            caveat='Weakly damped real-frequency diagnostic; no ion motion, collision frequency shift or bulk Doppler shift; time pairing uses nearest ~80.4 s cycle')
    add_snr(epoch)
    print(path, 'time offset',f['time_unix_s']-time,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epoch',type=int,default=1525460596)
    p.add_argument('--snr-only',action='store_true')
    args=p.parse_args()
    (add_snr if args.snr_only else prepare)(args.epoch)
