"""Plot products and quantitative checks for the native FDTD-driven fluid run."""
from pathlib import Path
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.constants import e,m_e,epsilon_0,c

root=Path(__file__).resolve().parent
with h5py.File(root/'output/fluid/fdtd_driven.h5') as h:
    b=h['background'];r=h['harmonic'];z=b['z'][:]-b.attrs['center'];zf=b['zf'][:]-b.attrs['center'];dz=b.attrs['dz']
    eta=r['density_phasor'][:];es=r['self_field_phasor_V_m'][:]
    source=b['source'][:];inc=b.attrs['incident'];match=b.attrs['match']-b.attrs['center']
    predk=np.sqrt(np.maximum(0,(b.attrs['D']-b['ne'][:]*e**2/(m_e*epsilon_0))/b.attrs['cs2']))
    lam=np.divide(2*np.pi,predk,out=np.full_like(predk,np.nan),where=predk>0)
    simlam=2*np.pi/abs(r['kz_rad_m'][:]);diag=h['transient/diagnostics'][:]
    rms=h['transient/density_amplitude_rms'][:]
    norm=max(abs(diag[-1,8]),1e-100)
    balance=abs(diag[-1,11])/norm
    off=int(np.searchsorted(diag[:,0],h['transient'].attrs['pulse_duration_s']))
    decay=diag[-1,3]/diag[off,3]
    herror=h['native_harmonic_check'].attrs['density_relative_l2_error']
    conv=h['convergence_grid_domain_relative'][:]
    assert balance<1e-3,(balance,'energy ledger')
    assert herror<1e-3,(herror,'harmonic agreement')
    assert max(conv)<.02,conv
    assert decay<.01,(decay,'post-pulse energy')
    flux=gaussian_filter1d(r['pressure_flux_W_m2'][:],.3/dz)
    select=(z<-1)&(z>-10)
    assert np.mean(flux[select])<0
    checks=dict(warm_source_power_over_incident=h.attrs['warm_source_power_over_incident'],cold_field_reconstruction_relative_error=h.attrs['cold_field_reconstruction_relative_error'],energy_balance_relative=balance,post_pulse_energy_ratio=decay,
                native_harmonic_relative_error=herror,bragg_distance_m=-match,
                bragg_wavelength_m=c/(2*430e6),incident_field_norm_V_m=inc,
                density_amplitude_at_bragg=float(np.interp(match,z,abs(eta))),
                source_off_s=h['transient'].attrs['pulse_duration_s'],first_post_off_sample_s=diag[off,0],final_time_s=diag[-1,0],
                collision_fraction_of_input=diag[-1,9]/norm,
                buffer_fraction_of_input=diag[-1,10]/norm)
    with h5py.File(root/'output/fluid/fdtd_fluid_plot.h5','w') as o:
        o.attrs.update(checks);o.attrs['provenance']='prepare_fdtd_fluid_figures.py; drive_fluid_from_fdtd.py; Rust native recorded evolution'
        for key,val in dict(z=z,zf=zf,density_real_percent=eta.real*100,density_imag_percent=eta.imag*100,
                density_amplitude_percent=abs(eta)*100,source_amplitude_mVm=abs(source)*1000,
                self_field_real_mVm=es.real*1000,self_field_imag_mVm=es.imag*1000,self_field_amplitude_mVm=abs(es)*1000,
                wavelength_theory_m=lam,wavelength_simulation_m=simlam,pressure_flux_nW_m2=flux*1e9,
                time_ms=diag[:,0]*1000,energy_relative=diag[:,3]/max(diag[:,3]),
                history_percent=rms.T*100,frequency_hz=b.attrs['frequency']).items():
            o[key]=val
    print(checks)
    for x in [-1,-2,match,-6,-10]:
        j=np.argmin(abs(z-x));print('z',z[j],'lambda',simlam[j],'amplitude %',abs(eta[j])*100,'downward flux',flux[j])
