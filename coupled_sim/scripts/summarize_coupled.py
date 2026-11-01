"""Scientific diagnostics/plot arrays for the shared Rust two-way simulation."""
import sys
import numpy as np
import h5py
from scipy.constants import e,m_e,epsilon_0,c
from scipy.integrate import trapezoid
from scipy.optimize import brentq
from coupled_arecibo import ROOT,OUT,PROFILE,CS,W
sys.path.insert(0,str(ROOT))
from radar import beam,RE,LAM,KR,PT,TSYS,kb

def radar(h):
    z=h['z'][:];zz=z-h.attrs['center_m'];dn=h['density_phasor_m3'][:]
    kx=float(h['kx_rad_m'][()]);freq=h.attrs['frequency_hz'];_,a,b,g=beam(z)
    J=a/(a-1j*b)*np.exp(-kx*kx/(4*(a-1j*b)))*np.exp(-2j*g)
    powers=[]
    for lo,hi in [(-80,20),(-100,30),(-110,40)]:
        win=np.sin(.5*np.pi*np.clip(np.minimum(zz-lo,hi-zz)/10,0,1))**2
        col=trapezoid(.5*dn*np.exp(1j*(2*KR+2*np.pi*freq/c)*zz)*J*win,z)
        powers.append(PT*RE**2*LAM**2*abs(col)**2)
    return np.array(powers)

def main():
    with h5py.File(ROOT/'data/radar_calibration.h5') as cal:
        C=cal.attrs['ion_line_measured_over_predicted'];oldflux=cal.attrs['required_incident_vertical_flux_W_m2']
        target=cal.attrs['fitted_on_frequency_snr'];tau=cal.attrs['pulse_duration_s']
        obsf=cal['line_frequency_hz'][:];obs=cal['line_measured_snr'][:];fit=cal['line_fit_fixed'][:]
    checks=[]
    for name in ['baseline','fine','finest','convergence','outer_mesh','larger_buffer','coupled','recorded']:
        with h5py.File(OUT/f'{name}.h5') as h:
            checks.append([h.attrs['finest_dz_m'],h.attrs['peak_Ez_V_m_per_sqrt_W_m2'],h.attrs['peak_density_fraction_per_sqrt_W_m2'],radar(h)[1],h.attrs['reflected_fraction'],h.attrs['physical_loss_between_planes']])
    checks=np.array(checks)
    # Plot the case actually advanced through the linked Rust libraries.
    with h5py.File(OUT/'recorded.h5') as h:
        z=h['z'][:];zz=z-h.attrs['center_m'];dn=h['density_phasor_m3'][:];ne=h['ne'][0];state=h['state_phasor'][:]
        kx=float(h['kx_rad_m'][()]);gyro=h['gyro'][:]*c
        kz=-(2*KR+W/c);kv=np.array([kx,0,kz]);D=W**2*(W**2-gyro@gyro)/(W**2-(gyro@kv)**2/(kv@kv))
        match=brentq(lambda z:PROFILE(z)*e*e/(m_e*epsilon_0)+CS**2*(kv@kv)-D,h.attrs['center_m']-30,h.attrs['center_m']+10)
        powers=radar(h);reference_snr=powers[1]*oldflux*C*tau/(kb*TSYS)
        flux=target*kb*TSYS/(tau*powers[1]*C);scale=np.sqrt(flux)
        phase_k=np.gradient(np.unwrap(np.angle(dn)),z)
        use=(zz>-20)&(zz<5)
        datasets=dict(height_m=zz[use],density_real_ppm=(dn/ne).real[use]*scale*1e6,
            density_amplitude_ppm=abs(dn/ne)[use]*scale*1e6,Ez_mV_m=abs(state[2,use])*scale*1e3,
            EM_flux=h['EM_flux_W_m2'][:][use],pressure_flux=h['pressure_flux_W_m2'][:][use],
            wavelength_phase_m=2*np.pi/np.maximum(-phase_k[use],1e-12),
            frequency_mhz=obsf/1e6,observed_snr=obs,
            spectrum_frequency_mhz=np.linspace(5.27,5.41,1001))
        datasets['spectrum_snr']=target*np.sinc((datasets['spectrum_frequency_mhz']*1e6-W/(2*np.pi))*tau)**2+fit[2]
        usewave=(zz>-12)&(zz<-.8)
        phase_match=z[usewave][np.argmin(abs(phase_k[usewave]-kz))]
        summary=dict(incident_flux_to_match_observed_W_m2=flux,required_density_at_bragg_ppm=float(np.interp(match,z,abs(dn)/ne)*scale*1e6),
            snr_at_previous_incident_flux=reference_snr,observed_snr=target,
            reference_incident_flux_W_m2=oldflux,bragg_height_relative_m=match-h.attrs['center_m'],
            solved_phase_bragg_height_relative_m=phase_match-h.attrs['center_m'],bragg_wavelength_m=2*np.pi/abs(kz),
            peak_Ez_mV_m=float(abs(state[2,use]).max()*scale*1e3),
            max_downward_pressure_flux_fraction=float(-h['pressure_flux_W_m2'][:].min()),
            reflected_fraction=h.attrs['reflected_fraction'],collisional_loss_fraction=h.attrs['physical_loss_between_planes'],
            boundary_balance_error=h.attrs['boundary_balance_error'],
            radar_window_power_spread=float(np.ptp(powers)/powers[1]),
            mesh_peak_density_relative_change=float(abs(checks[2,2]/checks[3,2]-1)),
            mesh_radar_power_relative_change=float(abs(checks[2,3]/checks[3,3]-1)))
    with h5py.File(OUT/'cold_control.h5') as h:
        cz=h['z'][:]-h.attrs['center_m'];ce=abs(h['E_phasor_V_m'][2])*scale*1e3
        datasets['cold_Ez_mV_m']=np.interp(datasets['height_m'],cz,ce)
    with h5py.File(OUT/'summary.h5','w') as h:
        h.attrs.update(summary);h.attrs.update(amplitude='conditionally normalized to observed plasma line using prior coherent-beam ion-line calibration; not independent amplitude prediction',
            caveat='collisional warm fluid, no Landau damping; continuous-wave initialization, not turn-on; cold control has similar total absorption, so absorption is not a conversion fraction',
            provenance='hf_fdtd/examples/summarize_coupled.py; plotter/src/bin/coupled.rs')
        for key,val in datasets.items():h['plot/'+key]=val
        h['checks']=checks;h['checks'].attrs['rows']='baseline,fine,finest,convergence,outer_mesh,larger_buffer,coupled,recorded'
        h['checks'].attrs['columns']='dz,peak_Ez,peak_density_fraction,radar_power,reflection,physical_loss'
    print(summary)

if __name__=='__main__':main()
