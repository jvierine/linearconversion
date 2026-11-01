"""Beam-filling HF inversion with an independent ion-line radiometric check.

Source provenance stays in HDF5 and code. Scientific assumptions:
1 MW / 150 K supplied by the experimenter. Nominal 61.1 dBi Gaussian radar
mode uses Mathews (2013), doi:10.5194/hgss-4-19-2013, not event beam calibration.
Finite-distance Gaussian spreading and round-trip Fresnel phase are retained.
The complex 1D density is extended horizontally as exp(i kx x); a Gaussian
transverse phase-correlation sensitivity is separate from beam filling.
Ion-line Doppler integration removes the unknown code phases by Parseval;
the remaining range weight is the overlap triangle for a 44 us pulse.
The +/-2 MHz integral is tested against 1.5 and 2.5 MHz windows. No deconvolved
plasma-line amplitude or arbitrarily normalized PSF is used as an absolute power.
"""
from pathlib import Path
import sys
import hashlib
import numpy as np
import h5py
from scipy.constants import c, k as kb, e, m_e, epsilon_0, physical_constants
from scipy.optimize import least_squares
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from deconvolve_plasma_line import load

ROOT=Path(__file__).resolve().parent
EPOCH=1525453430
PT=1e6; TSYS=150.; RADAR=430e6; TAU=44e-6; GAIN_DB=61.1
RE=physical_constants['classical electron radius'][0];LAM=c/RADAR;KR=2*np.pi/LAM


def beam(z,gain_db=GAIN_DB):
    # Normalized paraxial Gaussian mode: integral |u|^2 dA = 1.
    gain=10**(gain_db/10);w0=np.sqrt(gain)*LAM/(np.sqrt(8)*np.pi)
    zr=np.pi*w0*w0/LAM;w=w0*np.sqrt(1+(z/zr)**2)
    curvature=z*(1+(zr/z)**2)
    return w,2/w**2,KR/curvature,np.arctan(z/zr)


def main():
    f,z,p,bg,noise,path=load(EPOCH)
    roi=(z[:,None]>116)&(z[:,None]<119)&(f[None,:]>5.2)&(f[None,:]<5.5)
    iz,jf=np.unravel_index(np.argmax(np.where(roi,p-bg,-np.inf)),p.shape)
    q=p[iz]/bg-1;use=abs(f-5.34)<.075;ff=f[use]*1e6;yy=q[use]
    def curve(v,nu):return v[0]*np.sinc((nu-v[1])*v[2]*1e-6)**2+v[3]
    fit=least_squares(lambda v:curve(v,ff)-yy,[90.,5.34e6,44.,0.],
        bounds=([0,5.33e6,25,-1],[500,5.35e6,80,5]),x_scale=[100,10000,40,1])
    # The recovered effective duration is a check; use archived 44 us for scale.
    def fixed_curve(v,nu):return v[0]*np.sinc((nu-v[1])*TAU)**2+v[2]
    fixed=least_squares(lambda v:fixed_curve(v,ff)-yy,[90.,5.34e6,0.],
        bounds=([0,5.33e6,-1],[500,5.35e6,5]),x_scale=[100,10000,1])
    snr_cont,freq,bg_fit=fixed.x
    pr_measured=snr_cont*kb*TSYS/TAU

    # Separate pulse/ACF chain: frequency-calibrated densities and fitted Te/Ti.
    with h5py.File(ROOT/'data/eregion_fit/full_altitude_fits.h5') as h:
        tt=h['time_unix_s'][:];it=np.argmin(abs(tt-EPOCH));rr=h['height_km'][:]*1000
        valid=(rr>90000)&(rr<150000);rr=rr[valid];par=h['results'][it,valid,:4]
        ne,te,ti,vi=par.T;B=h['B_T'][:][valid];theta=h['angle_k_B_deg'][:][valid]
        acf_epoch=tt[it]
    ae=ne*e**2/(epsilon_0*kb*te*(2*KR)**2);ai=ae*te/ti
    ion_area=ae**2/((1+ae)*(1+ae+ai))
    # Cold-electron static approximation to integrated ion-source spectrum.
    # Check with the user's actual numerical ISR code at three smooth gates.
    sys.path.insert(0,str(ROOT/'isr_spec_validation'))
    from compare import load_juha, vierinen, JUHA_REV
    mod=load_juha(ROOT/'isr_spec_validation/.cache/vierinen/isr_spec.py')
    checks=[]
    for height in [129000.,133000.,138000.]:
        j=np.argmin(abs(rr-height));freqs=np.linspace(-40000.,40000.,401)
        case=dict(frequency_hz=freqs,ne=ne[j],te=te[j],ti=ti[j],radar_hz=RADAR,
                  ions=[('NO+',30.,1.)],B=B[j],theta=theta[j])
        spec=vierinen(mod,case,10000)
        numerical=trapezoid(spec[0],freqs)
        checks.append([rr[j],ion_area[j],numerical,(ion_area[j]/numerical-1)])
    checks=np.array(checks);assert max(abs(checks[:,3]))<.025

    with h5py.File(path) as h:
        ffull=h['freqs'][:]*1e6;rfull=h['range'][:]*1000;raw=h['PL'][:]
    df=ffull[1]-ffull[0];floor=np.median(raw[rfull<85000],axis=0)
    excess=raw/floor-1
    windows=np.array([1.5e6,2e6,2.5e6])
    il_observed=np.array([df*np.sum(excess[:,abs(ffull)<w],axis=1) for w in windows])
    width=c*TAU/2
    w,*_=beam(rr)
    weight=PT*RE**2*LAM**2/(kb*TSYS)*ne*ion_area/(np.pi*w*w)
    predicted=np.array([trapezoid(weight*np.maximum(0,1-abs(rr-r)/width),rr) for r in rfull])
    good=(rfull>=125000)&(rfull<=135000)
    ratio=il_observed[1,good]/predicted[good]
    cal=float(np.median(ratio));cal_scatter=float(1.4826*np.median(abs(ratio-cal)))
    assert np.all(np.isfinite(ratio)) and cal>0
    window_ratio=np.median(il_observed[:,good]/il_observed[1,good],axis=1)
    hold=(rfull>=110000)&(rfull<125000)
    heldout_ratio=il_observed[1,hold]/(cal*predicted[hold])

    # Mode overlap with a chirped, vertically propagating density perturbation.
    # n1 = Re[n_tilde exp(-i omega t)]; ONE sideband coefficient is n_tilde/2.
    fluid_path=ROOT/'output/fluid/fdtd_extended_kz.h5'
    with h5py.File(fluid_path) as h:
        zz=h['setup/z'][:];dn=h['full/dn_phasor_m3'][:];nmodel=h['setup/ne'][:];s=dict(h['setup'].attrs)
    dz=zz-s['center'];kx=s['kx']
    _,a,b,gouy=beam(zz)
    J=a/(a-1j*b)*np.exp(-kx*kx/(4*(a-1j*b)))*np.exp(-2j*gouy)
    integrals=[]
    for lo,hi in [(-80.,20.),(-100.,30.),(-110.,40.)]:
        window=np.sin(.5*np.pi*np.clip(np.minimum(dz-lo,hi-dz)/10,0,1))**2
        col=trapezoid(.5*dn*np.exp(2j*KR*dz)*J*window,zz)
        integrals.append(col)
    col=integrals[-1];pr_reference=PT*RE**2*LAM**2*abs(col)**2
    field_scale=np.sqrt(pr_measured/(pr_reference*cal))
    inc_peak=s['incident']*field_scale
    flux_z=s['incident_flux_W_m2']*field_scale**2
    with h5py.File(ROOT/'output/hf_fdtd/arecibo_hf_field.h5') as h:angle=np.deg2rad(float(h.attrs['angle_deg']))
    # Free-space equivalent only: vertical flux -> along-ray flux using vacuum
    # equivalent launch angle. Cold profile propagation already in model; extra
    # lower-path absorption multiplies EIRP separately, not counted twice.
    ranges=np.array([100e3,125e3,200e3,500e3])
    eirp=4*np.pi*ranges**2*flux_z/np.cos(angle)
    vacuum_rms=np.sqrt(physical_constants['characteristic impedance of vacuum'][0]*flux_z/np.cos(angle))
    # Across-beam phase-correlation sensitivity at the Bragg layer. This changes
    # phases only, not occupied beam area or local density-wave amplitude.
    match=s['match'];_,aa,bb,_=beam(np.array([match]));aa=float(aa[0]);bb=float(bb[0])
    wm=float(beam(np.array([match]))[0][0]);xx=np.linspace(-5*wm,5*wm,16001)
    numerical_J=aa/np.pi*trapezoid(np.exp(-(aa-1j*bb)*xx**2+1j*kx*xx),xx)*trapezoid(np.exp(-(aa-1j*bb)*xx**2),xx)
    analytic_J=aa/(aa-1j*bb)*np.exp(-kx*kx/(4*(aa-1j*bb)))
    gaussian_error=abs(numerical_J/analytic_J-1);assert gaussian_error<1e-7
    lc=np.logspace(-1,4,401)
    den=aa*aa+bb*bb+aa/lc**2
    coupling=aa*aa/den*np.exp(-kx*kx*aa/(2*den))
    coh=aa*aa/(aa*aa+bb*bb)*np.exp(-kx*kx*aa/(2*(aa*aa+bb*bb)))
    eirp_lc=eirp[1]*coh/coupling
    main=dict(measured_peak_snr_linear=float(q[jf]),measured_peak_snr_db=float(10*np.log10(q[jf])),
        peak_frequency_hz=float(f[jf]*1e6),fitted_line_frequency_hz=float(freq),
        fitted_on_frequency_snr=float(snr_cont),fitted_on_frequency_snr_db=float(10*np.log10(snr_cont)),
        measured_received_line_W=float(pr_measured),effective_fit_duration_us=float(fit.x[2]),
        fixed44_fit_rms=float(np.sqrt(np.mean((fixed_curve(fixed.x,ff)-yy)**2))),
        free_duration_fit_rms=float(np.sqrt(np.mean((curve(fit.x,ff)-yy)**2))),
        radar_peak_power_W=PT,system_temperature_K=TSYS,nominal_radar_gain_dBi=GAIN_DB,
        pulse_duration_s=TAU,noise_equivalent_bandwidth_hz=1/TAU,
        ion_line_measured_over_predicted=cal,ion_line_ratio_mad=cal_scatter,
        ion_line_heldout_median_ratio=float(np.median(heldout_ratio)),
        ion_line_heldout_relative_rms=float(np.sqrt(np.mean((heldout_ratio-1)**2))),
        gaussian_overlap_quadrature_relative_error=float(gaussian_error),
        acf_epoch_unix_s=float(acf_epoch),reference_incident_peak_V_m=s['incident'],
        reference_received_line_W=float(pr_reference),required_incident_peak_V_m=float(inc_peak),
        required_incident_vertical_flux_W_m2=float(flux_z),vacuum_equivalent_rms_V_m=float(vacuum_rms),
        vacuum_equivalent_angle_deg=float(np.rad2deg(angle)),eirp_at125km_W=float(eirp[1]),
        eirp_without_ion_check_W=float(eirp[1]*cal),
        required_bragg_density_fraction=float(np.interp(match,zz,abs(dn)/nmodel)*field_scale),
        vertical_window_power_spread=float(np.ptp(abs(np.array(integrals))**2)/abs(col)**2))
    print('SUMMARY',main,flush=True)
    print('ION_CHECKS',checks.tolist(),'bandwidth_ratios',window_ratio,flush=True)
    print('COHERENCE',[(x,float(np.interp(np.log10(x),np.log10(lc),eirp_lc))) for x in [.1,1,10,100,10000]],flush=True)
    out=ROOT/'output/fluid/hf_power_from_measured_pl.h5'
    with h5py.File(out,'w') as h:
        h.attrs.update(generator='estimate_hf_power.py',measurement_source=str(path),
            measurement_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fluid_source=str(fluid_path),isr_code_repository='https://github.com/jvierine/isr_analysis',
            isr_code_commit=JUHA_REV,
            assumptions='Beam filling=1. Fixed 5.33 MHz,300 K one-way warm fluid. Radar 1 MW,150 K supplied by user. Nominal Gaussian beam61.1dBi. Finite-distance Fresnel phase. Coherent horizontal exp(i kx x) baseline; random phase correlation sensitivity separate. Ion-line calibration is a common received-power multiplier with beam geometry fixed. Lossless free-space-equivalent HF path; additional loss and HF antenna gain explicit.',
            limitation='Conditional estimate, not identified transmitter power. Code envelope and 44us duration supported by archived processing and measured line shape; exact waveform unavailable. Ion integral +/-2MHz has finite-band and response assumptions. Ion density scale comes from independent broad plasma-line frequencies, not enhanced-line amplitude. Actual antenna pattern/near-field phase and transverse wave coherence not measured. Fluid HF frequency5.33MHz differs from fitted5.340MHz; response not recomputed at fitted center.')
        for key,v in main.items():h.attrs[key]=v
        arrays=dict(line_frequency_hz=ff,line_measured_snr=yy,line_model_snr=fixed_curve(fixed.x,ff),
            line_fit_free=fit.x,line_fit_fixed=fixed.x,ion_height_m=rfull,
            ion_integrated_snr_Hz=il_observed,ion_theory_integrated_snr_Hz=predicted,
            ion_scaled_theory_Hz=cal*predicted,ion_comparison_mask=good,
            ion_bandwidth_windows_hz=windows,ion_window_integral_ratio=window_ratio,
            isr_spectrum_integral_check=checks,coherence_length_m=lc,eirp125km_vs_coherence_W=eirp_lc,
            hf_slant_range_m=ranges,hf_eirp_W=eirp,vertical_integrals_m2=np.array(integrals),
            density_height_m=rr,density_ne_m3=ne,Te_K=te,Ti_K=ti,ion_spectrum_area=ion_area)
        for key,v in arrays.items():h.create_dataset(key,data=v,compression='gzip')
    print(out)


if __name__=='__main__':main()
