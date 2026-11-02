"""Independent source, radiometry, Fresnel-beam, and HF far-field checks.

FFT Fresnel propagation of finite apertures is independent of the analytic
Gaussian radar operator. Apertures are sensitivity cases, not measured feeds.
The archived ion spectra and current fitted plasma parameters recalibrate each
candidate radar mode. No existing measurement or model result is overwritten.
"""
from pathlib import Path
import gc,hashlib,sys
import h5py,numpy as np
from scipy.constants import c,epsilon_0,mu_0,k as kb,e
from scipy.fft import fft2,ifft2,fftfreq
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator
import coupled_arecibo as a
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from isr.coherent import RE,gaussian_parameters,gaussian_overlap
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data/power_calibration_audit.h5'
FR=430e6;LAM=c/FR;PT=1e6;TEMP=150.;GAIN=61.1;TAU=44e-6
with h5py.File(ROOT/'reference/coupled/minimum_incident.h5') as h:
    B=dict(h.attrs);z=h['z'][:];dn=h['density_phasor_m3'][:];kx=float(h['kx_rad_m'][()]);v=h['source_polarization'][:]
FP=B['frequency_hz'];CENTER=B['center_m'];F0=B['incident_vertical_flux_W_m2'];ANGLE=B['angle_deg']
with h5py.File(a.ROOT/'data/radar_calibration.h5') as h:
    C0=float(h.attrs['ion_line_measured_over_predicted']);TARGET=float(h.attrs['fitted_on_frequency_snr'])
    ri=h['ion_height_m'][:];obs=h['ion_integrated_snr_Hz'][1];EPOCH=h.attrs['acf_epoch_unix_s']
    archived_pred=h['ion_theory_integrated_snr_Hz'][:];fitted_duration=h.attrs['effective_fit_duration_us']
FITPATH=ROOT/'data/eregion_fit/full_altitude_fits.h5'
if FITPATH.exists():
    with h5py.File(FITPATH) as h:
        it=np.argmin(abs(h['time_unix_s'][:]-EPOCH));rr=h['height_km'][:]*1000
        keep=(rr>90000)&(rr<150000);rr=rr[keep];ne,te,ti,vi=h['results'][it,keep,:4].T
    FITSHA=hashlib.sha256(FITPATH.read_bytes()).hexdigest()
else:
    # The committed audit retains the one input profile needed here; the full
    # experiment fit cube need not be distributed just to repeat this check.
    with h5py.File(OUT) as h:
        rr=h['inputs/height_m'][:]
        ne,te,ti,vi=h['inputs/plasma_parameters'][:].T
        FITSHA=h['provenance'].attrs['plasma_fits_sha256']
calmask=(ri>=125000)&(ri<=135000);holdmask=(ri>=110000)&(ri<125000)
ae=ne*e**2/(epsilon_0*kb*te*(4*np.pi/LAM)**2);ai=ae*te/ti
area=ae**2/((1+ae)*(1+ae+ai))
W=np.maximum(0,1-abs(rr[:,None]-ri[None,:])/(c*TAU/2))
WEIGHT=PT*RE**2*LAM**2/(kb*TEMP)*ne*area
rel=z-CENTER;window=np.sin(.5*np.pi*np.clip(np.minimum(rel+100,30-rel)/10,0,1))**2
KB=(4*np.pi*FR+2*np.pi*FP)/c

def calibrated(H,J):
    pred=trapezoid((WEIGHT*H)[:,None]*W,rr,axis=0)
    C=float(np.median(obs[calmask]/pred[calmask]))
    amplitude=np.sqrt(PT)*RE*LAM*trapezoid(.5*dn*window*J*np.exp(1j*KB*rel),z)
    perflux=abs(amplitude)**2
    flux=TARGET*kb*TEMP/(C*TAU*perflux)
    eirp=4*np.pi*(125e3)**2*flux/np.cos(np.deg2rad(ANGLE))
    return dict(calibration_C=C,power_per_unit_flux_W=float(perflux),required_flux_W_m2=float(flux),EIRP125km_W=float(eirp),
        heldout_median_ratio=float(np.median(obs[holdmask]/(C*pred[holdmask]))),predicted_ion_Hz=pred)


def aperture_case(name,N=2048,dx=2.,phase=0.):
    x=(np.arange(N)-N//2)*dx;r2=x[:,None]**2+x[None,:]**2
    ff=fftfreq(N,d=dx);f2=ff[:,None]**2+ff[None,:]**2
    waist=np.sqrt(10**(GAIN/10))*LAM/(np.sqrt(8)*np.pi)
    if name=='Gaussian FFT':u=np.exp(-r2/waist**2).astype(complex)
    elif name=='Uniform 225 m':u=(r2<=112.5**2).astype(complex)
    else:u=np.exp((-np.log(10)/2+1j*phase)*r2/112.5**2)*(r2<=112.5**2)
    u/=np.sqrt(np.sum(abs(u)**2)*dx**2)
    gain=4*np.pi*abs(u.sum()*dx**2)**2/LAM**2;U=fft2(u);del u
    zz=np.unique(np.r_[np.arange(100e3,150001,5e3),CENTER-100,CENTER,CENTER+50])
    hh=[];jj=[];powers=[];transverse=np.exp(1j*kx*x[:,None])
    for height in zz:
        field=ifft2(U*np.exp(-1j*np.pi*LAM*height*f2))
        powers.append(np.sum(abs(field)**2)*dx**2)
        hh.append(np.sum(abs(field)**4)*dx**2)
        if abs(height-CENTER)<=101:
            jj.append(np.sum(field**2*transverse*np.exp(1j*np.pi*FP/(c*height)*r2))*dx**2)
        else:jj.append(np.nan+1j*np.nan)
    hh=np.asarray(hh);jj=np.asarray(jj);sel=np.isfinite(jj)
    H=np.exp(PchipInterpolator(zz,np.log(hh))(rr))
    J=np.interp(z,zz[sel],jj[sel].real)+1j*np.interp(z,zz[sel],jj[sel].imag)
    result=calibrated(H,J)
    result.update(gain_dBi=float(10*np.log10(gain)),grid_N=N,grid_dx_m=dx,aperture_edge_phase_rad=phase,
        max_propagation_energy_error=float(np.max(abs(np.asarray(powers)-1))))
    if name=='Gaussian FFT':
        Href=1/(np.pi*gaussian_parameters(zz)[0]**2)
        Jref=gaussian_overlap(zz[sel],kx,FP)
        result['gaussian_H_relative_error']=float(np.max(abs(hh/Href-1)))
        result['gaussian_J_relative_error']=float(np.max(abs(jj[sel]/Jref-1)))
    result['height_m']=zz;result['incoherent_beam_integral_m2']=hh;result['coherent_overlap']=jj
    del U,field,r2,f2;gc.collect()
    print(name,N,dx,{k:result[k] for k in ['gain_dBi','calibration_C','required_flux_W_m2','EIRP125km_W']},flush=True)
    return result


def save_case(h,name,result):
    g=h.create_group(name)
    for key,value in result.items():
        if np.asarray(value).ndim:g.create_dataset(key,data=value,compression='gzip')
        else:g.attrs[key]=value


def main():
    with h5py.File(OUT,'w') as h:
        h.attrs.update(frequency_hz=FP,radar_hz=FR,reference_flux_W_m2=F0,angle_deg=ANGLE,
            reference_center_m=CENTER,reference_calibration_C=C0,Pt_W=PT,Tsys_K=TEMP,
            assumptions='All beam alternatives are reciprocal paraxial scalar aperture models. Same fixed 1D density wave, full coherent filling, same local incident HF mode. Each mode is recalibrated against the measured ion line. No event-specific aperture phase map is available.',
            source_model_sha256=hashlib.sha256((ROOT/'reference/coupled/minimum_incident.h5').read_bytes()).hexdigest())
        H=1/(np.pi*gaussian_parameters(rr)[0]**2);J=gaussian_overlap(z,kx,FP)
        baseline=calibrated(H,J);save_case(h,'baseline',baseline)
        assert abs(baseline['calibration_C']/C0-1)<1e-12
        assert abs(baseline['required_flux_W_m2']/F0-1)<1e-10
        np.testing.assert_allclose(baseline['predicted_ion_Hz'],archived_pred,rtol=1e-12)
        # Independent SI Poynting flux: stored magnetic states equal c*B.
        ref=a.setup(dz=float(B['dz_m']),cap=float(B['cap_m']),angle=ANGLE,reference=True)
        inc=np.einsum('qnp,p->qn',ref['initial'],v)
        ix=np.argmin(abs(ref['z']-B['reference_plane_m']))
        Ex,Ey=(inc[:2]+np.roll(inc[:2],-1,axis=1))/2
        Bx,By=inc[3:5]/c
        pin=float(.5/mu_0*np.real(Ex*By.conj()-Ey*Bx.conj())[ix])
        erms=float(np.linalg.norm(inc[:3,ix])*np.sqrt(F0/2))
        h.require_group('source').attrs.update(upward_flux_for_normalized_source_W_m2=pin,scaled_flux_W_m2=pin*F0,
            computed_E_rms_V_m=erms,stored_E_rms_V_m=B['incident_reference_E_rms_V_m'])
        assert abs(pin-1)<1e-8;assert abs(erms/B['incident_reference_E_rms_V_m']-1)<1e-8
        del ref,inc;gc.collect()
        # A true far-field Gaussian limit, retaining propagation phase curvature.
        waist=np.sqrt(10**(GAIN/10))*LAM/(np.sqrt(8)*np.pi)
        wff=LAM*z/(np.pi*waist);ag=2/wff**2;bg=2*np.pi*FR/(c*z)+np.pi*FP/(c*z)
        Jff=ag/(ag-1j*bg)*np.exp(-kx*kx/(4*(ag-1j*bg))-1j*np.pi)
        Hff=1/(np.pi*(LAM*rr/(np.pi*waist))**2)
        save_case(h,'far_field_Gaussian',calibrated(Hff,Jff))
        # Deliberately omit radar wavefront curvature: a diagnostic, not a model.
        w,R,g=gaussian_parameters(z);ag=2/w**2;bg=np.pi*FP/(c*z)
        Jflat=ag/(ag-1j*bg)*np.exp(-kx*kx/(4*(ag-1j*bg))-2j*g)
        save_case(h,'flat_radar_phase_diagnostic',calibrated(H,Jflat))
        for name in ['Gaussian FFT','Uniform 225 m','Tapered 225 m']:
            save_case(h,'apertures/'+name,aperture_case(name))
        for label,phase in [('Minus45',-np.pi/4),('Plus45',np.pi/4)]:
            save_case(h,'apertures/'+label,aperture_case('Tapered 225 m',phase=phase))
        # Independent aperture-grid and periodic-domain checks.
        for label,N,dx in [('dx4',1024,4.),('wide_domain',2048,4.)]:
            save_case(h,'checks/'+label,aperture_case('Uniform 225 m',N,dx))
        # The received-power calibration removes a common Pt/Tsys scale.
        rows=[]
        for pt_scale,t_scale in [(1.,1.),(2.,1.),(1.,2.),(.5,1.5)]:
            cp=C0*t_scale/pt_scale
            flux=TARGET*kb*TEMP*t_scale/(cp*TAU*baseline['power_per_unit_flux_W']*pt_scale)
            rows.append([pt_scale,t_scale,cp,flux])
        h['common_scale_cancellation']=rows
        h['common_scale_cancellation'].attrs['columns']='Pt_scale,Tsys_scale,recalibrated_C,required_flux_W_m2'
        # Source far-field criteria and omitted HF spherical-wave phase.
        lamhf=c/FP;D=np.array([100.,300.,1000.,2000.])
        h['HF/source_aperture_D_m']=D;h['HF/Fraunhofer_distance_m']=2*D*D/lamhf
        h['HF'].attrs['D_at_Fraunhofer125km_m']=np.sqrt(125e3*lamhf/2)
        h['HF'].attrs['phase_over_300m_radius_rad']=2*np.pi/lamhf*300**2/(2*125e3)
        h.require_group('radar').attrs.update(Fraunhofer225m_m=2*225**2/LAM,Fraunhofer305m_m=2*305**2/LAM,
            Gaussian_waist_m=waist,Gaussian_Rayleigh_m=np.pi*waist**2/LAM,
            Gaussian_width_at_event_m=float(gaussian_parameters(CENTER)[0]),
            local_incoherent_near_over_far=float((LAM*CENTER/(np.pi*waist))**2/gaussian_parameters(CENTER)[0]**2),
            observed_rectangular_duration_us=fitted_duration)
        h['provenance/source']=Path(__file__).read_text()
        h['inputs/height_m']=rr
        h['inputs/plasma_parameters']=np.column_stack([ne,te,ti,vi])
        h['inputs/plasma_parameters'].attrs['columns']='ne_m-3,Te_K,Ti_K,vi_m_s'
        h['inputs'].attrs['calibration_acf_epoch_unix_s']=EPOCH
        h['provenance'].attrs['plasma_fits_sha256']=FITSHA
        for label,path in {
            'audit':Path(__file__),
            'coherent_operator':ROOT/'isr/coherent.py',
            'source_operator':Path(a.__file__),
            'calibration':a.ROOT/'data/radar_calibration.h5',
        }.items():
            h['provenance'].attrs[label+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    print(OUT,flush=True)
if __name__=='__main__':main()
