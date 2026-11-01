"""Distinguish pump gradient, prescribed force gradient and warm self-field.

Keep the measured density and collisions fixed. Gaussian smooth the complex
three-component imposed source, renormalize its peak effective vertical force,
then solve the same warm-fluid frequency-domain equations. These are prescribed
force sensitivity controls, not newly solved Maxwell propagation scenarios.
"""
import numpy as np
import h5py
from scipy.signal import fftconvolve
from scipy.constants import e,m_e,epsilon_0,k as kb
from drive_fluid_from_fdtd import ROOT,setup,harmonic,save_group
from estimate_hf_power import beam,PT,TSYS,RE,LAM,KR,TAU


def effective(s,include_buffer=False):
    a=s['nu']+(s['sponge'] if include_buffer else 0)-1j*s['omega'];bx,bz=s['bx'],s['bz'];v=s['source']
    return v[2]+bx*bz/(a*a+bz*bz)*v[0]+a*bx/(a*a+bz*bz)*v[1]


def power(s,r):
    zz=s['z'];dz=zz-s['center'];_,a,b,g=beam(zz)
    j=a/(a-1j*b)*np.exp(-s['kx']**2/(4*(a-1j*b)))*np.exp(-2j*g)
    taper=np.sin(.5*np.pi*np.clip(np.minimum(dz+110,40-dz)/10,0,1))**2
    col=np.trapezoid(.5*r['dn_phasor_m3']*np.exp(2j*KR*dz)*j*taper,zz)
    return PT*RE**2*LAM**2*abs(col)**2


def prepare_plot(path):
    with h5py.File(path,'a') as h:
        if 'plot' in h:del h['plot']
        g=h.create_group('plot');z=h['z_face_relative_m'][:]
        use=(z>=-6)&(z<=2);g['height_m']=z[use]
        for name,key in [('cold_gradient','cold_Ez_gradient'),('cold_envelope_gradient','cold_Ez_magnitude_gradient'),
                         ('warm_gradient','warm_total_Ez_gradient'),('source_gradient','complex_source_gradient'),
                         ('effective_gradient','complex_effective_gradient')]:
            g[name]=abs(h['baseline/'+key][:][use])
        scan=h['scan'][:];g['sigma_m']=scan[1:,0];g['scan_gradient_microV_m2']=scan[1:,1]*1e6
        g['scan_snr_db']=10*np.log10(scan[1:,3])
        g.attrs['baseline_gradient_microV_m2']=scan[0,1]*1e6
        g.attrs['baseline_snr_db']=10*np.log10(scan[0,3])
        # This amplitude-only family IS consistent with linear Maxwell scaling:
        # keep the recorded shape, density profile, losses and beam model fixed.
        if 'fixed_shape' in h:del h['fixed_shape']
        fs=h.create_group('fixed_shape');target=np.logspace(-3,3,301)
        factor=np.sqrt(target/h['baseline'].attrs['snr'])
        fs['snr_linear']=target
        fs['peak_cold_gradient_V_m2']=h['summary'].attrs['peak_cold_gradient_V_m2']*factor
        fs['peak_source_gradient_V_m2']=h['baseline'].attrs['peak_source_gradient_V_m2']*factor
        fs.attrs['meaning']='Required gradient at specified SNR, fixed complex spatial field shape and same coherent-beam ion-line calibration. A chosen SNR is not an established observational detection threshold.'


def metrics(s,r,scale,cal,target,core):
    z=s['zf']-s['center'];ex=s['source'][2]*scale;eff=effective(s)*scale
    grad=np.gradient(ex,s['dz']);ge=np.gradient(eff,s['dz'])
    # Exact force-divergence combination after eliminating velocity at fixed omega.
    a=s['nu']-1j*s['omega'];rz=(a*a+s['bz']**2)/(a*(a*a+s['bx']**2+s['bz']**2))
    comp=np.gradient(s['nf']*rz*eff,s['dz'])/(s['nf']*rz)
    snr=power(s,r)*scale**2*cal*TAU/(kb*TSYS)
    required=np.sqrt(target/snr)
    return dict(snr=snr,snr_db=10*np.log10(snr),amplitude_multiplier_for_observed=required,
        peak_source_Ez_V_m=float(max(abs(ex[core]))),
        peak_source_gradient_V_m2=float(max(abs(grad[core]))),
        peak_effective_E_V_m=float(max(abs(eff[core]))),
        peak_effective_gradient_V_m2=float(max(abs(ge[core]))),
        peak_force_divergence_equivalent_V_m2=float(max(abs(comp[core]))),
        required_source_gradient_V_m2=float(max(abs(grad[core])))*required,
        required_effective_gradient_V_m2=float(max(abs(ge[core])))*required,
        complex_source_Ez=ex,complex_effective_E=eff,complex_source_gradient=grad,
        complex_effective_gradient=ge,complex_force_divergence_equivalent=comp)


def main():
    with h5py.File(ROOT/'output/fluid/hf_power_from_measured_pl.h5') as h:
        scale=h.attrs['required_incident_peak_V_m']/h.attrs['reference_incident_peak_V_m']
        cal=h.attrs['ion_line_measured_over_predicted'];target=h.attrs['fitted_on_frequency_snr']
    s=setup(200000,-240.,160.);original=s['source'].copy();zf=s['zf']-s['center'];z=s['z']-s['center']
    core=(zf>-10)&(zf<10);plateau=(zf>-120)&(zf<60)
    eff=effective(s);peak=max(abs(eff[core]));r=harmonic(s)
    base=metrics(s,r,scale,cal,target,core)
    # Verify the effective drive gives the same density with horizontal forces
    # algebraically eliminated (the velocity/energy budgets need not be identical).
    reduced=dict(s);reduced['source']=np.zeros_like(original);reduced['source'][2]=effective(s,include_buffer=True)
    rr=harmonic(reduced)
    reduced_error=np.linalg.norm(rr['density_phasor']-r['density_phasor'])/np.linalg.norm(r['density_phasor'])
    assert reduced_error<1e-8
    out=ROOT/'output/fluid/hf_field_gradient.h5'
    sigmas=[0.,.02,.05,.1,.2,.5,1.,2.,5.,10.,20.]
    rows=[]
    with h5py.File(out,'w') as h:
        h.attrs.update(generator='scan_hf_field_gradient.py',
            definition='Gradients are magnitude of complex phasor derivative, not derivative of magnitude. Units V/m^2. Source is the corrected 1D-compatible drive; cold total Ez is diagnostic only.',
            calibration='Same ion-line-referenced coherent-beam normalization as hf_power_from_measured_pl.h5',
            source_control='Gaussian smoothing of complex vector drive; scalar rescale to retain peak effective vertical force within +/-10m. Density, temperature, B and collisions held fixed. Not self-consistent Maxwell re-solves.',
            cold_warning='Large cold-field resonance gradient includes cold polarization. It is not an independently prescribed source or a threshold.',
            flat_control='Constant complex vector within physical domain; taper only in distant numerical buffers. Chosen to retain effective-force peak. Illustrates compression from density gradient even with dEz_source/dz=0.',
            effective_reduction_relative_error=reduced_error,
            incident_normalization_factor=scale,target_snr_linear=target,
            bragg_height_relative_m=s['match']-s['center'])
        h['z_face_relative_m']=zf;h['z_cell_relative_m']=z;h['ne_face_m3']=s['nf']
        save_group(h.create_group('baseline'),base)
        cold=s['total'][2]*scale
        h['baseline/cold_Ez']=cold;h['baseline/cold_Ez_gradient']=np.gradient(cold,s['dz'])
        h['baseline/cold_Ez_magnitude_gradient']=np.gradient(abs(cold),s['dz'])
        warm=(s['source'][2]+r['self_field_phasor_V_m'])*scale
        h['baseline/warm_total_Ez']=warm;h['baseline/warm_total_Ez_gradient']=np.gradient(warm,s['dz'])
        h['baseline/self_Ez_gradient']=np.gradient(r['self_field_phasor_V_m']*scale,s['dz'])
        summary={key:float(max(abs(v[core]))) for key,v in [
            ('peak_cold_Ez_V_m',cold),('peak_cold_gradient_V_m2',np.gradient(cold,s['dz'])),
            ('peak_cold_magnitude_gradient_V_m2',np.gradient(abs(cold),s['dz'])),
            ('peak_warm_total_gradient_V_m2',np.gradient(warm,s['dz']))]}
        summary.update(bragg_warm_gradient_V_m2=float(np.interp(s['match'],s['zf'],abs(np.gradient(warm,s['dz'])))),
            bragg_density_fraction=float(np.interp(s['match'],s['z'],abs(r['density_phasor'])*scale)),
            density_scale_length_at_resonance_m=float(np.interp(s['center'],s['zf'],s['nf']/np.gradient(s['nf'],s['dz']))))
        save_group(h.create_group('summary'),summary)
        for sigma in sigmas:
            if sigma:
                xx=np.arange(-int(np.ceil(5*sigma/s['dz'])),int(np.ceil(5*sigma/s['dz']))+1)*s['dz']
                ker=np.exp(-.5*(xx/sigma)**2);ker/=ker.sum()
                src=np.array([fftconvolve(v,ker,mode='same') for v in original])
                s['source']=src;normalizer=peak/max(abs(effective(s)[core]));s['source']*=normalizer
            else:s['source']=original.copy();normalizer=1.
            rr=harmonic(s);m=metrics(s,rr,scale,cal,target,core)
            rows.append([sigma,m['peak_source_gradient_V_m2'],m['peak_effective_gradient_V_m2'],m['snr'],m['amplitude_multiplier_for_observed'],m['required_source_gradient_V_m2'],m['required_effective_gradient_V_m2'],normalizer])
            if sigma in [0,.1,1.,10.,20.]:
                g=h.create_group(f'sigma_{sigma:g}m');save_group(g,m)
                g.create_dataset('density_phasor',data=rr['density_phasor']*scale,compression='gzip')
            print('sigma',sigma,'gradient',m['peak_source_gradient_V_m2'],'SNR',m['snr'],'required amplitude',m['amplitude_multiplier_for_observed'],flush=True)
        # Flatten the external vector in the conversion region, not n0 itself.
        j=np.where(core)[0][np.argmax(abs(eff[core]))]
        s['source']=np.tile(original[:,j,None],(1,len(zf)))
        edge=np.minimum(zf-zf[0],zf[-1]-zf)
        s['source']*=np.sin(.5*np.pi*np.clip(edge/60.,0,1))**2
        s['source']*=peak/max(abs(effective(s)[core]))
        rr=harmonic(s);flat=metrics(s,rr,scale,cal,target,core)
        save_group(h.create_group('flat'),flat)
        assert flat['peak_source_gradient_V_m2']<1e-12
        assert flat['snr']>0
        # Move boundaries far away: a flat source can create edge compression,
        # which must not be confused with conversion at the density gradient.
        big=setup(330000,-400.,260.)
        vector=s['source'][:,np.argmin(abs(zf))].copy()
        edge_big=np.minimum(big['zf']-big['zf'][0],big['zf'][-1]-big['zf'])
        big['source']=vector[:,None]*np.sin(.5*np.pi*np.clip(edge_big/60.,0,1))[None,:]**2
        rb=harmonic(big)
        flat_power_ratio=power(big,rb)/power(s,rr)
        assert abs(flat_power_ratio-1)<.005
        homogeneous=dict(big)
        nn=float(np.interp(s['center'],big['z'],big['ne']))*.99
        homogeneous['ne']=np.full_like(big['ne'],nn);homogeneous['nf']=np.full_like(big['nf'],nn)
        homogeneous['nu']=np.full_like(big['nu'],np.interp(s['center'],big['zf'],big['nu']))
        rh=harmonic(homogeneous)
        interior=abs(big['z']-big['center'])<10
        homogeneous_amplitude=float(max(abs(rh['density_phasor'][interior]))*scale)
        flat_amplitude=float(max(abs(rb['density_phasor'][interior]))*scale)
        assert homogeneous_amplitude<1e-4*flat_amplitude
        controls=dict(flat_enlarged_domain_power_ratio=flat_power_ratio,
            flat_inhomogeneous_peak_density_fraction=flat_amplitude,
            flat_homogeneous_peak_density_fraction=homogeneous_amplitude)
        save_group(h.create_group('controls'),controls)
        h['scan']=np.array(rows)
        h['scan'].attrs['columns']='sigma_m,peak_source_Ez_gradient_V_m2,peak_effective_gradient_V_m2,snr,amplitude_multiplier_to_observed,required_source_gradient_V_m2,required_effective_gradient_V_m2,peak_normalization'
        print('BASE', {k:v for k,v in base.items() if np.isscalar(v)},flush=True)
        print('SUMMARY',summary,'FLAT',{k:v for k,v in flat.items() if np.isscalar(v)},'CONTROLS',controls,flush=True)
    prepare_plot(out)
    print(out)


if __name__=='__main__':main()
