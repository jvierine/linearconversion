"""Archive diagnostic definitions, comparisons and plotting arrays in HDF5."""
from pathlib import Path
import numpy as np,h5py
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from gradient_comparison import OUT,KB

def read(case,fine=False):
    p=OUT/f'{case}_16.75_0.0025_0.5_1.5.h5'
    if not fine or not p.exists():p=OUT/f'{case}_16.75_0.005_1_1.h5'
    return h5py.File(p)
with read('measured',True) as h:reference=h.attrs['bragg_column_per_sqrt_flux']
with h5py.File(OUT/'summary.h5','w') as out:
    out.attrs.update(reference_bragg_column_per_sqrt_flux=reference,window_sigma_m=10.,
        diagnostic='|integral dn(z) exp(i kB(z-zB)) exp(-(z-zB)^2/(2 sigma^2)) dz|^2 / S_inc',
        note='Local vertical Bragg-response proxy, not calibrated radar SNR. Same incident upward flux; polarization optimized separately; no Landau damping.',
        plotted_floor_dB=-120.,frequency_Hz=5340028.553893307,B_T=3.506470278825485e-5,B_angle_deg=46.815539793042)
    for family,cases in [('E',['E0.5','E1','E2','E5','E10','E20']),('F',['F5','F20','F40'])]:
        vals=[]
        for case in cases:
            with read(case,True) as h:
                a=dict(h.attrs);a['relative_response_dB']=20*np.log10(a['bragg_column_per_sqrt_flux']/reference)
                g=out.create_group('cases/'+case);g.attrs.update(a)
                vals.append([a['Ln_m']/1000,max(-120,a['relative_response_dB'])])
                print(case,'dB',a['relative_response_dB'],'Bragg m',a['bragg_distance_m'])
        out[f'plot/{family}_Ln_km']=np.array(vals)[:,0];out[f'plot/{family}_response_dB']=np.array(vals)[:,1]
    for case in ['measured','E5','F20']:
        vals=[]
        for p in OUT.glob(f'{case}_*_0.005_1_1.h5'):
            with h5py.File(p) as h:vals.append([h.attrs['angle_deg'],max(-120,20*np.log10(h.attrs['bragg_column_per_sqrt_flux']/reference))])
        vals=np.array(sorted(vals));out[f'plot/{case}_angles']=vals[:,0];out[f'plot/{case}_angular_dB']=vals[:,1]
    with read('measured',True) as h:
        out.create_group('cases/measured').attrs.update(dict(h.attrs))
        out['plot/measured_point']=np.array([h.attrs['Ln_m']/1000,0.])
    # Save a real-wave detail of the F-region counterexample at the same
    # incident flux used to normalize the event model; not fitted to F data.
    with read('F20',True) as h:
        z=h['z_m'][:]-h.attrs['center_m'];dn=h['density_phasor_m3'][:];ne=h['ne_m3'][:]
        match=-h.attrs['bragg_distance_m'];idx=np.argmin(abs(z-match))
        dn*=np.exp(-1j*np.angle(dn[idx]))
        v=(z>match-2)&(z<match+2);scale=np.sqrt(1.19649e-11)*1e6
        out['plot/F_wave_height_m']=z[v]-match;out['plot/F_wave_real_ppm']=(dn/ne).real[v]*scale;out['plot/F_wave_amplitude_ppm']=abs(dn/ne)[v]*scale
        # Average local phase gradient over a one-metre Gaussian to remove
        # wavelength-scale beating with the weak forced background.
        v=(z>-355)&(z<-195);zcut=z[v];k=h['phase_kz_rad_m'][:][v]
        k=gaussian_filter1d(k,1./np.median(np.diff(zcut)))
        keep=np.flatnonzero((zcut>-350)&(zcut<-200))[::20]
        out['plot/F_height_m']=zcut[keep];out['plot/F_wavelength_m']=np.where(k[keep]<-1,2*np.pi/-k[keep],np.nan)
        out.attrs['wavelength_phase_smoothing_sigma_m']=1.
        out.attrs['F_bragg_relative_m']=match;out.attrs['bragg_wavelength_m']=2*np.pi/KB
    checks=[]
    for case in ['measured','E1','E5','F5','F20']:
        with read(case) as coarse,read(case,True) as fine:
            change=(fine.attrs['bragg_column_per_sqrt_flux']/coarse.attrs['bragg_column_per_sqrt_flux'])**2-1
            checks.append(change); print('refinement',case,'power change',change)
    out['refinement_power_fraction_change']=checks
    out['refinement_power_fraction_change'].attrs['cases']='measured,E1,E5,F5,F20'
    with h5py.File(OUT/'noEs_16.75_0.005_0.5_1.h5') as h:
        out.create_group('cases/noEs').attrs.update(dict(h.attrs))
    # Preserve all successful run settings and numerical checks compactly.
    for i,p in enumerate(sorted(OUT.glob('*.h5'))):
        if p.name=='summary.h5':continue
        with h5py.File(p) as h:
            if 'bragg_column_per_sqrt_flux' in h.attrs:
                out.create_group(f'runs/{i}').attrs.update(dict(h.attrs))
    out['provenance/study_source']=(Path(__file__).parent/'gradient_comparison.py').read_text()
    out['provenance/shared_solver_source']=(Path(__file__).parent/'coupled_arecibo.py').read_text()
    out['provenance/summary_source']=Path(__file__).read_text()
