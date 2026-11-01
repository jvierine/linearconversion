"""Extract waves ONLY from saved total electron density.
Keep background, total density, density differences, and horizontal harmonics
separate. Spatial phase during switch-on is not a steady temporal spectrum.
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
from scipy.constants import c

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('label');a=p.parse_args()
source=ROOT/'output'/f'{a.label}.h5';target=ROOT/'reference'/f'{a.label}_diagnostics.h5'
with h5py.File(source) as f,h5py.File(target,'w') as h:
    z=f['z_m'][:];n_eq=f['equilibrium_density_m3'][:];time=f['time_s'][:];omega=f.attrs['omega_si']
    zr=float(f['resonance_m'][()]);local=(z>zr-10)&(z<zr+5);nx=len(f['x_m'])
    phase=np.exp(-2j*np.pi*np.arange(nx)/nx)
    wave=[];total=[];fields=[];peak=[];gauss=f['gauss_max_normalized'][:]
    for i in range(len(time)):
        n=f['electron_density_m3'][i]
        delta=n-n_eq[None,:]
        harmonic=2*np.mean(delta*phase[:,None],axis=0)
        wave.append(harmonic[local]);total.append(n[0,local])
        peak.append(np.max(abs(harmonic[local])/n_eq[local]))
        ef=f['state'][i,:,:,4:7]*f.attrs['electric_si']
        fields.append(2*np.mean(ef*phase[:,None,None],axis=0))
    wave=np.array(wave);total=np.array(total);fields=np.array(fields)
    adjacent_phase=np.angle(wave[-1,1:]*wave[-1,:-1].conj())
    grid_alternation_fraction=float(np.mean(abs(adjacent_phase)>.9*np.pi))
    assert np.array(peak).max()>0 and np.isfinite(wave).all()
    if f.attrs['initialization']=='equilibrium':
        # Python/Rust physical-unit conversion can differ by a final bit.
        np.testing.assert_allclose(f['electron_density_m3'][0],np.broadcast_to(n_eq,(nx,len(z))),rtol=2e-15,atol=0.)
        assert np.max(abs(wave[0])/n_eq[local])<2e-15
    h['time_s']=time;h['z_m']=z;h['height_relative_m']=z[local]-zr;h['equilibrium_density_m3']=n_eq
    h['total_density_x0_m3']=total
    h['delta_density_x0_m3']=total-n_eq[local]
    h['density_horizontal_fundamental_m3']=wave
    h['electric_horizontal_fundamental_V_m']=fields
    h['peak_fractional_wave']=peak;h['source_envelope']=f['source_envelope'][:]
    h['gauss_max_normalized']=gauss
    h['final_total_density_x0_full_profile_m3']=f['electron_density_m3'][-1,0]
    h['final_delta_density_x0_full_profile_m3']=f['electron_density_m3'][-1,0]-n_eq
    h.attrs.update(initialization=f.attrs['initialization'],cycles=omega*time[-1]/(2*np.pi),resonance_m=zr,
                   minimum_total_density_m3=float(np.min(f['electron_density_m3'][-1])),
                   max_gauss_residual=float(gauss.max()),nominal_bragg_wavelength_m=c/(2*430e6),
                   grid_alternation_fraction=grid_alternation_fraction,
                   scope='Transient total-density run; not kinetic Landau damping or a converged event RCS estimate')
    if f.attrs['initialization']=='harmonic':
        ref=f['harmonic_reference_normalized'][0,local]*f.attrs['density_si']*np.exp(-1j*omega*time[-1])
        h.attrs['final_complex_wave_change_from_linear_reference']=np.linalg.norm(wave[-1]-ref)/np.linalg.norm(ref)
    h['provenance/source']=Path(__file__).read_text();h.attrs['source_data_file']=str(source)
    print({k:v for k,v in h.attrs.items() if k not in ['source_data_file']})
    print('Initial and final peak fractional wave:',peak[0],peak[-1])
    if grid_alternation_fraction>.1:
        print('UNRESOLVED: near-Nyquist vertical phase increments; do not interpret this transient as a resolved plasma wave.')
print(target)
