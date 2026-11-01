"""Compare total-density and linearized runs; retain compact HDF5 diagnostics."""
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
with h5py.File(ROOT/'output/scalar_total.h5') as full,h5py.File(ROOT/'output/scalar_total_linear.h5') as linear:
    z=full['z_m'][:];mask=(z>117930)&(z<117960)
    n0=full['equilibrium_density_m3'][:]
    nf=full['electron_density_m3'][-1];nl=linear['electron_density_m3'][-1]
    error=np.linalg.norm((nf-nl)[:,mask])/np.linalg.norm((nl-n0)[:,mask])
    sol=full['initial_phasor_normalized'][:]
    phase=2*np.pi*np.arange(nf.shape[0])/nf.shape[0]-full.attrs['omega_si']*full['time_s'][-1]
    expected=np.real(sol[0,None,:]*np.exp(1j*phase[:,None]))*full.attrs['density_si']
    drift=np.linalg.norm((nl-n0-expected)[:,mask])/np.linalg.norm(expected[:,mask])
    assert nf.min()>0 and error<.01 and drift<1e-5,(error,drift)
    with h5py.File(ROOT/'reference/scalar_total_check.h5','w') as h:
        h.attrs.update(relative_density_wave_difference=error,relative_linear_harmonic_drift=drift,
            cycles=full.attrs['omega_si']*full['time_s'][-1]/(2*np.pi),
            initial_fractional_density_amplitude=float(np.max(abs(sol[0])*full.attrs['density_si']/n0)),
            scope='One-period weak-amplitude comparison on one grid, not spatial convergence or steady state')
        h['z_m']=z[mask];h['x_m']=full['x_m'][:];h['equilibrium_density_m3']=n0[mask]
        h['total_electron_density_m3']=nf[:,mask];h['linear_total_electron_density_m3']=nl[:,mask]
        h['provenance/source']=Path(__file__).read_text()
    print('relative density-wave difference',error,'linear harmonic drift',drift)
