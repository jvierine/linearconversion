"""Check causal snapshot reconstruction, phase geometry, and spatial resolution."""
from pathlib import Path
import h5py
import numpy as np

DATA=Path(__file__).resolve().parents[2]/'data/steady_spatial_cuts.h5'

def test_horizontal_phase_and_units():
    with h5py.File(DATA) as h:
        z=h['height_relative_m'][:]
        for name,key,units in [('Ez','Ez_envelope_V_m',1e3),('density','density_envelope_m3',1e-6)]:
            xe=h[name+'/x_edges_m'][:];ze=h[name+'/z_edges_m'][:]
            x=(xe[:-1]+xe[1:])/2;zc=(ze[:-1]+ze[1:])/2
            a=np.interp(zc,z,h[key][:])
            expected=np.real(a[:,None]*np.exp(1j*h.attrs['kx_m1']*x-2j*np.pi*h.attrs['carrier_hz']*.0015))*units
            np.testing.assert_allclose(h[name+'/instantaneous'][:],expected,rtol=1e-12,atol=1e-12)
        assert h.attrs['bragg_window_relative_L2_density_error']<.05

def test_bragg_wave_resolution_and_contraction():
    with h5py.File(DATA) as h:
        assert np.max(np.diff(h['density/z_edges_m'][:])) < .346/30
        z=h['height_relative_m'][:];n=h['density_envelope_m3'][:]
        kz=np.gradient(np.unwrap(np.angle(n)),z)
        k_high=np.median(kz[(z>-2.5)&(z<-2)])
        k_low=np.median(kz[(z>-5.5)&(z<-5)])
        assert k_low<k_high<0, (k_low,k_high)
