"""Check controlled illumination, harmonic solve, and field mesh convergence."""
from pathlib import Path
import h5py
import numpy as np
P=Path(__file__).resolve().parents[2]/'reference/coupled/gradient_electric_field.h5'

def test_controlled_fields():
    with h5py.File(P) as h:
        for name in ['L0p5','L1','L2','L5','check_L1','check_L5']:
            g=h[name]
            assert g.attrs['harmonic_residual']<1e-9
            np.testing.assert_allclose(g.attrs['incident_upward_flux_W_m2'],1.,rtol=1e-10)
            np.testing.assert_allclose(g['Ez_over_Einc'][:],abs(g['Ez_complex_over_Einc'][:]),rtol=1e-12)
            assert np.all(np.diff(g['height_relative_m'][:])>0)
            assert np.all(np.isfinite(g['Ez_over_Einc'][:]))
            assert g.attrs['collision_rate_s1']==h['L1'].attrs['collision_rate_s1']

def test_field_refinement():
    with h5py.File(P) as h:
        for name in ['L1','L5']:
            a,b=h[name],h['check_'+name]
            assert abs(a.attrs['peak_Ez_over_Einc']/b.attrs['peak_Ez_over_Einc']-1)<.002
            z=np.linspace(-100,30,26001)
            x=np.interp(z,a['height_relative_m'][:],a['Ez_over_Einc'][:])
            y=np.interp(z,b['height_relative_m'][:],b['Ez_over_Einc'][:])
            assert np.linalg.norm(x-y)/np.linalg.norm(y)<.02
