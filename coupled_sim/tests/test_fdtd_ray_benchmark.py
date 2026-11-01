"""Independent WKB/Airy comparison and native/mesh checks for saved FDTD runs."""
from pathlib import Path
import h5py
import numpy as np
P=Path(__file__).resolve().parents[2]/'reference/coupled/fdtd_ray_benchmark.h5'

def test_scalar_reflection_and_asymptotic_fields():
    with h5py.File(P) as h:
        g=h['scalar_fine'];a=g.attrs
        assert abs(a['reflected_EM_fraction']-1)<3e-6
        assert a['transmitted_EM_fraction']<1e-10
        assert a['WKB_field_NRMSE_3_10_airy']<.01
        assert a['WKB_field_NRMSE_1_3_airy']<.04
        assert a['Airy_field_NRMSE_within_one_length']<.01
        assert 40<a['airy_length_m']<45
        coarse=h['scalar_coarse'];z=g['z_m'][:];v=g['E_over_incident'][:]
        other=np.interp(z,coarse['z_m'][:],coarse['E_over_incident'][:]);m=(z>a['turn_height_m']-500)&(z<a['turn_height_m']+100)
        assert np.linalg.norm((v-other)[m])/np.linalg.norm(v[m])<.002

def test_native_recurrence_and_magnetized_refinement():
    with h5py.File(P) as h:
        for key in ['scalar_coarse','scalar_fine','magnetized_coarse','magnetized_fine']:
            assert h[key].attrs['max_all_recorded_E_error']<1e-8
        a=h['magnetized_refined'];b=h['magnetized_finest']
        assert abs(a.attrs['reflected_EM_fraction']-b.attrs['reflected_EM_fraction'])<2e-6
        z=b['z_m'][:];v=b['Ez_over_incident'][:];w=np.interp(z,a['z_m'][:],a['Ez_over_incident'][:]);m=abs(z-b.attrs['resonance_height_m'])<100
        assert np.linalg.norm((v-w)[m])/np.linalg.norm(v[m])<.02
