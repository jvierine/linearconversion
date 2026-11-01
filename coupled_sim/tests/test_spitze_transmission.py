from pathlib import Path
import h5py
import numpy as np
P=Path(__file__).resolve().parents[2]/'reference/coupled/spitze_transmission.h5'

def test_flux_and_heating_close_in_both_windows():
    with h5py.File(P) as h:
        for name in ['upward_window','downward_window','upward_window_fine','downward_window_fine']:
            a=h[name].attrs
            assert a['harmonic_residual']<1e-8
            assert a['mode_boundary_relative_error']<2e-4
            assert abs(a['flux_minus_heat_pressure'])<1e-4
            np.testing.assert_allclose(h[name]['absorbed_fraction_per_cell'][:].sum(),a['absorbed_heat_fraction'],atol=1e-10)

def test_mesh_and_signed_direction():
    with h5py.File(P) as h:
        for name in ['upward_window','downward_window']:
            for key in ['reflected_EM_fraction','transmitted_EM_fraction','absorbed_heat_fraction']:
                assert abs(h[name].attrs[key]-h[name+'_fine'].attrs[key])<.001
        assert h['upward_window_fine'].attrs['angle_deg']<0<h['downward_window_fine'].attrs['angle_deg']
        assert h.attrs['frequency_hz']<h.attrs['peak_plasma_frequency_hz']<h.attrs['field_aligned_L_max_plasma_frequency_hz']
