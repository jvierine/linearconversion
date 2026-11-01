"""Measured-profile identity, energy conservation and mesh convergence."""
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'reference/coupled/es_fregion_electric_field.h5'

def test_launch_and_energy():
    with h5py.File(P) as h:
        for name in ['Es','F','Es_vertical','F_vertical','Es_fine','F_fine','F_vertical_fine']:
            a=h[name].attrs
            assert a['harmonic_residual']<1e-9
            assert a['mode_boundary_relative_error']<4e-4
            assert abs(a['incident_EM_flux_W_m2']-1)<1e-10
            assert 0<a['reflected_EM_fraction']<1
            assert 0<a['absorbed_heat_fraction']<1
            assert abs(a['flux_minus_heat_pressure'])<5e-4
            assert np.all(h[name]['absorbed_fraction_per_cell'][:]>=0)

def test_actual_fregion_not_transplanted_es():
    with h5py.File(P) as h,h5py.File(ROOT/'coupled_sim/data/fregion_observed_background.h5') as inputs:
        for key in ['Ln_m','temperature_K','center_m']:
            np.testing.assert_allclose(h['F'].attrs[key],inputs['smoothing_10km'].attrs[key],rtol=1e-12)
        assert h['F'].attrs['Ln_m']>50*h['Es'].attrs['Ln_m']
        assert h['F'].attrs['nu_res_s1']<h['Es'].attrs['nu_res_s1']/10

def test_resolution():
    with h5py.File(P) as h:
        for name in ['Es','F','F_vertical']:
            a,b=h[name].attrs,h[name+'_fine'].attrs
            assert abs(a['reflected_EM_fraction']-b['reflected_EM_fraction'])<.001
            assert abs(a['absorbed_heat_fraction']-b['absorbed_heat_fraction'])<.001
            assert abs(a['peak_Ez_over_Einc']/b['peak_Ez_over_Einc']-1)<.002
