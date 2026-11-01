"""Physical limits and conservation checks for the linear conversion solver."""
import numpy as np
import pytest
from isr.conversion import Parameters, matrix, modes, scatter, layer, matching_ratio, optimal_polarization


def test_unmagnetized_modes_obey_maxwell_and_bohm_gross():
    par = Parameters(magnetic_t=0, q=.2)
    p = .65
    mu, v, _, _ = modes(p, par)
    assert np.max(abs(matrix(p, 0, 0, par)@v-v*mu)) < 1e-8
    expected = np.sort([1-p-par.q**2]*4+[(1-p)/par.beta-par.q**2]*2)
    np.testing.assert_allclose(np.sort(-mu**2).real, expected, rtol=1e-12)


def test_uniform_magnetized_plasma_has_no_conversion_or_reflection():
    par = Parameters()
    z = np.linspace(-2, 2, 801)
    r = scatter(z, np.full_like(z, .65), np.zeros_like(z), np.zeros_like(z), par)
    assert r['conversion'] < 1e-20
    assert r['reflection'] < 1e-20
    assert r['transmission'] == pytest.approx(1, abs=1e-10)


def test_unmagnetized_normal_incidence_cannot_convert():
    z = np.linspace(-60, 60, 12001)
    r = scatter(z, *layer(z, 10), Parameters(magnetic_t=0, q=0))
    assert r['conversion'] < 1e-20
    assert r['reflection'] == pytest.approx(1, abs=1e-10)


def test_s_polarization_cannot_convert_in_unmagnetized_layer():
    z = np.linspace(-60, 60, 12001)
    r = scatter(z, *layer(z, 10), Parameters(magnetic_t=0), branch=1)
    assert r['conversion'] < 1e-20
    assert r['reflection'] == pytest.approx(1, abs=1e-10)


def test_pressure_flux_restores_energy_balance_with_refinement():
    errors, conversion = [], []
    for dz in [.01, .005, .0025]:
        z = np.linspace(-60, 60, round(120/dz)+1)
        r = scatter(z, *layer(z, 10), Parameters())
        errors.append(abs(r['energy_error']))
        conversion.append(r['conversion'])
    assert errors[-1] < .001
    assert errors[1] < .35*errors[0] and errors[2] < .35*errors[1]
    assert abs(conversion[2]-conversion[1]) < .002


def test_exact_local_match_reduces_to_unmagnetized_dispersion():
    par = Parameters(magnetic_t=0)
    k = np.array([.03, 0, -18.024])
    p, = matching_ratio(k, par)
    assert p == pytest.approx(1-par.beta*(k@k)/par.k0**2, abs=1e-9)


def test_collisions_remove_positive_wave_power():
    z = np.linspace(-60, 60, 48001)
    r = scatter(z, *layer(z, 10, collision_s=1000), Parameters())
    assert r['absorption'] > 0
    assert abs(r['energy_error']) < .001


def test_coherent_optimum_is_flux_normalized_and_conserves_energy():
    z = np.linspace(-60,60,48001)
    r = optimal_polarization(z,*layer(z,10),Parameters())
    assert np.sum(abs(r['incoming_weights'])**2) == pytest.approx(1, abs=1e-12)
    assert 0 < r['conversion'] < 1
    assert r['reflection'] >= 0
    assert abs(r['energy_error']) < .002


def test_oblique_hf_and_small_surface_tilt_keep_radar_exactly_vertical():
    from vertical_radar_conversion import parameters, K_RADAR, B_VERTICAL_DEG
    for q in [-.5, 0, .5]:
        par, delta = parameters(q)
        a = np.deg2rad(delta)
        rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                             [-np.sin(a), 0, np.cos(a)]])
        kr = np.array([q*par.k0, 0, -np.sqrt(K_RADAR**2-(q*par.k0)**2)])
        np.testing.assert_allclose(rotation@kr, [0, 0, -K_RADAR], atol=1e-12)
        b = rotation@par.gyro
        assert np.rad2deg(np.arctan2(b[0], b[2])) == pytest.approx(B_VERTICAL_DEG)
        assert abs(delta) < .18
