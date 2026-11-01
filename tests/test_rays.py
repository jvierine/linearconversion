"""Published Muldrew benchmark and independent ray identities."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.constants import e, m_e, k as kb
from isr.conversion import Parameters, matching_ratio
from isr.landau import dielectric, spatial_root
from isr.rays import (spitze_angle_deg, cold_horizontal_group_point,
                      electromagnetic_dispersion, kinetic_group_velocity,
                      electrostatic_ray, electromagnetic_ray)


def test_muldrew_1993_cold_benchmark():
    par=Parameters(frequency_hz=8.175e6,temperature_k=550,
                   magnetic_t=2*np.pi*1e6*m_e/e,magnetic_tilt_deg=42.)
    angle=spitze_angle_deg(par)
    par=replace(par,q=np.sin(np.deg2rad(angle)))
    result=cold_horizontal_group_point(par)
    assert angle==pytest.approx(12.762,abs=.0005)
    assert result['refractive_index']==pytest.approx(4.743,abs=.0005)
    assert result['density_ratio']==pytest.approx(.992882,abs=5e-7)


def test_warm_tensor_agrees_with_independent_full_wave_density_eigenvalue():
    par=Parameters(frequency_hz=5.33e6,temperature_k=300,magnetic_t=35e-6,q=-.3)
    k=np.array([par.q*par.k0,0.,-18.])
    p=matching_ratio(k,par)[0]
    D=electromagnetic_dispersion(k,p,par)
    assert np.max(abs(D-D.conj().T))<1e-10
    assert min(abs(np.linalg.eigvalsh(D)))<1e-9


def test_unmagnetized_kinetic_group_direction_and_bohm_gross_limit():
    par=Parameters(frequency_hz=5.33e6,temperature_k=300,magnetic_t=0,q=-.3)
    kd=18.;p=-1/(dielectric(kd,1.,par).real-1)
    vg=kinetic_group_velocity(par,p,kd)
    kvec=np.array([par.q*par.k0,-kd])
    assert vg[0]/vg[1]==pytest.approx(kvec[0]/kvec[1],rel=1e-4)
    assert vg[1]==pytest.approx(3*kb*300/m_e*kvec[1]/par.omega,rel=.02)


def test_ray_clock_and_spatial_temporal_damping_agree():
    par=Parameters(frequency_hz=5.33e6,temperature_k=300,magnetic_t=0,q=-.3)
    kd=18.;p=-1/(dielectric(kd,1.,par).real-1)
    root=spatial_root(p,par,kd+1e-9j)
    z=np.linspace(0,-10,101)
    ray=electrostatic_ray(z,np.full_like(z,p),np.full(z.shape,root),par,np.full_like(z,.1))
    assert ray['travel_time_s'][-1]>0
    assert ray['path_length_m'][-1]>=10
    gamma=ray['collisional_temporal_amplitude_loss_s1'][0]
    assert ray['collisional_power_survival'][-1]==pytest.approx(np.exp(-2*gamma*ray['travel_time_s'][-1]))
    assert ray['collisional_power_survival'][-1]==pytest.approx(np.exp(-2))


def test_incoming_em_ray_matches_analytic_unmagnetized_parabola():
    par=Parameters(frequency_hz=5.33e6,magnetic_t=0,q=.3)
    p0=.2;slope=1e-4;kz=par.k0*np.sqrt(1-p0-par.q**2)
    ray=electromagnetic_ray(0,kz,lambda z:(p0+slope*z,slope),par,max_step_s=2e-6)
    from scipy.constants import c
    t=ray['time_s'];expected_z=c*kz/par.k0*t-c*c*slope*t*t/4
    assert np.max(abs(ray['height_m']-expected_z))<.01
    assert np.max(abs(ray['horizontal_m']-c*par.q*t))<.01
    assert max(abs(ray['dispersion_residual']))<1e-6
    assert ray['stop_reason']=='returned to lower boundary'
