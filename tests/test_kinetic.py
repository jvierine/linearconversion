import numpy as np
import pytest
from scipy.constants import e,m_e,k as kb,epsilon_0
from scipy.integrate import quad
from isr.kinetic import *

def test_free_particle_normalization_and_debye_limit():
    s=Species('e',1e11,2000,m_e,-e); k=radar_k()
    scale=k*s.sigma/(2*np.pi)
    f=np.linspace(-9*scale,9*scale,40001)
    chi,q=maxwellian_response(f,k,s)
    assert abs(np.trapezoid(q,f)-1)<1e-12
    c0,_=maxwellian_response([0.],k,s)
    assert c0[0].real==pytest.approx(s.density*e**2/(epsilon_0*kb*s.temperature*k*k))

def test_equal_temperature_static_structure_factor():
    # Independent equilibrium sum rule, includes ion AND plasma lines.
    es,ions=oxygen_plasma(ne=2e10,Te=2000,Ti=2000)
    k=radar_k(); a=es[0].wp2/(k*es[0].sigma)**2
    def integrand(x): return density_spectrum(np.array([x]),k,es,ions)['total'][0]
    area=2*sum(quad(integrand,l,r,epsabs=1e-9,limit=400)[0] for l,r in
               [(0,2e4),(2e4,1e6),(1e6,2e6),(2e6,5e6),(5e6,30e6)])
    assert area==pytest.approx((1+a)/(1+2*a),rel=2e-6)

def test_symmetry_and_parallel_limit():
    es,ions=oxygen_plasma(); k=radar_k(); f=np.linspace(-10000,10000,1001)
    a=density_spectrum(f,k,es,ions)['total']
    b=density_spectrum(f,k,es,ions,B=35e-6,theta_deg=0)['total']
    np.testing.assert_allclose(a,a[::-1],rtol=1e-12)
    np.testing.assert_allclose(a,b,rtol=1e-12)
    assert np.all(a>=0)

def test_bgk_static_and_perpendicular():
    s=Species('e',1e11,2000,m_e,-e,100.)
    k=radar_k(); chi,noise=maxwellian_response([0.,100.,1e6],k,s,35e-6,90)
    assert chi[0].real==pytest.approx(s.wp2/(k*s.sigma)**2)
    assert np.all(noise>=0)
    with pytest.raises(ValueError,match='delta'):
        maxwellian_response([100.],k,Species('e',1e11,2000,m_e,-e),35e-6,90)

def test_drift():
    es,ions=oxygen_plasma(); u=300.; k=radar_k(); f=np.linspace(-10000,10000,500)
    from dataclasses import replace
    shifted=density_spectrum(f+k*u/(2*np.pi),k,[replace(es[0],drift=u)],[replace(ions[0],drift=u)])
    np.testing.assert_allclose(shifted['total'],density_spectrum(f,k,es,ions)['total'],rtol=1e-12)
