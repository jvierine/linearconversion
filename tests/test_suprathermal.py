import numpy as np
import pytest
from scipy.constants import e,m_e
from isr.suprathermal import *
from isr.kinetic import radar_k

def test_distribution_normalization():
    E=np.linspace(1,100,9901); p=energy_distribution(E)
    assert np.trapezoid(p,E)==pytest.approx(1.)
    assert np.all(p>=0)

def test_independent_tail_susceptibility():
    E=np.linspace(1,100,19801); p=energy_distribution(E)
    for angle in [0,25,60,85]:
        a=tail_terms(2*np.pi*6e6,radar_k(),35e-6,angle,E,p,1e-4,30,'parts')
        b=tail_terms(2*np.pi*6e6,radar_k(),35e-6,angle,E,p,1e-4,30,'derivative')
        np.testing.assert_allclose(a[0],b[0],rtol=1e-13)
        np.testing.assert_allclose(a[1],b[1],rtol=5e-4)

def test_harmonic_and_energy_convergence():
    E=np.linspace(1,100,9901); p=energy_distribution(E)
    a=plasma_line(5e11,1500,35e-6,40,E,p,nmax=24)
    b=plasma_line(5e11,1500,35e-6,40,E,p,nmax=36)
    assert a['enhancement']==pytest.approx(b['enhancement'],rel=1e-6)
    E2=np.linspace(1,100,19801); p2=energy_distribution(E2)
    c=plasma_line(5e11,1500,35e-6,40,E2,p2)
    assert a['enhancement']==pytest.approx(c['enhancement'],rel=5e-4)

def test_weak_tail_zero_limit():
    E=np.linspace(1,100,2001); p=energy_distribution(E)
    a=plasma_line(5e11,1500,35e-6,40,E,p,eta=0)
    assert a['enhancement']==pytest.approx(1.)
    assert a['stable']

def test_parallel_independent_of_B():
    E=np.linspace(1,100,2001); p=energy_distribution(E)
    a=plasma_line(5e11,1500,20e-6,0,E,p)
    b=plasma_line(5e11,1500,50e-6,0,E,p)
    assert a['area']==pytest.approx(b['area'],rel=1e-10)
