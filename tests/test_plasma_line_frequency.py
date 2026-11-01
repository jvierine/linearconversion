"""Physical limits and parity with the independently used calibration response."""
import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import e,m_e,epsilon_0,k as kb,c
from isr.plasma_line_frequency import frequency_hz,dielectric
from fit_eregion import response


def test_parallel_field_has_no_frequency_correction():
    for te in [300.,1000.]:
        assert_allclose(frequency_hz(3e11,te,35e-6,0.),frequency_hz(3e11,te,0.,0.),rtol=1e-10)


def test_unmagnetized_bohm_gross_limit():
    ne=3e11;te=30.;k=4*np.pi*430e6/c
    leading=np.sqrt(ne*e**2/(epsilon_0*m_e)+3*k*k*kb*te/m_e)/(2*np.pi)
    assert_allclose(frequency_hz(ne,te,0.,0.,sideband=0),leading,rtol=2e-7)


def test_cold_oblique_upper_branch():
    # Exact cold longitudinal dispersion; a small finite Te avoids cancellation.
    ne=3e11;B=35e-6;angle=47.;wp2=ne*e**2/(epsilon_0*m_e);oc2=(e*B/m_e)**2
    cold=np.sqrt(.5*(wp2+oc2+np.sqrt((wp2+oc2)**2-4*wp2*oc2*np.cos(np.deg2rad(angle))**2)))/(2*np.pi)
    assert_allclose(frequency_hz(ne,1.,B,angle,sideband=0),cold,rtol=1e-5)


def test_matches_calibration_dielectric():
    for ne,te in [(1e11,300.),(3e11,1000.)]:
        f=frequency_hz(ne,te,35e-6,47.,sideband=0)
        chi,_=response(np.array([2*np.pi*f]),ne,te,m_e,0.,35e-6,47.)
        assert abs((1+chi[0]).real)<1e-8
        assert_allclose(1+chi[0],dielectric(f,ne,te,35e-6,47.,sideband=0),atol=1e-11)
        assert f>frequency_hz(ne,te,0.,0.,sideband=0)
