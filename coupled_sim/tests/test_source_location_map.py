"""Independent geometric checks and azimuth-rotation regression checks."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import source_location_map as s
from minimum_incident_power import solve

def test_vacuum_ray_and_source_reversal():
    z=np.linspace(0,118000,501)
    x,y,top,_=s.ray(z,np.zeros_like(z),14,53,0,0)
    assert x==pytest.approx(118000*np.tan(np.deg2rad(14)),rel=1e-7)
    assert abs(y)<1e-6
    assert top==118000
    # Upward ray goes north, hence source is south of its interaction point.
    beta=0.;north=-(x*np.cos(beta)+y*np.sin(beta))
    assert north<0

def test_unmagnetized_stratified_integral():
    z=np.linspace(0,118000,2001);q=.24
    X=.7*z/z[-1]
    ne=X*s.epsilon_0*s.m_e*s.a.W**2/s.e**2
    x,y,_,_=s.ray(z,ne,np.degrees(np.arcsin(q)),0,0,0)
    expected=2*q*z[-1]/.7*(np.sqrt(1-q*q)-np.sqrt(1-q*q-.7))
    assert x==pytest.approx(expected,rel=1e-7)
    assert abs(y)<1e-5

def test_rotating_incidence_plane_matches_signed_meridian():
    # Rotating both horizontal axes by pi maps positive to negative kx.
    p=s.coupling(14,180)[0];n=s.coupling(-14,0)[0]
    assert p==pytest.approx(n,rel=1e-8)
    # The new azimuth parameter defaults exactly to the original equations.
    base=s.a.setup(angle=14,dz=.04,cap=2.)
    rot=s.a.setup(angle=14,dz=.04,cap=2.,scenario={'field_azimuth_deg':0})
    assert np.array_equal(base['initial'],rot['initial'])

def test_cold_mode_satisfies_maxwell_dispersion():
    eps=s.dielectric(np.array([1e10,1e11,2e11]),[.1,.04,-.14])
    qx,qy=.24,.03;nz=s.ordinary_nz(eps,qx,qy)
    for ee,nn in zip(eps,nz):
        n=np.array([qx,qy,nn]);D=ee+np.outer(n,n)-np.dot(n,n)*np.eye(3)
        assert abs(np.linalg.det(D))<1e-12
