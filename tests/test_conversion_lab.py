"""Independent analytic checks of the coherent radar observation operator."""
import numpy as np
import pytest
from conversion_lab import coherent_gate,validated


def test_gaussian_gate_selects_downward_wave_and_sets_electron_column():
    z=np.linspace(-4,4,16001);sigma=.4;k=3.;n=2e11;a=.003+.002j
    up,down,column=coherent_gate(z,np.full_like(z,n),a*np.exp(-1j*k*z),0.,sigma,k)
    assert up==pytest.approx(a,rel=1e-10)
    assert down==pytest.approx(np.conj(a)*np.exp(-2*(k*sigma)**2),rel=1e-9)
    assert column==pytest.approx(2*np.sqrt(np.pi)*sigma*n,rel=1e-10)


def test_radar_gate_power_scales_with_drive_and_rejects_mismatched_k():
    z=np.linspace(-4,4,16001);n=np.ones_like(z);a=np.exp(-8j*z)
    u,_,_=coherent_gate(z,n,a,0.,.5,8.)
    u2,_,_=coherent_gate(z,n,2*a,0.,.5,8.)
    off,_,_=coherent_gate(z,n,a,0.,.5,10.)
    assert abs(u2)**2==pytest.approx(4*abs(u)**2)
    assert abs(off)**2/abs(u)**2==pytest.approx(np.exp(-1),rel=1e-9)


def test_resource_limits_reject_excessive_meshes():
    with pytest.raises(ValueError,match='200,000-cell'):
        validated(dict(scale_m=40,dz_m=.00125,profile_mode=0))
