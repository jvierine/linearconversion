"""Density smoothing must preserve Es and supply consistent smooth gradients."""
import numpy as np
import pytest
from isr.observed_layer import fit_observed_layer,profile,scattering_profile
from conversion_lab import bottomside_match


def test_observed_peak_is_preserved_and_background_is_smoothed():
    f=fit_observed_layer()
    z=f['source_height_m']; raw=f['source_ne_m3']
    model,derivative,_,_=profile(z,f['coefficients'],f['knots_km'])
    center=f['es_center_m'];outside=abs(z-center)>2500
    assert np.max(model)/np.max(raw)==pytest.approx(1,abs=.08)
    assert abs(z[np.argmax(model)]-z[np.argmax(raw)])<=300
    assert np.std(np.diff(model[outside])) < .2*np.std(np.diff(raw[outside]))
    numerical=(profile(z+.01,f['coefficients'],f['knots_km'])[0]-profile(z-.01,f['coefficients'],f['knots_km'])[0])/.02
    np.testing.assert_allclose(derivative,numerical,rtol=1e-6,atol=1e-2)
    lower=np.linspace(z[0],center-3*f['es_sigma_m'],1001)
    assert np.min(profile(lower,f['coefficients'],f['knots_km'])[1])>=-1e-5


def test_boundary_continuation_is_flat_and_keeps_interior_fit():
    f=fit_observed_layer();c=f['es_center_m'];w=f['es_sigma_m']
    z=np.linspace(c-5*w,c+5*w,10001)
    n,dn=scattering_profile(z,f['coefficients'],f['knots_km'])
    assert dn[0]==dn[-1]==0
    inside=(z>z[0]+400)&(z<z[-1]-400)
    np.testing.assert_array_equal(n[inside],profile(z[inside],f['coefficients'],f['knots_km'])[0])
    assert np.all(n>0)


def test_bottomside_match_selects_rising_flank_of_finite_layer():
    z=np.arange(5.);n=np.array([1.,2.,4.,2.,1.])
    assert bottomside_match(z,n,3.)==1.5
    with pytest.raises(ValueError,match='not rescaled'):
        bottomside_match(z,n,5.)
