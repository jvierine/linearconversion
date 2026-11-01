"""Checks of the numerical inverse and exclusion of the second enhancement."""
import numpy as np
from scipy.signal import fftconvolve
import deconvolve_plasma_line as d

def test_excluded_feature_cannot_change_psf(monkeypatch):
    original=d.load
    before=d.estimate()[0]
    def altered(epoch):
        f,z,p,bg,sigma,path=original(epoch)
        p[:,(f>=4)&(f<=5)]+=1e30
        return f,z,p,bg,sigma,path
    monkeypatch.setattr(d,'load',altered)
    np.testing.assert_array_equal(d.estimate()[0],before)

def test_inverse_two_sources_and_sidelobes():
    k=np.zeros((9,11));k[4,5]=.7;k[1,7]=.15;k[7,3]=.15
    truth=np.zeros((40,60));truth[19,30]=10.;truth[22,19]=2.
    a=fftconvolve(truth,k,mode='same')
    x,res,*_=d.invert(a,k,1e-6,iterations=250)
    assert np.linalg.norm(res)/np.linalg.norm(a)<1e-5
    np.testing.assert_allclose(x,truth,atol=2e-5)
