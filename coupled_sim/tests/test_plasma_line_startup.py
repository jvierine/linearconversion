"""Causal reconstruction tested against independent analytical systems."""
import sys
from pathlib import Path
import numpy as np
import h5py
from scipy.special import erf, erfc

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from plasma_line_startup import turn_on, thresholds


def test_instantaneous_transfer_recovers_source_ramp():
    df=250.; offsets=np.arange(-160000.,160001.,df);sigma=10e-6/np.sqrt(8*np.log(2))
    t,A=turn_on(offsets,np.ones(len(offsets)),df,sigma)
    expected=.5*(1+erf(t/(np.sqrt(2)*sigma)))
    valid=(t>-40e-6)&(t<.0035)
    np.testing.assert_allclose(A[valid].real,expected[valid],atol=5e-5,rtol=0)
    assert max(abs(A[valid].imag))<1e-12


def test_causal_relaxation_recovers_analytical_convolution():
    df=250.;offsets=np.arange(-160000.,160001.,df);sigma=10e-6/np.sqrt(8*np.log(2));tau=100e-6
    t,A=turn_on(offsets,1/(1-2j*np.pi*offsets*tau),df,sigma)
    expected=.5*(erfc(-t/(np.sqrt(2)*sigma))-
        np.exp(-t/tau+.5*(sigma/tau)**2)*erfc((sigma*sigma/tau-t)/(np.sqrt(2)*sigma)))
    valid=(t>-30e-6)&(t<.0035)
    np.testing.assert_allclose(A[valid].real,expected[valid],atol=3e-6,rtol=0)


def test_settling_requires_remaining_in_band_not_first_crossing():
    t=np.arange(8,dtype=float)
    result=thresholds(t,np.array([0.,1.,1.2,1.,.9,1.,1.,1.]),valid_end=7.)
    assert result['first_90pct_s']==1
    assert result['settle_5pct_s']==5


def test_coupled_result_converges_in_space_and_frequency():
    root=Path(__file__).resolve().parents[2]
    with h5py.File(root/'data/plasma_line_startup.h5') as h:
        t=h['ramp_10us/time_s'][:];p=h['ramp_10us/power_normalized'][:]
        use=(t>=0)&(t<.0018)
        coarse=np.interp(t[use],h['checks/coarse_time_s'][:],h['checks/coarse_power'][:])
        df500=np.interp(t[use],h['checks/df500_time_s'][:],h['checks/df500_power'][:])
        assert max(abs(p[use]-coarse))<.008
        assert max(abs(p[use]-df500))<1e-5
        assert max(abs(p[(t>.003)&(t<.0035)]-1))<1e-5
        assert max(p[(t>=0)&(t<.0008)])<1e-8
        for key in ['first_90pct_s','settle_5pct_s']:
            assert abs(h['ramp_10us'].attrs[key]-h['checks/coarse'].attrs[key])<.3e-6
