"""Independent solver/mesh checks and causal-pulse checks retained with products."""
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def test_transient_is_causal_and_has_requested_duration():
    with h5py.File(ROOT/'reference/coupled/gaussian_pulse_validation.h5') as h:
        assert h.attrs['carrier_hz']==5.34e6
        assert abs(h.attrs['incident_field_envelope_FWHM_s']/10e-6-1)<.03
        assert h.attrs['prepulse_relative_Ez']<.002
        assert h.attrs['late_relative_Ez']<.002
        time=h['time_s'][:]
        assert time[-1]>.003 and np.all(np.diff(time)>0)
        for key in ['Ez','density','Jx','Jz','EM_flux','pressure_flux']:
            assert np.all(np.isfinite(h[key][:]))

def test_refined_frequency_responses_agree_with_rust_and_mesh():
    with h5py.File(ROOT/'reference/coupled/gaussian_pulse_validation.h5') as h:
        for key in ['-50000.0','0.0','50000.0']:
            a=h['mesh_and_Rust_steps/'+key].attrs
            assert a['amplitude_mesh_error']<.05
            assert a['complex_mesh_error']<.1
            assert a['finite']
            assert a['relative_max_e_error']<1e-10
            assert a['relative_max_q_error']<1e-10
