"""Independent physical normalization and diffraction checks in the saved audit.

Regenerate with coupled_sim/scripts/audit_power_calibration.py when an input
changes. These checks enforce analytical limits and grid convergence, not an
assumed watt estimate for the unidentified transmitter.
"""
from pathlib import Path
import hashlib
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def test_source_and_radiometry():
    with h5py.File(ROOT / 'data/power_calibration_audit.h5') as h:
        src = h['source'].attrs
        assert abs(src['upward_flux_for_normalized_source_W_m2'] - 1) < 1e-8
        np.testing.assert_allclose(src['computed_E_rms_V_m'], src['stored_E_rms_V_m'], rtol=1e-8)
        np.testing.assert_allclose(h['common_scale_cancellation'][:, 3], h.attrs['reference_flux_W_m2'], rtol=1e-12)
        assert abs(h['baseline'].attrs['heldout_median_ratio'] - 1) < .02
        np.testing.assert_allclose(h['baseline'].attrs['required_flux_W_m2'], h.attrs['reference_flux_W_m2'], rtol=1e-10)


def test_fresnel_analytic_limit_and_convergence():
    with h5py.File(ROOT / 'data/power_calibration_audit.h5') as h:
        g = h['apertures/Gaussian FFT'].attrs
        assert g['gaussian_H_relative_error'] < 1e-10
        assert g['gaussian_J_relative_error'] < 1e-10
        for group in ['apertures', 'checks']:
            for result in h[group].values():
                assert result.attrs['max_propagation_energy_error'] < 1e-10
        fine = h['apertures/Uniform 225 m'].attrs['required_flux_W_m2']
        coarse = h['checks/dx4'].attrs['required_flux_W_m2']
        wide = h['checks/wide_domain'].attrs['required_flux_W_m2']
        assert abs(coarse / fine - 1) < .002
        assert abs(wide / coarse - 1) < .006


def test_audit_matches_current_inputs():
    inputs = {
        'audit': ROOT / 'coupled_sim/scripts/audit_power_calibration.py',
        'coherent_operator': ROOT / 'isr/coherent.py',
        'source_operator': ROOT / 'coupled_sim/scripts/coupled_arecibo.py',
        'calibration': ROOT / 'coupled_sim/data/radar_calibration.h5',
        'plasma_fits': ROOT / 'data/eregion_fit/full_altitude_fits.h5',
    }
    with h5py.File(ROOT / 'data/power_calibration_audit.h5') as h:
        for label, path in inputs.items():
            if label == 'plasma_fits' and not path.exists():
                assert h['inputs/plasma_parameters'].shape == (len(h['inputs/height_m']), 4)
                continue
            assert hashlib.sha256(path.read_bytes()).hexdigest() == h['provenance'].attrs[label + '_sha256'], f'Regenerate power audit: {label} changed'
