"""Compare regenerated scientific diagnostics with the archived paper case."""
from pathlib import Path
import h5py
import numpy as np

root = Path(__file__).resolve().parents[1]
for name, ref, fields in [
    ('summary.h5', 'coupled_summary.h5', [
        'bragg_height_relative_m', 'incident_flux_to_match_observed_W_m2',
        'required_density_at_bragg_ppm', 'reflected_fraction',
        'collisional_loss_fraction', 'max_downward_pressure_flux_fraction']),
    ('losses.h5', 'losses.h5', [
        'electron_ion_loss_fraction', 'electron_neutral_loss_fraction',
        'cold_collision_loss_fraction'])]:
    with h5py.File(root/'output/coupled'/name) as got, h5py.File(root/'reference'/ref) as expected:
        for field in fields:
            np.testing.assert_allclose(got.attrs[field], expected.attrs[field], rtol=1e-7, atol=1e-12)
            print(field, got.attrs[field], 'PASS')
