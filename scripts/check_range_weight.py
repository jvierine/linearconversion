"""Compare a rectangular-bit voltage triangle with the thin-layer approximation.
The numerical edge taper only truncates the resolved integration domain.
"""
from pathlib import Path
import sys
import h5py
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isr.coherent import sideband_amplitude

source = ROOT / 'reference/coupled/minimum_incident.h5'
with h5py.File(source) as h:
    z = h['z'][:]
    density = h['density_phasor_m3'][:]
    kx = float(h['kx_rad_m'][()])
    frequency = float(h.attrs['frequency_hz'])
    resonance = float(h.attrs['center_m'])
relative = z - resonance
integration_taper = np.sin(.5*np.pi*np.clip(np.minimum(relative+100, 30-relative)/10, 0, 1))**2

def power(weight):
    return abs(sideband_amplitude(z, density, frequency, kx, integration_taper*weight,
                                  reference_range_m=resonance))**2

baseline = power(1.)
offsets = np.array([0., -4.1, -150., 150.])
ratios = np.array([power(np.maximum(1-abs(relative-offset)/300., 0))/baseline for offset in offsets])
assert np.isfinite(ratios).all() and (ratios > 0).all()
out = ROOT / 'reference/coupled/range_weight_check.h5'
with h5py.File(out, 'w') as h:
    h['gate_center_relative_to_resonance_m'] = offsets
    h['triangle_to_unit_received_power_ratio'] = ratios
    h['power_change_dB'] = 10*np.log10(ratios)
    h.attrs.update(bit_duration_s=2e-6, triangle_half_support_m=300.,
                   scope='Central voltage main lobe only; same numerical taper in both integrals; original model and its unit-weight power normalization unchanged.',
                   source=str(source.relative_to(ROOT)), bragg_gate_offset_m=-4.1)
    h['provenance/script'] = Path(__file__).read_text()
print(out)
for offset, ratio in zip(offsets, ratios):
    print(f'Gate offset {offset:g} m: power ratio {ratio:.8f}, {10*np.log10(ratio):.5f} dB')
