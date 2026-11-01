"""Place the coupled electric field at the ray's plotted resonance crossing.

Align the arbitrary complex eigenvector phases before interpolating between
the stored ray samples. Choose the phase with Ez real and positive.
Retain the cold-ray polarization for comparison, but plot the coupled field,
including the resonant response. Interpolate Yee components at the same height.
"""
from pathlib import Path
import hashlib
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'coupled_sim/reference/resonant_access_audit.h5'
target = ROOT / 'data/resonant_access_polarization.h5'
with h5py.File(source) as h:
    z = h['ray_3/height_m'][:] - h.attrs['resonance_height_m']
    x = h['ray_3/horizontal_m'][:]
    x -= x[np.argmax(z)]
    e = h['ray_3/electric_polarization'][:]
    j = np.flatnonzero((z[:-1] < 0) & (z[1:] >= 0))[0]
    fraction = -z[j] / (z[j+1] - z[j])
    aligned = e[j:j+2] * np.exp(-1j*np.angle(e[j:j+2, 2]))[:, None]
    field = (1-fraction)*aligned[0] + fraction*aligned[1]
    field /= np.linalg.norm(field)
    assert np.max(abs(field[[0, 2]].imag)) < 1e-10
    cold_field = field.copy()
    anchor = [(1-fraction)*x[j]+fraction*x[j+1], 0.]
    center = float(h.attrs['resonance_height_m'])
solution = ROOT / 'coupled_sim/output/coupled/minimum_incident/best_model.h5'
with h5py.File(solution) as h:
    z = h['z'][:]
    widths = h['dz_cells'][:]
    electric = h['E_phasor_V_m'][:]
    positions = [z, z, z+widths/2]
    reference_index = np.argmin(abs(positions[2]-115100.))
    incident_amplitude = np.linalg.norm(h['incident_reference_state'][:3, reference_index])
    with h5py.File(source) as audit:
        np.testing.assert_allclose(abs(electric[2])/incident_amplitude,
                                   audit['Ez_over_incident_peak'][:], rtol=1e-12)
    assert abs(center-h.attrs['center_m']) < 1e-6
    field = np.array([np.interp(center, positions[j], electric[j].real)
                      +1j*np.interp(center, positions[j], electric[j].imag)
                      for j in range(3)])
    field *= np.exp(-1j*np.angle(field[2]))
    field /= np.linalg.norm(field)
projection = field[[0, 2]].real
projection /= np.linalg.norm(projection)
with h5py.File(target, 'w') as out:
    out['anchor_xz_m'] = anchor
    out['unit_projection_xz'] = projection
    out['unit_electric_phasor_xyz'] = field
    out['cold_ray_unit_electric_phasor_xyz'] = cold_field
    for j, name in enumerate(['Ex', 'Ey', 'Ez']):
        height = positions[j]-center
        keep = (height > -251) & (height < 61)
        out[f'fields/{name}_height_m'] = height[keep]
        out[f'fields/{name}_over_incident'] = abs(electric[j, keep])/incident_amplitude
    out.attrs.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     solution_sha256=hashlib.sha256(solution.read_bytes()).hexdigest(),
                     source_code=Path(__file__).read_text(),
                     meaning='Coupled full-wave instantaneous x-z electric-field projection at resonance, anchored at the incident ray crossing; phase chosen with Ez positive; normalized direction only.')
print('Anchor:', anchor, 'unit E:', field, 'projected direction:', projection)
