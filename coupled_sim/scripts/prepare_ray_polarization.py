"""Extract the incident O-ray polarization at the plotted resonance height.

Align the arbitrary complex eigenvector phases before interpolating between
the stored ray samples. Choose the phase with Ez real and positive. The arrow
represents the x-z projection at that phase, not the full-wave field amplitude.
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
    projection = field[[0, 2]].real
    projection /= np.linalg.norm(projection)
    anchor = [(1-fraction)*x[j]+fraction*x[j+1], 0.]
with h5py.File(target, 'w') as out:
    out['anchor_xz_m'] = anchor
    out['unit_projection_xz'] = projection
    out['unit_electric_phasor_xyz'] = field
    out.attrs.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     source_code=Path(__file__).read_text(),
                     meaning='Incident cold O-ray instantaneous x-z electric-field projection at resonance; phase chosen with Ez positive; normalized direction only.')
print('Anchor:', anchor, 'unit E:', field, 'projected direction:', projection)
