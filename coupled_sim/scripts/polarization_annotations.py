"""Map saved lower-boundary cold eigenvectors onto the polarization sweep."""
from pathlib import Path
import hashlib
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'coupled_sim/reference/resonant_access_audit.h5'
out = ROOT / 'data/polarization_annotations.h5'
with h5py.File(source) as src, h5py.File(out, 'w') as dst:
    with h5py.File(ROOT / 'reference/coupled/radar_sensitivity.h5') as sweep:
        assert np.isclose(src.attrs['angle_deg'], sweep.attrs['reference_angle_deg'])
    dst.attrs['incidence_angle_deg'] = src.attrs['angle_deg']
    dst.attrs['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    dst.attrs['provenance'] = Path(__file__).read_text()
    dst.attrs['definition'] = 'alpha=atan(abs(Ey/Ex)); phase=arg(Ey/Ex); cold collisionless incident eigenvectors'
    for name, key, expected in [('O', 'ray_3', 'ordinary'), ('X', 'ray_2', 'extraordinary')]:
        g = src[key]
        assert g.attrs['cold_mode'] == expected
        assert np.isclose(g['height_m'][0], 115300.)
        field = g['electric_polarization'][0]
        ratio = field[1] / field[0]
        point = [np.degrees(np.arctan(abs(ratio))), np.degrees(np.angle(ratio))]
        dst[name] = point
        dst[name].attrs['reference_height_m'] = float(g['height_m'][0])
        print(name, point)
