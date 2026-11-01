"""Animate the article's exact saved harmonic solution, without a startup claim."""
from pathlib import Path
import hashlib
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/gaussian_pulse'
source = ROOT / 'coupled_sim/output/coupled/minimum_incident/best_model.h5'
if not source.exists():
    source = ROOT / 'reference/coupled/minimum_incident.h5'
with h5py.File(source) as model:
    attrs = dict(model.attrs)
    z = model['z'][:]
    spacing = model['dz_cells'][:] if 'dz_cells' in model else np.r_[np.diff(z), z[-1]-z[-2]]
    zf = z + spacing / 2
    center = float(attrs['center_m'])
    ne = model['ne'][:]
    target = np.interp(center, z, ne)
    crossings = np.where((z[:-1] > center + 100) & ((ne[:-1]-target)*(ne[1:]-target) < 0))[0]
    top = float('nan')
    if len(crossings):
        i = crossings[0]
        top = np.interp(target, ne[i:i+2][::-1], z[i:i+2][::-1])
    heights = center + np.arange(-50, 50.001, .0125)
    scale = np.sqrt(attrs['incident_vertical_flux_W_m2'])
    ez = np.interp(heights, zf, model['E_phasor_V_m'][2]) * scale
    density = np.interp(heights, z, model['density_phasor_m3'][:]) * scale
    frequency = float(attrs['frequency_hz'])
    times = np.arange(8 * 48) / (48 * frequency)
    with h5py.File(OUT / 'steady_movie.h5', 'w') as h:
        h.attrs.update(attrs)
        h.attrs.update(carrier_hz=frequency, kx_m1=float(model['kx_rad_m'][()]),
            bottom_resonance_m=center, top_resonance_m=top,
            source_label='Continuous illumination: steady state',
            display_note='48 frames per radio-wave cycle; no skipped cycles; sustained response',
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        h['time_s'] = times
        h['height_m'] = heights
        h['views/closeup'] = [0, len(heights)]
        h.create_dataset('Ez', data=np.broadcast_to(ez, (len(times), len(ez))), compression='gzip')
        h.create_dataset('density', data=np.broadcast_to(density, (len(times), len(ez))), compression='gzip')
        h['provenance/source'] = Path(__file__).read_text()
        assert np.allclose(np.exp(-2j*np.pi*frequency*times[48]),1)
        assert np.max(abs(ez)) <= attrs['peak_Ez_V_m'] * 1.001
        print('Article steady state:', frequency, 'Hz;', len(times), 'frames;', times[-1]*1e6, 'microseconds')
