"""Measure transient phase mixing separately from wave-energy transport."""
from pathlib import Path
import h5py
import numpy as np
from scipy.constants import e, epsilon_0, m_e

ROOT = Path(__file__).resolve().parents[2]
with h5py.File(ROOT / 'output/gaussian_pulse/movie.h5') as h:
    a, b = h['views/closeup'][:]
    z = h['height_m'][a:b]
    ne = h['ne_m3'][a:b]
    t = h['time_s'][:]
    dn = h['density'][:, a:b].astype(complex)
    relative_z = z - h.attrs['bottom_resonance_m']
    kz = np.gradient(np.unwrap(np.angle(dn), axis=1), z, axis=1)
    gradient = np.gradient(np.sqrt(ne * e**2 / (epsilon_0 * m_e)), z)
    j = np.argmin(abs(relative_z + 10))
    selected = (t > 150e-6) & (t < 800e-6)
    slope, _ = np.polyfit(t[selected], kz[selected, j], 1)
with h5py.File(ROOT / 'reference/coupled/gaussian_phase_mixing.h5', 'w') as out:
    out['time_s'] = t
    out['height_from_resonance_m'] = relative_z
    out['kz_m1'] = kz
    out['density_amplitude_m3'] = abs(dn)
    out['cold_plasma_frequency_gradient_rad_s_m'] = gradient
    out.attrs['kz_time_fit_height_m'] = relative_z[j]
    out.attrs['kz_time_fit_slope_m1_s1'] = slope
    out.attrs['cold_phase_mixing_prediction_m1_s1'] = -gradient[j]
    out.attrs['interpretation'] = (
        'A short pulse excites a finite altitude band. Differential plasma '
        'frequency generates temporal phase mixing in addition to warm-fluid '
        'transport. Time-dependent wavelength contraction alone does not '
        'demonstrate propagation. Phase gradients at negligible amplitude '
        'must not be interpreted as meaningful wave numbers.'
    )
    out['provenance/source'] = Path(__file__).read_text()
print('Measured dkz/dt:', slope, 'cold prediction:', -gradient[j])
