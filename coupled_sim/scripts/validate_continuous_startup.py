"""Check zero initial wave state and convergence of the causal switch-on movie."""
from pathlib import Path
import h5py
import numpy as np
from scipy.constants import epsilon_0, k as kb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/gaussian_pulse'
with h5py.File(OUT/'response_dz0.002_z60.h5') as response, h5py.File(OUT/'startup_movie.h5') as movie:
    z = response['height_m'][:]
    ov = slice(*response['views/overview'][:])
    bo = slice(*response['views/bottomside'][:])
    relative_z = z[bo] - response.attrs['bottom_resonance_m']
    indices = np.r_[np.arange(len(z))[ov], np.arange(len(z))[bo][(relative_z>=-50)&(relative_z<=50)], np.arange(len(z))[bo][::20]]
    np.testing.assert_allclose(movie['height_m'][:], z[indices], atol=1e-9)
    state = response['response/+0.000'][:][:, indices]
    expected = {'Ez': state[2], 'density': state[9]*np.sqrt(epsilon_0*movie['ne_m3'][:]/(3*kb*300))}
    with h5py.File(ROOT/'reference/coupled/continuous_startup_validation.h5', 'w') as out:
        out.attrs.update(dict(movie.attrs))
        out['time_s'] = movie['time_s'][:]
        assert movie['time_s'][0] == 0
        for key in ['Ex','Ey','Ez','Jx','Jy','Jz','density','EM_flux','pressure_flux']:
            assert np.max(abs(movie[key][0])) == 0, key
        out.attrs['all_stored_wave_fields_zero_initially'] = True
        for name, target in expected.items():
            initial = movie[name][0]
            final = movie[name][-1]
            initial_error = np.max(abs(initial)) / np.max(abs(target))
            final_error = np.max(abs(final-target)) / np.max(abs(target))
            out.attrs[name+'_initial_relative_error'] = initial_error
            out.attrs[name+'_final_relative_error'] = final_error
            assert initial_error == 0
            assert final_error < .005, (name, final_error)
            print(name, 'initial', initial_error, 'final harmonic error', final_error)
            # Compact temporal traces retain both sides of the resonance.
            close = movie['height_m'][:] - movie.attrs['bottom_resonance_m']
            for offset in [-10., -4., 0., 10.]:
                i = int(np.argmin(abs(close-offset)))
                out[f'{name}/height_{offset:+g}m'] = movie[name][:,i]
        out['provenance/source'] = Path(__file__).read_text()
