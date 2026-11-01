"""Conditional radiometric errors for the paired plasma-line spectra.

The recovered lp_an.py averages nipp-10 nonoverlapping pulse spectra.
For independent Gaussian Fourier amplitudes, sigma(P)=P/sqrt(M).
This includes Gaussian signal self-noise; it is conservative relative to a
constant coherent tone with the same power in additive Gaussian receiver noise.
It does not estimate intermittency, common noise-reference uncertainty, or
calibration uncertainty. Zero padding correlates neighboring frequency bins.
"""
from pathlib import Path
import hashlib
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    intervals = ROOT / 'radar/legacy_2018/intervals.h5'
    with h5py.File(intervals) as h:
        times = h['txs'][:] / 1e6
        counts = h['nipp'][:]
    with h5py.File(ROOT / 'output/fluid/plasma_line_asymmetry.h5', 'a') as h:
        for epoch in ['1525453430', '1525453540', '1525460596']:
            g = h[epoch]
            raw = ROOT / f'evidence/seid_event/results4/spl-{epoch}.h5'
            with h5py.File(raw) as s:
                t = float(s['t0'][()])
                z, f, power = s['range'][:], -s['freqs'][:], s['PL'][:]
            i = int(np.argmin(abs(times-t)))
            assert abs(times[i]-t) < 1e-5
            m = int(counts[i])-10
            assert m > 0
            off = (z < 105) | (z > 130)
            noise = np.median(power[off], axis=0)
            iz = int(np.argmin(abs(z-g.attrs['height_km'])))
            for name, sign in [('enhanced', 1), ('opposite', -1)]:
                b = g[name]
                idx = np.array([np.argmin(abs(f-sign*x)) for x in b['frequency_magnitude_mhz'][:]])
                q = power[iz, idx]/noise[idx]-1
                np.testing.assert_allclose(q, b['signed_excess_snr'][:], rtol=1e-12, atol=1e-12)
                sigma = (q+1)/np.sqrt(m)
                if 'snr_sampling_sigma' in b:
                    del b['snr_sampling_sigma']
                b['snr_sampling_sigma'] = sigma
                print(epoch, name, 'M=', m, 'max sigma=', float(sigma.max()))
            g.attrs['spectra_averaged'] = m
            g.attrs['sampling_error_assumptions'] = '1 sigma; independent Gaussian pulse spectra; noise reference held fixed; adjacent frequency bins correlated'
            g.attrs['sampling_count_source_sha256'] = hashlib.sha256(intervals.read_bytes()).hexdigest()
        h.attrs['sampling_error_script'] = Path(__file__).read_text()

if __name__ == '__main__':
    main()
