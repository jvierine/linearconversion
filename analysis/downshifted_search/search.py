"""Audit the downshifted branches without the old bottomside range restriction.

Positive display frequency is upshifted by the event's established convention.
The archive's frequency sign is reversed. No source spectra are modified.
Counts below refer to operational detections, not a formal false-discovery rate.
"""
from pathlib import Path
import sys
import hashlib
from datetime import datetime, timezone
import h5py
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis/plasma_line_rcs'))
from estimate import fit_core, CENTERS


def main():
    output = ROOT / 'data/downshifted_search.h5'
    with h5py.File(ROOT / 'data/plasma_line_rcs_inputs.h5') as inp, h5py.File(ROOT / 'data/plasma_line_rcs.h5') as cat, h5py.File(output, 'w') as out:
        t = inp['time_unix_s'][:]
        z = inp['height_km'][:]
        es = cat['es_peak_height_km'][:]
        out['time_unix_s'] = t
        out['height_km'] = z
        out['es_peak_height_km'] = es
        out.attrs.update(
            scope='All 1407 cached native integrations; 4.51 and 5.34 MHz downshifted bands over 100-139 km. Independent strong-feature screening over 2-8 MHz uses range maxima over 100-130 km.',
            detection='Local altitude maxima: downshifted fitted amplitude >6 off-range MAD scales; paired excess >6 paired MAD scales; amplitude >3 fit RMS; frequency within9.5kHz of nominal; amplitude >half absolute baseline. MAD scales are empirical, not formal confidence intervals.',
            altitude_caution='Density gates300m; plasma-line samples150m. Small differences from the fitted density peak do not establish a resolved topside location.',
            frequency_convention='Minus native archived frequency; negative is downshifted in the established event interpretation.',
            input_sha256=hashlib.sha256((ROOT / 'data/plasma_line_rcs_inputs.h5').read_bytes()).hexdigest(),
            source=Path(__file__).read_text())
        rows = []
        for j in [2, 3]:
            a = cat[f'band{j}/amplitude_by_height'][:]
            b = cat[f'band{j-2}/amplitude_by_height'][:]
            diff = a-b
            off = z < 100
            sigma = 1.4826*np.median(abs(a[:, off]-np.median(a[:, off], axis=1)[:, None]), axis=1)
            dsigma = 1.4826*np.median(abs(diff[:, off]-np.median(diff[:, off], axis=1)[:, None]), axis=1)
            f = inp[f'band{j}/frequency_mhz'][:]
            fo = inp[f'band{j-2}/frequency_mhz'][:]
            for i in range(len(t)):
                peaks, _ = find_peaks(a[i], distance=3)
                for iz in peaks:
                    if not (100 < z[iz] < 139 and a[i, iz] > 6*sigma[i]):
                        continue
                    v, rms, *_ = fit_core(f, inp[f'band{j}/excess'][i, iz], CENTERS[j])
                    vo, *_ = fit_core(fo, inp[f'band{j-2}/excess'][i, iz], CENTERS[j-2])
                    amp, freq, baseline, slope = v
                    passed = amp > 6*sigma[i] and amp > 3*rms and abs(freq-CENTERS[j]) < .0095 and amp > .5*abs(baseline) and amp-vo[0] > 6*dsigma[i]
                    rows.append([j, i, t[i], z[iz], es[i], freq, amp, vo[0], rms, sigma[i], dsigma[i], float(passed)])
        out['screened_rows'] = rows
        out['screened_rows'].attrs['columns'] = 'band,record,time_unix_s,height_km,Es_peak_km,frequency_MHz,down_amplitude,up_amplitude,fit_RMS,down_MAD_scale,paired_MAD_scale,passed'
        # Screen all cached frequency bins for strong additional features.
        f = inp['frequency_mhz'][:]
        sel = np.flatnonzero((f < -2) & (f > -8))
        sel = sel[np.argsort(f[sel])]
        broad = []
        for i, q in enumerate(inp['max_excess']):
            y = q[sel]
            peaks, _ = find_peaks(y, prominence=.5, distance=4)
            broad.extend([[i, t[i], f[sel[k]], y[k]] for k in peaks if y[k] > .75])
        out['broad_strong_screen'] = broad
        out['broad_strong_screen'].attrs['columns'] = 'record,time_unix_s,frequency_MHz,maximum_excess_SNR'
        accepted = [r for r in rows if r[-1]]
        assert len(accepted) == 1
        row = accepted[0]
        i = int(row[1]); iz = int(np.argmin(abs(z-row[3])))
        out['candidate/row'] = row
        for j, name in [(3, 'downshifted'), (1, 'upshifted')]:
            freq = abs(inp[f'band{j}/frequency_mhz'][:]); order = np.argsort(freq)
            out[f'candidate/{name}/frequency_mhz'] = freq[order]
            out[f'candidate/{name}/spectrum'] = inp[f'band{j}/excess'][i, iz][order]
            out[f'candidate/{name}/amplitude_by_height'] = cat[f'band{j}/amplitude_by_height'][i]
        out['candidate/adjacent_times'] = t[i-1:i+2]
        out['candidate/adjacent_down_spectra'] = inp['band3/excess'][i-1:i+2, iz]
        with h5py.File(ROOT / 'data/density_reference_debye.h5') as density:
            td = 1525392000 + 3600*density['map/time_utc_hour'][:]
            zd = density['map/height_km'][:]
            ne = density['map/ne_m3'][:]
            profile = np.array([np.interp(t[i], td, y) for y in ne])
            out['candidate/density_height_km'] = zd
            out['candidate/density_m3'] = profile
            out['candidate/density_bracketing_times'] = td[np.searchsorted(td,t[i])-1:np.searchsorted(td,t[i])+1]
        for r in rows:
            print('PASS' if r[-1] else 'reject', datetime.fromtimestamp(r[2], timezone.utc).isoformat(), 'frequency', r[5], 'height', r[3], 'peak', r[4], flush=True)
        print('Strong broad-band screen:', broad, flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    main()
