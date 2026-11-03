"""Causal turn-on of coherent plasma-line power from the coupled resolvent.

Use the saved frequency responses of the actual Yee/fluid recurrence, not a
time envelope imposed on its harmonic solution. Retardation is evaluated at
each sideband frequency, with the common radar return delay removed. All
source and diagnostic provenance is retained in HDF5, never in the figure.
"""
from pathlib import Path
import argparse, hashlib, sys
import h5py
import numpy as np
from scipy.constants import epsilon_0, k as kb
from scipy.fft import fft
from scipy.integrate import cumulative_trapezoid
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isr.coherent import sideband_amplitude

OUT = ROOT / 'data/plasma_line_startup.h5'


def turn_on(offset, transfer, df, sigma, center=60e-6, nfft=32768):
    """Convolve causal linear response with an error-function source ramp."""
    t = np.arange(nfft) / (df * nfft)
    weights = df * sigma * np.sqrt(2*np.pi) * np.exp(-.5*(2*np.pi*offset*sigma)**2)
    weights = weights * np.exp(2j*np.pi*offset*center)
    spectrum = np.zeros(nfft, complex)
    spectrum[np.rint(offset/df).astype(int) % nfft] = transfer * weights
    pulse = fft(spectrum)
    step = cumulative_trapezoid(pulse, t, initial=0) / (sigma*np.sqrt(2*np.pi))
    return t-center, step


def thresholds(t, power, valid_end=.0035):
    keep = (t >= 0) & (t <= valid_end)
    tt, pp = t[keep], power[keep]
    result = {}
    for level in [.1, .5, .9]:
        ix = np.flatnonzero(pp >= level)
        result[f'first_{int(level*100)}pct_s'] = float(tt[ix[0]]) if len(ix) else np.nan
    for tolerance in [.05, .01]:
        bad = np.flatnonzero(abs(pp-1) > tolerance)
        settled = bool(len(bad) == 0 or bad[-1] < len(tt)-1)
        value = tt[bad[-1]+1] if settled and len(bad) else tt[0] if settled else np.nan
        result[f'settle_{int(tolerance*100)}pct_s'] = float(value)
    result.update(peak_power_ratio=float(pp.max()), peak_time_s=float(tt[np.argmax(pp)]),
                  final_power_ratio=float(pp[-1]), valid_end_s=valid_end)
    return result


def extract(path):
    with h5py.File(path) as h:
        assert h.attrs['complete']
        f0 = float(h.attrs['carrier_hz']); df = float(h.attrs['frequency_spacing_hz'])
        center = float(h.attrs['bottom_resonance_m']); kx = float(h.attrs['kx_m1'])
        start, stop = h['views/bottomside'][:]
        z = h['height_m'][start:stop]; ne = h['ne_m3'][start:stop]
        rel = z-center
        window = np.sin(.5*np.pi*np.clip(np.minimum(rel+100,30-rel)/10,0,1))**2
        use = np.flatnonzero((rel >= -100) & (rel <= 30))
        lo, hi = int(use[0]), int(use[-1]+1)
        z, ne, window = z[lo:hi], ne[lo:hi], window[lo:hi]
        offsets = np.array(sorted(float(k) for k in h['response']))
        response = []
        for offset in offsets:
            density = h[f'response/{offset:+.3f}'][9, start+lo:start+hi]
            density = density.astype(complex)*np.sqrt(epsilon_0*ne/(3*kb*300))
            response.append(sideband_amplitude(z,density,f0+offset,kx,window,reference_range_m=center))
        attrs = {k:h.attrs[k] for k in ['carrier_hz','angle_deg','peak_incident_vertical_flux_W_m2',
                                        'bottom_resonance_m','dz_m','dt_s']}
    response = np.asarray(response)
    print(path.name, 'frequencies', len(offsets), 'samples', len(z), flush=True)
    return offsets, response, df, attrs


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--skip-coarse',action='store_true')
    args=parser.parse_args()
    offsets,H,df,attrs=extract(ROOT/'output/gaussian_pulse/response_dz0.002_z60.h5')
    H0=H[np.argmin(abs(offsets))]
    normalized=H/H0
    with h5py.File(OUT,'w') as out:
        out.attrs.update(attrs)
        out.attrs.update(frequency_spacing_hz=df,normalization='Received coherent upshifted sideband power / its continuous-wave asymptote',
            time_origin='Source-current switch-on midpoint at lower numerical source; common radar-return delay removed',
            physics='Fixed measured smooth profile; magnetized coupled Yee/electron fluid; 300 K; en+ei collisions; no kinetic Landau damping',
            steady_received_model_W=float(abs(H0)**2),
            diagnostic='Same coherent antenna overlap and -100 to +30 m integration window as article; instantaneous sideband-envelope power',
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        out['frequency_offset_Hz']=offsets;out['received_sideband_transfer']=H
        for width in [10e-6,20e-6]:
            sigma=width/np.sqrt(8*np.log(2));t,A=turn_on(offsets,normalized,df,sigma)
            power=abs(A)**2
            # Trailing coherent 44 us average, rather than averaging power.
            samples=round(44e-6/(t[1]-t[0]))
            integ=cumulative_trapezoid(A,t,initial=0)
            filtered=np.zeros_like(A)
            filtered[samples:]=(integ[samples:]-integ[:-samples])/(samples*(t[1]-t[0]))
            g=out.create_group(f'ramp_{round(width*1e6)}us');g.attrs['derivative_FWHM_s']=width
            g.attrs.update(thresholds(t,power))
            g['time_s']=t;g['received_amplitude_normalized']=A;g['power_normalized']=power
            g['source_power_envelope']=(.5*(1+erf(t/(np.sqrt(2)*sigma))))**2
            g['power_44us_normalized']=abs(filtered)**2
            g.require_group('smoothed_metrics').attrs.update(thresholds(t,abs(filtered)**2))
            print(g.name,dict(g.attrs),flush=True)
        # Frequency spacing / periodic tail check using every other response.
        t2,A2=turn_on(offsets[::2],normalized[::2],df*2,10e-6/np.sqrt(8*np.log(2)))
        out['checks/df500_time_s']=t2;out['checks/df500_power']=abs(A2)**2
        if not args.skip_coarse:
            off,Hc,dfc,ac=extract(ROOT/'output/gaussian_pulse/response_dz0.01.h5')
            tc,Ac=turn_on(off,Hc/Hc[np.argmin(abs(off))],dfc,10e-6/np.sqrt(8*np.log(2)))
            out['checks/coarse_time_s']=tc;out['checks/coarse_power']=abs(Ac)**2
            out.require_group('checks/coarse').attrs.update(thresholds(tc,abs(Ac)**2))
            out['checks'].attrs['coarse_dz_m']=ac['dz_m']
        out['provenance/source']=Path(__file__).read_text()
    print(OUT,flush=True)


if __name__=='__main__': main()
