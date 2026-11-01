"""Temporal high-pass of the archived CSP density reconstruction.

Source data are filtered directly; no screenshot manipulation, clipping of
negative input estimates, spatial filtering, gap filling or quality masking.
HDF5 retains the original and filtered arrays, filter coefficients and response.
"""
from pathlib import Path
import hashlib
import h5py
import numpy as np
from scipy.signal import butter,sosfiltfilt,sosfreqz

ROOT=Path(__file__).parent
src=ROOT/'evidence/seid_event/ne_result.h5'
out=ROOT/'data/eregion_fit/density_highpass.h5'
with h5py.File(src) as h:
    t=h['unix_seconds'][:];z=h['range_km'][:];ne=h['Ne_m3'][:]
dt=float(np.median(np.diff(t)))
assert np.isfinite(ne).all()
assert np.max(abs(np.diff(t)/dt-1))<.005
with h5py.File(out,'w') as h:
    h['time_unix_s']=t;h['height_km']=z
    h.create_dataset('ne_original_m3',data=ne,compression='gzip')
    h.attrs['source_path']=str(src);h.attrs['source_sha256']=hashlib.sha256(src.read_bytes()).hexdigest()
    h.attrs['provenance']='filter_eregion_density.py; plotter/src/bin/eregion_highpass.rs'
    h.attrs['input_status']='archived CSP density reconstruction, not the new kinetic fit'
    h.attrs['method']='Fourth-order Butterworth high-pass, forward and backward; temporal only, at every native range gate'
    h.attrs['sampling_dt_s']=dt;h.attrs['timestamp_convention']='original ne_result.h5 timestamps retained'
    h.attrs['maximum_cadence_deviation_s']=float(np.max(abs(np.diff(t)-dt)))
    h.attrs['edges']='odd-reflection padding; interpret boundary transients cautiously; no pixels removed'
    for minutes in [20,30,60]:
        g=h.create_group(f'cutoff_{minutes}_min')
        sos=butter(4,1/(60*minutes),btype='highpass',fs=1/dt,output='sos')
        hp=sosfiltfilt(sos,ne,axis=0)
        background=ne-hp
        # Fractional view is rendered only above 95 km, where background>0.
        frac=np.divide(100*hp,background,out=np.full_like(ne,np.nan),where=background>0)
        for k,v in [('delta_ne_m3',hp),('background_ne_m3',background),('fractional_percent',frac)]:g.create_dataset(k,data=v,compression='gzip')
        g['sos']=sos;g.attrs['cutoff_period_min']=minutes
        g.attrs['cutoff_gain']='forward-backward amplitude is 0.5 (-6.02 dB) at the specified cutoff'
        periods=np.geomspace(2*dt/60*1.001,180,300)
        _,response=sosfreqz(sos,worN=1/(periods*60),fs=1/dt)
        g['response_period_min']=periods;g['amplitude_response']=abs(response)**2
        use=(z>=95)&(z<=140)
        assert np.isfinite(frac[:,use]).all()
        print(minutes,'min: delta Ne percentile [1,99]',np.percentile(hp[:,use],[1,99]),'fraction %',np.percentile(frac[:,use],[1,99]))
print(out)
