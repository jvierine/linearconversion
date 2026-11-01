"""Package existing HDF5 equilibrium as JSON transport for the browser solver.

Scientific source remains HDF5. Do not fit/rescale density for a chosen pump.
"""
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np
from scipy.special import airy

ROOT = Path(__file__).resolve().parent
profiles = []
prior = ROOT / 'data/eregion_fit/full_altitude_priors.h5'
prior_hash = hashlib.sha256(prior.read_bytes()).hexdigest()
def neutral_profile(epoch):
    with h5py.File(prior) as h:
        times = h['time_unix_s'][:]
        j = np.clip(np.searchsorted(times, epoch)-1, 0, len(times)-2)
        fraction = np.clip((epoch-times[j])/(times[j+1]-times[j]), 0, 1)
        z = h['msis/height_km'][:]
        keep = (z >= 90) & (z <= 150)
        out = dict(z=(z[keep]*1000).tolist())
        for name in ['N2_m3', 'O2_m3', 'O_m3', 'Tn_K']:
            values = h['msis/'+name][j:j+2, keep]
            out[name] = ((1-fraction)*values[0]+fraction*values[1]).tolist()
        out['provenance'] = dict(source=str(prior.relative_to(ROOT)), sha256=prior_hash,
            time_bracket_unix_s=times[j:j+2].tolist(), time_fraction=float(fraction),
            model='MSIS 2.1; saved historical atmospheric prior, not a neutral measurement',
            altitude_convention='Saved prior identifies vertical radar range with model altitude; station elevation correction omitted')
        return out
for epoch in (1525453430, 1525460596):
    source = ROOT / f'data/smooth_observed_layer_{epoch}.h5'
    with h5py.File(source) as h:
        profiles.append(dict(
            epoch=epoch, time=h.attrs['time'],
            z=h['model_height_m'][:].tolist(),
            ne=h['model_ne_m3'][:].tolist(),
            dne=h['model_dne_dz_m4'][:].tolist(),
            measured_z=h['source_height_m'][:].tolist(),
            measured_ne=h['source_ne_m3'][:].tolist(),
            magnetic_t=float(h.attrs['magnetic_t']),
            field_angle_deg=float(h.attrs['field_angle_deg']),
            limitation=h.attrs['limitation'],
            neutrals=neutral_profile(epoch),
            provenance=dict(source=str(source.relative_to(ROOT)),
                            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            calibrated_source_sha256=h.attrs['source_sha256'])))
(ROOT/'web/fluid/profiles.json').write_text(json.dumps(profiles, separators=(',', ':')))
print('Packaged two existing calibrated density profiles.')
# Dimensionless Airy solution; JSON is browser transport, HDF5 scientific data.
x=np.linspace(-50,15,32501)
peak=float(airy(-1.018792971647471)[0])
values=airy(x)[0]/peak
with h5py.File(ROOT/'data/fluid_airy_reference.h5','w') as h:
    h['argument']=x;h['signed_field_over_peak']=values
    h.attrs['reference']='https://farside.ph.utexas.edu/teaching/plasma/Plasma/node76.html; equations 6.19-6.21'
    h.attrs['provenance']='prepare_fluid_browser.py; scipy.special.airy'
    h.attrs['normalization']='Ai(x) / Ai(-1.018792971647471); signed phasor, not absolute value'
(ROOT/'web/fluid/airy.json').write_text(json.dumps(dict(x=x.tolist(),field=values.tolist()),separators=(',',':')))
