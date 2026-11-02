"""Prepare Figure 3 from native spectra; retain unfiltered data and provenance."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
root=Path(__file__).resolve().parents[1]
source=root/'data/plasma_line_movie_native.h5'
with h5py.File(source) as src, h5py.File(root/'data/paper_plasma_lines.h5','w') as out:
    t=src['time_unix_s'][:]; f=src['frequency_mhz'][:]; z=src['height_km'][:]
    out['frequency_mhz']=f; out['height_km']=z
    for name,date in [('enhanced','2018-05-04T17:03:50+00:00'),('ordinary','2018-05-04T14:52:31+00:00')]:
        i=int(np.argmin(abs(t-datetime.fromisoformat(date).timestamp())))
        q=src['snr'][i].astype(float)
        background=gaussian_filter1d(q,.100/np.median(np.diff(f)),axis=1,mode='nearest')
        g=out.create_group(name)
        g['excess_snr']=q; g['smooth_background']=background
        g['filtered_excess_snr']=q-.85*background
        g.attrs.update(time_unix_s=t[i],record_index=i,utc=datetime.fromtimestamp(t[i],timezone.utc).isoformat())
        for key in ['source_file','source_sha256']:
            if key in src: g.attrs[key]=src[key][i]
    with h5py.File(root/'data/conversion_overlay_1525453430.h5') as overlay:
        for key in ['ion_height_km','ion_plasma_line_mhz','model_height_km','model_plasma_line_mhz']:
            out[key]=overlay[key][:]
    out.attrs.update(gaussian_sigma_mhz=.100,background_subtraction_fraction=.85,filter_axis='frequency',display_floor_db=-20.,display_ceiling_db=5.,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    out['provenance/script']=Path(__file__).read_text()
print(root/'data/paper_plasma_lines.h5')
