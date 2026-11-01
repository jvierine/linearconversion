"""Measurement-background comparison for the coupled Arecibo calculation.
Density palette: user-requested nipy_spectral; saved blog palette kept separately.
The density map uses restored CSP information; the previously used smooth
propagation profile remains separately identified and is not silently replaced.
"""
from pathlib import Path
import hashlib
import h5py
import numpy as np
from matplotlib import colormaps
ROOT=Path(__file__).resolve().parent
out=ROOT/'output/coupled/density_comparison.h5'
source=ROOT/'data/eregion_fit/full_altitude_fits.h5'
with h5py.File(source) as h, h5py.File(ROOT/'data/smooth_observed_layer_1525453430.h5') as s, h5py.File(ROOT/'output/coupled/recorded.h5') as sim,h5py.File(out,'w') as o:
    z=h['height_km'][:]; select=(z>=49.8)&(z<=140.1)
    t=h['time_unix_s'][:];ne=h['results'][:,select,0]
    assert np.all(np.isfinite(ne))
    it=int(s.attrs['source_time_index']);zs=s['source_height_m'][:]/1000
    np.testing.assert_allclose(h['results'][it,:,0][(z>=100)&(z<=135)],s['source_ne_m3'][:],rtol=1e-13)
    o['map/time_utc_hour']=(t-1525392000)/3600
    o['map/time_edges_utc_hour']=(h['time_edges_unix_s'][:]-1525392000)/3600
    o['map/height_km']=z[select];o['map/old_ne_m3']=ne.T
    with h5py.File(ROOT/'data/eregion_fit/density_native_csp.h5') as d:
        take=(z[select]>=d['z'][0]-1e-6)&(z[select]<=d['z'][-1]+1e-6)
        np.testing.assert_allclose(z[select][take],d['z'][:],atol=1e-8)
        np.testing.assert_array_equal(t,d['t'][:])
        ne[:,take]=d['ne_m3'][:]
        o['profile/csp_height_km']=d['z'][:];o['profile/csp_ne_m3']=d['ne_m3'][it]
        o.attrs['density_method_above_94_5_km']=d.attrs['method']
        o.attrs['density_gain_csp']=d.attrs['density_gain']
    o['map/ne_m3']=ne.T
    o['palette/rgb']=colormaps['plasma'](np.linspace(0,1,256))[:,:3]
    o['palette/density_rgb']=colormaps['nipy_spectral'](np.linspace(0,1,256))[:,:3]
    o['profile/height_km']=zs;o['profile/ne_m3']=s['source_ne_m3'][:]
    # Saved source sigma includes the 5% floor used for smoothing, not formal error.
    keep=(z>=100)&(z<=135)
    o['profile/conditional_sigma_m3']=h['results'][it,keep,4]
    o['profile/valid']=s['source_valid'][:]
    o['smooth/height_km']=s['model_height_m'][:]/1000;o['smooth/ne_m3']=s['model_ne_m3'][:]
    with h5py.File(ROOT/'data/smooth_csp_layer_1525453430.h5') as revised:
        o['smooth_csp/height_km']=revised['model_height_m'][:]/1000
        o['smooth_csp/ne_m3']=revised['model_ne_m3'][:]
    zz=sim['z'][:];cut=(zz>=116500)&(zz<=119500)
    o['simulation/height_km']=zz[cut]/1000;o['simulation/ne_m3']=sim['ne'][0,cut]
    o.attrs.update(selected_utc_hour=(t[it]-1525392000)/3600,selected_time_unix_s=t[it],
        resonance_height_km=sim.attrs['center_m']/1000,
        bragg_height_km=(sim.attrs['center_m']-4.087166)/1000,
        density_calibration_log_scatter=float(s.attrs['calibration_log_scatter']),
        background_fit_residual_rms=float(s.attrs['residual_rms']),
        cmap='nipy_spectral; exact 256-entry Matplotlib lookup table',
        color_limits_log10_m3=[8.,12.],
        blog_url='http://www.radio-science.net/2018/06/day-time-d-and-e-region-at-arecibo.html',
        blog_palette_evidence='Saved arecibo_d_e.png colorbar RGB RMSE: plasma 0.00119, nipy_spectral 0.4167. Archived Python script has a different palette from the published image.',
        provenance=Path(__file__).read_text(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        caveat='Density amplitude at native sampling; CLP shape pooled over 3 cycles x 5 gates above94.5km. Below94.5km previous CSP fits retained. Propagation profile from previous full-parameter fits, separately compared. Log display saturates below 1e8; no white-pixel masks. Conditional errors exclude common calibration uncertainty.')
print(out)
