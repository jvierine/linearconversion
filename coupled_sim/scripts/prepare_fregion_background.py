"""Recreate the F-region smooth profiles from the bundled native fit samples."""
from pathlib import Path
import h5py,numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.constants import e,m_e,epsilon_0
ROOT=Path(__file__).resolve().parents[1]
with h5py.File(ROOT/'data/fregion_observed_background.h5','a') as h:
    z=h['height_m'][:];raw=h['native_ne_Te_Ti'][:]
    b=h['B_T'][:];theta=h['field_angle_deg'][:];w=2*np.pi*h.attrs['frequency_Hz']
    for sigma in [5000.,10000.,15000.]:
        ne=gaussian_filter1d(np.median(raw[:,:,0],axis=0),sigma/300,mode='reflect')
        te=gaussian_filter1d(np.median(raw[:,:,1],axis=0),sigma/300,mode='reflect')
        prof=CubicSpline(z,ne)
        def residual(zz):
            om=e*np.interp(zz,z,b)/m_e;oz=om*np.cos(np.deg2rad(np.interp(zz,z,theta)))
            return prof(zz)*e*e/(m_e*epsilon_0)-w*w*(w*w-om*om)/(w*w-oz*oz)
        zr=brentq(residual,210000,270000);key=f'smoothing_{sigma/1000:g}km'
        if key in h:del h[key]
        g=h.create_group(key);g['ne_m3']=ne;g['Te_K']=te
        g.attrs.update(sigma_m=sigma,center_m=zr,Ln_m=float(prof(zr)/prof(zr,1)),
            temperature_K=float(np.interp(zr,z,te)),magnetic_t=float(np.interp(zr,z,b)),
            field_angle_deg=float(np.interp(zr,z,theta)))
        print(dict(g.attrs))
    key='provenance/portable_profile_preparation'
    if key in h:del h[key]
    h[key]=Path(__file__).read_text()
