"""Archive the event's own F-region background, without imposing an Es gradient."""
from pathlib import Path
import hashlib
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.constants import e, m_e, epsilon_0

ROOT=Path(__file__).resolve().parent
src=ROOT/'data/eregion_fit/full_altitude_fits.h5'
with h5py.File(src) as h:
    t=h['time_unix_s'][:]; z=h['height_km'][:]*1000
    i=int(np.argmin(abs(t-1525453430)))
    use=(z>=150000)&(z<=350000)
    raw=h['results'][i-1:i+2,:,:3][:,use,:]
    b=h['B_T'][:][use];theta=h['angle_k_B_deg'][:][use]
    times=t[i-1:i+2];z=z[use]
with h5py.File(ROOT/'data/eregion_fit/full_altitude_priors.h5') as h:
    j=int(np.argmin(abs(h['time_unix_s'][:]-times[1])))
    zn=h['msis/height_km'][:]*1000
    neutrals={k:np.exp(np.interp(z,zn,np.log(np.maximum(h['msis/'+k][j],1e-30)))) for k in ['N2_m3','O2_m3','O_m3']}
f=5.340028553893307e6;w=2*np.pi*f
dest=ROOT/'coupled_sim/data/fregion_observed_background.h5'
with h5py.File(dest,'w') as h:
    h['height_m']=z;h['time_unix_s']=times;h['native_ne_Te_Ti']=raw
    h['B_T']=b;h['field_angle_deg']=theta
    for k,v in neutrals.items():h['neutrals/'+k]=v
    for sigma in [5000.,10000.,15000.]:
        ne=gaussian_filter1d(np.median(raw[:,:,0],axis=0),sigma/300,mode='reflect')
        te=gaussian_filter1d(np.median(raw[:,:,1],axis=0),sigma/300,mode='reflect')
        prof=CubicSpline(z,ne)
        def residual(zz):
            om=e*np.interp(zz,z,b)/m_e;oz=om*np.cos(np.deg2rad(np.interp(zz,z,theta)))
            return prof(zz)*e*e/(m_e*epsilon_0)-w*w*(w*w-om*om)/(w*w-oz*oz)
        zr=brentq(residual,210000,270000)
        g=h.create_group(f'smoothing_{sigma/1000:g}km');g['ne_m3']=ne;g['Te_K']=te
        g.attrs.update(sigma_m=sigma,center_m=zr,Ln_m=float(prof(zr)/prof(zr,1)),
            temperature_K=float(np.interp(zr,z,te)),magnetic_t=float(np.interp(zr,z,b)),
            field_angle_deg=float(np.interp(zr,z,theta)))
        print(dict(g.attrs))
    h.attrs.update(frequency_Hz=f,method='Median of three native long-pulse fits, then Gaussian altitude smoothing; no density rescaling or gradient replacement.',
        interpretation='Event-conditioned F-region comparison, not a global climatology. Density/temperature fit uncertainties and smoothing sensitivity remain.',
        fit_source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        model_domain_m=[190000.,274000.],selected_sigma_m=10000.)
    h['provenance/preparation_source']=Path(__file__).read_text()
