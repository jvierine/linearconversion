"""Data-level checks of the recovered native CSP density information."""
from pathlib import Path
import json,sys
import h5py,numpy as np
from PIL import Image
from matplotlib import colormaps
from scipy.constants import c,e,m_e,epsilon_0
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq
from full_altitude_model import FullModel
import recover_density_information as r

ROOT=Path(__file__).resolve().parent
r.prepare();r.init()
with h5py.File(ROOT/'data/eregion_fit/density_native_csp.h5') as h:
    z=h['z'][:];t=h['t'][:];ne=h['ne_m3'][:];old=h['old_ne_m3'][:]
    shape=h['physical_shape'][:];gain=float(h.attrs['density_gain']);power=h['power'][:]
    cal=h['calibration/heldout_density_ratio'][:]
    snr=np.sqrt(h['power_var'][:])/power
use=(z>=125)&(z<=138)
mad=lambda a:1.4826*np.median(abs(a-np.median(a)))
stats={}
for axis,name in [(1,'altitude'),(0,'time')]:
    a=mad(np.diff(np.log(old[:,use]),axis=axis));b=mad(np.diff(np.log(ne[:,use]),axis=axis))
    stats[f'{name}_log_difference_scatter_old']=a;stats[f'{name}_log_difference_scatter_new']=b
    stats[f'{name}_scatter_reduction']=a/b
stats.update(heldout_density_median_ratio=np.median(cal),heldout_log_scatter=np.std(np.log(cal)),median_short_lag_fractional_sigma=np.median(snr[:,use]))
with h5py.File(ROOT/'data/eregion_fit/density_shape_sensitivity.h5') as h:
    stats['shape_window_fractional_change_p05'],stats['shape_window_fractional_change_p50'],stats['shape_window_fractional_change_p95']=np.percentile(h['ne_m3'][:]/ne-1,[5,50,95])

# Reproduce the 2018 polynomial estimator from archived complex ACFs.
with h5py.File(ROOT/'data/eregion_fit/all_de_inputs.h5') as h:
    l=h['csp/lags_us'][:];x=np.log10(np.where(l==0,22,l));counts=h['csp/count'][0]
    a=h['csp/mean'][:,:500].real;zz=h['range_km'][:500]
aa=a.copy()
aa[:,3:266]=(a[:,2:265]+a[:,3:266]+a[:,4:267])/3
recon=np.zeros(a.shape[:2]);weight=np.sqrt(counts)
for order in [1,2]:
    take=np.arange(500)<=266 if order==1 else np.arange(500)>266
    X=np.vander(x,order+1);v=np.vander([np.log10(22)],order+1)[0]
    estimator=v@np.linalg.pinv(X*weight[:,None])*weight
    recon[:,take]=aa[:,take]@estimator
recon*=42859649122.80702*zz[None,:]**2
with h5py.File(ROOT/'evidence/seid_event/ne_result.h5') as h:blog=h['Ne_m3'][:]
stats['legacy_reproduction_relative_norm']=np.linalg.norm(recon-blog)/np.linalg.norm(blog)
assert stats['legacy_reproduction_relative_norm']<1e-12

# Verify the first-four-CLP-bin shortcut against the full independent CSP
# integration, including its broad electron spectrum and receiver response.
full=FullModel('csp',r.D['lags']);errors=[];closure=[]
for i,j in zip(np.linspace(0,len(t)-1,18,dtype=int),np.linspace(0,len(z)-1,18,dtype=int)):
    ti,te,v=shape[i,j];ns=[r.D[k][i,j] for k in ['N2_m3','O2_m3','O_m3']]
    exact=full(ti,te,v,ne[i,j],r.D['B'][j],r.D['theta'][j],ns,np.array([0,0,0,1,0,0]))[0].real
    approx=(r.model(ti,te,v,ne[i,j],i,j)[:4]@np.array([6,10,10,10])/36).real
    errors.append(abs(approx/exact-1));closure.append(abs(ne[i,j]*exact/(gain*z[j]**2*power[i,j])-1))
stats['max_full_csp_response_difference']=max(errors);stats['max_power_closure_difference']=max(closure)
assert max(errors)<1e-3 and max(closure)<1e-3

# The saved blog colorbar, rather than its stale script, fixes the palette.
im=np.asarray(Image.open(ROOT/'evidence/seid_event/arecibo_d_e.png').convert('RGB'))
y=np.arange(66,476);rgb=im[y,637]/255;v=(480-y)/420
for name in ['plasma','nipy_spectral']:
    stats[f'blog_colorbar_{name}_rgb_rmse']=np.sqrt(np.mean((rgb-colormaps[name](v)[:,:3])**2))
i=np.argmin(abs(t-1525453430));es=(z>116)&(z<120)
stats.update(selected_old_peak_ne_m3=old[i,es].max(),selected_new_peak_ne_m3=ne[i,es].max(),
             old_peak_height_km=z[es][np.argmax(old[i,es])],new_peak_height_km=z[es][np.argmax(ne[i,es])])
with h5py.File(ROOT/'data/eregion_fit/density_recovery_audit.h5','w') as h:
    h.attrs.update(stats);h['legacy_reconstructed_ne_m3']=recon
    h['provenance/audit_source']=Path(__file__).read_text()
    h.attrs['roughness_interpretation']='Robust adjacent-sample log differences in diffuse 125-138 km background, not an independent instrumental noise estimate.'
print(json.dumps(stats,indent=2))
