"""Re-run the coupled model with the restored CSP density, retaining baseline."""
from pathlib import Path
import sys,h5py,numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq
from scipy.constants import e,m_e,epsilon_0,c,k as kb
ROOT=Path(__file__).resolve().parents[1]
import coupled_arecibo as a
from summarize_coupled import radar
with h5py.File(ROOT/'data/smooth_csp_layer_1525453430.h5') as h:
    fit={k:h[k][:] for k in ['model_height_m','model_ne_m3','model_dne_dz_m4']}
a.PROFILE=CubicHermiteSpline(fit['model_height_m'],fit['model_ne_m3'],fit['model_dne_dz_m4'])
original_setup=a.setup
def pivoted_setup(*args,**kwargs):
    kwargs['scenario']=dict(kwargs.get('scenario') or {},ordering='COLAMD')
    return original_setup(*args,**kwargs)
a.setup=pivoted_setup
a.run(dz=.005,cap=.5,native_cycles=0,label='density_csp_sensitivity')
a.run(dz=.0025,cap=.5,native_cycles=0,label='density_csp_mesh_check')
with h5py.File(a.OUT/'density_csp_sensitivity.h5','a') as h,h5py.File(a.OUT/'density_sensitivity_summary.h5','a') as audit,h5py.File(ROOT/'data/radar_calibration.h5') as cal:
    gyro=h['gyro'][:]*c;kx=h['kx_rad_m'][()];kz=-(4*np.pi*430e6/c+a.W/c);kv=np.array([kx,0,kz])
    D=a.W**2*(a.W**2-gyro@gyro)/(a.W**2-(gyro@kv)**2/(kv@kv));center=h.attrs['center_m']
    z=brentq(lambda z:a.PROFILE(z)*e*e/(m_e*epsilon_0)+a.CS**2*(kv@kv)-D,center-30,center+10)
    flux=cal.attrs['fitted_on_frequency_snr']*kb*150/(cal.attrs['pulse_duration_s']*radar(h)[1]*cal.attrs['ion_line_measured_over_predicted'])
    values=dict(csp_model_resonance_height_m=center,csp_model_bragg_distance_m=center-z,
        csp_model_scale_length_m=float(a.PROFILE(center)/a.PROFILE(center,1)),
        csp_model_required_flux_W_m2=flux,csp_model_collisional_loss=h.attrs['physical_loss_between_planes'])
    audit.attrs.update(values);h.attrs.update(values);h['provenance/density_sensitivity_source']=Path(__file__).read_text()
    with h5py.File(a.OUT/'density_csp_mesh_check.h5') as fine:
        change=float(abs(radar(fine)[1]/radar(h)[1]-1))
        audit.attrs['csp_model_mesh_radar_power_relative_change']=change
        assert change<.01,change
    print(values)
