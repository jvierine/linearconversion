"""Fixed integrated drive: check scalar harmonic discretization on measured profile.
This checks the linear reference used for initializing/scaling total-density runs,
not nonlinear steady-state or RCS convergence. No per-grid amplitude normalization.
"""
from argparse import Namespace
from pathlib import Path
import hashlib
import h5py
import numpy as np
from run_scalar_total import prepare

root=Path(__file__).resolve().parents[1]
cases=[('coarse',.08,8,10.,.04),('base',.04,8,10.,.04),('fine',.02,8,10.,.04),('outer',.02,16,5.,.02),('fine_outer',.02,16,2.5,.01),('one_cm',.01,16,2.5,.01),('half_cm',.005,16,2.5,.01),('quarter_cm',.0025,16,2.5,.01)]
results=[]
for name,dz,nx,outer,grading in cases:
    a=Namespace(dz=dz,nx=nx,outer_spacing=outer,grading=grading,drive_scale=1.,amplitude=1e-4,start='harmonic',cycles=1.,frames=1,ramp_cycles=0.,linear=True,label='grid_not_run')
    cfg,sol,src,meta,D,residual,scale,unit=prepare(a)
    results.append((name,cfg,sol,meta,residual))
path=root/'reference/scalar_total_grid.h5'
with h5py.File(path,'w') as h:
    h.attrs['scope']='Fixed integrated source; scalar harmonic reference; not nonlinear time-domain or radar-power convergence'
    prev=None
    for name,cfg,sol,meta,residual in results:
        g=h.create_group(name);z=meta['z_m'];mask=(z>meta['resonance_m']-10)&(z<meta['resonance_m']+3)
        g['z_m']=z[mask];g['density_phasor_normalized']=sol[0,mask];g['Ez_phasor_normalized']=sol[6,mask]
        g.attrs.update(source_height_m=meta['source_height_m'],linear_residual=residual,peak_fractional_wave=float(np.max(abs(sol[0])/cfg['n0'])))
        if prev is not None:
            oldz,oldsol=prev
            interp=np.interp(oldz,z,sol[0].real)+1j*np.interp(oldz,z,sol[0].imag)
            err=np.linalg.norm(interp-oldsol)/np.linalg.norm(interp)
            g.attrs['relative_complex_change_from_previous']=err
            gain=np.vdot(oldsol,interp)/np.vdot(oldsol,oldsol)
            g.attrs['amplitude_ratio_from_previous']=abs(gain)
            g.attrs['phase_shift_rad_from_previous']=np.angle(gain)
            g.attrs['shape_change_after_complex_rescaling']=np.linalg.norm(interp-gain*oldsol)/np.linalg.norm(interp)
            print(name,'relative complex change',err,'source height',meta['source_height_m'],flush=True)
        prev=(z[mask],sol[0,mask])
    h['provenance/source']=Path(__file__).read_text()
print(path)
