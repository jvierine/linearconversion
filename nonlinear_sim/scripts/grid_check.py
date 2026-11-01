"""Spatial convergence of the independently solved linear initial state.
Same horizontal grid; phase/amplitude alignment separates local shape from
incident-path phase. This is not nonlinear steady-state convergence.
"""
from run import setup,ROOT
import h5py,numpy as np
with h5py.File(ROOT/'reference/grid_check.h5','w') as h:
 previous=None
 for step in [.04,.02,.01,.005,.0025]:
  cfg,sol,src,meta=setup(step,16);z=meta['z_m']-meta['resonance_m'];a=sol[0]/cfg['n0'];a/=max(abs(a));mask=(z>-5)&(z<1)
  g=h.create_group(str(step));g['z']=z;g['density_phasor_peak_normalized']=a;g.attrs['linear_residual']=meta['linear_residual']
  if previous:
   zz,aa=previous;v=np.interp(z[mask],zz,aa);w=a[mask];scale=np.vdot(v,w)/np.vdot(v,v)
   g.attrs['aligned_complex_error']=np.linalg.norm(scale*v-w)/np.linalg.norm(w)
   g.attrs['envelope_error']=np.linalg.norm(abs(v)-abs(w))/np.linalg.norm(w)
   g.attrs['passes_one_percent_shape_criterion']=g.attrs['aligned_complex_error']<.01
   print(step,dict(g.attrs),flush=True)
  previous=(z,a)
 h.attrs['comparison']='Successive vertical grids; nx=16 fixed; region -5 to +1 m about resonance; independent peak normalization and complex phase alignment'
