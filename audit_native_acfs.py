"""Pulse-mode overlap and exact cycle timing audit; read-only source use."""
from pathlib import Path
import h5py,numpy as np
root=Path('/mnt/data/juha/arecibo/plasma_fit_20261029')
with h5py.File(root/'all_altitude_inputs.h5') as h:
 z=h['range_km'][:];tt=h['time_unix_s'][:];edges=h['time_edges_unix_s'][:]
 print('coverage',z[0],z[-1],len(z),'cadence',np.median(np.diff(tt)),flush=True)
 print('cycle span first',edges[:2], 'CSP starts',h['csp/record_start_unix_s'][:8],flush=True)
 print('CLP integration median s',np.median(h['clp/record_end_unix_s'][:]-h['clp/record_time_unix_s'][:]),flush=True)
 r=(z>=95)&(z<=140);c=h['csp/mean'][:,r,0];l=h['clp/mean'][:,r,0]
 with h5py.File(root/'pulse_mode_overlap.h5','w') as o:
  o['height_km']=z[r];o['time_unix_s']=tt;o['csp_first_lag']=c;o['clp_first_lag']=l
  o['ratio_of_time_means']=np.mean(c.real,axis=0)/np.mean(l.real,axis=0)
  o.attrs['interpretation']='CSP 4..39 us versus CLP 4..9 us: different lag responses; ratios are not calibration factors without a spectral model'
 for hgt in [99,105,115,125,135]:
  j=np.argmin(abs(z[r]-hgt));print(hgt,'CSP/CLP mean first lag',c[:,j].real.mean()/l[:,j].real.mean(),flush=True)
