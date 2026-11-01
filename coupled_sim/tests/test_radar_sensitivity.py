"""Check the delivered sweeps against independent mesh/window cases."""
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'reference/coupled/radar_sensitivity.h5'

def test_grid_and_window_stability():
 with h5py.File(PATH) as h:
  table=h['gradient/table'][:]
  for ln in [1.,2.,5.]:
   coarse=table[np.argmin(abs(table[:,0]-ln)),1:]
   fine=h[f'checks/L{ln:g}_0.0025_0.5_windowcheck'][:]
   assert abs(10*np.log10(fine[0]/coarse[0]))<.35
   assert np.ptp(10*np.log10(fine))<.02
  fine=h['checks/measured_0.0025_0.5_windowcheck'][:]
  assert abs(fine[0]/h.attrs['reference_RCS_m2']-1)<.001
  assert np.ptp(10*np.log10(fine))<.002

def test_polarization_bound_and_power_linearity():
 with h5py.File(PATH) as h:
  pol=h['polarization/RCS_m2'][:];ref=h.attrs['reference_RCS_m2']
  assert pol.max()<=ref*(1+1e-9)
  assert pol.max()/ref>.999
  assert pol.min()/ref<.001
  p=h['power/EIRP_W'][:]
  for angle in [0,14,25]:
   r=h[f'power/RCS_{angle}_m2'][:]
   assert np.allclose(r/p,(r/p)[0],rtol=1e-12)
  angle=h['angle/degrees'][:];rcs=h['angle/RCS_m2'][:]
  assert 13<angle[np.argmax(rcs)]<15
  assert np.all(np.diff(h['gradient/table'][:,1])<0)
