"""Physical heat/port-energy closure, launch matching, and mesh checks."""
from pathlib import Path
import h5py,numpy as np
P=Path(__file__).resolve().parents[2]/'reference/coupled/hf_absorption_profile.h5'
def test_physical_deposition_and_launch():
 with h5py.File(P) as h:
  for family in ['event','event_coarse','normal','oblique25','half_nu','double_nu']:
   for g in h[family].values():
    a=g.attrs;cum=g['cumulative_absorbed_fraction'][:]
    assert np.all(g['absorption_per_m'][:]>=0)
    assert np.all(np.diff(cum)>=-1e-14)
    np.testing.assert_allclose(cum[-1],a['absorbed_heat_fraction'],atol=1e-12)
    assert abs(1-a['reflected_EM_fraction']-a['transmitted_EM_fraction']-cum[-1])<3e-4
    assert a['mode_boundary_relative_error']<2e-4
    assert 0<cum[-1]<1

def test_refinement_and_resonant_localization():
 with h5py.File(P) as h:
  for name in ['O','X','optimized']:
   a=h['event/'+name].attrs;b=h['event_coarse/'+name].attrs
   assert abs(a['absorbed_heat_fraction']-b['absorbed_heat_fraction'])<5e-5
  a=h['event/O'].attrs
  assert a['incident_fraction_absorbed_within_30m']/a['absorbed_heat_fraction']>.9
