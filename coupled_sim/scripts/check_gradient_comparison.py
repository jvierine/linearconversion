import sys,numpy as np,h5py
from pathlib import Path
from scipy.constants import e,epsilon_0,c
p=Path(__file__).resolve().parents[1]/'output/coupled/gradient_study';checks=[]
for case in ['measured','E1','E5','F5','F20']:
 with h5py.File(p/f'{case}_16.75_0.0025_0.5_1.5.h5') as h:
  a=h.attrs;z=h['z_m'][:]-a['center_m'];dz=h['dz_cells_m'][:];dual=(dz+np.roll(dz,1))/2
  E=h['E_phasor_V_m'][:];dn=h['density_phasor_m3'][:];kx=2*np.pi*a['frequency_Hz']/c*np.sin(np.deg2rad(a['angle_deg']))
  div=1j*kx*E[0]+(E[2]-np.roll(E[2],1))/dual
  use=(z>-a['bragg_distance_m']-20)&(z<20)
  error=np.linalg.norm((div+e/epsilon_0*dn)[use])/np.linalg.norm((e/epsilon_0*dn)[use])
  assert error<1e-7 and abs(a['balance'])<1e-3 and a['harmonic_residual']<1e-7
  checks.append([error,a['balance'],a['harmonic_residual']]);print(case,checks[-1])
with h5py.File(p/'summary.h5','a') as h:
 if 'physics_checks' in h:del h['physics_checks']
 h['physics_checks']=checks;h['physics_checks'].attrs['rows']='measured,E1,E5,F5,F20';h['physics_checks'].attrs['columns']='Gauss_relative_residual,flux_balance_residual,harmonic_residual'
