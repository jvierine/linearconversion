"""Controlled-gradient electric fields using the paper's coupled FDTD/fluid.
Fix physical incident tangential polarization, angle, flux, T, B and nu.
"""
from pathlib import Path
import numpy as np,h5py
from scipy.constants import c,e,m_e,epsilon_0
from scipy.optimize import brentq
import coupled_arecibo as a
from minimum_incident_power import metric
ROOT=Path(__file__).resolve().parents[2]
with h5py.File(ROOT/'reference/coupled/minimum_incident.h5') as h:
 ANGLE=float(h.attrs['angle_deg']);CENTER=float(h.attrs['center_m']);OPT=h['source_polarization'][:]
NRES=float(a.PROFILE(CENTER));NU=float(a.cold.rates(CENTER,NRES))

def solve(ln,dz,polarization):
 profile=lambda z:NRES*(1+.6*np.tanh((np.asarray(z)-CENTER)/(.6*ln)))
 extent=max(4000.,1.5*ln)
 cfg=dict(profile=profile,center=CENTER,ordering='COLAMD',bounds=(CENTER-extent-1000,CENTER+extent+1000),fine_bounds=(-150.,40.),sample_bounds=(CENTER-extent,CENTER+extent),reference_z=CENTER-extent,source_z=CENTER-extent-100.,temperature=300.,rates=lambda z,ne:np.zeros_like(ne)+NU)
 s=a.setup(angle=ANGLE,dz=dz,cap=.5,scenario=cfg);r=a.setup(angle=ANGLE,dz=dz,cap=.5,scenario=cfg,reference=True)
 z=s['z'];zf=z+s['dz_cells']/2;lo=np.argmin(abs(zf-(cfg['reference_z']+100)))
 v=np.linalg.solve(r['initial'][:2,lo,:],polarization);Q=metric(r['initial'],lo);v/=np.sqrt(np.vdot(v,Q@v).real)
 state=np.einsum('qnp,p->qn',s['initial'],v);inc=np.einsum('qnp,p->qn',r['initial'],v)
 norm=np.linalg.norm(inc[:3,lo]);turn=brentq(lambda zz:profile(zz)*e*e/(epsilon_0*m_e)-a.W*a.W,CENTER,CENTER+.3*ln)
 cut=(zf>CENTER-300)&(zf<CENTER+160)
 x=abs(state[2])/norm
 stats=dict(Ln_m=ln,dz_m=dz,angle_deg=ANGLE,center_m=CENTER,O_turn_relative_m=turn-CENTER,incident_E_peak_V_m=norm,incident_upward_flux_W_m2=a.electromagnetic_flux(inc)[lo],incident_port_relative_m=zf[lo]-CENTER,collision_rate_s1=NU,peak_Ez_over_Einc=float(x[cut].max()),harmonic_residual=s['harmonic_residual'])
 print(stats,flush=True)
 return dict(height_relative_m=zf[cut]-CENTER,Ez_over_Einc=x[cut],Ez_complex_over_Einc=state[2,cut]/norm,source_polarization=v),stats

def main():
 r=a.setup(angle=ANGLE,dz=.005,cap=.5,reference=True,scenario={'ordering':'COLAMD'})
 lo=np.argmin(abs(r['z']+r['dz_cells']/2-115100));target=r['initial'][:2,lo,:]@OPT;target/=np.linalg.norm(target)
 path=ROOT/'reference/coupled/gradient_electric_field.h5'
 with h5py.File(path,'w') as h:
  h.attrs.update(frequency_hz=a.F,temperature_K=300.,magnetic_T=float(a.P['magnetic_t']),angle_deg=ANGLE,normalization='Total incident electric-field phasor magnitude at lower homogeneous reference plane; no normalization to curve maxima.',scope='Same prescribed tanh profiles as E-region sensitivity controls. Warm coupled FDTD-fluid, constant physical collision rate, fixed incident tangential polarization and upward flux. Landau damping absent.',profile='nres*(1+0.6*tanh((z-zres)/(0.6*Ln)))')
  h['fixed_incident_tangential_polarization']=target;h['provenance/source']=Path(__file__).read_text()
  for name,ln,dz in [('L0p5',500.,.005),('L1',1000.,.005),('L2',2000.,.005),('L5',5000.,.005),('check_L1',1000.,.0025),('check_L5',5000.,.0025)]:
   arrays,stats=solve(ln,dz,target);g=h.create_group(name);g.attrs.update(stats)
   for key,val in arrays.items():g.create_dataset(key,data=val,compression='gzip')
   h.flush()
 print(path,flush=True)
if __name__=='__main__':main()
