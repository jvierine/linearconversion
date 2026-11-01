"""Coherent radar sensitivity of the existing coupled harmonic Yee/fluid model.
No changes to the evolution equations. All scientific arrays use HDF5.
"""
from pathlib import Path
import sys,hashlib,time,shutil
import h5py,numpy as np
from scipy.constants import c,epsilon_0,m_e,k as kb,e
from scipy.optimize import brentq
import coupled_arecibo as a
from minimum_incident_power import metric
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from isr.coherent import sideband_amplitude
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reference/coupled/radar_sensitivity';OUT.mkdir(parents=True,exist_ok=True)
with h5py.File(ROOT/'reference/coupled/minimum_incident.h5') as h:B=dict(h.attrs)
ANGLE=B['angle_deg'];CENTER=B['center_m'];FLUX=B['incident_vertical_flux_W_m2']
NRES=float(a.PROFILE(CENTER));LN=float(NRES/a.PROFILE.derivative()(CENTER));NU=float(a.cold.rates(CENTER,NRES))
G=10**(61.1/10);LAMBDA=c/430e6
RCS_FACTOR=(4*np.pi)**3*CENTER**4/(1e6*G**2*LAMBDA**2)

def response(ln=None,dz=.005,cap=1.):
 name='measured' if ln is None else f'L{ln/1000:g}'
 path=OUT/f'{name}_{dz:g}_{cap:g}_windowcheck.h5'
 if path.exists():return path
 cfg=None
 if ln is not None:
  profile=lambda z:NRES*(1+.6*np.tanh((np.asarray(z)-CENTER)/(.6*ln)))
  extent=max(4000.,1.5*ln)
  cfg=dict(profile=profile,center=CENTER,ordering='COLAMD',bounds=(CENTER-extent-1000,CENTER+extent+1000),fine_bounds=(-150.,40.),sample_bounds=(CENTER-extent,CENTER+extent),reference_z=CENTER-extent,source_z=CENTER-extent-100.,temperature=300.,rates=lambda z,ne:np.zeros_like(ne)+NU)
 s=a.setup(angle=ANGLE,dz=dz,cap=cap,scenario=cfg);ref=a.setup(angle=ANGLE,dz=dz,cap=cap,scenario=cfg,reference=True)
 z=s['z'];rel=z-CENTER;lower=115100. if cfg is None else cfg['reference_z']+100
 ix=np.argmin(abs(z+s['dz_cells']/2-lower));Q=metric(ref['initial'],ix)
 n=s['initial'][9]*np.sqrt(epsilon_0*s['ne'][0]/(m_e*a.CS**2))[:,None]
 responses=[]
 # Vary taper endpoints independently of the physical background.
 for extra in [0.,20.,40.]:
  window=np.sin(.5*np.pi*np.clip(np.minimum(rel+100+extra,30-rel)/10,0,1))**2
  responses.append([sideband_amplitude(z,n[:,i],a.F,s['kx'],window,reference_range_m=CENTER) for i in range(2)])
 responses=np.asarray(responses);powers=[]
 for r in responses:powers.append(np.vdot(r.conj(),np.linalg.solve(Q,r.conj())).real)
 assert np.linalg.eigvalsh(Q).min()>0
 with h5py.File(path,'w') as h:
  h['incident_tangential_E_basis']=ref['initial'][:2,ix,:];h['incident_flux_metric']=Q;h['radar_response_basis']=responses;h['optimum_power_per_flux_W']=powers
  h.attrs.update(angle_deg=ANGLE,dz_m=dz,cap_m=cap,Ln_m=LN if ln is None else ln,collision_rate_s1=NU,temperature_K=300.,center_m=CENTER,harmonic_residual=s['harmonic_residual'],pressure_adjoint_error=s['pressure_adjoint_error'],cells=s['n'],profile='measured smooth profile' if ln is None else 'Nres*(1+0.6*tanh((z-zres)/(0.6*Ln)))',polarization='optimized for coherent radar response at each Ln',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 print(name,dz,'power per flux',powers,'cells',s['n'],flush=True)
 return path

def main():
 measured=response();paths=[response(x*1000) for x in [.5,.75,1.,1.25,1.5,2.,3.,5.]]
 # Selected independent grid checks, in addition to the event optimum audit.
 checks=[response(None,.0025,.5),response(1000.,.0025,.5),response(2000.,.0025,.5),response(5000.,.0025,.5)]
 with h5py.File(measured) as h:
  Q=h['incident_flux_metric'][:];r=h['radar_response_basis'][0];p0=h['optimum_power_per_flux_W'][0];T=h['incident_tangential_E_basis'][:]
 Ti=np.linalg.inv(T);r=r@Ti;Q=Ti.conj().T@Q@Ti
 out=ROOT/'data/radar_sensitivity.h5'
 with h5py.File(out,'w') as h:
  h.attrs.update(reference_flux_W_m2=FLUX,reference_angle_deg=ANGLE,reference_Ln_m=LN,reference_RCS_m2=FLUX*p0*RCS_FACTOR,reference_EIRP125km_W=4*np.pi*(125e3)**2*FLUX/np.cos(np.deg2rad(ANGLE)),rcs_factor=RCS_FACTOR,frequency_hz=a.F,temperature_K=300.,polarization_coordinates='tan(alpha)=abs(Ey/Ex), phase=arg(Ey/Ex), incident field at lower reference plane; x in magnetic meridian',scope='Coherent beam-filled linear collisional fluid response; optimized polarization except the explicit polarization sweep; no Landau damping',provenance='radar_sensitivity_sweeps.py; same sideband_amplitude operator as manuscript')
  with h5py.File(ROOT/'data/minimum_incident_plots.h5') as scan:
   ang=scan['scan/angle_deg'][:];fl=scan['scan/incident_vertical_flux_W_m2'][:];dz=scan['scan/dz_m'][:]
   ix=np.argsort(ang);h['angle/degrees']=ang[ix];h['angle/RCS_m2']=B['radar_power_per_unit_flux_W']*B['incident_vertical_flux_W_m2']/fl[ix]*FLUX*RCS_FACTOR
   h['angle/required_flux_W_m2']=fl[ix]
  alpha=np.linspace(0,90,91);phase=np.linspace(-180,180,181);aa,pp=np.meshgrid(np.deg2rad(alpha),np.deg2rad(phase));v=np.stack([np.cos(aa),np.exp(1j*pp)*np.sin(aa)],axis=-1)
  denom=np.einsum('...i,ij,...j->...',v.conj(),Q,v).real
  power=abs(np.einsum('i,...i->...',r,v))**2/denom
  h['polarization/amplitude_angle_deg']=alpha;h['polarization/relative_phase_deg']=phase;h['polarization/RCS_m2']=power*FLUX*RCS_FACTOR
  opt=np.linalg.solve(Q,r.conj());ratio=opt[1]/opt[0];h['polarization/optimum']=np.array([np.rad2deg(np.arctan(abs(ratio))),np.rad2deg(np.angle(ratio))])
  eirp=np.logspace(-3,4,200);h['power/EIRP_W']=eirp
  for angle in [0.,14.,25.]:
   i=np.argmin(abs(ang-angle));flux=eirp/(4*np.pi*(125e3)**2)*np.cos(np.deg2rad(angle));h[f'power/RCS_{angle:g}_m2']=B['radar_power_per_unit_flux_W']*B['incident_vertical_flux_W_m2']/fl[i]*flux*RCS_FACTOR
  vals=[]
  for path in paths:
   with h5py.File(path) as p:vals.append([p.attrs['Ln_m']/1000,*list(p['optimum_power_per_flux_W'][:]*FLUX*RCS_FACTOR)])
  h['gradient/table']=vals
  for path in checks:
   with h5py.File(path) as p:h[f'checks/{path.stem}']=p['optimum_power_per_flux_W'][:]*FLUX*RCS_FACTOR
  h['provenance/source']=Path(__file__).read_text()
 shutil.copy2(out,ROOT/'reference/coupled/radar_sensitivity.h5')
 print(out,flush=True)
if __name__=='__main__':main()
