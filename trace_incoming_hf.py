"""Incoming HF Hamiltonian rays on the measured smooth density profile.

Sources: Muldrew 1993 doi:10.1029/92RS00273. Cold EM ray propagation is
appropriate outside the conversion zone; no evanescent gap is bridged by a
fabricated ray. Full-wave conversion solutions provide the plasma response.
"""
import h5py
import numpy as np
from scipy.constants import c,e,epsilon_0,m_e
from isr.conversion import Parameters,modes
from isr.observed_layer import fit_observed_layer,profile as density_profile
from isr.rays import electromagnetic_ray

def run():
 fit=fit_observed_layer(1525453430)
 start=100000.
 with h5py.File('data/incoming_hf_rays.h5','w') as h:
  h.attrs.update(provenance='trace_incoming_hf.py; isr/rays.py',reference='https://doi.org/10.1029/92RS00273',
    epoch_unix_s=1525453430,hf_frequency_hz=5.33e6,
    density_source_sha256=fit['source_sha256'],temperature_k=300.,magnetic_t=35.025e-6,field_vertical_deg=46.815,
    polarization_definition='Unit complex electric phasor; in-plane arrow evaluated at maximum Ez; arrow length is not field strength',
    interpretation='Cold local electromagnetic Hamilton rays; full-wave Maxwell solution handles conversion/tunnelling. Each incident mode has its own x=0 at the lower boundary. No ray is drawn through an evanescent interval.')
  h['density_coefficients']=fit['coefficients'];h['density_knots_km']=fit['knots_km']
  for angle in [-20.,20.,-16.5,16.5]:
   q=np.sin(np.deg2rad(angle));delta=np.arcsin(q*5.33e6/(2*430e6))
   par=Parameters(frequency_hz=5.33e6,temperature_k=300,magnetic_t=35.025e-6,
                  magnetic_tilt_deg=46.815-np.rad2deg(delta),q=q)
   def profile(z):
    ne,dn,*_=density_profile(z,fit['coefficients'],fit['knots_km'])
    scale=e*e/(epsilon_0*m_e*par.omega**2)
    return float(ne*scale),float(dn*scale)
   mu,v,d,p=modes(profile(start)[0],par)
   em=sorted([j for j in range(6) if d[j]==1 and p[j] and abs(mu[j])<5],key=lambda j:abs(mu[j]))
   for branch,j in enumerate(em):
    ray=electromagnetic_ray(start,mu[j].imag*par.k0,profile,par,max_step_s=2e-7)
    g=h.create_group(f'{angle:+05.1f}/{branch}')
    for k,value in ray.items():
     if isinstance(value,str):g.attrs[k]=value
     else:g[k]=value
    g.attrs.update(angle_deg=angle,branch=branch,maximum_height_m=max(ray['height_m']),
                    maximum_dispersion_residual=max(abs(ray['dispersion_residual'])))
    E=ray['electric_polarization'];phase=np.exp(-1j*np.angle(E[:,2]))
    axis=(E*phase[:,None]).real[:,[0,2]]
    axis/=np.linalg.norm(axis,axis=1)[:,None]
    g['electric_inplane_axis']=axis
    kv=np.column_stack([ray['kx_m1'],ray['kz_m1']])
    g['phase_unit_vector']=kv/np.linalg.norm(kv,axis=1)[:,None]
    g['electric_vertical_fraction']=abs(E[:,2])
    j=np.argmax(ray['height_m'])
    g.attrs['electric_vertical_fraction_at_turn']=abs(E[j,2])
    print(angle,branch,dict(g.attrs),flush=True)
if __name__=='__main__':run()
