"""Run the observed F-region profile with its own gradient and local parameters.

Uses the existing coupled solver and identical local Bragg diagnostic.
The constant temperature and field are local values at F-region resonance;
collision rates follow the smoothed density and MSIS neutrals with height.
No Landau damping. Outputs are local response proxies, not radar SNR.
"""
from pathlib import Path
import argparse
import numpy as np
import h5py
from scipy.interpolate import CubicSpline
from scipy.constants import k as kb,e,m_e
import gradient_comparison as g

p=argparse.ArgumentParser();p.add_argument('--angle',type=float,default=16.75)
p.add_argument('--dz',type=float,default=.02);p.add_argument('--cap',type=float,default=2.)
p.add_argument('--sigma-km',type=int,default=10)
p.add_argument('--phase-grid',action='store_true')
p.add_argument('--ordering',choices=['COLAMD','NATURAL'],default='COLAMD');v=p.parse_args()
with h5py.File(g.a.ROOT/'data/fregion_observed_background.h5') as h:
    z=h['height_m'][:];sg=h[f'smoothing_{v.sigma_km}km'];a=dict(sg.attrs)
    profile=CubicSpline(z,sg['ne_m3'][:]);neutrals={k:h['neutrals/'+k][:] for k in ['N2_m3','O2_m3','O_m3']}
t=a['temperature_K'];ln=a['Ln_m'];center=a['center_m']
def rates(zz,ne):
    tev=kb*t/e;ncm=np.maximum(ne*1e-6,1e-30)
    ei=2.91e-6*ncm*(23-np.log(np.sqrt(ncm)*tev**-1.5))/tev**1.5
    n2,o2,o=[np.exp(np.interp(zz,z,np.log(neutrals[k]))) for k in ['N2_m3','O2_m3','O_m3']]
    en=2.33e-17*n2*(1-1.21e-4*t)*t+1.82e-16*o2*(1+.036*np.sqrt(t))*np.sqrt(t)+8.9e-17*o*(1+5.7e-4*t)*np.sqrt(t)
    return ei+en
d=ln*(3*kb*t/m_e)*g.KB**2/g.a.W**2
cfg=dict(profile=profile,center=center,ordering=v.ordering,bounds=(190000.,274000.),
    sample_bounds=(191000.,273000.),reference_z=192000.,source_z=191900.,
    fine_bounds=(-max(150.,1.1*d+400),40.),temperature=t,rates=rates,
    magnetic_t=a['magnetic_t'],field_angle_deg=a['field_angle_deg'])
if v.phase_grid:
    cfg['phase_grid_distance_m']=d
    cfg['fine_bounds']=(-1.05*d-120,40.)
name=f'Fobserved{v.sigma_km}'+('p' if v.phase_grid else '')
g.parameters=lambda _: (cfg,t,ln,float(rates(center,profile(center))))
g.run(name,v.angle,v.dz,v.cap)
dest=g.OUT/f'{name}_{v.angle:g}_{v.dz:g}_{v.cap:g}_1.h5'
with h5py.File(dest,'a') as h:
    h.attrs.update(profile='Smoothed contemporaneous F-region density; no Es profile transplanted',
        B_T=a['magnetic_t'],B_angle_deg=a['field_angle_deg'],smoothing_sigma_m=v.sigma_km*1000.,
        fine_bounds_m=cfg['fine_bounds'],matrix_ordering=cfg['ordering'],
        physics='Coupled Maxwell warm-fluid CW recurrence; measured F-region background, local constant Te and B, height-dependent collisions; no Landau damping')
    h['provenance/observed_F_driver']=Path(__file__).read_text()
