"""Controlled reflection-layer comparisons using the shared coupled recurrence.
All scenarios are prescribed sensitivity cases, not measured F-region profiles.
Outputs retain fields, parameters, source and energy residuals in HDF5.
"""
from pathlib import Path
import argparse,sys,time
import numpy as np
import h5py
from scipy.constants import c,e,m_e,epsilon_0,k as kb
from scipy.optimize import brentq
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator
import coupled_arecibo as a

OUT=a.OUT/'gradient_study';OUT.mkdir(exist_ok=True,parents=True)
KB=(4*np.pi*430e6+a.W)/c
with h5py.File(a.OUT/'recorded.h5') as h: CENTER=float(h.attrs['center_m'])
NRES=float(a.PROFILE(CENTER)); LN=float(NRES/a.PROFILE.derivative()(CENTER))
NU=float(a.cold.rates(CENTER,NRES))

def parameters(name):
    if name=='measured':return None,300.,LN,NU
    if name=='noEs':
        with h5py.File(a.ROOT/'data/smooth_observed_layer_1525453430.h5') as h:
            profile=PchipInterpolator(h['model_height_m'][:],h['background_ne_m3'][:])
        return dict(profile=profile,center=CENTER,ordering='COLAMD'),300.,np.nan,NU
    if name.startswith('E'):
        ln=float(name[1:])*1000;temp=300.;nu=NU;center=CENTER
    else:
        ln=float(name[1:])*1000;temp=1000.;center=250000.
        tev=kb*temp/e;ncm=NRES*1e-6
        nu=2.91e-6*ncm*(23-np.log(np.sqrt(ncm)*tev**-1.5))/tev**1.5+10.
    profile=lambda z:NRES*(1+.6*np.tanh((np.asarray(z)-center)/(.6*ln)))
    # Resolve past the local Bragg match by at least several collisional
    # damping lengths. The outer grid is only for the EM wave and buffers.
    cs2=3*kb*temp/m_e
    d=ln*cs2*KB**2/a.W**2
    fine_lo=-max(100.,2*d+150.)
    extent=max(4000.,1.5*ln)
    cfg=dict(profile=profile,center=center,ordering='COLAMD',bounds=(center-extent-1000,center+extent+1000),
             fine_bounds=(fine_lo,40.),sample_bounds=(center-extent,center+extent),
             reference_z=center-extent,source_z=center-extent-100.,temperature=temp,
             rates=lambda z,ne:np.zeros_like(ne)+nu)
    return cfg,temp,ln,nu

def metric_flux(s,states,height):
    i=np.argmin(abs(s['z']+s['dz_cells']/2-height))
    vals=[]
    for v in [np.array([1,0]),np.array([0,1]),np.array([1,1]),np.array([1,1j])]:
        vals.append(sum(a.flux(s,np.einsum('qnp,p->qn',states,v)))[i])
    cross=(vals[2]-vals[0]-vals[1])/2+1j*(vals[0]+vals[1]-vals[3])/2
    return np.array([[vals[0],cross],[cross.conjugate(),vals[1]]])

def run(name,angle,dz=.02,cap=4.,extra=1.):
    start=time.monotonic();cfg,temp,ln,nu=parameters(name)
    if cfg is not None and 'fine_bounds' in cfg:
        cfg['fine_bounds']=(cfg['fine_bounds'][0]*extra,cfg['fine_bounds'][1])
    s=a.setup(dz=dz,angle=angle,cap=cap,scenario=cfg)
    ref=a.setup(dz=dz,angle=angle,cap=cap,scenario=cfg,reference=True)
    center=s['center'];z=s['z']; cs2=3*kb*temp/m_e
    profile=a.PROFILE if cfg is None else cfg['profile']
    gyro=s['gyro']*c;kv=np.array([s['kx'],0.,-KB])
    D=a.W**2*(a.W**2-gyro@gyro)/(a.W**2-(gyro@kv)**2/(kv@kv))
    match=center-4.087166131619597 if name=='noEs' else brentq(lambda zz:profile(zz)*e*e/(m_e*epsilon_0)+cs2*(kv@kv)-D,center-min(ln*.5,3000),center+10)
    lower=115100. if cfg is None or name=='noEs' else cfg['reference_z']+100.
    upper=120800. if cfg is None or name=='noEs' else cfg['sample_bounds'][1]-100.
    Q=metric_flux(ref,ref['initial'],lower)
    assert np.linalg.eigvalsh(Q).min()>0
    density_basis=s['initial'][9]*np.sqrt(epsilon_0*s['ne'][0]/(m_e*cs2))[:,None]
    window=np.exp(-.5*((z-match)/10.)**2)
    # Same 10-m Gaussian localization in every case. Fourier projection,
    # unlike pointwise density, rejects the slowly varying forced response.
    d=trapezoid(density_basis*(window*np.exp(1j*KB*(z-match)))[:,None],z,axis=0)
    # Optimize this local Bragg column for unit upward incident flux.
    v=np.linalg.solve(Q,d.conj());v/=np.sqrt(np.vdot(v,Q@v).real)
    optimum=abs(d@v)**2
    state=np.einsum('qnp,p->qn',s['initial'],v)
    inc=np.einsum('qnp,p->qn',ref['initial'],v)
    dn=state[9]*np.sqrt(epsilon_0*s['ne'][0]/(m_e*cs2))
    em,pf=a.flux(s,state);zz=z+s['dz_cells']/2
    ia=np.argmin(abs(zz-lower));ib=np.argmin(abs(zz-upper))
    incflux=sum(a.flux(ref,inc))[ia]
    assert abs(incflux-1)<1e-8
    reflection=-sum(a.flux(s,state-inc))[ia]
    ch=np.cos(s['omega']*s['dt']/2)
    weights=np.array([s['dual'],s['dual'],s['dz_cells']])
    coll=.5*epsilon_0*c*ch*np.sum(s['physical_nu']*abs(state[6:9])**2*weights,axis=0)
    loss=coll[(z>zz[ia])&(z<zz[ib])].sum()
    transmission=(em+pf)[ib]
    phase=np.gradient(np.unwrap(np.angle(dn)),z)
    stats=dict(case=name,Ln_m=ln,temperature_K=temp,collision_rate_s1=nu,angle_deg=angle,
        dz_m=dz,cap_m=cap,extra_fine_extent=extra,center_m=center,bragg_distance_m=(np.nan if name=='noEs' else center-match),bragg_exists=(name!='noEs'),
        bragg_column_per_sqrt_flux=float(np.sqrt(optimum)),window_sigma_m=10.,
        density_at_bragg_per_sqrt_flux=float(abs(np.interp(match,z,dn))/profile(match)),
        phase_kz_at_bragg=float(np.interp(match,z,phase)),
        pressure_flux_at_bragg=float(np.interp(match,zz,pf)),
        max_downward_pressure_flux=float(-pf[(z>center-2000)&(z<center+30)].min()),
        reflection=float(reflection),transmission=float(transmission),collision_loss=float(loss),
        balance=float(1-reflection-transmission-loss),harmonic_residual=s['harmonic_residual'],
        pressure_adjoint_error=s['pressure_adjoint_error'],cells=s['n'])
    path=OUT/f'{name}_{angle:g}_{dz:g}_{cap:g}_{extra:g}.h5'
    with h5py.File(path,'w') as h:
        h.attrs.update(stats);h.attrs.update(profile=('measured spline' if cfg is None else 'smooth background with Es removed' if name=='noEs' else 'nres*(1+0.6*tanh((z-zres)/(0.6*Ln)))'),
          normalization='one W/m2 upward incident flux at lower diagnostic plane; optimize source polarization for local Gaussian Bragg column',
          physics='same coupled Maxwell warm-fluid discrete CW recurrence; collisions; no Landau damping; F cases prescribed, not observations',
          frequency_Hz=a.F,B_T=a.P['magnetic_t'],B_angle_deg=a.P['field_angle_deg'])
        for k,val in dict(z_m=z,ne_m3=s['ne'][0],density_phasor_m3=dn,E_phasor_V_m=state[:3],
                pressure_flux_W_m2=pf,EM_flux_W_m2=em,phase_kz_rad_m=phase,source_polarization=v,
                collision_loss_per_cell_W_m2=coll,dz_cells_m=s['dz_cells']).items():h[k]=val
        h['provenance/study_source']=Path(__file__).read_text();h['provenance/shared_preparation_source']=Path(a.__file__).read_text()
    print(stats,'seconds',time.monotonic()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',default='measured');p.add_argument('--angle',type=float,default=16.75);p.add_argument('--dz',type=float,default=.02);p.add_argument('--cap',type=float,default=4.);p.add_argument('--extra',type=float,default=1.);v=p.parse_args();run(v.case,v.angle,v.dz,v.cap,v.extra)
