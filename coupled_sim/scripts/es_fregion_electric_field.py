"""Measured Es/F backgrounds: coupled CW fields and physical energy budgets.

Common ordinary-mode illumination; actual profiles, temperatures, B and collision
rates for each region. Independent illumination below each layer, not a ray
through the Es layer into the F region. A near-vertical control tests reflection.
"""
from pathlib import Path
import gc
import h5py
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0, k as kb
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import coupled_arecibo as a
from hf_absorption_profile import modes, ANGLE

ROOT=Path(__file__).resolve().parents[2]

def configuration(region):
    if region=='Es':
        return dict(ordering='COLAMD',fine_bounds=(-150.,40.)),300.,115100.,120900.
    with h5py.File(a.ROOT/'data/fregion_observed_background.h5') as h:
        z=h['height_m'][:];g=h['smoothing_10km'];at=dict(g.attrs)
        profile=CubicSpline(z,g['ne_m3'][:])
        neutrals={key:h['neutrals/'+key][:] for key in ['N2_m3','O2_m3','O_m3']}
    temp=at['temperature_K'];ln=at['Ln_m'];center=at['center_m']
    def rates(zz,ne):
        tev=kb*temp/e;ncm=np.maximum(ne*1e-6,1e-30)
        ei=2.91e-6*ncm*(23-np.log(np.sqrt(ncm)*tev**-1.5))/tev**1.5
        n2,o2,o=[np.exp(np.interp(zz,z,np.log(neutrals[key]))) for key in ['N2_m3','O2_m3','O_m3']]
        en=2.33e-17*n2*(1-1.21e-4*temp)*temp+1.82e-16*o2*(1+.036*np.sqrt(temp))*np.sqrt(temp)+8.9e-17*o*(1+5.7e-4*temp)*np.sqrt(temp)
        return ei+en
    d=ln*(3*kb*temp/m_e)*((4*np.pi*430e6+a.W)/c)**2/a.W**2
    cfg=dict(profile=profile,center=center,ordering='COLAMD',bounds=(190000.,274000.),
             sample_bounds=(193000.,273000.),reference_z=193000.,source_z=191900.,
             fine_bounds=(-1.1*d-400.,40.),phase_grid_distance_m=d,
             temperature=temp,rates=rates,magnetic_t=at['magnetic_t'],field_angle_deg=at['field_angle_deg'])
    return cfg,temp,192100.,273100.

def solve(group,region,angle,dz,cap,config_override=None):
    cfg,temp,lower,upper=configuration(region)
    if config_override:cfg.update(config_override)
    print('Solve',group.name,angle,dz,cap,flush=True)
    s=a.setup(angle=angle,dz=dz,cap=cap,scenario=cfg)
    ref=a.setup(angle=angle,dz=dz,cap=cap,scenario=cfg,reference=True)
    z=s['z'];width=s['dz_cells'];dual=s['dual'];zf=z+width/2
    lo=np.argmin(abs(zf-lower));hi=np.argmin(abs(zf-upper))
    v,err=modes(s,ref,lo)['O']
    inc=np.einsum('qnp,p->qn',ref['initial'],v)
    v/=np.sqrt(a.electromagnetic_flux(inc)[lo])
    state=np.einsum('qnp,p->qn',s['initial'],v)
    inc=np.einsum('qnp,p->qn',ref['initial'],v)
    budget=a.hf_port_budget(state,inc,lo,hi)
    pin=budget['incident_EM_flux_W_m2'];ch=np.cos(s['omega']*s['dt']/2)
    weights=np.array([dual,dual,width])
    cells=.5*epsilon_0*c*ch*np.sum(s['physical_nu']*abs(state[6:9])**2*weights,axis=0)/pin
    mask=(z>zf[lo])&(z<zf[hi]);heat=cells[mask].sum()
    em,pf=a.flux(s,state);_,pref=a.flux(s,state-inc);_,pinc=a.flux(ref,inc)
    # Total energy crossing ports includes any outgoing pressure flux.
    pressure_out=(pf[hi]-pref[lo]-pinc[lo])/pin
    residual=1-budget['reflected_EM_fraction']-budget['transmitted_EM_fraction']-pressure_out-heat
    profile=cfg.get('profile',a.PROFILE);center=s['center']
    ln=float(profile(center)/profile.derivative()(center))
    turn=brentq(lambda zz:profile(zz)*e*e/(epsilon_0*m_e)-a.W*a.W,center,center+.15*ln)
    # Collocate transverse E with Ez before taking total incident magnitude.
    ince=np.array([(inc[0]+np.roll(inc[0],-1))/2,(inc[1]+np.roll(inc[1],-1))/2,inc[2]])
    norm=np.linalg.norm(ince[:,lo]);mag=abs(state[2])/norm
    # Save full extent (and mesh) for diagnostics; only plot the reflection region.
    stats=dict(budget,region=region,angle_deg=angle,dz_m=dz,cap_m=cap,temperature_K=temp,
               center_m=center,Ln_m=ln,O_turn_relative_m=turn-center,
               incident_E_peak_V_m=norm,peak_Ez_over_Einc=mag[mask].max(),
               absorbed_heat_fraction=heat,outgoing_pressure_fraction=pressure_out,
               flux_minus_heat_pressure=residual,mode_boundary_relative_error=err,
               nu_res_s1=float(np.interp(center,z,s['physical_nu'][0])*c),
               magnetic_T=np.linalg.norm(s['gyro'])*m_e*c/e,
               lower_port_m=zf[lo],upper_port_m=zf[hi],harmonic_residual=s['harmonic_residual'])
    group.attrs.update(stats)
    for k,val in dict(height_relative_m=zf-center,Ez_over_Einc=mag,Ez_complex_over_Einc=state[2]/norm,
                      ne_m3=s['ne'][0],source_polarization=v,z_nodes_m=z,dz_cells_m=width,
                      absorbed_fraction_per_cell=cells*mask,EM_flux_over_incident=em/pin,
                      pressure_flux_over_incident=pf/pin).items():
        group.create_dataset(k,data=val,compression='gzip')
    print(stats,flush=True)
    del s,ref;gc.collect()

def export(full,out):
    """Keep exact plotted fields and conservative 10-m heating sums portable."""
    with h5py.File(full) as src,h5py.File(out,'w') as dst:
        dst.attrs.update(dict(src.attrs));src.copy('provenance',dst)
        for name in src:
            if name=='provenance':continue
            s=src[name];g=dst.create_group(name);g.attrs.update(dict(s.attrs))
            z=s['z_nodes_m'][:];edges=np.arange(np.floor(z.min()/10)*10,np.ceil(z.max()/10)*10+10,10)
            heat,_=np.histogram(z,edges,weights=s['absorbed_fraction_per_cell'][:])
            g['heating_bin_edges_m']=edges;g['absorbed_fraction_per_cell']=heat
            g.attrs['heating_bins']='Conservative sums in 10 m bins; zero outside diagnostic ports.'
            if name in ['Es_fine','F_fine']:
                height=s['height_relative_m'][:];turn=s.attrs['O_turn_relative_m']
                keep=(height-turn>-3000)&(height-turn<2000)
                for key in ['height_relative_m','Ez_over_Einc']:
                    g.create_dataset(key,data=s[key][:][keep],compression='gzip')

def main():
    out=a.OUT/'es_fregion_electric_field_full.h5'
    with h5py.File(out,'w') as h:
        h.attrs.update(frequency_hz=a.F,scope='Coupled warm-fluid/FDTD CW. Independently illuminated measured Es and F-region backgrounds. Ordinary-mode input; collisional damping, no Landau damping. No retuning of polarization for maximum RCS.',normalization='Magnitude of total incident E phasor at homogeneous lower reference port; unit upward incident EM flux')
        h['provenance/source']=Path(__file__).read_text()
        for name,reg,ang,dz,cap in [('Es','Es',ANGLE,.005,.5),('F','F',ANGLE,.02,2.),('Es_vertical','Es',0.,.005,.5),('F_vertical','F',0.,.02,2.),('Es_fine','Es',ANGLE,.0025,.5),('F_fine','F',ANGLE,.01,1.),('F_vertical_fine','F',0.,.01,1.)]:
            solve(h.create_group(name),reg,ang,dz,cap);h.flush()
    portable=ROOT/'reference/coupled/es_fregion_electric_field.h5'
    export(out,portable)
    print(portable,flush=True)
if __name__=='__main__':main()
