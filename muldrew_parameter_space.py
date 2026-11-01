"""Signed-incidence and damping window on the measured smooth Es profile.

Muldrew (1993), doi:10.1029/92RS00273 supplies the ray-geometry benchmark,
not event parameters. Full-wave launch is solved for both incident EM modes.
Their coherent radar amplitudes define the favorable polarization envelope.
Tables retain separate eigenpolarization responses; no assumption that an
unknown transmitter supplies the optimum is hidden in a pass/fail flag.
"""
import argparse
import gc
from pathlib import Path
import h5py
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0
from scipy.optimize import brentq
from isr.conversion import Parameters, scatter, physical_density
from isr.landau import closure_profile, bragg_match
from isr.observed_layer import fit_observed_layer, scattering_profile, profile
from isr.rays import spitze_angle_deg, cold_horizontal_group_point, electrostatic_ray
from conversion_lab import coherent_gate

OUT=Path('data/muldrew_parameter_space.h5')
KR=4*np.pi*430e6/c


def initialize():
    with h5py.File(OUT, 'a') as h:
        h.attrs.update(provenance='muldrew_parameter_space.py; isr/rays.py; isr/conversion.py; isr/landau.py',
            reference='https://doi.org/10.1029/92RS00273', epoch_unix_s=1525453430,
            magnetic_t=35.025e-6, field_vertical_deg=46.815, radar_frequency_hz=430e6,
            geometry='Coplanar signed incidence; positive x is horizontal projection of upward magnetic axis. For Arecibo this is magnetic south. Negative incidence is northward. Exact vertical radar matching uses delta=asin(kx/kB).',
            normalization='Reference measured peak divided by its previous 5.33 MHz, -20 deg, Te300 K, nu1000/s full-wave prediction; all new runs use the same incident normal flux.',
            polarization='Envelope maximizes the coherent Gaussian-gate amplitude over unit-flux incident EM polarizations independently at each point; separate modal powers are also saved.',
            limitation='Conditional detectability with a 95 m sigma coherent Gaussian gate and infinite horizontal plane illumination. Finite beam overlap and the actual ambiguity function are not established.')
        with h5py.File('data/gradient_detectability.h5') as ref:
            h.attrs['detection_threshold_fraction']=ref.attrs['detection_threshold_fraction']
            h.attrs['reference_power']=abs(ref['1.000/combined/radar_amplitude'][1])**2
            h.attrs['reference_density_gradient_m4']=ref.attrs['reference_density_gradient_m4']
        # Muldrew's printed numbers are an independent cold-dispersion check.
        p=Parameters(frequency_hz=8.175e6,temperature_k=550,
                     magnetic_t=2*np.pi*1e6*m_e/e,magnetic_tilt_deg=42.)
        p=Parameters(**(p.__dict__|dict(q=np.sin(np.deg2rad(spitze_angle_deg(p))))))
        g=h.require_group('muldrew1993_benchmark')
        g.attrs.update(cold_horizontal_group_point(p),critical_angle_deg=spitze_angle_deg(p),
                       reported_refractive_index=4.743,reported_density_ratio=.992882,
                       reported_critical_angle_deg=12.762)


def evaluate(angle=-20.,gradient=1.,temperature=300.,collision=1000.,freq=5.33,dz=.02):
    fit=fit_observed_layer(1525453430)
    q=np.sin(np.deg2rad(angle));delta=np.arcsin(q*2*np.pi*freq*1e6/c/KR)
    par=Parameters(frequency_hz=freq*1e6,temperature_k=temperature,magnetic_t=35.025e-6,
                   magnetic_tilt_deg=46.815-np.rad2deg(delta),q=q)
    pc,alpha=bragg_match(par,KR)
    Y2=par.gyro@par.gyro;ca=np.cos(np.deg2rad(par.magnetic_tilt_deg))
    pres=(1-Y2)/(1-Y2*ca*ca)
    def ratio(h):return profile(h,fit['coefficients'],fit['knots_km'])[0]*e**2/(epsilon_0*m_e*par.omega**2)
    grid=fit['model_height_m'];pp=ratio(grid)
    def height(target):
        ix=np.flatnonzero((pp[:-1]<target)&(pp[1:]>=target))
        if not len(ix):raise ValueError('No bottomside density match')
        j=ix[0];return brentq(lambda h:ratio(h)-target,grid[j],grid[j+1])
    res=height(pres);zb=height(pc)
    lo=fit['es_center_m']-5*fit['es_sigma_m'];hi=fit['es_center_m']+5*fit['es_sigma_m']
    z=np.linspace((lo-res)/gradient,(hi-res)/gradient,round((hi-lo)/(gradient*dz))+1)
    base=res+gradient*z
    ne,dn=scattering_profile(base,fit['coefficients'],fit['knots_km'])
    p=ne*e**2/(epsilon_0*m_e*par.omega**2);grad=gradient*dn/ne
    u=np.clip(np.minimum(base-base[0],base[-1]-base)/400,0,1)
    nu=collision*(10*u**3-15*u**4+6*u**5);nu[[0,-1]]=0
    xi,_=closure_profile(base,p,par,points=650)
    center=(zb-res)/gradient
    amplitudes=[];errors=[]
    for branch in [0,1]:
        r=scatter(z,p,grad,nu,par,branch=branch,pressure_response=xi,cell_order=4)
        density=physical_density(r['h'],par,1e-6)
        amplitudes.append(coherent_gate(z,ne,density,center,95.,KR)[0])
        errors.append(r['energy_error']);del r,density;gc.collect()
    nB,dnB,*_=profile(zb,fit['coefficients'],fit['knots_km'])
    return dict(angle_deg=angle,gradient_factor=gradient,temperature_k=temperature,collision_s=collision,
                hf_mhz=freq,dz_m=dz,layer_tilt_deg=np.rad2deg(delta),spitze_angle_deg=spitze_angle_deg(par),
                density_gradient_m4=float(dnB*gradient),density_m3=float(nB),
                scale_height_m=float(nB/(dnB*gradient)),resonance_to_bragg_m=float((res-zb)/gradient),
                radar_amplitude=np.array(amplitudes),energy_error=np.array(errors),bragg_alpha_m1=alpha)


def run_case(section,label,**kwargs):
    key=section+'/'+label
    with h5py.File(OUT,'r') as h:
        if key in h:return
    result=evaluate(**kwargs)
    with h5py.File(OUT,'a') as h:
        g=h.create_group(key)
        for k,v in result.items():
            if np.isscalar(v):g.attrs[k]=v
            else:g[k]=v
        a=result['radar_amplitude'];reference=h.attrs['reference_power'];threshold=h.attrs['detection_threshold_fraction']
        power=float(np.vdot(a,a).real/reference)
        g['mode_relative_power']=abs(a)**2/reference
        g['optimal_weights']=np.conj(a)/np.linalg.norm(a)
        g.attrs['optimal_relative_power']=power
        if result['hf_mhz']==5.33:
            g.attrs.update(required_flux_relative_to_reference=threshold/power,
                           detectable_at_reference_flux=bool(power>=threshold))
        print(key,'relative',power,'Eerr',result['energy_error'],flush=True)


def ray_products():
    with h5py.File('data/landau_bragg_study.h5') as source,h5py.File(OUT,'a') as h:
        for freq in ['4.51','5.33']:
            key='rays/'+freq
            if key in h:del h[key]
            s=source['1525453430/'+freq];z=s['height_m'][:][::-1];p=s['density_ratio'][:][::-1]
            kd=-s['kz_real_m1'][:][::-1]+1j*s['alpha_m1'][:][::-1]
            keep=np.isfinite(kd.real)&(kd.real>=5)&(z>=s.attrs['bragg_height_m']-20)
            q=np.sin(np.deg2rad(-20));delta=np.arcsin(q*2*np.pi*float(freq)*1e6/c/KR)
            par=Parameters(frequency_hz=float(freq)*1e6,temperature_k=300,magnetic_t=35.025e-6,
                           magnetic_tilt_deg=46.815-np.rad2deg(delta),q=q)
            ac=s['damping/collisional/alpha_m1'][:][::-1][keep]
            ray=electrostatic_ray(z[keep],p[keep],kd[keep],par,ac)
            g=h.create_group(key)
            for k,v in ray.items():g[k]=v
            zb=s.attrs['bragg_height_m']
            g.attrs.update(bragg_height_m=zb,resonance_height_m=s.attrs['resonance_height_m'],
                origin='x=0 at first weak-loss ES sample with k>=5/m, not at transmitter or EM conversion start',
                local_approximation='Weak-loss electrostatic kinetic group velocity; launch amplitude and conversion are determined separately by full-wave Maxwell solution')
            for name in ['horizontal_displacement_m','travel_time_s','path_length_m']:
                g.attrs[name+'_to_bragg']=float(np.interp(zb,ray['height_m'][::-1],ray[name][::-1]))
            print(key,dict(g.attrs),flush=True)


def finalize():
    with h5py.File(OUT,'a') as h:
        h.attrs['detectability_frequency_hz']=5.33e6
        h.attrs['other_frequency_scope']='4.51 MHz curves compare modeled powers for equal incident normal flux; the 5.33 MHz receiver threshold is not a separately calibrated 4.51 MHz detection test.'
        for name in h['incidence']:
            g=h['incidence/'+name]
            if g.attrs['hf_mhz']!=5.33:
                for k in ['detectable_at_reference_flux','required_flux_relative_to_reference']:
                    if k in g.attrs:del g.attrs[k]
        angles=np.array([-30,-25,-20,-17.5,-15,-12.5,-10,-5,0,5,10,12.5,15,17.5,20,25,30.])
        gradients=np.array([.25,.5,1.]);values=[]
        for gradient in gradients:
            row=[]
            for angle in angles:
                key=f'incidence/5.33_{angle:+06.1f}' if gradient==1 else f'angle_gradient/{gradient:.3f}_{angle:+06.1f}'
                row.append(h[key].attrs['optimal_relative_power'])
            values.append(row)
        if 'allowed_map' in h:del h['allowed_map']
        g=h.create_group('allowed_map');g['angle_deg']=angles;g['gradient_factor']=gradients
        g['density_gradient_m4']=gradients*h.attrs['reference_density_gradient_m4']
        g['margin_db']=10*np.log10(np.asarray(values)/h.attrs['detection_threshold_fraction'])
        g.attrs.update(temperature_k=300,collision_s=1000,hf_mhz=5.33,
                       interpretation='Computed sample cells; colors do not establish exact continuous boundaries. Positive margin is detectable under reference illumination and favorable polarization.')
        for name,section,label in [('baseline','gradient_collision','1.000_1000'),('south','incidence','5.33_+020.0'),('threshold','gradient_collision','0.170_1000'),('boundary','gradient_collision','0.150_1000'),('low_collision','gradient_collision','1.000_100')]:
            if 'mesh_validation/'+name not in h:continue
            fine=h['mesh_validation/'+name].attrs['optimal_relative_power'];coarse=h[section+'/'+label].attrs['optimal_relative_power']
            h['mesh_validation/'+name].attrs['relative_power_mesh_change']=abs(fine/coarse-1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--section',default='all');args=parser.parse_args()
    initialize()
    if args.section in ['all','rays']:ray_products()
    if args.section in ['all','incidence']:
        for angle in [-30,-25,-20,-17.5,-15,-12.5,-10,-5,0,5,10,12.5,15,17.5,20,25,30]:
            for freq in [5.33,4.51]:run_case('incidence',f'{freq:.2f}_{angle:+06.1f}',angle=angle,freq=freq)
    if args.section in ['all','damping']:
        for gradient in [1.,.5,.25,.125]:
            for collision in [100.,1000.,3000.,10000.]:
                run_case('gradient_collision',f'{gradient:.3f}_{collision:.0f}',gradient=gradient,collision=collision)
    if args.section in ['all','temperature']:
        for temperature in [200.,300.,600.,1000.,1500.]:
            run_case('temperature',f'{temperature:.0f}',temperature=temperature)
    if args.section in ['all','angle_gradient']:
        for gradient in [.5,.25]:
            for angle in [-30,-25,-20,-17.5,-15,-12.5,-10,-5,0,5,10,12.5,15,17.5,20,25,30]:
                run_case('angle_gradient',f'{gradient:.3f}_{angle:+06.1f}',gradient=gradient,angle=angle)
    if args.section in ['all','refine']:
        for angle in [-33,-32,-31,-2,-1,15.5,16,16.5,31,32,33]:
            run_case('incidence',f'5.33_{angle:+06.1f}',angle=angle)
        for gradient in [.15,.17,.20]:run_case('gradient_collision',f'{gradient:.3f}_1000',gradient=gradient)
        for nu in [5000.,6000.,7000.]:run_case('gradient_collision',f'1.000_{nu:.0f}',collision=nu)
        for label,kw in [('baseline',{}),('south',{'angle':20}),('threshold',{'gradient':.17}),('boundary',{'gradient':.15}),('low_collision',{'collision':100})]:
            run_case('mesh_validation',label,dz=.01,**kw)
    if args.section in ['all','summary']:finalize()
