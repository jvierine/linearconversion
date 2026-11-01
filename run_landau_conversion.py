"""Reproducible measured-profile kinetic damping and Bragg-height study.

Source provenance is retained in HDF5. All heights are radar-relative.
The smooth fitted gradient is observationally constrained on 300 m samples;
meter-scale root separations are conditional model outputs, not resolution.
"""
import json,gc
from pathlib import Path
import h5py
import numpy as np
from scipy.constants import c,e,epsilon_0,m_e
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from conversion_lab import solve,DEFAULT
from isr.conversion import Parameters,matching_ratio
from isr.landau import bragg_match,branch_table,spatial_root
from isr.observed_layer import fit_observed_layer,profile

ROOT=Path(__file__).resolve().parent

def study():
    out=ROOT/'data/landau_bragg_study.h5'
    with h5py.File(out,'w') as h:
        h.attrs.update(provenance='run_landau_conversion.py; isr/landau.py; isr/conversion.py',
            radar_frequency_hz=430e6,temperature_k=300.,magnetic_t=35.025e-6,field_vertical_deg=46.815,
            hf_incidence_deg=-20.,height_definition='height above radar, z positive upward',
            limitation='Conditional model on smoothly fitted density. 300 m sampled measurements do not resolve the meter-scale separation. Maxwellian local longitudinal closure is not a full kinetic tensor.',
            reference='https://doi.org/10.1029/2009GL038379')
        for epoch in [1525453430,1525460596]:
            fitted=fit_observed_layer(epoch)
            for mhz in [4.51,5.33]:
                q=np.sin(np.deg2rad(-20.));kB=4*np.pi*430e6/c
                delta=np.arcsin(q*2*np.pi*mhz*1e6/c/kB)
                par=Parameters(frequency_hz=mhz*1e6,temperature_k=300.,magnetic_t=35.025e-6,magnetic_tilt_deg=46.815-np.rad2deg(delta),q=q)
                ratio=lambda zz: profile(zz,fitted['coefficients'],fitted['knots_km'])[0]*e*e/(epsilon_0*m_e*par.omega**2)
                grid=fitted['model_height_m'];pg=ratio(grid)
                def height(target):
                    i=np.where((pg[:-1]<target)&(pg[1:]>=target))[0][0]
                    return brentq(lambda zz:ratio(zz)-target,grid[i],grid[i+1],xtol=1e-8)
                Y2=par.gyro@par.gyro;ca=np.cos(np.deg2rad(par.magnetic_tilt_deg))
                pres=(1-Y2)/(1-Y2*ca*ca);pc,alpha=bragg_match(par,kB)
                zres=height(pres);zb=height(pc);zc=height(1.)
                pf=matching_ratio([q*par.k0,0,-np.sqrt(kB*kB-(q*par.k0)**2)],par)[0]
                zz=np.linspace(zres-200,zres,4001);pp=ratio(zz)
                tab=branch_table(par,min(pp)*.999,points=1400)
                pr=tab['density_ratio'];kk=tab['kdown_m1']
                interp=lambda a:PchipInterpolator(pr,a)(np.clip(pp,pr[0],pr[-1]))
                kr=interp(kk.real);atten=interp(kk.imag)
                # No claimed WKB damping at/above long-wave conversion zone.
                atten[pp>pr[-1]]=0.;kr[pp>pr[-1]]=np.nan
                tau=-cumulative_trapezoid(atten[::-1],zz[::-1],initial=0)[::-1]
                g=h.create_group(f'{epoch}/{mhz:.2f}')
                g.attrs.update(profile_time_unix_s=fitted['time_unix_s'],density_sha256=fitted['source_sha256'],
                    hf_hz=par.frequency_hz,bragg_height_m=zb,critical_height_m=zc,resonance_height_m=zres,
                    resonance_to_bragg_m=zres-zb,critical_to_bragg_m=zc-zb,
                    fluid_bragg_height_m=height(pf),bragg_wavelength_m=2*np.pi/kB,
                    bragg_alpha_m1=alpha,bragg_amplitude_efold_m=1/alpha if alpha>0 else np.inf,
                    amplitude_survival_resonance_to_bragg=float(np.exp(-np.interp(zb,zz,tau))),
                    table_residual=tab['maximum_dielectric_residual'])
                for name,value in dict(height_m=zz,height_relative_resonance_m=zz-zres,
                    density_ratio=pp,kz_real_m1=-kr,alpha_m1=atten,amplitude_survival=np.exp(-tau),
                    power_survival=np.exp(-2*tau),branch_density_ratio=pr,branch_kdown_m1=kk).items():g[name]=value
                print(epoch,mhz,dict(g.attrs),flush=True)
    # Full-wave fields, with normal electron drag and collisionless pressure loss
    # separately integrated in the energy budget. Matching summaries above use
    # exact spline roots rather than the full-wave grid's interpolation.
    for epoch in [1525453430,1525460596]:
        for mhz in [4.51,5.33]:
            result=solve(dict(profile_time_unix_s=epoch,frequency_mhz=mhz,landau_mode=1.))
            with open(ROOT/f'data/landau_conversion_{epoch}_{mhz:.2f}.json','w') as f:json.dump(result,f)
            with h5py.File(out,'a') as h:
                g=h[f'{epoch}/{mhz:.2f}'];g.attrs['full_wave_solution_id']=result['id']
                for name,val in result['budget'].items():g.attrs[name]=val
            print('FULL',epoch,mhz,result['id'],result['budget'],flush=True);gc.collect()
    coarse=solve(dict(profile_time_unix_s=1525453430,frequency_mhz=5.33,dz_m=.005,landau_mode=1.))
    with h5py.File(out,'a') as h:h['1525453430/5.33'].attrs['coarse_solution_id']=coarse['id']
    print('COARSE',coarse['id'],coarse['budget'],flush=True)

if __name__=='__main__':
    study()
    from prepare_landau_figures import prepare
    prepare()
