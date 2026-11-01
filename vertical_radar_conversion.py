"""HF incidence optimization with the user-confirmed vertical radar fixed.

Radar k is nearly normal to the Es layer. Tangential k conservation sets
the small local surface tilt delta=asin(q*k0/kB), not an arbitrary radar
pointing. In the coplanar case B/normal=assumed B/vertical-delta.
q=0 is the exactly horizontal-layer/vertical-radar limit.
"""
from dataclasses import replace
import h5py
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0
from scipy.optimize import minimize_scalar
from isr.conversion import Parameters, layer, optimal_polarization, matching_ratio, physical_density

K_RADAR=4*np.pi*430e6/c
B_VERTICAL_DEG=42.  # illustrative field geometry, not a 2018 field reconstruction


def parameters(q):
    base=Parameters()
    delta=np.rad2deg(np.arcsin(q*base.k0/K_RADAR))
    return replace(base,q=float(q),magnetic_tilt_deg=B_VERTICAL_DEG-delta),delta


def run(q, dz=.0025, scale=10., nu=0., extent_factor=6.):
    par,delta=parameters(q)
    z=np.linspace(-extent_factor*scale,extent_factor*scale,round(2*extent_factor*scale/dz)+1)
    r=optimal_polarization(z,*layer(z,scale,collision_s=nu),par)
    kz=-np.sqrt(K_RADAR**2-(par.q*par.k0)**2)
    pm,=matching_ratio([par.q*par.k0,0,kz],par)
    zm=scale*np.arctanh(2*(pm-.65)/.65-1)
    m=np.argmin(abs(z-zm))
    r.update(layer_normal_tilt_deg=delta,radar_zenith_deg=0.,
             magnetic_normal_deg=par.magnetic_tilt_deg,
             radar_match_z_m=zm,radar_match_ratio=pm,
             radar_match_density_m3=pm*par.omega**2*epsilon_0*m_e/e**2,
             fractional_density_at_match_per_sqrt_w_m2=abs(physical_density(r['h'][m],par,1.)),
             vacuum_incidence_deg=float(np.rad2deg(np.arcsin(q))),q=float(q))
    return r


def save_result(group,r,fields=True):
    for key,value in r.items():
        if np.isscalar(value):group.attrs[key]=value
        elif fields:group[key]=value


def main():
    with h5py.File('data/vertical_radar_conversion.h5','w') as f:
        f.attrs.update(radar_zenith_deg=0.,radar_frequency_hz=430e6,
            radar_geometry_source='User: radar vertical; k_B almost perpendicular to Es layer',
            assumed_acute_B_vertical_deg=B_VERTICAL_DEG,magnetic_t=40e-6,
            temperature_k=1000.,hf_frequency_hz=5.33e6,
            scope='Favorable linear warm-fluid benchmark, not event fit or transmitter identification',
            polarization='Coherent EM eigenmode mixture optimized at lower reservoir',
            provenance='vertical_radar_conversion.py; isr/conversion.py',
            pressure_closure='Scalar gamma=3; no kinetic pressure tensor/FLR or Landau damping',
            default_scale_m=10.,density_low_ratio=.65,density_high_ratio=1.3,
            geometry='Coplanar B, density normal and incident k; q=k_tangent/k0; signed normal tilt=asin(q*k0/kB)')
        q=np.linspace(-.5,.5,81)
        g=f.create_group('incidence');g['q']=q
        g['vacuum_incidence_deg']=np.rad2deg(np.arcsin(q))
        results=[run(qi,dz=.0025) for qi in q]
        for key in ['conversion','reflection','energy_error','layer_normal_tilt_deg']:
            g[key]=[r[key] for r in results]
        j=int(np.argmax(g['conversion'][:]))
        qb=q[j]
        opt=minimize_scalar(lambda x:-run(x,dz=.0025)['conversion'],
            bounds=(q[max(0,j-2)],q[min(len(q)-1,j+2)]),method='bounded',
            options={'xatol':1e-6})
        qb=float(opt.x)
        f.attrs.update(best_q=qb,optimizer_success=bool(opt.success),
                       best_hf_vacuum_angle_deg=np.rad2deg(np.arcsin(qb)))
        print('vertical radar best incidence',qb,np.rad2deg(np.arcsin(qb)),flush=True)
        for name,qv,dz,nu,extent in [
            ('horizontal_layer',0.,.000625,0.,6.),
            ('best_fine',qb,.00125,0.,6.),('best_finer',qb,.000625,0.,6.),
            ('best_extended',qb,.00125,0.,8.),
            ('best_drag_1000',qb,.000625,1000.,6.),
            ('best_drag_10000',qb,.000625,10000.,6.)]:
            r=run(qv,dz=dz,nu=nu,extent_factor=extent)
            g=f.create_group(name);g.attrs.update(dz_m=dz,collision_peak_s=nu,extent_factor=extent)
            save_result(g,r)
            print(name,{key:r[key] for key in ['conversion','reflection','absorption','energy_error',
                'layer_normal_tilt_deg','fractional_density_at_match_per_sqrt_w_m2']},flush=True)
            if name=='best_drag_1000':
                plot=f.create_group('plot');par,_=parameters(qv)
                zm=r['radar_match_z_m'];z=r['z_m'];h=r['h']
                keep=abs(z-zm)<=2.01
                density=physical_density(h,par,1e-6)
                phasek=np.imag(np.conj(h)*np.gradient(h,z))/np.maximum(abs(h)**2,1e-100)
                plot.attrs.update(illustrative_incident_normal_flux_w_m2=1e-6,
                                  collision_peak_s=1000.,q=qb)
                plot['z_relative_match_m']=(z-zm)[keep]
                plot['density_real_percent']=100*density.real[keep]
                plot['density_envelope_percent']=100*abs(density[keep])
                plot['phase_kz_m1']=-phasek[keep]
        scale_study(f)


def scale_study(f):
    if "scale_sensitivity" in f: del f["scale_sensitivity"]
    # Re-optimize incidence for broader ramps, rather than misinterpreting
    # a fixed off-window angle as failure of conversion in a broad layer.
    root=f.create_group('scale_sensitivity')
    for scale in [10.,30.,100.]:
        # Search both signed incidence directions before local refinement;
        # the best window can move substantially as the gradient broadens.
        coarse_q=np.linspace(-.5,.5,41)
        coarse_C=np.array([run(x,dz=.01,scale=scale)['conversion'] for x in coarse_q])
        j=int(np.argmax(coarse_C))
        opt=minimize_scalar(lambda x:-run(x,dz=.01,scale=scale)['conversion'],
            bounds=(coarse_q[max(0,j-1)],coarse_q[min(40,j+1)]),method='bounded',
            options={'xatol':2e-4,'maxiter':35})
        qv=float(opt.x)
        g=root.create_group(str(int(scale)));g.attrs.update(scale_m=scale,
            optimizer_success=bool(opt.success),coarse_optimization_dz_m=.01)
        g['coarse_q']=coarse_q;g['coarse_conversion']=coarse_C
        for name,dz in [('fine',.0025),('finer',.00125)]:
            r=run(qv,dz=dz,scale=scale)
            out=g.create_group(name);out.attrs['dz_m']=dz
            save_result(out,r,fields=False)
            print('scale',scale,name,'q',qv,'C',r['conversion'],'balance',r['energy_error'],flush=True)


if __name__=='__main__':
    import sys
    if "--scales-only" in sys.argv:
        with h5py.File("data/vertical_radar_conversion.h5", "r+") as f: scale_study(f)
    else: main()
