"""Refine best-case conversion including coherent EM eigenmode mixtures.

The 22--62 degree B/normal window is an illustrative Arecibo-accessible
screen: assumed acute B/vertical=42 deg, radar zenith <=20 deg, and radar
wavevector nearly parallel to the local density-gradient normal. It is not
a reconstruction of the event pointing or the 2018 geomagnetic field.
"""
from dataclasses import replace
import h5py
import numpy as np
from scipy.constants import c, epsilon_0, m_e, e
from scipy.optimize import minimize
from isr.conversion import Parameters, optimal_polarization, layer, matching_ratio, physical_density


def run(q, tilt, dz=.0025, nu=0., scale=10.):
    par=replace(Parameters(),q=float(q),magnetic_tilt_deg=float(tilt))
    z=np.linspace(-6*scale,6*scale,round(12*scale/dz)+1)
    return optimal_polarization(z,*layer(z,scale,collision_s=nu),par)


def main():
    with h5py.File('data/linear_conversion_geometry.h5','r+') as f:
        q=f['q'][:];a=f['magnetic_tilt_deg'][:]
        C=np.nansum(f['conversion'][:],axis=2)
        if 'coherent_upper_bound' in f:del f['coherent_upper_bound']
        f['coherent_upper_bound']=C
        if 'coherent' in f:del f['coherent']
        root=f.create_group('coherent')
        root.attrs.update(assumed_acute_B_vertical_deg=42.,assumed_radar_max_zenith_deg=20.,
            source_pointing_limit='https://aasarchives.blob.core.windows.net/archives/BAAS/v29n1/008701bas.pdf',
            scope='Best-case coherent lower-boundary illumination; not identified transmitter',
            provenance='optimize_polarization.py; isr/conversion.py')
        for name,bounds in [('unrestricted',(0.,90.)),('arecibo_screen',(22.,62.)),
                             ('horizontal_layer',(42.,42.))]:
            valid=(a>=bounds[0])&(a<=bounds[1])
            if bounds[0]==bounds[1]:
                q0=q[np.argmax(C[np.argmin(abs(a-bounds[0]))])];a0=bounds[0]
                opt=minimize(lambda x:-run(x[0],a0)['conversion'],[q0],
                    method='Nelder-Mead',bounds=[(-.5,.5)],
                    options={'xatol':1e-5,'fatol':1e-6,'maxiter':60})
                qb,ab=opt.x[0],a0
            else:
                score=np.where(valid[:,None],C,-np.inf)
                i,j=np.unravel_index(np.argmax(score),score.shape)
                opt=minimize(lambda x:-run(x[0],x[1])['conversion'],[q[j],a[i]],
                    method='Nelder-Mead',bounds=[(-.5,.5),bounds],
                    options={'xatol':1e-4,'fatol':1e-6,'maxiter':80})
                qb,ab=opt.x
            print(name,'q',qb,'tilt',ab,'success',opt.success,flush=True)
            group=root.create_group(name)
            group.attrs.update(q=qb,magnetic_tilt_deg=ab,
                vacuum_incidence_deg=np.rad2deg(np.arcsin(qb)),optimizer_success=bool(opt.success))
            par=replace(Parameters(),q=float(qb),magnetic_tilt_deg=float(ab))
            kB=4*np.pi*430e6/c;kx=qb*par.k0;kz=-np.sqrt(kB*kB-kx*kx)
            pm,=matching_ratio([kx,0,kz],par)
            for label,dz,nu in [('fine',.00125,0.),('finer',.000625,0.),
                                 ('drag_1000',.00125,1000.),('drag_10000',.00125,10000.)]:
                r=run(qb,ab,dz=dz,nu=nu)
                g=group.create_group(label);g.attrs.update(dz_m=dz,collision_peak_s=nu)
                for key,value in r.items():
                    if np.isscalar(value):g.attrs[key]=value
                    elif name=='arecibo_screen':g[key]=value
                zm=10*np.arctanh(2*(pm-.65)/.65-1)
                m=np.argmin(abs(r['z_m']-zm))
                g.attrs.update(radar_match_ratio=pm,radar_match_z_m=zm,
                    radar_match_density_m3=pm*par.omega**2*epsilon_0*m_e/e**2,
                    required_wave_tilt_from_normal_deg=np.rad2deg(np.arcsin(kx/kB)),
                    fractional_density_at_match_per_sqrt_w_m2=abs(physical_density(r['h'][m],par,1.)))
                print(name,label,'C',r['conversion'],'R',r['reflection'],
                      'A',r['absorption'],'balance',r['energy_error'],
                      'density transfer',g.attrs['fractional_density_at_match_per_sqrt_w_m2'],flush=True)


if __name__=='__main__':main()
