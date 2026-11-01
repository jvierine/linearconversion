"""Best-case in-plane geometry search; not event geometry reconstruction.

Angles refer to the local density-gradient normal, which can be inclined to
geographic vertical. q is conserved tangential wave number / vacuum k0.
We compare propagating incoming EM eigenpolarizations at the low reservoir.
"""
from dataclasses import replace
import h5py
import numpy as np
from scipy.optimize import minimize
from isr.conversion import Parameters, scatter, layer, matching_ratio


def evaluate(q, tilt, branch, dz=.005, scale=10., nu=0.):
    z=np.linspace(-6*scale,6*scale,round(12*scale/dz)+1)
    par=replace(Parameters(),q=float(q),magnetic_tilt_deg=float(tilt))
    return scatter(z,*layer(z,scale,collision_s=nu),par,branch=int(branch))


def main():
    q=np.linspace(-.5,.5,41)
    tilt=np.linspace(0.,90.,10)
    conversion=np.full((len(tilt),len(q),2),np.nan)
    balance=conversion.copy()
    for i,a in enumerate(tilt):
        for j,qi in enumerate(q):
            for branch in [0,1]:
                try:
                    r=evaluate(qi,a,branch)
                except ValueError as exc:
                    if 'unavailable' not in str(exc): raise
                    continue
                conversion[i,j,branch]=r['conversion']
                balance[i,j,branch]=r['energy_error']
        print('tilt',a,'max',np.nanmax(conversion[i]),flush=True)
    best=np.unravel_index(np.nanargmax(conversion),conversion.shape)
    a0,q0,branch=tilt[best[0]],q[best[1]],best[2]
    def objective(v):
        try: return -evaluate(v[0],v[1],branch,dz=.0025)['conversion']
        except ValueError as exc:
            if 'unavailable' not in str(exc): raise
            return 0.
    opt=minimize(objective,[q0,a0],method='Nelder-Mead',bounds=[(-.5,.5),(0.,90.)],
                 options={'xatol':1e-4,'fatol':1e-6,'maxiter':80})
    qb,ab=opt.x
    print('optimum',qb,ab,int(branch),-opt.fun,opt.success,flush=True)
    with h5py.File('data/linear_conversion_geometry.h5','w') as f:
        f.attrs.update(scope='Best-case 1-D local tilted-surface geometry; no measured pointing',
            provenance='optimize_conversion.py; isr/conversion.py',
            tilt_definition='Angle B to local density-gradient normal',
            q_definition='Conserved tangential k / vacuum HF k',
            coarse_dz_m=.005, optimization_dz_m=.0025,
            scale_m=10., temperature_k=1000., magnetic_t=40e-6,
            frequency_hz=5.33e6,low_density_ratio=.65,high_density_ratio=1.3,
            best_q=qb,best_magnetic_tilt_deg=ab,best_branch=int(branch),
            optimizer_success=bool(opt.success))
        f['q']=q;f['vacuum_incidence_deg']=np.rad2deg(np.arcsin(q));f['magnetic_tilt_deg']=tilt
        f['conversion']=conversion;f['energy_error']=balance
        # A unitary mixture of incident polarizations might improve upon a
        # single eigenbranch. This search does not yet optimize that mixture.
        f.attrs['polarization_limit']='Single incoming EM eigenmode; not arbitrary coherent mixture'
        for name,dz,L,nu in [('best_fine',.00125,10.,0.),('best_finer',.000625,10.,0.),
                             ('best_drag_1000',.00125,10.,1000.),
                             ('best_drag_10000',.00125,10.,10000.),
                             ('best_scale_20',.00125,20.,0.),
                             ('best_scale_50',.0025,50.,0.),
                             ('best_scale_100',.005,100.,0.)]:
            r=evaluate(qb,ab,branch,dz=dz,scale=L,nu=nu)
            g=f.create_group(name);g.attrs.update(dz_m=dz,scale_m=L,collision_peak_s=nu)
            for key,value in r.items():
                if np.isscalar(value):g.attrs[key]=value
                elif name=='best_finer':g[key]=value
            print(name,{k:v for k,v in r.items() if np.isscalar(v)},flush=True)


if __name__=='__main__':main()
