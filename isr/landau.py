"""Local Maxwellian kinetic pressure closure for the downward plasma branch.

exp(i kx x + i kz z - i omega t), kz=-(k_real+i alpha), alpha>=0.
The collisionless longitudinal Vlasov dielectric is analytically continued
from the Landau prescription. Its outgoing spatial root sets a complex scalar
pressure response for the full-wave fluid-Maxwell solver. This matches that
local electrostatic branch, not the full kinetic electromagnetic tensor.

Sources: the harmonic susceptibility also implemented in isr/kinetic.py;
Gondarenko et al. 2009 doi:10.1029/2009GL038379 motivates putting collisionless
loss in the pressure response, distinct from ordinary electron drag.
Strongly damped roots are a local closure continuation, not WKB wave packets.
"""
import numpy as np
from scipy.constants import e,m_e,k as kb
from scipy.special import iv,wofz
from scipy.optimize import root
from scipy.interpolate import PchipInterpolator


def dielectric(kdown, density_ratio, par):
    """Longitudinal collisionless dielectric, fixed real drive frequency.

    Analytic k continuation on the branch with positive effective parallel
    wavenumber. Restricted to the downward branch and B within 80 deg of z.
    Ions are immobile; electrons are isotropic Maxwellian.
    """
    v=np.sqrt(kb*par.temperature_k/m_e)
    kv=np.array([par.q*par.k0,0.,-complex(kdown)])
    k2=kv@kv
    if par.magnetic_t==0:
        kap=np.sqrt(k2)
        zz=par.omega/(np.sqrt(2)*kap*v)
        response=1+1j*np.sqrt(np.pi)*zz*wofz(zz)
    else:
        om=e*par.magnetic_t/m_e
        bhat=par.gyro/np.linalg.norm(par.gyro)
        kp=-(kv@bhat)
        if kp.real<0:kp=-kp
        if abs(kp.real)<1e-12:raise ValueError('Exact perpendicular kinetic continuation is not supported')
        b=(k2-kp*kp)*v*v/om**2
        nn=max(20,int(np.ceil(10*np.sqrt(abs(b)+1)+10)))
        orders=np.arange(-nn,nn+1)
        weights=np.exp(-b)*iv(orders,b)
        zz=(par.omega-orders*om)/(np.sqrt(2)*kp*v)
        response=1+1j*np.sqrt(np.pi)*par.omega/(np.sqrt(2)*kp*v)*np.sum(weights*wofz(zz))
    return 1+density_ratio*par.omega**2/(k2*v*v)*response


def pressure_for_root(kdown,p,par):
    """Scalar pressure factor that reproduces this local ES spatial root."""
    kv=np.array([par.q*par.k0,0.,-kdown]);k2=kv@kv
    if par.magnetic_t==0:A=1.
    else:
        bhat=par.gyro/np.linalg.norm(par.gyro)
        ct2=(kv@bhat)**2/k2;Y2=par.gyro@par.gyro
        A=ct2+(1-ct2)/(1-Y2)
    return (1-p*A)/(par.beta*k2/par.k0**2*A)


def spatial_root(p,par,guess):
    def fun(x):
        eps=dielectric(complex(*x),p,par)
        return [eps.real,eps.imag]
    sol=root(fun,[guess.real,guess.imag],tol=1e-9)
    if not np.isfinite(sol.fun).all() or np.linalg.norm(sol.fun)>2e-7:
        raise ArithmeticError(f'Kinetic spatial root failed at P={p}: residual {sol.fun}')
    k=complex(*sol.x)
    if k.real<=0 or k.imag<-1e-8:raise ArithmeticError('Wrong spatial branch')
    return complex(k.real,max(0.,k.imag))


def branch_table(par,minimum_ratio,points=900):
    # Below k=5 m^-1 Landau loss is negligible for the current event defaults.
    # Extend the first real pressure factor smoothly through conversion; do not
    # infer kinetic electromagnetic modes from a longitudinal dielectric.
    kstart=5.
    pstart=-1/(dielectric(kstart,1.,par).real-1)
    pmin=min(float(minimum_ratio),pstart*.99)
    if pmin<=0:raise ValueError('Density must be positive')
    pp=pstart-np.r_[0.,np.geomspace(1e-8,pstart-pmin,points-1)]
    roots=[];xi=[];k=complex(kstart,1e-12)
    for p in pp:
        k=spatial_root(p,par,k)
        xx=pressure_for_root(k,p,par)
        if xx.imag>1e-7 or xx.real<=0:raise ArithmeticError('Non-passive local pressure closure')
        roots.append(k);xi.append(complex(xx.real,min(0.,xx.imag)))
    return dict(density_ratio=pp[::-1],kdown_m1=np.array(roots)[::-1],
                pressure_response=np.array(xi)[::-1],maximum_dielectric_residual=max(abs(dielectric(k,p,par)) for k,p in zip(roots,pp)))


def closure_profile(z,p,par,taper_m=400.,points=900):
    table=branch_table(par,float(np.min(p))*.999,points)
    pp=table['density_ratio'];xx=table['pressure_response']
    pc=np.clip(p,pp[0],pp[-1])
    xi=PchipInterpolator(pp,xx.real)(pc)+1j*PchipInterpolator(pp,xx.imag)(pc)
    # Uniform reservoirs use the original collisionless fluid eigenmodes.
    u=np.clip(np.minimum(z-z[0],z[-1]-z)/taper_m,0,1)
    taper=10*u**3-15*u**4+6*u**5
    xi=1+taper*(xi-1);xi[[0,-1]]=1.
    return xi,table


def bragg_match(par,k_radar):
    kr=np.sqrt(k_radar**2-(par.q*par.k0)**2)
    p0=-1/(dielectric(kr,1.,par).real-1)
    def fun(x):
        eps=dielectric(kr+1j*x[1],x[0],par)
        return [eps.real,eps.imag]
    sol=root(fun,[p0,1e-8],tol=1e-10)
    if np.linalg.norm(sol.fun)>2e-8 or sol.x[1]<-1e-8:raise ArithmeticError('Kinetic Bragg match failed')
    return float(sol.x[0]),float(max(0.,sol.x[1]))
