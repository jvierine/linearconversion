"""Weak-tail magnetized plasma-line theory and stability diagnostics.

Implements the Lorentzian area in Longley et al. (2021), doi:2020JA028734,
with thermal first-order collisional terms and isotropic energy distributions.
The reference source is archived at doi:10.5281/zenodo.4041626 (CC-BY-4.0).
Our exp(-i omega t) convention is conjugate to the reference's convention.
Thermal response determines the real root: Re chi_tail is neglected, as in
that paper. This is a weak-tail plasma-line approximation, NOT a full
non-Maxwellian ion-line model or a nonlinear saturation calculation.
"""
import numpy as np
from scipy.constants import e,m_e,epsilon_0,k as kb,c
from scipy.special import ive,jv,dawsn
from scipy.optimize import root_scalar
from scipy.interpolate import PchipInterpolator


def energy_distribution(E_eV, peaks=(20.,), widths=(0.5,), strengths=(0.02,),
                        pedestal_power=2., pedestal=1.):
    """Energy probability density p(E) in eV^-1; integral p(E)dE=1.

    Same energy distribution definition as the archived e_dist.m, with its
    factor 4*pi absorbed into p(E). A peak is in isotropic ENERGY, not a
    drifting 1D electron beam. The low-energy cutoff is explicitly 1 eV.
    """
    E=np.asarray(E_eV,float)
    if E.ndim!=1 or np.any(E<=0) or np.any(np.diff(E)<=0): raise ValueError('positive increasing energy grid required')
    if not len(peaks)==len(widths)==len(strengths): raise ValueError('peak parameter lengths differ')
    if any(w<=0 for w in widths) or pedestal<0 or any(a<0 for a in strengths): raise ValueError('invalid distribution')
    p=pedestal*E**(-pedestal_power)
    for center,width,amp in zip(peaks,widths,strengths):
        p+=amp*np.sqrt(E)/width**1.5*np.exp(-.5*((E-center)/width)**2)
    p[E<1]=0
    return p/np.trapezoid(p,E)


def collision_rate(ne,Te):
    ld=np.sqrt(epsilon_0*kb*Te/(ne*e*e)); sig=np.sqrt(kb*Te/m_e)
    return np.sqrt(np.pi/9)*ne*e**4/(4*np.sqrt(2)*np.pi*epsilon_0**2*m_e**2*sig**3)*np.log(4*np.pi*epsilon_0*kb*Te*ld/e**2)


def tail_terms(w,k,B,theta,E_eV,p_eV,eta,nmax=30,method='parts'):
    """Return M_h=free PSD/2 and Im chi_h for an isotropic energy PDF.

    'parts': integration by parts, equivalent to published reference.
    'derivative': independent derivative quadrature for validation.
    All lower bounds are inserted exactly rather than snapped to a grid.
    Im chi_h is divided by omega_p(core)^2; caller multiplies by it.
    """
    if not 0<=theta<90 or B<=0: raise ValueError('requires 0 <= theta < 90 and B > 0')
    if eta<0: raise ValueError('tail fraction must be nonnegative')
    kp=k*np.cos(np.deg2rad(theta)); kt=k*np.sin(np.deg2rad(theta))
    om=e*B/m_e
    EJ=np.asarray(E_eV)*e
    # F(E)=p(E)/(4*pi) in the reference, normalized by energy rather than d^3v.
    F=np.asarray(p_eV)/(4*np.pi*e)
    interp=PchipInterpolator(EJ,F,extrapolate=False)
    mh=0.; im=0.; by_order=[]
    for n in range(-nmax,nmax+1) if theta else [0]:
        En=m_e/2*((w-n*om)/kp)**2
        lo=max(En,EJ[0]); hi=EJ[-1]
        if lo>=hi: continue
        xx=np.r_[lo,EJ[(EJ>lo)&(EJ<hi)],hi]
        ff=interp(xx); a=kt/om*np.sqrt(2*(xx-En)/m_e)
        J=jv(n,a)
        mh_n=2*np.pi**2/kp*eta*np.trapezoid(np.sqrt(m_e/(2*xx))*J*J*ff,xx)
        # F3(v)=m/v*F(E). Integration of dF3/dE (including cutoff jumps).
        if method=='derivative':
            D=m_e**1.5/np.sqrt(2)*(interp.derivative()(xx)/np.sqrt(xx)-.5*ff/xx**1.5)
            integral=np.trapezoid(J*J*D,xx)
            if En<EJ[0]: integral+=J[0]**2*m_e**1.5/np.sqrt(2*lo)*ff[0]
            integral-=J[-1]**2*m_e**1.5/np.sqrt(2*hi)*ff[-1]
            im_n=-2*np.pi**2*w/(k*k*kp)*eta*integral
        elif method=='parts':
            # integrate derivative by parts; d[J_n(a)^2]/dE regular at a=0
            djda=jv(n-1,a)-jv(n+1,a)
            ratio=np.divide(J*djda,a,out=np.zeros_like(a),where=a!=0)
            if a[0]==0:
                ratio[0]= -1. if n==0 else (.5 if abs(n)==1 else 0.)
            dJ2=ratio*kt**2/(m_e*om**2)
            integral=-np.trapezoid(dJ2*m_e**1.5/np.sqrt(2*xx)*ff,xx)
            if En>=EJ[0]: integral-=J[0]**2*m_e**1.5/np.sqrt(2*lo)*ff[0]
            im_n=-2*np.pi**2*w/(k*k*kp)*eta*integral
        else: raise ValueError('method must be parts or derivative')
        mh+=mh_n; im+=im_n
        by_order.append((n,mh_n,im_n))
    return mh,im,np.asarray(by_order)


def plasma_line(ne,Te,B,theta,E,p,eta=1e-4,radar_hz=430e6,
                nu_en=0.,nmax=30,bragg_correction=True):
    """Positive plasma-line resonance, area, linewidth, and instability flag.

    area = integral S(f) df, normalized per core electron (n_core).
    A nonpositive damping rate returns NaN area; never display an unstable
    stationary Lorentzian as a predicted finite power.
    theta=90 is deliberately excluded; use kinetic BGK or mode weights there.
    """
    if ne<=0 or Te<=0 or B<=0 or not 0<=theta<90: raise ValueError('invalid plasma-line parameters')
    wp2=ne*e*e/(epsilon_0*m_e); om=e*B/m_e
    vt=np.sqrt(2*kb*Te/m_e); ld=np.sqrt(epsilon_0*kb*Te/(ne*e*e))
    th=np.deg2rad(theta); ct=np.cos(th); st=np.sin(th)
    k0=4*np.pi*radar_hz/c; nu=collision_rate(ne,Te)+nu_en
    ns=np.arange(-nmax,nmax+1) if theta else np.array([0])
    def eps_real(w):
        k=k0+(w/c if bragg_correction else 0.)
        kp=k*ct; b=(k*st*vt/(np.sqrt(2)*om))**2
        yn=(w-ns*om)/(kp*vt)
        return 1+(k*ld)**-2*(1-w/(kp*vt)*np.sum(ive(ns,b)*2*dawsn(yn)))
    guess=np.sqrt(wp2+1.5*k0*k0*vt*vt+om*om*st*st)
    root=root_scalar(eps_real,x0=guess,x1=guess*1.00001,method='secant',xtol=1e-5,maxiter=60)
    if not root.converged or root.root<=0 or abs(eps_real(root.root))>1e-7:
        raise RuntimeError('Langmuir branch root did not converge')
    w=root.root; k=k0+(w/c if bragg_correction else 0.); kp=k*ct; kt=k*st
    aa=(k*ld)**-2; b=(kt*vt/(np.sqrt(2)*om))**2
    weights=ive(ns,b)
    if abs(weights.sum()-1)>1e-10: raise ValueError('thermal harmonics truncated')
    y=(w-ns*om)/(kp*vt); D=dawsn(y)
    Mm=np.sum(weights*np.sqrt(np.pi)*np.exp(-y*y))/(kp*vt)
    im_m=aa*w*Mm
    # First order in nu, matching the archived collisional theory.
    xa=np.sum(weights*(2*y*D-1))
    xb=np.sum(weights*(2*D+1j*np.sqrt(np.pi)*np.exp(-y*y)))
    Mc=nu/(kp*vt)**2*(2*xa-abs(xb)**2)
    im_c=-aa*nu/(kp*vt)*np.sum(weights*(4*(1-y*w/(kp*vt))*D-(kp*vt/w-2*w/(kp*vt))))
    dreal=np.sum(-2*aa/(kp*vt)*weights*((1-2*y*w/(kp*vt))*D+w/(kp*vt)))
    dreal+=2*np.sqrt(np.pi)*nu*aa/(kp*vt)**2*np.sum(weights*(w/(kp*vt)+3*y-2*y*y*w/(kp*vt))*np.exp(-y*y))
    Mh,im_h,orders=tail_terms(w,k,B,theta,E,p,eta,nmax)
    im_h*=wp2
    eps_im=im_m+im_c+im_h
    damping=eps_im/dreal  # positive -> decay in exp(-i omega t)
    stable=damping>0 and Mm+Mc+Mh>=0 and dreal>0
    area=(Mm+Mc+Mh)/abs(eps_im*dreal) if stable else np.nan
    thermal_area=(Mm+Mc)/abs((im_m+im_c)*dreal)
    return dict(f_hz=w/(2*np.pi),omega=w,k=k,damping_s=damping,
                hwhm_hz=damping/(2*np.pi),area=area,thermal_area=thermal_area,
                enhancement=area/thermal_area,stable=bool(stable),
                M_thermal=Mm,M_collision=Mc,M_tail=Mh,
                im_thermal=im_m,im_collision=im_c,im_tail=im_h,
                derivative=dreal,nu_s=nu,orders=orders)


def lorentzian(f_hz,line):
    if not line['stable']: raise ValueError('growing mode: stationary spectrum is undefined')
    g=line['hwhm_hz']
    return line['area']/np.pi*g/((np.asarray(f_hz)-line['f_hz'])**2+g*g)
