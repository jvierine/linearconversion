"""Independent harmonic reference for the Rust fluid-Poisson discretization.

Same prescribed E source, staggered density/face fluxes, zero exterior induced
field and velocity sponge. Eliminates ux,uy,uz and Poisson to a tridiagonal
equation for p=n0*uz. Neutrality keeps both exterior induced fields zero.
This isolates propagation/damping; it does NOT solve incident HF enhancement.
All scientific products are HDF5. JSON is GUI transport only.
"""
from pathlib import Path
import json
import h5py
import numpy as np
from scipy.constants import e, m_e, epsilon_0, k as kb, c
from scipy.interpolate import CubicHermiteSpline
from scipy.linalg import solve_banded
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parent

def solve(gradient=1.,nu=1000.,cells=16384,span=160.,frequency=5.33e6,te=300.,amplitude=.0005):
    with h5py.File(ROOT/'data/smooth_observed_layer_1525453430.h5') as h:
        density=CubicHermiteSpline(h['model_height_m'][:],h['model_ne_m3'][:],h['model_dne_dz_m4'][:])
    omega=2*np.pi*frequency;gyro=e*35.065e-6/m_e;theta=np.deg2rad(46.816)
    bx=gyro*np.sin(theta);bz=gyro*np.cos(theta);cs2=3*kb*te/m_e
    D=1/(np.cos(theta)**2/omega**2+np.sin(theta)**2/(omega**2-gyro**2))
    critical=D*epsilon_0*m_e/e**2;kr=4*np.pi*430e6/c
    center=brentq(lambda z:density(z)-critical,117000,118100)
    original_match=brentq(lambda z:density(z)-(D-cs2*kr**2)*epsilon_0*m_e/e**2,117000,center)
    match=(original_match-center)/gradient if gradient else np.nan
    dz=span/cells;zf=np.linspace(-span/2,span/2,cells+1);z=(zf[:-1]+zf[1:])/2
    ne=density(center+gradient*z);nf=np.r_[ne[0],(ne[:-1]+ne[1:])/2,ne[-1]]
    sponge=2e6*np.maximum(0,1-np.minimum(zf-zf[0],zf[-1]-zf)/(.15*span))**3
    a=nu+sponge[1:-1]-1j*omega
    rzz=(a*a+bz*bz)/(a*(a*a+gyro*gyro))
    # Vertical, Gaussian 1 m sigma source. Same peak and width for every case.
    drive=amplitude*np.exp(-.5*zf*zf)
    lap=cs2/(ne*dz*dz)
    diagonal=-(lap[:-1]+lap[1:])+1j*omega/(nf[1:-1]*rzz)-e*e/(m_e*epsilon_0)
    matrix=np.zeros((3,cells-1),complex);matrix[1]=diagonal
    matrix[0,1:]=lap[1:-1];matrix[2,:-1]=lap[1:-1]
    force=-1j*omega*e/m_e
    yy=solve_banded((1,1),matrix,force*drive[1:-1])
    p=np.r_[0,yy,0]
    dn=np.diff(p)/(1j*omega*dz);eta=dn/ne
    eself=1j*e/(omega*epsilon_0)*p
    uz=p/nf;uy=np.zeros_like(p);ux=np.zeros_like(p)
    uy[1:-1]=-a*bx*uz[1:-1]/(a*a+bz*bz)
    ux[1:-1]=bx*bz*uz[1:-1]/(a*a+bz*bz)
    # Source and losses must balance for the stationary closed-domain solution.
    u2=abs(ux)**2+abs(uy)**2+abs(uz)**2
    collision=.5*m_e*nu*np.sum(nf*u2)*dz
    buffer=.5*m_e*np.sum(nf*sponge*u2)*dz
    source=-.5*e*np.sum(nf*np.real(np.conj(uz)*drive))*dz
    balance=(source-collision-buffer)/max(abs(source),1e-100)
    energy=.25*dz*(m_e*np.sum(nf*u2)+epsilon_0*np.trapezoid(abs(eself)**2)+m_e*cs2*np.sum(abs(dn)**2/ne))
    amplitude_bragg=float(np.interp(match,z,abs(eta))) if np.isfinite(match) else np.nan
    return dict(gradient=gradient,nu_s1=nu,z_m=z,ne_m3=ne,density_phasor=eta,
                match_m=match,amplitude_at_match=amplitude_bragg,
                amplitude_at_original_match=float(np.interp(original_match-center,z,abs(eta))),
                source_power_W_m2=source,collision_power_W_m2=collision,buffer_power_W_m2=buffer,
                balance_relative=balance,energy_J_m2=energy,center_m=center,source_peak_V_m=amplitude,
                n_cells=cells,domain_span_m=span,frequency_hz=frequency,temperature_K=te)

def main():
    path=ROOT/'data/fluid_gradient_validation.h5';rows=[]
    with h5py.File(path,'w') as h:
        h.attrs['provenance']='validate_fluid_gradient.py; independent harmonic reference for fluid_browser/src/lib.rs'
        h.attrs['scope']='Prescribed vertical Gaussian source; no incident electromagnetic-wave calculation; no Landau damping; counterfactual profile stretching about fixed resonance density'
        for nu in [0.,1000.,3000.,10000.]:
            for gradient in [1.,.5,.25,0.]:
                result=solve(gradient,nu);g=h.create_group(f'nu_{nu:g}/gradient_{gradient:g}')
                for key,value in result.items():
                    if isinstance(value,np.ndarray):g[key]=value
                    else:g.attrs[key]=value
                assert abs(result['balance_relative'])<1e-6,result['balance_relative']
                row={key:result[key] for key in ['gradient','nu_s1','match_m','amplitude_at_match','amplitude_at_original_match','balance_relative']}
                rows.append({k:(float(v) if np.isfinite(v) else None) for k,v in row.items()})
                print(row)
        # Resolution and boundary-distance checks at the actual Bragg location.
        reference=solve(1.,3000.)
        fine=solve(1.,3000.,cells=32768)
        large=solve(1.,3000.,cells=32768,span=320.)
        errors=np.array([abs(fine['amplitude_at_match']/reference['amplitude_at_match']-1),abs(large['amplitude_at_match']/reference['amplitude_at_match']-1)])
        h['relative_sensitivity_grid_domain']=errors
        print('Bragg amplitude sensitivity, grid/domain',errors)
        assert np.all(errors<.03)
        checks=[]
        for nu in (0.,3000.):
            for gradient in (1.,.5,.25):
                ref=solve(gradient,nu)
                fine=solve(gradient,nu,cells=32768)
                large=solve(gradient,nu,cells=32768,span=320.)
                errors=[abs(fine['amplitude_at_match']/ref['amplitude_at_match']-1),
                        abs(large['amplitude_at_match']/ref['amplitude_at_match']-1)]
                assert max(errors)<.006
                checks.append([nu,gradient,*errors])
        h['comparison_convergence_nu_gradient_grid_domain']=np.array(checks)
    (ROOT/'web/fluid/gradient-validation.json').write_text(json.dumps(rows))

if __name__=='__main__':main()
