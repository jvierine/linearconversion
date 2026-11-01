"""Scalar-pressure audit; does not replace the production Rust solution.

Bohm & Gross (1949), doi:10.1103/PhysRev.75.1851.
Willes & Cairns (2001), Eqs. 5--6, doi:10.1071/AS01051.
Gondarenko et al. (2009), paragraph 6, doi:10.1029/2009GL038379.
"""
from pathlib import Path
import sys
import numpy as np
import h5py
from scipy.constants import c,e,m_e,epsilon_0,k as kb
from scipy.optimize import root
from scipy.special import wofz
from coupled_arecibo import setup,ROOT,F,CS,PROFILE,hf_port_budget
sys.path.insert(0,str(ROOT.parent))
from isr.coherent import sideband_amplitude

def kinetic_frequency(kld):
    # Unmagnetized Maxwellian longitudinal dielectric function, omega/omega_pe.
    def residual(v):
        w=complex(*v);z=w/(np.sqrt(2)*kld)
        eps=1+(1+z*1j*np.sqrt(np.pi)*wofz(z))/kld**2
        return [eps.real,eps.imag]
    r=root(residual,[np.sqrt(1+3*kld**2),-1e-8],tol=1e-10)
    assert np.linalg.norm(residual(r.x))<1e-8
    return complex(*r.x)

def main():
    records=[];arrays={}
    with h5py.File(ROOT/'data/illumination.h5') as h:pol=h['source_polarization'][:]
    for dz in [.01,.005]:
        ref=setup(dz=dz,cap=.5,reference=True)
        inc=np.einsum('qnp,p->qn',ref['initial'],pol)
        for alternative in [False,True]:
            s=setup(dz=dz,cap=.5,scenario={'pressure_gradient_audit':alternative})
            v=np.einsum('qnp,p->qn',s['initial'],pol);z=s['z'];center=s['center']
            face=z+s['dz_cells']/2;lo=np.argmin(abs(face-115100));hi=np.argmin(abs(face-120800))
            budget=hf_port_budget(v,inc,lo,hi);pin=budget['incident_EM_flux_W_m2']
            dn=v[9]*np.sqrt(epsilon_0*s['ne'][0]/(m_e*CS**2))/np.sqrt(pin)
            d=z-center;win=np.sin(.5*np.pi*np.clip(np.minimum(d+100,30-d)/10,0,1))**2
            power=abs(sideband_amplitude(z,dn,F,s['kx'],win,reference_range_m=center))**2
            records.append([dz,float(alternative),power,budget['reflected_EM_fraction'],s['harmonic_residual']])
            arrays[(dz,alternative)]=(z,dn)
            print('case',records[-1],flush=True)
    a=np.array(records);power_changes=a[[1,3],2]/a[[0,2],2]-1
    assert np.max(abs(power_changes))<.01, power_changes
    assert abs(power_changes[1]-power_changes[0])<1e-4
    center=s['center'];zbr=center-4.09;ne=float(PROFILE(zbr));ld=np.sqrt(epsilon_0*kb*300/(ne*e**2))
    k=(4*np.pi*430e6+2*np.pi*F)/c;kld=k*ld
    exact=kinetic_frequency(kld);bg=np.sqrt(1+3*kld**2)
    error=abs(bg/exact.real-1)
    assert error<.001
    path=ROOT.parent/'reference/coupled/scalar_closure_check.h5'
    with h5py.File(path,'w') as h:
        h['cases']=a
        h['cases'].attrs['columns']='dz_m,physical_equilibrium_gradient,echo_power_per_unit_flux,reflected_EM_fraction,harmonic_residual'
        h['relative_echo_power_change']=power_changes
        h.attrs.update(k_lambda_D=kld,lambda_D_m=ld,kinetic_real_omega_over_wp=exact.real,
            bohm_gross_omega_over_wp=bg,relative_dispersion_error=error,
            density_scale_m=ne/float(PROFILE.derivative()(zbr)),
            gradient_force_relative_scale=2/(3*k*(ne/float(PROFILE.derivative()(zbr)))),
            interpretation='Gradient sensitivity of scalar harmonic models; unmagnetized kinetic dispersion check only. Not validation of magnetized tensor response, Landau damping, or absolute heating.',
            production_closure='-3 kB Te/me grad(delta_ne/ne); constant Te; effective barotropic background support')
        for alternative in [False,True]:
            z,dn=arrays[(.005,alternative)];mask=abs(z-center)<20
            g=h.create_group('physical_gradient' if alternative else 'production');g['z_m']=z[mask];g['density_phasor_m3']=dn[mask]
        h['provenance/source']=Path(__file__).read_text()
    print('relative echo changes',power_changes,'k lambdaD',kld,'BG frequency error',error,flush=True)

if __name__=='__main__':main()
