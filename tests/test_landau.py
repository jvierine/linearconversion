"""Kinetic consistency, passivity, and full-wave energy tests."""
import numpy as np
import pytest
from scipy.constants import epsilon_0,m_e,e,c
from isr.kinetic import Species,maxwellian_response
from isr.conversion import Parameters,matrix,components,flux,scatter,layer
from isr.landau import dielectric,spatial_root,closure_profile,bragg_match

@pytest.mark.parametrize('B',[0.,35.025e-6])
def test_real_k_matches_independent_existing_susceptibility(B):
    par=Parameters(frequency_hz=5.33e6,temperature_k=300,magnetic_t=B,q=-.342)
    kv=np.array([par.q*par.k0,0,-80.]);kk=np.linalg.norm(kv)
    theta=0. if B==0 else np.rad2deg(np.arccos(abs(kv@par.gyro)/(kk*np.linalg.norm(par.gyro))))
    s=Species('e',par.omega**2*epsilon_0*m_e/e**2,300,m_e,-e)
    chi,_=maxwellian_response(par.frequency_hz,kk,s,B,theta)
    assert dielectric(80.,.8,par)==pytest.approx(1+.8*chi,abs=2e-12)

def test_complex_spatial_root_matches_weak_damping_expansion():
    par=Parameters(frequency_hz=5.33e6,temperature_k=300,magnetic_t=0,q=0)
    kr=100.;p=-1/(dielectric(kr,1,par).real-1)
    dk=.001;der=(dielectric(kr+dk,p,par).real-dielectric(kr-dk,p,par).real)/(2*dk)
    weak=-dielectric(kr,p,par).imag/der
    root=spatial_root(p,par,complex(kr,weak))
    assert root.imag==pytest.approx(weak,rel=.02)
    assert root.imag>0

def test_pressure_loss_has_correct_sign_and_energy_factor():
    par=Parameters();rng=np.random.default_rng(42);y=rng.normal(size=6)+1j*rng.normal(size=6)
    p=.8;nu=.002;xi=1.3-.2j;dy=matrix(p,0,nu,par,xi)@y
    dt=1e-6;der=(flux(y+dt*dy,p,par)-flux(y-dt*dy,p,par))/(2*dt)
    u=components(y,p,nu,par,xi)[2]
    loss=p*nu*np.sum(abs(u)**2)+p*np.imag(1/xi)*abs(y[5])**2
    assert der==pytest.approx(-loss,rel=1e-8)

def test_no_spurious_transverse_landau_absorption():
    par=Parameters(magnetic_t=0,q=0,temperature_k=300)
    z=np.linspace(-60,60,12001);p,g,nu=layer(z,10)
    xi=1-.2j*np.sin(np.pi*(z-z[0])/(z[-1]-z[0]))**2
    r=scatter(z,p,g,nu,par,pressure_response=xi)
    assert r['absorption_landau']<1e-20
    assert abs(r['energy_error'])<1e-8

def test_kinetic_closure_refines_and_conserves_energy():
    par=Parameters(frequency_hz=5.33e6,temperature_k=300,magnetic_t=35.025e-6,magnetic_tilt_deg=46.9,q=-.342)
    errors=[]
    for dz in [.005,.0025]:
        z=np.linspace(-60,60,round(120/dz)+1);p,g,nu=layer(z,10,collision_s=1000)
        xi,_=closure_profile(z,p,par,taper_m=10,points=300)
        r=scatter(z,p,g,nu,par,pressure_response=xi)
        assert r['absorption_landau']>0
        assert r['absorption_collision']>0
        errors.append(abs(r['energy_error']))
    assert errors[-1]<.002
    assert errors[-1]<.4*errors[0]

def test_fourth_order_cell_and_fixed_polarization():
    from isr.conversion import optimal_polarization
    par=Parameters();z=np.linspace(-60,60,12001);p,g,nu=layer(z,10,collision_s=1000)
    optimum=optimal_polarization(z,p,g,nu,par,target_z_m=0)
    direct=scatter(z,p,g,nu,par,incident_weights=optimum['incoming_weights'])
    np.testing.assert_allclose(direct['h'],optimum['h'],rtol=1e-9,atol=1e-7)
    fine_z=np.linspace(-60,60,48001);pf,gf,nf=layer(fine_z,10,collision_s=1000)
    fine=scatter(fine_z,pf,gf,nf,par,cell_order=4)
    coarse_z=np.linspace(-60,60,6001);pc,gc,nc=layer(coarse_z,10,collision_s=1000)
    coarse=scatter(coarse_z,pc,gc,nc,par,cell_order=4)
    assert abs(coarse['reflection']-fine['reflection'])<1e-4
    assert abs(coarse['energy_error'])<.002
