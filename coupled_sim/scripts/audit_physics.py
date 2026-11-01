"""Audit saved fields using the equations, not stored pass flags or fit outputs.

Phasors use exp(i kx x - i omega t). Time is c*t; all rates are per metre.
Density-weighted Yee operators are evaluated here without importing the solver.
"""
from pathlib import Path
import h5py
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0, k as kb
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]

def audit(path):
    with h5py.File(path) as h:
        z=h['z'][:]; d=h['dz_cells'][:]; dual=(d+np.roll(d,1))/2
        E,H,J=h['state_phasor'][:9].reshape(3,3,-1); q=h['state_phasor'][9]
        rho=np.sqrt(h['ne'][:]); wp=rho*e/np.sqrt(epsilon_0*m_e)/c
        physical_nu=h['physical_nu'][:]; sponge=h['sponge'][:]
        gyro=h['gyro'][:]; kx=float(h['kx_rad_m'][()])
        beta=np.sqrt(3*kb*h.attrs['electron_temperature_K']/m_e)/c
        if not h.attrs['thermal_pressure_enabled']: beta=0.
        omega=2*np.pi*h.attrs['frequency_hz']/c; dt=h.attrs['dt_s']*c
        wd=2*np.sin(omega*dt/2)/dt; ch=np.cos(omega*dt/2)
        def df(a):return (np.roll(a,-1)-a)/d
        def db(a):return (a-np.roll(a,1))/dual
        def cf(v):return np.array([-df(v[1]),df(v[0])-1j*kx*v[2],1j*kx*v[1]])
        def cb(v):return np.array([-db(v[1]),db(v[0])-1j*kx*v[2],1j*kx*v[1]])
        jzf=(J[2]*d+np.roll(J[2]*d,1))/(2*dual)
        jx=(J[0]+np.roll(J[0],-1))/2; jy=(J[1]+np.roll(J[1],-1))/2
        G=np.array([gyro[1]*jzf-gyro[2]*J[1],gyro[2]*J[0]-gyro[0]*jzf,gyro[0]*jy-gyro[1]*jx])
        pressure=np.array([1j*kx*beta*q,np.zeros_like(q),beta*rho[2]*df(q/rho[0])])
        continuity=beta*(1j*kx*J[0]+db(rho[2]*J[2])/rho[0])
        ampere=-1j*wd*E-cb(H)+wp*J+ch*sponge[:3]*E
        # Only this transverse current sheet is externally driven.
        source_index=np.argmin(abs(z-114850.))
        source=np.zeros_like(E)
        source[:2,source_index]=h['source_polarization'][:]/dual[source_index]
        def relative(residual,*terms):
            return float(np.linalg.norm(residual)/max(sum(np.linalg.norm(v) for v in terms),1e-100))
        results={
            'Ampere_relative_residual':relative(ampere-source,wd*E,cb(H),wp*J,source),
            'Faraday_relative_residual':relative(-1j*wd*H+cf(E)+ch*sponge[3:]*H,wd*H,cf(E)),
            'momentum_relative_residual':relative(-1j*wd*J+ch*(physical_nu+sponge[:3])*J-ch*G-wp*E-pressure,wd*J,wp*E,pressure),
            'continuity_relative_residual':relative(-1j*wd*q+ch*sponge[0]*q-continuity,wd*q,continuity),
        }
        active=(abs(z-h.attrs['center_m'])<20)
        if beta:
            results['Gauss_relative_residual']=relative((1j*kx*E[0]+db(E[2])+wp[0]/beta*q)[active],(wp[0]/beta*q)[active])
        for key,value in results.items():assert np.isfinite(value) and value<1e-7,(path,key,value)
        weights=np.array([dual,dual,d]); hw=np.array([d,d,dual])
        collision=.5*epsilon_0*c*ch*np.sum(physical_nu*abs(J)**2*weights)
        buffer=.5*epsilon_0*c*ch*(np.sum(sponge[:3]*(abs(E)**2+abs(J)**2)*weights)+np.sum(sponge[3:]*abs(H)**2*hw)+np.sum(sponge[0]*abs(q)**2*dual))
        source_power=.5*epsilon_0*c*np.sum(np.real(E[:,source_index].conj()*source[:,source_index])*weights[:,source_index])
        results['global_energy_relative_residual']=abs(source_power-collision-buffer)/source_power
        assert source_power>0 and collision>=0 and buffer>=0
        assert results['global_energy_relative_residual']<1e-7
        # Independent outgoing-wave accounting; reflection is outgoing energy.
        inc=h['incident_reference_state'][:]
        def poynting(v):
            ex=(v[0]+np.roll(v[0],-1))/2;ey=(v[1]+np.roll(v[1],-1))/2
            return .5*epsilon_0*c*np.real(ex*v[4].conj()-ey*v[3].conj())
        state=np.vstack([E,H,J,q]); faces=z+d/2
        lo=np.argmin(abs(faces-115100));hi=np.argmin(abs(faces-120800))
        assert not np.any(sponge[:,[lo,hi]])
        pin=poynting(inc)[lo];R=-poynting(state-inc)[lo]/pin;T=poynting(state)[hi]/pin
        results.update(reflected_fraction=float(R),transmitted_fraction=float(T),hf_nonreturn_fraction=float(1-R-T))
        assert pin>0 and R>=-1e-8 and T>=-1e-8 and -1e-6<=1-R-T<=1+1e-6
        if beta:
            dn=h['density_phasor_m3'][:]
            predicted=q*np.sqrt(epsilon_0*h['ne'][0]/(m_e*(beta*c)**2))
            np.testing.assert_allclose(dn,predicted,rtol=1e-12,atol=1e-12)
            kz=np.gradient(np.unwrap(np.angle(dn)),z)
            match_k=-(4*np.pi*430e6/c+omega)
            # Local electrostatic warm dispersion, with the actual magnetic field.
            def dispersion(height):
                k2=kx*kx+match_k*match_k
                along=(gyro[0]*kx+gyro[2]*match_k)**2/k2
                target=omega**2*(omega**2-gyro@gyro)/(omega**2-along)
                local_wp2=np.interp(height,z,wp[0]**2)
                return local_wp2+beta*beta*k2-target
            center=float(h.attrs['center_m'])
            analytic=brentq(dispersion,center-20,center)
            # Interpolate the crossing in the solved phase without using the
            # analytic height as the phase estimate.
            valid=(z>center-12)&(z<center-2)
            zz=z[valid];kk=kz[valid];crossings=np.where((kk[:-1]-match_k)*(kk[1:]-match_k)<0)[0]
            assert len(crossings)==1,('ambiguous Bragg crossing',crossings)
            i=crossings[0];phase=zz[i]+(match_k-kk[i])*(zz[i+1]-zz[i])/(kk[i+1]-kk[i])
            medians=[np.median(kz[(z>center+a)&(z<center+a+1)]) for a in range(-12,-2)]
            assert np.all(kk<0) and np.all(np.diff(medians)>0), 'downward wave must shorten downward'
            assert abs(phase-analytic)<.1,('Bragg height discrepancy',phase-analytic)
            results.update(Bragg_dispersion_height_m=analytic-center,Bragg_phase_height_m=phase-center,Bragg_height_difference_m=phase-analytic,Bragg_wavelength_m=2*np.pi/abs(match_k))
    return results

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('file',type=Path);args=parser.parse_args()
    for k,v in audit(args.file).items():print(k,v)


def native_recording(path):
    """Check every recorded E/q value, not the executable's final report."""
    with h5py.File(path) as h:
        assert 'time_s' in h, 'No native timesteps were saved; harmonic solve alone is not a timestep validation'
        t=h['time_s'][:];assert len(t)>=32 and np.all(np.diff(t)>0)
        phase=np.exp(-2j*np.pi*h.attrs['frequency_hz']*t)
        initial=h['state_phasor'][[0,1,2,9]]
        actual=h['E_q_time_real'][:]+1j*h['E_q_time_quadrature'][:]
        expected=phase[:,None,None]*initial[None,:,:]
        errors=np.max(abs(actual-expected),axis=(0,2))/np.maximum(np.max(abs(initial),axis=1),1e-100)
        assert np.all(np.isfinite(actual)) and np.max(errors)<5e-8,errors
        return {'frames':len(t),'periods':float(t[-1]*h.attrs['frequency_hz']),'maximum_relative_error':float(errors.max())}


def projected_power(h,lo=-100.,hi=30.):
    """Independent Gaussian-beam coherent integral, including both radar legs."""
    from scipy.constants import physical_constants
    z=h['z'][:];x=z-h.attrs['center_m'];dn=h['density_phasor_m3'][:]
    wavelength=c/430e6;kr=2*np.pi/wavelength;gain=10**(61.1/10)
    waist=np.sqrt(gain)*wavelength/(np.sqrt(8)*np.pi);rayleigh=np.pi*waist**2/wavelength
    width=waist*np.sqrt(1+(z/rayleigh)**2);curvature=z+rayleigh**2/z
    a=2/width**2;b=kr/curvature;kx=float(h['kx_rad_m'][()])
    gaussian=a/(a-1j*b)*np.exp(-kx**2/(4*(a-1j*b))-2j*np.arctan(z/rayleigh))
    window=np.sin(np.pi/2*np.clip(np.minimum(x-lo,hi-x)/10,0,1))**2
    density_wave=.5*dn*np.exp(1j*(2*kr+2*np.pi*h.attrs['frequency_hz']/c)*x)*gaussian*window
    # Explicit trapezoid weights, without importing the production radar code.
    column=np.sum((density_wave[:-1]+density_wave[1:])*np.diff(z)/2)
    re=physical_constants['classical electron radius'][0]
    return 1e6*re**2*wavelength**2*abs(column)**2


def convergence(out):
    values=[]
    for name in ['baseline','fine','finest','convergence','outer_mesh','larger_buffer']:
        with h5py.File(out/(name+'.h5')) as h:
            center=h.attrs['center_m'];mask=abs(h['z'][:]-center)<20
            density=np.max(abs(h['density_phasor_m3'][:][mask])/h['ne'][0][mask])
            values.append([density,projected_power(h)])
    values=np.array(values)
    # Successive halving of the fine conversion-region mesh, with fixed outer mesh.
    differences=abs(np.diff(values[:4],axis=0))
    ratios=differences[:-1]/differences[1:]
    assert np.all(ratios>3),('mesh error is not approximately second order',ratios)
    mesh=abs(values[2]/values[3]-1)
    assert mesh[0]<.005 and mesh[1]<.002,('fine-grid change',mesh)
    outer=abs(values[4]/values[1]-1);buffer=abs(values[5]/values[1]-1)
    assert np.max(outer)<.001 and np.max(buffer)<.001,(outer,buffer)
    with h5py.File(out/'convergence.h5') as h:
        powers=np.array([projected_power(h,*bounds) for bounds in [(-80,20),(-100,30),(-110,40)]])
    window=np.ptp(powers)/powers[1];assert window<.002
    return {'density_mesh_relative_change':float(mesh[0]),'radar_power_mesh_relative_change':float(mesh[1]),'minimum_mesh_refinement_ratio':float(ratios.min()),'outer_mesh_relative_change':float(outer.max()),'buffer_relative_change':float(buffer.max()),'radar_window_relative_spread':float(window)}
