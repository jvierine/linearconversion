"""Independent checks on the saved two-sided conversion and native recordings."""
from pathlib import Path
import json
import h5py,numpy as np
from scipy.constants import c,epsilon_0,e,m_e
from scipy.optimize import brentq
import coupled_arecibo as a
from lmode_animation import OUT,BOTTOM,TOP,G,TILT

def read(label):
    with h5py.File(OUT/(label+'.h5')) as h:
        data={k:h[k][:] for k in ['z_m','z_faces_m','ne_m3','density_phasor_m3','E_phasor_V_m','incident_state','dz_cells_m','pressure_flux_W_m2']}
        data['attrs']=dict(h.attrs);data['kx']=float(h['kx_rad_m'][()])
    return data

def interp(zz,z,v):return np.interp(zz,z,v.real)+1j*np.interp(zz,z,v.imag)

def inspect(label,fine,side):
    x=read(label);y=read(fine);z=x['z_m'];zf=x['z_faces_m'];dn=x['density_phasor_m3'];E=x['E_phasor_V_m'];at=x['attrs']
    res=TOP if side>0 else BOTTOM
    offsets=side*np.linspace(1,8,1401);zz=res+offsets
    phi=lambda a:np.angle(interp(115100,a['z_m'],a['incident_state'][0]))
    low=interp(zz,z,dn)*np.exp(-1j*phi(x));high=interp(zz,y['z_m'],y['density_phasor_m3'])*np.exp(-1j*phi(y))
    amp_error=np.linalg.norm(abs(low)-abs(high))/np.linalg.norm(abs(high))
    complex_error=np.linalg.norm(low-high)/np.linalg.norm(high)
    # Actual density phase, without a guessed WKB wave or geometric cartoon.
    kz=np.gradient(np.unwrap(np.angle(dn)),z)
    interval=(side*(z-res)>1)&(side*(z-res)<8)
    phase_correct=float(np.mean(side*kz[interval]>0))
    assert phase_correct>.99,(label,phase_correct)
    kr=4*np.pi*430e6/c
    zr=np.linspace(res+side*1,res+side*8,3000)
    phase_k=np.interp(zr,z,kz)
    i=np.where((abs(phase_k[:-1])-kr)*(abs(phase_k[1:])-kr)<0)[0][0]
    observed=float(zr[i]+(zr[i+1]-zr[i])*(kr-abs(phase_k[i]))/(abs(phase_k[i+1])-abs(phase_k[i])))
    kvec=np.array([x['kx'],0,side*np.sqrt(kr**2-x['kx']**2)])
    b=np.array([np.sin(TILT),0,-np.cos(TILT)])
    mu=np.dot(kvec,b)/kr
    # Independent local electrostatic warm-fluid dispersion, retaining B and
    # the actual signed k. This is a WKB comparison outside the conversion core.
    target=1/((1-mu*mu)/(a.W*a.W-G*G)+mu*mu/(a.W*a.W))-a.CS*a.CS*kr*kr
    theory=brentq(lambda z:a.PROFILE(z)*e*e/(m_e*epsilon_0)-target,*sorted([res,res+side*20]))
    bm_error=abs((observed-res)/(theory-res)-1)
    assert bm_error<.05,(label,observed-res,theory-res)
    assert abs(np.interp(res+side*8,z,kz))>abs(np.interp(res+side*2,z,kz))
    # Compare each curve's own amplitude at increasing outward distance;
    # no automatic normalization of the attenuated tail.
    attenuation=float(abs(interp(res+side*12,z,dn))/abs(interp(res+side*2,z,dn)))
    assert attenuation<.2,(label,attenuation)
    widths=x['dz_cells_m'];dual=(widths+np.roll(widths,1))/2
    div=1j*x['kx']*E[0]+(E[2]-np.roll(E[2],1))/dual
    charge=e*dn/epsilon_0
    gauss=np.linalg.norm((div+charge)[interval])/np.linalg.norm(charge[interval])
    assert gauss<1e-6,(label,gauss)
    # Pressure energy flux must point away from the conversion layer.
    outgoing=side*np.interp(res+side*4,zf,x['pressure_flux_W_m2'])
    assert outgoing>0,(label,outgoing)
    assert amp_error<.02,(label,amp_error)
    assert abs(at['energy_residual'])<2e-4
    report=json.loads(at['native_report'])
    assert report['finite'] and report['relative_max_e_error']<1e-7 and report['relative_max_q_error']<1e-7
    with h5py.File(OUT/(label+'.h5')) as h:
        tt=h['native/time_s'][:]
        actual=h['native/E_q_real'][:]+1j*h['native/E_q_imag'][:]
        predE=E[None,:,:]*np.exp(-1j*a.W*tt)[:,None,None]
        err=np.linalg.norm((actual[:,:3]-predE)[:,:,interval])/np.linalg.norm(predE[:,:,interval])
        actualdn=actual[:,3]*np.sqrt(epsilon_0*x['ne_m3']/(m_e*a.CS**2))[None,:]
        preddn=dn[None,:]*np.exp(-1j*a.W*tt)[:,None]
        dnerr=np.linalg.norm((actualdn-preddn)[:,interval])/np.linalg.norm(preddn[:,interval])
        assert max(err,dnerr)<1e-7
    stats=dict(amplitude_mesh_relative_error=amp_error,complex_mesh_relative_error=complex_error,
        correct_phase_direction_fraction=phase_correct,measured_bragg_offset_m=observed-res,
        local_dispersion_bragg_offset_m=theory-res,bragg_relative_error=bm_error,
        density_amplitude_12m_over_2m=attenuation,Gauss_relative_error=gauss,
        outward_pressure_flux_at_4m=outgoing,native_E_relative_error=err,native_density_relative_error=dnerr)
    print(label,stats,flush=True)
    return stats

if __name__=='__main__':
    with h5py.File(OUT/'validation.h5','w') as out:
        for label,fine,side in [('topside_optimum','topside_refined',1),('bottomside_optimum','bottomside_refined',-1),('topside_spitze','spitze_refined',1)]:
            out.create_group(label).attrs.update(inspect(label,fine,side))
        out['provenance/source']=Path(__file__).read_text()
