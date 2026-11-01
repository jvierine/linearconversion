"""Audit current optimum's HF loss and cold-ray access on the same background.

Cold local rays diagnose refraction only; no tunnelling is assigned to rays.
The independently solved full-wave fields determine access and power loss.
References: Muldrew (1993); Gondarenko et al. (2003, 2009).
"""
from pathlib import Path
import sys,itertools
import numpy as np
import h5py
from scipy.constants import c,e,m_e,epsilon_0
from scipy.optimize import brentq
from numpy.polynomial import Polynomial as Poly
import coupled_arecibo as a
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from isr.conversion import Parameters
from isr.rays import electromagnetic_dispersion,electromagnetic_ray,spitze_angle_deg


def cold_roots(p,par):
    base=electromagnetic_dispersion([par.q*par.k0,0,0],p,par,cold=True)
    mat=[[Poly([base[i,j]]) for j in range(3)] for i in range(3)]
    mat[0][0]+=Poly([0,0,-1]);mat[1][1]+=Poly([0,0,-1])
    mat[0][2]+=Poly([0,par.q]);mat[2][0]+=Poly([0,par.q])
    det=Poly([0])
    for perm in itertools.permutations(range(3)):
        sign=(-1)**sum(perm[i]>perm[j] for i in range(3) for j in range(i+1,3))
        det+=sign*mat[0][perm[0]]*mat[1][perm[1]]*mat[2][perm[2]]
    return det.trim(tol=1e-13).roots()


def main():
    inp=a.OUT/'minimum_incident/best_model.h5'
    out=a.ROOT/'reference/resonant_access_audit.h5'
    with h5py.File(inp) as h:
        attrs=dict(h.attrs);z=h['z'][:];width=h['dz_cells'][:];state=h['state_phasor'][:]
        inc=h['incident_reference_state'][:];nu=h['physical_nu'][:];v=h['source_polarization'][:]
    zf=z+width/2;dual=(width+np.roll(width,1))/2;center=attrs['center_m'];angle=attrs['angle_deg']
    lo=np.argmin(abs(zf-115100));hi=np.argmin(abs(zf-120800));mask=(z>zf[lo])&(z<zf[hi])
    # Centered current velocities are normalized as in coupled_arecibo.py.
    dt=c/a.F/np.ceil(c/a.F/(.4*width.min()))
    q=.5*epsilon_0*c*np.cos(a.W/c*dt/2)*(nu*abs(state[6:9])**2*np.array([dual,dual,width])).sum(axis=0)
    budget=a.hf_port_budget(state,inc,lo,hi);pin=budget['incident_EM_flux_W_m2']
    heat=q[mask].sum()/pin;near=abs(z-center)<30
    stats=dict(budget,angle_deg=angle,total_collision_loss_fraction=heat,
      loss_balance_residual=budget['hf_nonreturn_fraction']-heat,
      fraction_heating_within_30m=q[mask&near].sum()/q[mask].sum(),
      nu_at_resonance_s1=np.interp(center,z,nu[0])*c,
      peak_Ez_over_incident_total_peak=attrs['peak_Ez_V_m']/(np.sqrt(2)*attrs['incident_reference_E_rms_V_m']))
    assert abs(stats['loss_balance_residual'])<2e-5
    par=Parameters(frequency_hz=a.F,temperature_k=300,magnetic_t=float(a.P['magnetic_t']),
          magnetic_tilt_deg=180-float(a.P['field_angle_deg']),q=np.sin(np.deg2rad(angle)))
    fac=e*e/(epsilon_0*m_e*a.W*a.W)
    def profile(zz):return float(a.PROFILE(zz)*fac),float(a.PROFILE(zz,1)*fac)
    crit=brentq(lambda zz:profile(zz)[0]-1,117000,118100)
    scalar_turn=brentq(lambda zz:profile(zz)[0]-np.cos(np.deg2rad(angle))**2,117000,118100)
    Ln=1/profile(crit)[1];airy=(Ln/par.k0**2)**(1/3)
    stats.update(critical_height_m=crit,resonance_height_m=center,spitze_angle_deg=spitze_angle_deg(par),
      unmagnetized_turn_height_m=scalar_turn,unmagnetized_airy_length_m=airy,
      unmagnetized_gap_m=crit-scalar_turn,gyrofrequency_hz=e*par.magnetic_t/(2*np.pi*m_e))
    with h5py.File(out,'w') as h:
      h.attrs.update(stats);h.attrs['provenance']=Path(__file__).read_text()
      h.attrs['scope']='Energy audit of saved optimized warm solution; separate same-source cold control; cold rays neglect collisions and do not carry conversion coefficients.'
      h['height_relative_m']=z-center;h['collision_loss_per_m']=q/dual/pin
      h['Ez_over_incident_peak']=abs(state[2])/np.linalg.norm(inc[:3,lo])
      h['Ez_height_relative_m']=zf-center
      h['cumulative_collision_loss']=np.cumsum(q*mask)/pin
      # Same source, grid, density and physical collisions for a cold control.
      cold=a.setup(dz=attrs['dz_m'],cap=attrs['cap_m'],angle=angle,thermal=False,scenario={'ordering':'COLAMD'})
      ref=a.setup(dz=attrs['dz_m'],cap=attrs['cap_m'],angle=angle,thermal=False,reference=True,scenario={'ordering':'COLAMD'})
      cs=np.einsum('qnp,p->qn',cold['initial'],v);ci=np.einsum('qnp,p->qn',ref['initial'],v)
      cb=a.hf_port_budget(cs,ci,lo,hi)
      for k,val in cb.items():h.attrs['cold_'+k]=val
      print('BUDGET',stats,'COLD',cb,flush=True)
      start=115300.
      for j,root in enumerate(cold_roots(profile(start)[0],par)):
        if abs(root.imag)>1e-8:continue
        kz=root.real*par.k0
        vals,vec=np.linalg.eigh(electromagnetic_dispersion([par.q*par.k0,0,kz],profile(start)[0],par,cold=True))
        E=vec[:,np.argmin(abs(vals))];S=np.cross(E,np.cross(np.array([par.q,0,root.real]),E.conj())).real[2]
        if S<=0:continue
        ray=electromagnetic_ray(start,kz,profile,par,max_step_s=2e-7,max_time_s=.0005,max_index=7.)
        g=h.create_group(f'ray_{j}')
        for k,val in ray.items():
          if isinstance(val,str):g.attrs[k]=val
          else:g[k]=val
        g.attrs['maximum_height_relative_m']=ray['height_m'].max()-center
        g.attrs['maximum_dispersion_residual']=abs(ray['dispersion_residual']).max()
        assert g.attrs['maximum_dispersion_residual']<1e-6
        g.attrs['cold_mode']='ordinary' if ray['height_m'].max()>center else 'extraordinary'
        if g.attrs['cold_mode']=='ordinary':
            assert abs(ray['height_m'].max()-crit)<.01
        print('RAY',j,dict(g.attrs),flush=True)
    print(out,flush=True)
if __name__=='__main__':main()
