"""One-way cold Maxwell -> warm vertical electron fluid; scientific output HDF5.

The 1D-compatible drive is Ex,cold, Ey,cold and Ez,cold + i Jz,cold/(omega eps0).
This removes exactly the cold vertical polarization response that the 1D
Poisson solver regenerates. The Fourier-x Helmholtz E_T split is NOT used here:
omitting horizontal continuity with that split produces spurious compression.
Horizontal fluid derivatives are omitted; the imposed HF pump is held fixed.
Native time stepping uses fluid_browser/src/lib.rs, not a Python integrator.
The independent banded harmonic calculation is for the identical discretization.
"""
from pathlib import Path
import argparse
import ctypes as C
import hashlib
import json
import time
import h5py
import numpy as np
from scipy.constants import e, m_e, epsilon_0, k as kb, c
from scipy.interpolate import CubicHermiteSpline
from scipy.linalg import solve_banded
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent
FIELD = ROOT/'output/hf_fdtd/arecibo_hf_field.h5'
PROFILE = ROOT/'data/smooth_observed_layer_1525453430.h5'
OUT = ROOT/'output/fluid/fdtd_driven.h5'


def export_browser_field():
    """JSON is browser transport; the canonical scientific field remains HDF5."""
    with h5py.File(FIELD) as h:
        j=np.argmin(abs(h['z_nodes_m'][:]-115100.))
        reference=np.linalg.norm(h['incident_reference_E_V_m'][:,j])
        v=h['E_phasor_V_m'][:];v[2]+=1j*h['current_phasor_A_m2'][2]/(2*np.pi*h.attrs['frequency_hz']*epsilon_0);v/=reference;b=h['magnetic_field_T'][:]
        payload=dict(z=h['z_E_components_m'][:].tolist(),real=v.real.tolist(),imag=v.imag.tolist(),
                     frequency_MHz=float(h.attrs['frequency_hz'])/1e6,
                     magnetic_uT=float(np.linalg.norm(b)*1e6),
                     bangle_deg=float(np.degrees(np.arctan2(b[0],b[2]))),
                     normalization='complex vector incident norm at 115.1 km',
                     phasor_convention='Re(E exp(-i omega t))',
                     source='1D-compatible Maxwell drive: Ex,cold, Ey,cold, Ez,cold+i Jz,cold/(omega epsilon0); one-way vertical approximation')
    (ROOT/'web/fluid/fdtd-field.json').write_text(json.dumps(payload,separators=(',',':')))


def setup(cells=2048, lo=-24., hi=16., incident=0.0005, gradient=1.):
    with h5py.File(PROFILE) as h:
        zz, nn, dd = (h[k][:] for k in ['model_height_m','model_ne_m3','model_dne_dz_m4'])
    density = CubicHermiteSpline(zz,nn,dd)
    with h5py.File(FIELD) as h:
        frequency=float(h.attrs['frequency_hz']); te=float(h.attrs['electron_temperature_K'])
        b=h['magnetic_field_T'][:]; omega=2*np.pi*frequency
        bx,bz=e*b[[0,2]]/m_e; gyro2=bx*bx+bz*bz
        D=omega**2*(omega**2-gyro2)/(omega**2-bz*bz)
        cs2=3*kb*te/m_e; kr=4*np.pi*430e6/c
        center=brentq(lambda z:density(z)*e*e/(m_e*epsilon_0)-D,117000,118100)
        match=brentq(lambda z:density(z)*e*e/(m_e*epsilon_0)+cs2*kr*kr-D,117000,center)
        zf=np.linspace(center+lo,center+hi,cells+1); z=(zf[:-1]+zf[1:])/2;dz=(hi-lo)/cells
        ne=density(center+gradient*(z-center)); nf=np.r_[ne[0],(ne[:-1]+ne[1:])/2,ne[-1]]
        ez=h['z_E_components_m'][:]; ph=h['E_phasor_V_m'][:];ph[2]+=1j*h['current_phasor_A_m2'][2]/(omega*epsilon_0)
        j=np.argmin(abs(h['z_nodes_m'][:]-115100.))
        scale=incident/np.linalg.norm(h['incident_reference_E_V_m'][:,j])
        source=np.array([np.interp(zf,x,y.real)+1j*np.interp(zf,x,y.imag) for x,y in zip(ez,ph)])*scale
        total=np.array([np.interp(zf,x,y.real)+1j*np.interp(zf,x,y.imag) for x,y in zip(ez,h['E_phasor_V_m'][:])])*scale
        nu=np.interp(zf,ez[2],h['collision_rate_s1'][2])
        kx=float(h['kx_rad_m'][()]);incident_flux=float(h.attrs['incident_vertical_flux_W_m2'])*scale**2
    edge=np.minimum(zf-zf[0],zf[-1]-zf)
    sponge=2e6*np.maximum(0,1-edge/(.15*(hi-lo)))**3
    # The source is unmodified in the physical interior. Taper it smoothly only
    # inside the numerical buffers to avoid driving a closed end face.
    taper=np.sin(.5*np.pi*np.clip(edge/(.15*(hi-lo)),0,1))**2
    source*=taper
    return dict(cells=cells,z=z,zf=zf,dz=dz,ne=ne,nf=nf,nu=nu,sponge=sponge,
                source=source,total=total,center=center,match=match,frequency=frequency,
                te=te,b=b,bx=bx,bz=bz,omega=omega,cs2=cs2,D=D,kx=kx,incident=incident,
                profile=np.r_[zz,nn,dd],profile_count=len(zz),gradient=gradient,incident_flux_W_m2=incident_flux)


def harmonic(s):
    """Eliminate velocity polarization and Gauss law to a tridiagonal flux solve."""
    n=s['cells'];w=s['omega'];a=s['nu'][1:-1]+s['sponge'][1:-1]-1j*w
    bx,bz=s['bx'],s['bz'];gyro2=bx*bx+bz*bz
    rzz=(a*a+bz*bz)/(a*(a*a+gyro2))
    effective=s['source'][2,1:-1]+bx*bz/(a*a+bz*bz)*s['source'][0,1:-1]+a*bx/(a*a+bz*bz)*s['source'][1,1:-1]
    lap=s['cs2']/(s['ne']*s['dz']**2)
    matrix=np.zeros((3,n-1),complex)
    matrix[1]=-(lap[:-1]+lap[1:])+1j*w/(s['nf'][1:-1]*rzz)-e*e/(m_e*epsilon_0)
    matrix[0,1:]=lap[1:-1];matrix[2,:-1]=lap[1:-1]
    p=np.r_[0,solve_banded((1,1),matrix,-1j*w*e/m_e*effective),0]
    dn=np.diff(p)/(1j*w*s['dz']);eta=dn/s['ne'];eself=1j*e/(w*epsilon_0)*p
    uz=p/s['nf'];fx,fy=-e/m_e*s['source'][:2,1:-1]
    ux=np.r_[0,(a*fx-bz*fy+bz*bx*uz[1:-1])/(a*a+bz*bz),0]
    uy=np.r_[0,(bz*fx+a*fy-a*bx*uz[1:-1])/(a*a+bz*bz),0]
    u=np.array([ux,uy,uz]);u2=np.sum(abs(u)**2,axis=0)
    collision=.5*m_e*np.sum(s['nf']*s['nu']*u2)*s['dz']
    buffer=.5*m_e*np.sum(s['nf']*s['sponge']*u2)*s['dz']
    work=-.5*e*np.sum(s['nf']*np.real(np.sum(u.conj()*s['source'],axis=0)))*s['dz']
    pressure=m_e*s['cs2']*dn
    flux=.5*np.real(pressure*np.conj((uz[:-1]+uz[1:])/2))
    # Phase-gradient derivative without finite-difference sine compression.
    k=np.gradient(np.unwrap(np.angle(eta)),s['dz'])
    return dict(density_phasor=eta,dn_phasor_m3=dn,velocity_phasor_m_s=u,
                self_field_phasor_V_m=eself,pressure_flux_W_m2=flux,kz_rad_m=k,
                collision_power_W_m2=collision,buffer_power_W_m2=buffer,
                source_power_W_m2=work,balance_relative=(work-collision-buffer)/work)


def save_group(g,values):
    for key,value in values.items():
        if isinstance(value,np.ndarray):g.create_dataset(key,data=value,compression='gzip',shuffle=True)
        elif np.isscalar(value):g.attrs[key]=value


def native(s,steady,stop=0.004,duration=0.0015,ramp=50e-6,initial='zero',output=OUT):
    lib=C.CDLL(str(ROOT/'fluid_browser/target/release/libarecibo_fluid.dylib'))
    ptr=C.c_void_p;fp=C.POINTER(C.c_double);u=C.c_size_t;f=C.c_double
    def bind(name,args,result):
        fn=getattr(lib,name);fn.argtypes=args;fn.restype=result;return fn
    create=bind('create',[u]+[f]*7+[fp,u],ptr)
    source=bind('source',[ptr]+[f]*7,C.c_uint)
    phasor=bind('source_phasor',[ptr,fp,fp,u],C.c_uint)
    collisions=bind('collision_profile',[ptr,fp,u],C.c_uint)
    advance=bind('advance',[ptr,u],C.c_uint)
    snapshot=bind('snapshot',[ptr],fp)
    state=bind('state_ptr',[ptr],fp)
    destroy=bind('destroy',[ptr],None)
    def cp(a):return np.ascontiguousarray(a,dtype=np.float64)
    def p(a):return a.ctypes.data_as(fp)
    n=s['cells'];profile=cp(s['profile']);nu=cp(s['nu'])
    sim=create(n,s['zf'][0],s['zf'][-1],s['te'],np.linalg.norm(s['b']),np.arctan2(s['bx'],s['bz']),0.,0.,p(profile),s['profile_count'])
    assert sim
    assert source(sim,0.,s['frequency'],0.,s['center'],1.,ramp,duration)
    re,im=cp(s['source'].real.ravel()),cp(s['source'].imag.ravel())
    assert phasor(sim,p(re),p(im),re.size)
    assert collisions(sim,p(nu),n+1)
    if initial=='harmonic':
        yy=np.ctypeslib.as_array(state(sim),shape=(4*n+3,))
        yy[:n]=steady['dn_phasor_m3'].real/1e11
        yy[n:]=steady['velocity_phasor_m_s'].real.ravel()
        # For a stationary validation, the phasor drive must be fully on at t=0.
        # Native source supports a zero-rise initialization through tiny ramp;
        # start with the exact phase at t=0, whose one RK endpoint omission is tested.
    out=np.ctypeslib.as_array(snapshot(sim),shape=(13*n+12,)).copy()
    dt=out[13*n+1];steps=int(round(stop/dt));stride=max(1,int(round(10e-6/dt)))
    print('Native',n,'cells, dt',dt,'steps',steps,flush=True)
    with h5py.File(output,'a') as h:
        name='transient' if initial=='zero' else 'native_harmonic_check'
        if name in h:del h[name]
        g=h.create_group(name);g.attrs.update(dt_s=dt,pulse_duration_s=duration,ramp_s=ramp,initial=initial)
        # Full mesh retained; display reduction belongs to the figure, not data.
        # Time-height samples are float32; final state and harmonic arrays float64.
        ds={key:g.create_dataset(key,shape=(0,n),maxshape=(None,n),chunks=(1,n),compression='gzip') for key in ['fractional_density','density_amplitude_rms','self_field_V_m','velocity_z_m_s']}
        diag=g.create_dataset('diagnostics',shape=(0,12),maxshape=(None,12),chunks=True)
        diag.attrs['columns']='time_s,dt_s,max_fraction,energy_J_m2,net_charge_number_m2,bragg_projection,dz_m,source_envelope,source_work_J_m2,collision_loss_J_m2,buffer_loss_J_m2,energy_residual_J_m2'
        done=0;frame=0;wall=time.monotonic();last=wall
        while True:
            out=np.ctypeslib.as_array(snapshot(sim),shape=(13*n+12,)).copy()
            for key,row in [('fractional_density',2),('density_amplitude_rms',12),('self_field_V_m',4),('velocity_z_m_s',6)]:
                ds[key].resize(frame+1,axis=0);ds[key][frame]=out[row*n:(row+1)*n]
            diag.resize(frame+1,axis=0);diag[frame]=out[13*n:];frame+=1;h.flush()
            if done>=steps:break
            batch=min(stride,steps-done)
            remaining=batch
            while remaining:
                k=min(10000,remaining);assert advance(sim,k);remaining-=k
            done+=batch
            if time.monotonic()-last>15:
                print(f'{out[13*n]*1e3:.3f}/{stop*1e3:.3f} ms; wall {time.monotonic()-wall:.1f}s',flush=True);last=time.monotonic()
        yy=np.ctypeslib.as_array(state(sim),shape=(4*n+3,)).copy()
        g['final_state']=yy
        if initial=='harmonic':
            expected=np.r_[steady['dn_phasor_m3']/1e11,steady['velocity_phasor_m_s'].ravel()]*np.exp(-1j*s['omega']*out[13*n])
            err=np.linalg.norm(yy[:n]-expected[:n].real)/np.linalg.norm(steady['dn_phasor_m3']/1e11)
            g.attrs['density_relative_l2_error']=err;print('Native harmonic error',err,flush=True)
        print('Final ledger',out[13*n:],flush=True)
    destroy(sim)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--check',action='store_true');ap.add_argument('--stop-ms',type=float,default=4.);args=ap.parse_args()
    s=setup();r=harmonic(s);OUT.parent.mkdir(exist_ok=True)
    if not args.run and not args.check:
        export_browser_field()
        fine=setup(4096);rf=harmonic(fine);large=setup(4096,-44.,36.);rl=harmonic(large)
        at=lambda ss,rr:float(np.interp(ss['match'],ss['z'],abs(rr['density_phasor'])))
        errs=np.array([abs(at(fine,rf)/at(s,r)-1),abs(at(large,rl)/at(s,r)-1)])
        print('Convergence grid/domain',errs,'balance',r['balance_relative'])
        assert abs(r['balance_relative'])<1e-6
        cold={**s,'cs2':0.};rc=harmonic(cold)
        interior=abs(s['zf']-s['center'])<3.
        restored=rc['self_field_phasor_V_m']+s['source'][2]
        cold_error=np.linalg.norm((restored-s['total'][2])[interior])/np.linalg.norm(s['total'][2,interior])
        # Component interpolation between the staggered Maxwell and fluid grids
        # limits this cold reconstruction; report the error rather than fitting it.
        assert cold_error<.005,cold_error
        power_ratio=r['source_power_W_m2']/s['incident_flux_W_m2']
        assert 0<power_ratio<1.05,power_ratio
        print('Cold field reconstruction error',cold_error,'warm loading / incident flux',power_ratio)
        with h5py.File(OUT,'w') as h:
            h.attrs.update(model='1D3V warm barotropic electron momentum-continuity-Poisson, prescribed complex FDTD field with cold vertical polarization removed',
                scope='One-way forcing matched to the vertical cold current; horizontal fluid derivatives omitted. No Landau damping, feedback on HF, saturation, or inferred event amplitude.',
                field_sha256=hashlib.sha256(FIELD.read_bytes()).hexdigest(),
                profile_sha256=hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
                phasor_convention='Re(phasor exp(-i omega t)); all components retain phase',
                source_provenance='drive_fluid_from_fdtd.py; fluid_browser/src/lib.rs; hf_fdtd/examples/arecibo_layer.py')
            save_group(h.create_group('background'),s);save_group(h.create_group('harmonic'),r)
            h['convergence_grid_domain_relative']=errs
            h.attrs['cold_field_reconstruction_relative_error']=cold_error
            h.attrs['warm_source_power_over_incident']=power_ratio
            for name,ss,rr in [('fine',fine,rf),('large',large,rl)]:
                g=h.create_group('validation/'+name);g['z']=ss['z'];save_group(g,rr)
            for key in ['drive_fluid_from_fdtd.py','fluid_browser/src/lib.rs']:
                h.create_dataset('provenance/'+Path(key).name,data=Path(ROOT/key).read_text(),dtype=h5py.string_dtype())
        print('Match',s['match']-s['center'],'m; amplitude',at(s,r))
        for height in [-1,-2,-4.05,-6,-10]:
            j=np.argmin(abs(s['z']-s['center']-height));print(height,'eta',abs(r['density_phasor'][j]),'kz',r['kz_rad_m'][j],'flux',r['pressure_flux_W_m2'][j])
    elif args.check:native(s,r,stop=10e-6,duration=0,ramp=1e-20,initial='harmonic')
    else:native(s,r,stop=args.stop_ms*1e-3)


if __name__=='__main__':main()
