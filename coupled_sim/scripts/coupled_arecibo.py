"""Coupled warm-fluid / Maxwell Yee recurrence, with a Rust CW evolution.

One Fourier-x mode, three velocities and fields, measured smooth n0. Current
feeds Ampere; pressure feeds momentum; full Fourier-x continuity evolves dn.
This is a linear warm-fluid closure with collisions, NOT a kinetic solver.
The density-weighted pressure/divergence operators are energy adjoints.
Gyro interpolation uses the shared energy-conserving Yee current stencil.
"""
from pathlib import Path
import argparse,json,subprocess,hashlib,time
import numpy as np
import h5py
from scipy.constants import c,epsilon_0,e,m_e,k as kb
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq
from scipy.sparse import bmat,diags,eye,coo_matrix
from scipy.sparse.linalg import splu
from scipy.linalg import solve_banded
import model_inputs as cold

ROOT=cold.ROOT;REPO=cold.REPO;OUT=ROOT/'output/coupled';OUT.mkdir(exist_ok=True,parents=True)
F=5.340028553893307e6;W=2*np.pi*F;TE=300.;CS=np.sqrt(3*kb*TE/m_e)
with h5py.File(ROOT/'data/smooth_observed_layer_1525453430.h5') as h:
    PROFILE=CubicHermiteSpline(h['model_height_m'][:],h['model_ne_m3'][:],h['model_dne_dz_m4'][:])
P=cold.P

def setup(dz=.02,angle=16.75,reference=False,collision_scale=1.,buffer=600.,thermal=True,cap=2.,cfl=.4,scenario=None):
    cfg={} if scenario is None else scenario
    frequency=cfg.get('frequency_hz',F)
    w=2*np.pi*frequency
    tilt=np.deg2rad(cfg.get('field_angle_deg',P['field_angle_deg']))
    field_azimuth=np.deg2rad(cfg.get('field_azimuth_deg',0.))
    gyro=e*cfg.get('magnetic_t',P['magnetic_t'])/m_e/c*np.array([
        np.sin(tilt)*np.cos(field_azimuth),
        np.sin(tilt)*np.sin(field_azimuth),-np.cos(tilt)])
    om=w/c;D=w**2*(w**2-np.sum((c*gyro)**2))/(w**2-(c*gyro[2])**2)
    profile=cfg.get('profile',PROFILE)
    center=cfg.get('center',None)
    if center is None:center=brentq(lambda z:profile(z)*e*e/(m_e*epsilon_0)-D,117000,118100)
    lo,hi=cfg.get('bounds',(114000-(buffer-600),122000+(buffer-600)))
    fine_lo,fine_hi=cfg.get('fine_bounds',(-80.,20.))
    fine_intervals=cfg.get('fine_intervals',[(fine_lo,fine_hi)])
    points=[center]
    for direction,bound in [(1,hi),(-1,lo)]:
        v=center
        while (v-bound)*direction<0:
            d=v-center
            # Resolve outgoing waves near one or both resonance crossings.
            outside=min(max(a-d,d-b,0.) for a,b in fine_intervals)
            base=dz
            if 'phase_grid_distance_m' in cfg:
                # Keep roughly fixed cells per local Langmuir wavelength
                # along a long, smooth propagation path. The shared Yee
                # equations use the actual widths; only the grid changes.
                length=cfg['phase_grid_distance_m']
                base=dz*min(10.,np.sqrt(length/max(length/100.,-d)))
            step=min(cap,base+.015*outside)
            v+=direction*step;points.append(v)
    z=np.array(sorted(points));n=len(z);hi=z[-1]+cap;lo=z[0]
    widths=np.diff(np.r_[z,hi]);dual=(widths+np.roll(widths,1))/2;zf=z+widths/2
    positions=np.array([z,z,zf]);clipped=np.clip(positions,*cfg.get('sample_bounds',(115300,120800)))
    ne=profile(clipped)
    if reference:ne[:]=profile(cfg.get('reference_z',115300.))
    wp=np.sqrt(ne*e*e/(epsilon_0*m_e))/c;sqrt_ne=np.sqrt(ne)
    physical_nu=cfg.get('rates',cold.rates)(clipped,ne)*collision_scale/c
    sph=np.array([z,z,zf,zf,zf,z]);depth=np.maximum((lo+buffer-sph)/buffer,(sph-(hi-buffer))/buffer).clip(0,1)
    sponge=40/buffer*depth**3
    nu=physical_nu+sponge[:3]
    spp=int(np.ceil(c/frequency/(cfl*widths.min())));dt=cfg.get('fixed_dt_m',c/frequency/spp)
    wd=2*np.sin(om*dt/2)/dt;ch=np.cos(om*dt/2);kx=cfg.get('fixed_kx_m1',om*np.sin(np.deg2rad(angle)))
    i=np.arange(n);S=coo_matrix((np.ones(n),(i,(i+1)%n)),shape=(n,n)).tocsr();I=eye(n,format='csr');Z=I*0
    if cfg.get('banded_solver',False):
        # Terminate behind the absorbing buffers instead of joining their
        # distant endpoints. The interior Yee/fluid operators are unchanged.
        S=coo_matrix((np.ones(n-1),(i[:-1],i[1:])),shape=(n,n)).tocsr()
    df=diags(1/widths)@(S-I);db=diags(1/dual)@(I-S.T);av=(I+S)/2;ab=diags(1/dual)@av.T@diags(widths)
    cf=bmat([[Z,-df,Z],[df,Z,-1j*kx*I],[Z,1j*kx*I,Z]],format='csr')
    cb=bmat([[Z,-db,Z],[db,Z,-1j*kx*I],[Z,1j*kx*I,Z]],format='csr')
    gx,gy,gz=gyro
    G=bmat([[Z,-gz*I,gy*ab],[gz*I,Z,-gx*ab],[-gy*av,gx*av,Z]],format='csr')
    beta=np.sqrt(3*kb*cfg.get('temperature',TE)/m_e)/c if thermal else 0.
    pressure=bmat([[1j*kx*beta*I],[Z],[beta*diags(sqrt_ne[2])@df@diags(1/sqrt_ne[0])]],format='csr')
    # Audit only: p0=n0*kT, p1=3*kT*n1 gives an additional acceleration
    # -(2/3)*cs^2*(n1/n0)*grad(log n0). Default remains the conservative
    # effective barotropic closure used by the Rust evolution and paper.
    pressure_gradient_audit=cfg.get('pressure_gradient_audit',False)
    if pressure_gradient_audit:
        dlog=df@np.log(ne[0])
        correction=(2/3)*beta*diags(sqrt_ne[2]*dlog)@av@diags(1/sqrt_ne[0])
        pressure=pressure+bmat([[Z],[Z],[correction]],format='csr')
    continuity=bmat([[1j*kx*beta*I,Z,beta*diags(1/sqrt_ne[0])@db@diags(sqrt_ne[2])]],format='csr')
    we=diags(wp.ravel());Le=diags((-1j*wd+ch*sponge[:3]).ravel());Lh=diags((-1j*wd+ch*sponge[3:]).ravel())
    Lj=diags((-1j*wd+ch*nu).ravel())-ch*G;Lq=diags(-1j*wd+ch*sponge[0])
    mat=bmat([[Le,-cb,we,None],[cf,Lh,None,None],[-we,None,Lj,-pressure],[None,None,-continuity,Lq]],format='csc')
    source=np.zeros((10*n,2),complex);ii=np.argmin(abs(z-cfg.get('source_z',114850.)));source[ii,0]=1/dual[ii];source[n+ii,1]=1/dual[ii]
    perm=np.arange(10*n).reshape(10,n).T.ravel()
    if cfg.get('banded_solver',False):
        block=mat[perm,:][:,perm].tocoo()
        lower=int(np.max(block.row-block.col));upper=int(np.max(block.col-block.row))
        bands=np.zeros((lower+upper+1,10*n),complex)
        bands[upper+block.row-block.col,block.col]=block.data
        tmp=solve_banded((lower,upper),bands,source[perm],overwrite_ab=True,check_finite=False)
    else:
        lu=splu(mat[perm,:][:,perm].tocsc(),permc_spec=cfg.get('ordering','NATURAL'))
        tmp=lu.solve(source[perm])
    sol=np.empty_like(tmp);sol[perm]=tmp
    residual=np.linalg.norm(mat@sol-source)/np.linalg.norm(source)
    # These identities are the spatial discrete energy check, independent of
    # the solution and of the relative size of electric vs thermal energy.
    wj=diags(np.array([dual,dual,widths]).ravel());wq=diags(dual)
    adj=wj@pressure+continuity.conj().T@wq
    adjerr=np.max(abs(adj.data)) if adj.nnz else 0.
    assert residual<1e-6, residual
    if not pressure_gradient_audit:assert adjerr<1e-10, adjerr
    return dict(n=n,z=z,dz_cells=widths,dual=dual,dt=dt,omega=om,kx=kx,gyro=gyro,wp=wp,nu=nu,
                physical_nu=physical_nu,sponge=sponge,sqrt_ne=sqrt_ne,beta=beta,initial=sol.reshape(10,n,2),
                source=source.reshape(10,n,2),spp=spp,ne=ne,center=center,D=D,
                harmonic_residual=residual,pressure_adjoint_error=adjerr)

def electromagnetic_flux(v):
    """Cycle-averaged Poynting flux at Yee faces, including both polarizations."""
    ex,ey=(v[:2]+np.roll(v[:2],-1,axis=1))/2
    return .5*epsilon_0*c*np.real(ex*v[4].conj()-ey*v[3].conj())

def hf_port_budget(state,incident,lower,upper):
    """HF non-return from outgoing EM power; subtract fields before flux.

    Ports must be outside the conversion/near-field region and before PMLs.
    This planar single-kx model has no net lateral escape. Finite-beam models
    must also count outgoing side flux. These are steady-state power ratios.
    """
    pin=electromagnetic_flux(incident)[lower]
    assert pin>0
    reflected=-electromagnetic_flux(state-incident)[lower]/pin
    transmitted=electromagnetic_flux(state)[upper]/pin
    return dict(incident_EM_flux_W_m2=float(pin),reflected_EM_fraction=float(reflected),
        transmitted_EM_fraction=float(transmitted),
        hf_nonreturn_fraction=float(1-reflected-transmitted))

def flux(s,v):
    # Flux at half cells. q and E_x/E_y are node centered; jz, Hx, Hy faces.
    em=electromagnetic_flux(v)
    qp=(v[9]/s['sqrt_ne'][0]+np.roll(v[9]/s['sqrt_ne'][0],-1))/2
    pressure=-.5*epsilon_0*c*s['beta']*np.real(qp*s['sqrt_ne'][2]*v[8].conj())
    return em,pressure

def run(dz=.02,collision_scale=1.,buffer=600.,cap=2.,angle=16.75,native_cycles=0,label='baseline',thermal=True):
    wall=time.monotonic();print('Setup',label,dz,collision_scale,flush=True)
    s=setup(dz,angle,collision_scale=collision_scale,buffer=buffer,cap=cap,thermal=thermal)
    ref=setup(dz,angle,reference=True,collision_scale=collision_scale,buffer=buffer,cap=cap,thermal=thermal)
    with h5py.File(ROOT/'data/illumination.h5') as h:v=h['source_polarization'][:]
    state=np.einsum('qnp,p->qn',s['initial'],v);inc=np.einsum('qnp,p->qn',ref['initial'],v)
    z=s['z'];zf=z+s['dz_cells']/2;ia=np.argmin(abs(zf-115100));ib=np.argmin(abs(zf-120800))
    incident=sum(flux(ref,inc))[ia];assert incident>0
    state/=np.sqrt(incident);inc/=np.sqrt(incident);v/=np.sqrt(incident)
    source=np.einsum('qnp,p->qn',s['source'],v)
    weights=np.array([s['dual'],s['dual'],s['dz_cells']])
    ch=np.cos(s['omega']*s['dt']/2)
    coll=.5*epsilon_0*c*ch*np.sum(s['physical_nu']*abs(state[6:9])**2*weights,axis=0)
    buffers=.5*epsilon_0*c*ch*(np.sum(s['sponge'][:3]*abs(state[:3])**2*weights,axis=0)+np.sum(s['sponge'][3:]*abs(state[3:6])**2*np.array([s['dz_cells'],s['dz_cells'],s['dual']]),axis=0)+np.sum(s['sponge'][:3]*abs(state[6:9])**2*weights,axis=0)+s['sponge'][0]*abs(state[9])**2*s['dual'])
    work=.5*epsilon_0*c*np.sum(np.real(state[:3].conj()*source[:3])*weights)
    balance=(work-coll.sum()-buffers.sum())/work
    assert abs(balance)<1e-7
    em,pf=flux(s,state);refl=sum(flux(s,state-inc))[ia];net=em+pf
    cut=(z>zf[ia])&(z<zf[ib]);loss=coll[cut].sum()
    density=state[9]*np.sqrt(epsilon_0*s['ne'][0]/(m_e*CS**2))
    divergence=1j*s['kx']*state[0]+(state[2]-np.roll(state[2],1))/s['dual']
    if not thermal:density=-epsilon_0/e*divergence
    mask=(z>s['center']-20)&(z<s['center']+10)
    stats=dict(frequency_hz=F,angle_deg=angle,azimuth_deg=0.,electron_temperature_K=TE,
        collision_scale=collision_scale,finest_dz_m=dz,grid_cells=s['n'],dt_s=s['dt']/c,
        incident_flux_W_m2=1.,reflected_fraction=-refl,transmitted_fraction=net[ib],
        physical_loss_between_planes=loss,flux_loss_between_planes=net[ia]-net[ib],
        boundary_balance_error=1+refl-net[ib]-loss,
        total_source_work_W_m2=work,global_balance_relative=balance,
        peak_density_fraction_per_sqrt_W_m2=float(np.max(abs(density[mask])/s['ne'][0,mask])),
        peak_Ez_V_m_per_sqrt_W_m2=float(abs(state[2,mask]).max()),
        center_m=s['center'],thermal_pressure_enabled=thermal,
        gauss_law_relative_error=float(np.linalg.norm((divergence+e*density/epsilon_0)[mask])/np.linalg.norm(e*density[mask]/epsilon_0)),
        harmonic_residual=s['harmonic_residual'],pressure_adjoint_error=s['pressure_adjoint_error'])
    stats.update(hf_port_budget(state,inc,ia,ib))
    stats.update(lower_EM_port_m=float(zf[ia]),upper_EM_port_m=float(zf[ib]))
    print('Solved',label,stats,'seconds',time.monotonic()-wall,flush=True)
    path=OUT/f'{label}.h5'
    with h5py.File(path,'w') as h:
        h.attrs.update(stats);h.attrs.update(model=('two-way linear warm electron fluid + Maxwell Yee' if thermal else 'cold Maxwell-current control')+', one Fourier x mode, 3 velocity components; no Landau damping',
            initialization='exact discrete continuous-wave solution, not transmitter switch-on',
            polarization='previous favorable cold-field polarization held fixed; not optimized for warm response',
            pressure_closure='-m cs2 grad(n1/n0), cs2=3 kB Te/me; prescribed stationary n0',
            gyro_interpolation='energy-conserving Yee interpolation of normalized currents; consistency checked by mesh refinement',
            convention='Re{phasor exp(i kx x-i omega t)}',repository='https://github.com/jvierine/fdtd')
        for key in ['z','dz_cells','ne','physical_nu','sponge','gyro']:
            h[key]=s[key]
        h['kx_rad_m']=s['kx'];h['state_phasor']=state;h['incident_reference_state']=inc;h['source_polarization']=v
        h['density_phasor_m3']=density;h['E_phasor_V_m']=state[:3];h['J_phasor_A_m2']=epsilon_0*c*s['wp']*state[6:9]
        h['EM_flux_W_m2']=em;h['pressure_flux_W_m2']=pf;h['collision_loss_per_cell_W_m2']=coll
        h['provenance/python_source']=Path(__file__).read_text();h['provenance/rust_source']=(ROOT/'src/main.rs').read_text();h['provenance/fluid_adapter']=(cold.FLUID/'src/maxwell_coupling.rs').read_text();h['provenance/yee_fields']=(REPO/'src/stratified.rs').read_text()
        h.attrs['source_sha256']=hashlib.sha256((Path(__file__).read_text()+(ROOT/'src/main.rs').read_text()).encode()).hexdigest()
        if native_cycles:
            cfg={key:(s[key].tolist() if isinstance(s[key],np.ndarray) else s[key]) for key in ['n','dz_cells','dt','omega','kx','gyro','wp','nu','sponge','sqrt_ne','beta']}
            pairs=lambda v:np.stack([v.real,v.imag],axis=-1).tolist()
            raw=OUT/f'{label}.bin';config=OUT/f'{label}.json'
            cfg.update(initial=pairs(state),source=pairs(source[:3]),steps=native_cycles*s['spp'],record_stride=max(1,s['spp']//32),output=str(raw))
            config.write_text(json.dumps(cfg,separators=(',',':')))
            print('Rust evolving',cfg['steps'],'steps',flush=True)
            result=subprocess.run([str(cold.TARGET/'release/arecibo-coupled'),str(config)],capture_output=True,text=True)
            print('Rust',result.stdout,result.stderr,flush=True);result.check_returncode();report=json.loads(result.stdout)
            rows=np.fromfile(raw,dtype='<f8').reshape(-1,1+8*s['n']);frames=rows[:,1:].reshape(-1,4,s['n'],2)
            h['time_s']=rows[:,0];h.create_dataset('E_q_time_real',data=frames[...,0],compression='gzip');h.create_dataset('E_q_time_quadrature',data=frames[...,1],compression='gzip')
            h.attrs['native_report']=json.dumps(report);config.unlink();raw.unlink()
    return stats

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dz',type=float,default=.02);p.add_argument('--collisions',type=float,default=1.);p.add_argument('--buffer',type=float,default=600.);p.add_argument('--cap',type=float,default=2.);p.add_argument('--angle',type=float,default=16.75);p.add_argument('--cycles',type=int,default=1);p.add_argument('--label',default='baseline');a=p.parse_args()
    run(a.dz,a.collisions,a.buffer,a.cap,a.angle,a.cycles,a.label)
