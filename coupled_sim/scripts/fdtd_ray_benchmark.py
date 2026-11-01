"""Native Rust FDTD vs geometrical-optics/WKB reflection on measured Es profile.
Scalar B=nu=0 TE control isolates the approximation; magnetized collisional
case keeps event parameters. CW initialization is not a pulse turn-on test.
"""
from pathlib import Path
import json, subprocess, hashlib
import numpy as np
import h5py
from scipy.constants import c,e,m_e,epsilon_0
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from scipy.special import airy
import coupled_arecibo as a
from model_inputs import REPO,TARGET
from fdtd_only_control import magnetic_from_electric

OUT=a.ROOT.parent/'reference/coupled/fdtd_ray_benchmark.h5'
TMP=a.OUT/'ray_benchmark';TMP.mkdir(exist_ok=True,parents=True)
ANGLE=13.955755836943089

def native(s,state,source,name,cycles=3):
    pair=lambda x:np.stack([x.real,x.imag],axis=-1).tolist()
    raw=TMP/f'{name}.bin';cfgpath=TMP/f'{name}.json'
    cfg=dict(n=s['n'],dz=float(s['dz_cells'].min()),dz_cells=s['dz_cells'].tolist(),dt=s['dt'],omega=s['omega'],kx=s['kx'],gyro=s['gyro'].tolist(),wp=s['wp'].tolist(),nu=s['nu'].tolist(),sponge=s['sponge'].tolist(),initial=pair(state),source=pair(source[:3]),steps=cycles*s['spp'],record_start=0,record_stride=s['spp']//4,output=str(raw))
    cfgpath.write_text(json.dumps(cfg,separators=(',',':')))
    p=subprocess.run([str(TARGET/'release/stratified'),str(cfgpath)],capture_output=True,text=True,check=True)
    report=json.loads(p.stdout);rows=np.fromfile(raw,dtype='<f8').reshape(-1,1+6*s['n'])
    t=rows[:,0];v=rows[:,1:].reshape(-1,3,s['n'],2);E=(v[...,0]+1j*v[...,1])*np.exp(1j*a.W*t[:,None,None])
    ph=E.mean(axis=0);H=magnetic_from_electric(ph,s['dz_cells'],s['kx'],s['dt'],s['omega'],s['sponge'])
    report['max_all_recorded_E_error']=float(np.max(abs(E-state[:3]))/np.max(abs(state[:3])))
    raw.unlink();cfgpath.unlink()
    print(name,report,flush=True)
    return np.concatenate([ph,H]),report

def scalar(dz):
    cfg=dict(magnetic_t=0.,ordering='COLAMD')
    s=a.setup(dz=dz,cap=dz,thermal=False,collision_scale=0.,angle=ANGLE,scenario=cfg)
    r=a.setup(dz=dz,cap=dz,thermal=False,collision_scale=0.,angle=ANGLE,scenario=cfg,reference=True)
    state=s['initial'][:,:,1];inc=r['initial'][:,:,1];src=s['source'][:,:,1]
    z=s['z'];zf=z+s['dz_cells']/2;lo=np.argmin(abs(zf-115100));hi=np.argmin(abs(zf-120800))
    field,report=native(s,state,src,f'scalar_{dz:g}')
    budget=a.hf_port_budget(field,inc[:6],lo,hi)
    k0=a.W/c;kx=s['kx'];fac=e*e/(epsilon_0*m_e*a.W*a.W)
    Q=lambda zz:k0*k0*(1-a.PROFILE(zz)*fac)-kx*kx
    zt=brentq(Q,117000.,118100.);ell=(k0*k0*a.PROFILE(zt,1)*fac)**(-1/3)
    kb=np.sqrt(Q(115300.));amp=abs(inc[1,lo]);q=Q(z)
    zz=np.linspace(115300.,zt,50001);kk=np.sqrt(np.maximum(Q(zz),0));S=cumulative_trapezoid(kk,zz,initial=0);phase=np.interp(z,zz,S[-1]-S)
    wkb=np.full(len(z),np.nan);valid=(z<zt)&(z>=115300.)
    wkb[valid]=2*np.sqrt(kb/np.sqrt(q[valid]))*abs(np.sin(phase[valid]+np.pi/4))
    near=2*np.sqrt(np.pi*kb*ell)*abs(airy((z-zt)/ell)[0])
    measured=abs(field[1])/amp
    stats=dict(budget,**report,turn_height_m=zt,airy_length_m=ell,dz_m=dz)
    for start,end in [(1,3),(3,10),(10,30)]:
      m=((zt-z)/ell>=start)&((zt-z)/ell<end)&valid
      stats[f'WKB_field_NRMSE_{start}_{end}_airy']=float(np.linalg.norm(measured[m]-wkb[m])/np.linalg.norm(measured[m]))
    m=abs(z-zt)<ell
    stats['Airy_field_NRMSE_within_one_length']=float(np.linalg.norm(measured[m]-near[m])/np.linalg.norm(measured[m]))
    stats['field_peak_relative_to_turn_m']=float(z[np.argmax(measured*(abs(z-zt)<3*ell))]-zt)
    print('scalar stats',stats,flush=True)
    return dict(z_m=z,ne_m3=s['ne'][0],E_over_incident=measured,WKB_over_incident=wkb,Airy_over_incident=near,WKB_parameter=abs(k0*k0*a.PROFILE(z,1)*fac)/(2*np.maximum(abs(q),1e-30)**1.5)),stats

def magnetized(dz,advance=True):
    s=a.setup(dz=dz,cap=.5,thermal=False,angle=ANGLE,scenario={'ordering':'COLAMD'})
    r=a.setup(dz=dz,cap=.5,thermal=False,angle=ANGLE,scenario={'ordering':'COLAMD'},reference=True)
    with h5py.File(a.ROOT.parent/'reference/coupled/minimum_incident.h5') as h:
      # Archived optimum source polarization, preserved with the measured-profile solution.
      v=h['source_polarization'][:]
    state=np.einsum('qnp,p->qn',s['initial'],v);inc=np.einsum('qnp,p->qn',r['initial'],v);src=np.einsum('qnp,p->qn',s['source'],v)
    if advance:field,report=native(s,state,src,f'magnetized_{dz:g}',cycles=1)
    else:field=state[:6];report={'harmonic_residual':s['harmonic_residual'],'method':'exact harmonic solution of the discrete FDTD timestep'}
    z=s['z'];zf=z+s['dz_cells']/2;lo=np.argmin(abs(zf-115100));hi=np.argmin(abs(zf-120800))
    budget=a.hf_port_budget(field,inc[:6],lo,hi)
    return dict(z_m=zf,Ez_over_incident=abs(field[2])/np.linalg.norm(inc[:3,lo])),dict(**budget,**report,resonance_height_m=s['center'],dz_m=dz)

def main():
    subprocess.run(['cargo','build','--release','--locked','--manifest-path',str(REPO/'Cargo.toml'),'--bin','stratified','--target-dir',str(TARGET)],check=True)
    with h5py.File(OUT,'w') as h:
      h.attrs.update(frequency_hz=a.F,vacuum_incidence_from_vertical_deg=ANGLE,scope='Local planar 114-122 km measured Es profile; one horizontal Fourier mode; no ground path or finite beam. Scalar control B=nu=0; event cold plasma retains B and collisions. CW initialization, native time stepping.',fdtd_revision='346135fe3ee5eba8e4df89fc7c3e5678fa24f69c',native_executable_sha256=hashlib.sha256((TARGET/'release/stratified').read_bytes()).hexdigest())
      h['provenance/source']=Path(__file__).read_text()
      for name,fun,dz in [('scalar_coarse',scalar,.5),('scalar_fine',scalar,.25),('magnetized_coarse',magnetized,.05),('magnetized_fine',magnetized,.025),('magnetized_refined',lambda dz:magnetized(dz,False),.005),('magnetized_finest',lambda dz:magnetized(dz,False),.0025)]:
        arrays,stats=fun(dz);g=h.create_group(name);g.attrs.update(stats)
        for k,v in arrays.items():g.create_dataset(k,data=v,compression='gzip')
        h.flush()
    print(OUT,flush=True)
if __name__=='__main__':main()
