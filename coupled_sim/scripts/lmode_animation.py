"""Resolved two-sided Es conversion, O-input angle scan and native CW movies.

Same linear Yee + electron-fluid equations as coupled_arecibo. Multiple fine
intervals resolve both resonance layers without resolving the overdense gap at
millimetre spacing. No synthetic upward wave is prescribed. All amplitudes scale
linearly; incident vertical flux is the normalization used for conversion scans.
"""
from pathlib import Path
import argparse, json, subprocess, time, gc
import numpy as np
import h5py
from scipy.constants import c, epsilon_0, e, m_e
from scipy.optimize import brentq, minimize_scalar
import coupled_arecibo as a
from hf_absorption_profile import modes

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/lmode_animation'
OUT.mkdir(parents=True,exist_ok=True)
G=e*float(a.P['magnetic_t'])/m_e
TILT=np.deg2rad(float(a.P['field_angle_deg']))
D=a.W**2*(a.W**2-G**2)/(a.W**2-(G*np.cos(TILT))**2)
RES=lambda z:float(a.PROFILE(z))*e*e/(m_e*epsilon_0)-D
BOTTOM=brentq(RES,117000,118100)
TOP=brentq(RES,118100,119000)
SPITZE=np.rad2deg(np.arcsin(np.sqrt(G/(a.W+G))*np.sin(TILT)))

def solve(angle,dz=.02,cap=.5,save=None,native=False):
    wall=time.monotonic()
    cfg={'ordering':'COLAMD','fine_intervals':[(-60.,12.),(TOP-BOTTOM-12.,TOP-BOTTOM+60.)]}
    s=a.setup(angle=angle,dz=dz,cap=cap,scenario=cfg)
    r=a.setup(angle=angle,dz=dz,cap=cap,scenario=cfg,reference=True)
    z=s['z'];zf=z+s['dz_cells']/2
    lo=np.argmin(abs(zf-115100));hi=np.argmin(abs(zf-120900))
    v,err=modes(s,r,lo)['O']
    inc=np.einsum('qnp,p->qn',r['initial'],v)
    v/=np.sqrt(a.electromagnetic_flux(inc)[lo])
    state=np.einsum('qnp,p->qn',s['initial'],v)
    inc=np.einsum('qnp,p->qn',r['initial'],v)
    em,pf=a.flux(s,state)
    dn=state[9]*np.sqrt(epsilon_0*s['ne'][0]/(m_e*a.CS**2))
    budget=a.hf_port_budget(state,inc,lo,hi)
    mask=(z>zf[lo])&(z<zf[hi])
    heat=.5*epsilon_0*c*np.cos(s['omega']*s['dt']/2)*np.sum(s['physical_nu']*abs(state[6:9])**2*np.array([s['dual'],s['dual'],s['dz_cells']]),axis=0)
    topwin=(zf>TOP+3)&(zf<TOP+6)
    botwin=(zf>BOTTOM-6)&(zf<BOTTOM-3)
    # Signed outward pressure energy flux, averaged over 3--6 m to suppress
    # interference; physical collisional damping is retained at these planes.
    up=np.average(pf[topwin],weights=s['dz_cells'][topwin])
    down=-np.average(pf[botwin],weights=s['dz_cells'][botwin])
    stats=dict(angle_deg=float(angle),dz_m=dz,cap_m=cap,grid_cells=s['n'],
               upward_pressure_fraction=float(up),downward_pressure_fraction=float(down),
               bottom_resonance_m=BOTTOM,top_resonance_m=TOP,
               harmonic_residual=s['harmonic_residual'],mode_error=err,
               heating_fraction=heat[mask].sum(),**budget)
    stats['energy_residual']=1-budget['reflected_EM_fraction']-budget['transmitted_EM_fraction']-heat[mask].sum()-(pf[hi]-pf[lo])
    print(json.dumps(stats), 'seconds',round(time.monotonic()-wall,2),flush=True)
    assert abs(stats['energy_residual'])<2e-4,stats
    if save:
        path=OUT/(save+'.h5')
        with h5py.File(path,'w') as h:
            h.attrs.update(stats)
            h.attrs.update(frequency_hz=a.F,spitze_magnitude_deg=SPITZE,
                model='Linear coupled Rust Yee FDTD + electron fluid; fixed ions; measured smooth Es; 300K; physical electron-ion and electron-neutral collisions; no Landau damping',
                initialization='Discrete continuous-wave solution; sustained oscillations, not switch-on',
                normalization='Phasors per sqrt(W/m2) upward incident EM flux',
                angle_convention='Vacuum-equivalent signed angle from vertical, kx positive along B horizontal component, Bz negative',
                polarization='Incident local positive-flux ordinary eigenmode, not optimized arbitrary polarization')
            arrays=dict(z_m=z,z_faces_m=zf,ne_m3=s['ne'][0],density_phasor_m3=dn,
                E_phasor_V_m=state[:3],state_phasor=state,incident_state=inc,
                pressure_flux_W_m2=pf,EM_flux_W_m2=em,collision_loss_per_cell_W_m2=heat,
                physical_nu_s1=s['physical_nu']*c,source_polarization=v,
                dz_cells_m=s['dz_cells'],kx_rad_m=s['kx'])
            for k,value in arrays.items():h.create_dataset(k,data=value,compression='gzip' if np.ndim(value) else None)
            h['provenance/analysis_source']=Path(__file__).read_text()
            h['provenance/coupled_driver_source']=Path(a.__file__).read_text()
            h['provenance/rust_driver_source']=(a.ROOT/'src/main.rs').read_text()
            h['provenance/coupled_timestep_source']=(a.ROOT/'src/lib.rs').read_text()
            h['provenance/Cargo.lock']=(a.ROOT/'Cargo.lock').read_text()
            if native:
                pairs=lambda x:np.stack([x.real,x.imag],axis=-1).tolist()
                config={k:(s[k].tolist() if isinstance(s[k],np.ndarray) else s[k]) for k in ['n','dz_cells','dt','omega','kx','gyro','wp','nu','sponge','sqrt_ne','beta']}
                source=np.einsum('qnp,p->qn',s['source'],v)
                raw=OUT/(save+'.bin');jp=OUT/(save+'.json')
                config.update(initial=pairs(state),source=pairs(source[:3]),steps=s['spp'],record_stride=max(1,s['spp']//96),output=str(raw))
                jp.write_text(json.dumps(config,separators=(',',':')))
                print('NATIVE',save,'cells',s['n'],'steps',s['spp'],flush=True)
                run=subprocess.run([str(a.cold.TARGET/'release/arecibo-coupled'),str(jp)],capture_output=True,text=True)
                print(run.stdout,run.stderr,flush=True);run.check_returncode()
                rows=np.fromfile(raw,dtype='<f8').reshape(-1,1+8*s['n'])
                frames=rows[:,1:].reshape(-1,4,s['n'],2)
                # Save resolved fields around both conversion layers and full
                # phasors. Native recording retained at all mesh cells here.
                h['native/time_s']=rows[:,0]
                h.create_dataset('native/E_q_real',data=frames[...,0],compression='gzip')
                h.create_dataset('native/E_q_imag',data=frames[...,1],compression='gzip')
                h.attrs['native_report']=run.stdout.strip()
                jp.unlink();raw.unlink()
    del s,r;gc.collect()
    return stats

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['scan','case','record']);p.add_argument('--angle',type=float);p.add_argument('--dz',type=float,default=.005);p.add_argument('--cap',type=float,default=.25);p.add_argument('--label',default='case');p.add_argument('--native',action='store_true');args=p.parse_args()
    if args.stage=='case':solve(args.angle,args.dz,args.cap,args.label,args.native);return
    if args.stage=='record':
        with h5py.File(OUT/'scan.h5') as h:
            up=float(h['angle_deg'][np.argmax(h['upward_pressure_fraction'][:])])
            down=float(h['angle_deg'][np.argmax(h['downward_pressure_fraction'][:])])
        for label,angle in [('topside_optimum',up),('bottomside_optimum',down),('topside_spitze',-SPITZE)]:
            solve(angle,args.dz,args.cap,label,True)
            fine={'topside_optimum':'topside_refined','bottomside_optimum':'bottomside_refined','topside_spitze':'spitze_refined'}[label]
            solve(angle,args.dz/2,args.cap/2,fine,False)
        return
    rows=[]
    def calc(theta):
        s=solve(theta,args.dz,args.cap);rows.append(s)
        with h5py.File(OUT/'scan.h5','w') as h:
            for k in s:h[k]=[r[k] for r in rows]
            h.attrs.update(spitze_magnitude_deg=SPITZE,metric='Maximum outward pressure flux averaged 3--6 m from each resonance, per incident vertical EM flux. O-mode input, magnetic meridian only.')
        return s
    for ang in [-24,-22,-20,-18,-SPITZE,-15,-13,-11,-8,0,10,12,14,SPITZE,19,22,24]:calc(ang)
    for metric,side in [('upward_pressure_fraction',-1),('downward_pressure_fraction',1)]:
        candidates=[r for r in rows if r['angle_deg']*side>0]
        best=max(candidates,key=lambda r:r[metric]);angle=best['angle_deg']
        opt=minimize_scalar(lambda th:-calc(th)[metric],bounds=(angle-2,angle+2),method='bounded',options={'xatol':.03})
        best=calc(opt.x)
        print('OPTIMUM',metric,best,flush=True)
    with h5py.File(OUT/'scan.h5','a') as h:h['provenance/source']=Path(__file__).read_text()

if __name__=='__main__':main()
