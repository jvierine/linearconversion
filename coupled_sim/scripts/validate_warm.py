"""Independent analytic periodic wave tests of the coupled Rust updates."""
import json,subprocess
import numpy as np
import h5py
from scipy.constants import c,e,m_e,epsilon_0,k as kb
from coupled_arecibo import OUT,REPO,ROOT,cold

def run():
    records=[]
    for mode in ['langmuir','electromagnetic']:
        for n in [32,64,128]:
            ne=3.5e11;wp=np.sqrt(ne*e*e/(m_e*epsilon_0))/c
            beta=np.sqrt(3*kb*300/m_e)/c
            length=c/(2*430e6) if mode=='langmuir' else 100.
            dz=length/n;dt=.4*dz;k=2*np.pi/length;kd=2*np.sin(k*dz/2)/dz
            wd=np.sqrt(wp*wp+(beta*kd)**2) if mode=='langmuir' else np.sqrt(wp*wp+kd*kd)
            om=2*np.arcsin(.5*dt*wd)/dt
            continuum=np.sqrt(wp*wp+(beta*k)**2) if mode=='langmuir' else np.sqrt(wp*wp+k*k)
            node=np.exp(1j*k*np.arange(n)*dz);face=node*np.exp(.5j*k*dz)
            state=np.zeros((10,n),complex)
            if mode=='langmuir':
                state[2]=face;state[8]=1j*wd/wp*face;state[9]=-1j*beta*kd/wp*node
            else:
                state[0]=node;state[4]=kd/wd*face;state[6]=1j*wp/wd*node
            pair=lambda x:np.stack([x.real,x.imag],axis=-1).tolist()
            steps=int(np.ceil(4*2*np.pi/(om*dt)))
            cfg=dict(n=n,dz_cells=[dz]*n,dt=dt,omega=om,kx=0.,gyro=[0.,0.,0.],wp=np.full((3,n),wp).tolist(),nu=np.zeros((3,n)).tolist(),sponge=np.zeros((6,n)).tolist(),sqrt_ne=np.full((3,n),np.sqrt(ne)).tolist(),beta=beta,initial=pair(state),source=pair(np.zeros((3,n),complex)),steps=steps,record_stride=steps,output=str(OUT/'validation.bin'))
            path=OUT/'validation.json';path.write_text(json.dumps(cfg))
            result=subprocess.run([str(cold.TARGET/'release/arecibo-coupled'),str(path)],check=True,capture_output=True,text=True)
            report=json.loads(result.stdout);err=abs(om/continuum-1)
            records.append([int(mode=='electromagnetic'),n,err,report['relative_max_e_error'],report['relative_max_q_error']])
            print(mode,n,err,report,flush=True)
            path.unlink();(OUT/'validation.bin').unlink()
    with h5py.File(OUT/'analytic_validation.h5','w') as h:
        h['results']=records;h['results'].attrs['columns']='electromagnetic_mode,cells,dispersion_relative_error,evolution_E_error,evolution_q_error'
        h.attrs['provenance']='hf_fdtd/examples/validate_warm.py; analytic Yee electromagnetic and warm Langmuir eigenmodes, 4 periods, zero collisions, no source'
    for offset in [0,3]:
        assert records[offset+2][2]<records[offset][2]/10
    assert np.max(np.array(records)[:,3:])<1e-8

if __name__=='__main__':run()
