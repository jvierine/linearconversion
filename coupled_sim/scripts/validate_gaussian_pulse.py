"""Check the broadband operator against native Rust steps and a refined mesh."""
from pathlib import Path
import json,subprocess,argparse
import numpy as np,h5py
from scipy.constants import c,e,m_e,epsilon_0
from scipy.optimize import brentq
import coupled_arecibo as a
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/gaussian_pulse'
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--dz",type=float,default=.002);dz=parser.parse_args().dz
 with h5py.File(ROOT/'reference/coupled/minimum_incident.h5') as h:
  angle=h.attrs['angle_deg'];pol=h['source_polarization'][:]
 f0=5.34e6;w=2*np.pi*f0;g=e*a.P['magnetic_t']/m_e;t=np.deg2rad(a.P['field_angle_deg'])
 n=w*w*(w*w-g*g)/(w*w-(g*np.cos(t))**2)*m_e*epsilon_0/e**2
 bottom=brentq(lambda z:a.PROFILE(z)-n,117000,118100);top=brentq(lambda z:a.PROFILE(z)-n,118100,119000)
 cfg=dict(ordering='COLAMD',center=bottom,frequency_hz=f0,fixed_kx_m1=w/c*np.sin(np.deg2rad(angle)),fine_intervals=[(-180.,80.),(top-bottom-80.,top-bottom+180.)])
 base=a.setup(angle=angle,dz=dz,cap=.5,scenario=cfg);cfg['fixed_dt_m']=base['dt']
 with h5py.File(OUT/'validation.h5','w') as out:
  for offset in [-50000.,0.,50000.]:
   cfg['frequency_hz']=f0+offset
   s=a.setup(angle=angle,dz=dz,cap=.5,scenario=cfg)
   fine=a.setup(angle=angle,dz=dz/2,cap=.25,scenario=cfg)
   v=np.einsum('qnp,p->qn',s['initial'],pol);vf=np.einsum('qnp,p->qn',fine['initial'],pol)
   ztest=np.linspace(bottom-8,bottom-1,1401)
   low=np.interp(ztest,s['z'],v[9]);high=np.interp(ztest,fine['z'],vf[9])
   amp=np.linalg.norm(abs(low)-abs(high))/np.linalg.norm(abs(high))
   # Source plane snaps to the respective grid; remove only that common phase
   # when comparing spatial structures, not any height-dependent phase.
   shift=np.vdot(high,low);aligned=low*np.exp(-1j*np.angle(shift))
   complex_error=np.linalg.norm(aligned-high)/np.linalg.norm(high)
   pairs=lambda x:np.stack([x.real,x.imag],axis=-1).tolist()
   config={k:(s[k].tolist() if isinstance(s[k],np.ndarray) else s[k]) for k in ['n','dz_cells','dt','omega','kx','gyro','wp','nu','sponge','sqrt_ne','beta']}
   config.update(initial=pairs(v),source=pairs(np.einsum('qnp,p->qn',s['source'],pol)[:3]),steps=32,record_stride=32,output=str(OUT/'validation_native.bin'))
   path=OUT/'validation_native.json';path.write_text(json.dumps(config))
   run=subprocess.run([str(a.cold.TARGET/'release/arecibo-coupled'),str(path)],capture_output=True,text=True,check=True)
   report=json.loads(run.stdout);print('VALIDATION',offset,amp,complex_error,report,flush=True)
   assert amp<.05 and complex_error<.1,(amp,complex_error)
   out.require_group(str(offset)).attrs.update(amplitude_mesh_error=amp,complex_mesh_error=complex_error,**report)
   path.unlink();(OUT/'validation_native.bin').unlink()
  out['provenance/source']=Path(__file__).read_text()
if __name__=='__main__':main()
