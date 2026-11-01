"""Real total-density scalar fluid/Maxwell run on the measured Arecibo profile.
Start either at equilibrium with a ramped source, or at a weak harmonic state.
No tensor closure or kinetic Landau damping is used.
"""
import argparse,json,subprocess,hashlib
from pathlib import Path
import numpy as np
import h5py
from scipy.sparse import eye,diags,bmat,csr_matrix
from scipy.sparse.linalg import spsolve
from run import setup,ROOT,BETA,NREF,OMEGA,LENGTH,EREF

def prepare(a):
 cfg,_,oldsource,meta=setup(a.dz,a.nx,background_only=True,outer_spacing=a.outer_spacing,grading=a.grading);cfg['eq_ez']*=3
 n=len(cfg['z']);inv=np.zeros(n);inv[1:-1]=1/(cfg['z'][2:]-cfg['z'][:-2])
 D=diags(inv)@(diags(np.ones(n-1),1)-diags(np.ones(n-1),-1));I=eye(n,format='csr');Z=csr_matrix((n,n));A=[[Z.copy() for _ in range(10)] for _ in range(10)]
 def add(i,j,v):A[i][j]=A[i][j]+v
 kd=meta['kx_discrete_per_m']*LENGTH;deriv=[1j*kd*I,Z,D];g=cfg['gyro'];L=np.array([[0,-g[2],g[1]],[g[2],0,-g[0]],[-g[1],g[0],0]])
 for j in [0,2]:
  add(0,1+j,-BETA*deriv[j]);add(0,4+j,BETA*deriv[j]@diags(cfg['sponge']))
 for i in range(3):
  add(1+i,0,-3*BETA*deriv[i]);add(1+i,4+i,-diags(cfg['n0']));add(1+i,1+i,-diags(cfg['nu']+cfg['sponge']))
  if i==2:add(1+i,0,-diags(cfg['eq_ez']))
  for j in range(3):add(1+i,1+j,L[i,j]*I)
 curl=[[Z,-D,Z],[D,Z,-1j*kd*I],[Z,1j*kd*I,Z]]
 for i in range(3):
  add(4+i,1+i,I);add(4+i,4+i,-diags(cfg['sponge']))
  for j in range(3):add(4+i,7+j,curl[i][j]);add(7+i,4+j,-curl[i][j])
 op=-1j*eye(10*n,format='csc')-bmat(A,format='csc');src=np.vstack([oldsource[:4],oldsource[10:]])
 perm=np.arange(10*n).reshape(10,n).T.ravel();sol=np.empty(10*n,complex);sol[perm]=spsolve(op[perm,:][:,perm],src.ravel()[perm],permc_spec='NATURAL');sol=sol.reshape(10,n)
 residual=np.linalg.norm(op@sol.ravel()-src.ravel())/np.linalg.norm(src);assert residual<1e-8
 unit_solution=sol.copy()
 scale=a.drive_scale if a.drive_scale is not None else a.amplitude/np.max(abs(sol[0])/cfg['n0']);sol*=scale;src*=scale
 initial=np.zeros((16,n),complex);source=initial.copy();indices=[0,1,2,3,10,11,12,13,14,15];initial[indices]=sol if a.start=='harmonic' else 0.;source[indices]=src
 dt=.3*min(np.diff(cfg['z']).min(),cfg['dx']);steps=int(np.ceil(a.cycles*2*np.pi/dt));dt=a.cycles*2*np.pi/steps
 out=ROOT/'output';out.mkdir(exist_ok=True);path=out/(a.label+'.h5')
 cfg.update(dt=dt,steps=steps,stride=max(1,steps//a.frames),ramp_time=a.ramp_cycles*2*np.pi,initial=np.stack([initial.real,initial.imag],axis=-1),source=np.stack([source.real,source.imag],axis=-1),linear=a.linear,output=str(path),omega_si=OMEGA,length_si=LENGTH,density_si=NREF,electric_si=EREF,altitude_origin_m=0.)
 return cfg,sol,src,meta,D,residual,scale,unit_solution

def main():
 a=arguments();cfg,sol,src,meta,D,residual,scale,unit_solution=prepare(a)
 path=Path(cfg['output']);out=path.parent
 config=out/(a.label+'.json');config.write_text(json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in cfg.items()}))
 subprocess.run([str(ROOT/'target/release/scalar_total'),str(config)],check=True)
 with h5py.File(path,'a') as h:
  h['equilibrium_density_m3']=cfg['n0']*NREF;h['z_m']=meta['z_m'];h['x_m']=np.arange(a.nx)*cfg['dx']*LENGTH;h['harmonic_reference_normalized']=sol;h['source_phasor_normalized']=src
  h['initial_phasor_normalized']=sol if a.start=='harmonic' else np.zeros_like(sol)
  h['unit_drive_harmonic_normalized']=unit_solution
  for key,value in meta.items():
   if key!='z_m':h[key]=value
  h['equilibrium_Ez_V_m']=cfg['eq_ez']*EREF
  # Ions support the static field; they have no dynamic current.
  h['ion_density_m3']=(cfg['n0']+BETA*(D@cfg['eq_ez']))*NREF
  h.attrs.update(pressure_closure='p_eff=3 n_e kB T_ref; constant T_ref=300 K; total density and momentum evolved',state_columns='n_total,nu_x,nu_y,nu_z,E_x,E_y,E_z,H_x,H_y,H_z (normalized); fields relative to equilibrium',linearized=int(a.linear),harmonic_initial_residual=residual,initialization=a.start,drive_scale=scale,requested_harmonic_peak_dn_over_n=a.amplitude,ramp_cycles=a.ramp_cycles,outer_spacing_m=a.outer_spacing,grid_grading=a.grading,scope='Real x-z total-density evolution; not an event RCS or loss convergence study',driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  h['provenance/rust_source']=(ROOT/'src/scalar.rs').read_text()
  h.attrs['executable_sha256']=hashlib.sha256((ROOT/'target/release/scalar_total').read_bytes()).hexdigest()
  h.attrs['cargo_lock_sha256']=hashlib.sha256((ROOT/'Cargo.lock').read_bytes()).hexdigest()
 config.unlink();print(path,flush=True)

def arguments():
 p=argparse.ArgumentParser()
 p.add_argument('--dz',type=float,default=.04)
 p.add_argument('--nx',type=int,default=8)
 p.add_argument('--cycles',type=float,default=1.)
 p.add_argument('--amplitude',type=float,default=1e-4,help='Peak fractional amplitude of the harmonic reference, not the starting density when --start equilibrium')
 p.add_argument('--drive-scale',type=float,help='Fixed integrated current-sheet scale, for comparing grids without amplitude renormalization')
 p.add_argument('--linear',action='store_true')
 p.add_argument('--label',default='scalar_total')
 p.add_argument('--start',choices=['equilibrium','harmonic'],default='harmonic')
 p.add_argument('--ramp-cycles',type=float,default=0.)
 p.add_argument('--frames',type=int,default=32)
 p.add_argument('--outer-spacing',type=float,default=10.)
 p.add_argument('--grading',type=float,default=.04)
 a=p.parse_args()
 if a.ramp_cycles>0 and a.start!='equilibrium':p.error('A source ramp requires --start equilibrium')
 if min(a.dz,a.nx,a.cycles,a.amplitude,a.frames,a.outer_spacing,a.grading)<=0 or a.ramp_cycles<0:p.error('Positive grid, amplitude, duration, and frame values are required')
 return a
if __name__=='__main__':main()
