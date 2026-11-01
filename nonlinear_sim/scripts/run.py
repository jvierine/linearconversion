"""Prepare the measured profile, solve the independent linearized ten-moment
boundary problem, then evolve the FULL nonlinear Rust equations in real x,z.
The harmonic state is an initial condition, not a nonlinear steady solution.
Scientific data and diagnostics are HDF5; temporary JSON is solver configuration.
"""
from pathlib import Path
import argparse,json,subprocess,hashlib
import numpy as np
import h5py
from scipy.constants import c,e,m_e,epsilon_0,k as kb
from scipy.sparse import diags,bmat,eye,csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicHermiteSpline
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'coupled_sim/data'
PAIRS=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
MI={p:4+i for i,p in enumerate(PAIRS)};MI.update({(j,i):v for (i,j),v in list(MI.items())})
FREQ=5.340028553893307e6;OMEGA=2*np.pi*FREQ;LENGTH=c/OMEGA
V=np.sqrt(kb*300/m_e);BETA=V/c;NREF=epsilon_0*m_e*OMEGA**2/e**2;EREF=m_e*V*OMEGA/e

def setup(dz=.04,nx=8,*,background_only=False,outer_spacing=10.,grading=.04):
 with h5py.File(DATA/'smooth_observed_layer_1525453430.h5') as h:
  profile=CubicHermiteSpline(h['model_height_m'][:],h['model_ne_m3'][:],h['model_dne_dz_m4'][:])
 with h5py.File(DATA/'background_prior.h5') as h:
  bt=float(h['magnetic_t'][()]);angle=float(h['field_angle_deg'][()]);p={k:h['neutrals/'+k][:] for k in ['z','N2_m3','O2_m3','O_m3']}
 from scipy.optimize import brentq
 gyro=e*bt/(m_e*OMEGA)*np.array([np.sin(np.deg2rad(angle)),0,-np.cos(np.deg2rad(angle))])
 target=NREF*(1-np.dot(gyro,gyro))/(1-gyro[2]**2)
 zr=brentq(lambda z:profile(z)-target,117000,118100)
 # Resolve the conversion region and several metres below Bragg. Outer
 # grid grading is tested separately; physical altitude extent stays 8 km.
 zz=[zr]
 for direction,bound in [(1,122000.),(-1,114000.)]:
  z=zr
  while (z-bound)*direction<0:
   outside=max(zr-12-z,z-(zr+5),0.)
   z+=direction*min(outer_spacing,dz+grading*outside);zz.append(z)
 zsi=np.array(sorted(zz));z=zsi/LENGTH;n=len(z)
 sample=np.clip(zsi,115300,120800);n0=profile(sample)/NREF
 inv=np.zeros(n);inv[1:-1]=1/(z[2:]-z[:-2]);D=diags(inv)@(diags(np.ones(n-1),1)-diags(np.ones(n-1),-1))
 eq=-BETA*(D@n0)/n0
 tev=kb*300/e;ncm=n0*NREF*1e-6
 ei=2.91e-6*ncm*(23-np.log(np.sqrt(ncm)*tev**-1.5))/tev**1.5
 n2,o2,o=[np.exp(np.interp(sample,p['z'],np.log(p[k]))) for k in ['N2_m3','O2_m3','O_m3']]
 t=300.;en=2.33e-17*n2*(1-1.21e-4*t)*t+1.82e-16*o2*(1+.036*np.sqrt(t))*np.sqrt(t)+8.9e-17*o*(1+5.7e-4*t)*np.sqrt(t)
 nu=(ei+en)/OMEGA
 # Electric-conductivity absorber, with a consistent artificial charge flux.
 # It is not a physical collision term and is excluded from physical diagnostics.
 depth=np.maximum((114700-zsi)/700,(zsi-121300)/700).clip(0,1)
 sig=40*LENGTH/700*depth**3
 kx=np.sin(np.deg2rad(16.75));dx=2*np.pi/(kx*nx)
 kd=(8*np.sin(kx*dx)-np.sin(2*kx*dx))/(6*dx)
 if background_only:
  source=np.zeros((16,n),complex)
  iz=np.argmin(abs(zsi-114850))
  # Integrated source-sheet strength is independent of the local grid width.
  width=(z[iz+1]-z[iz-1])/2
  source[10,iz]=1./width;source[11,iz]=(.11986+1.01797j)/width
  return dict(z=z,n0=n0,eq_ez=eq,nu=nu,sponge=sig,beta=BETA,gyro=gyro,dx=dx,nx=nx),None,source,dict(z_m=zsi,kx_per_m=kx/LENGTH,kx_discrete_per_m=kd/LENGTH,resonance_m=zr,nu_ei_s=ei,nu_en_s=en,source_height_m=zsi[iz])
 I=eye(n,format='csr');Z=csr_matrix((n,n));A=[[Z.copy() for _ in range(16)] for _ in range(16)]
 def add(i,j,v):A[i][j]=A[i][j]+v
 deriv=[1j*kd*I,Z,D];L=np.array([[0,-gyro[2],gyro[1]],[gyro[2],0,-gyro[0]],[-gyro[1],gyro[0],0]])
 for a in [0,2]:
  add(0,1+a,-BETA*deriv[a]);add(0,10+a,BETA*deriv[a]@diags(sig))
 for i in range(3):
  for a in [0,2]:add(1+i,MI[i,a],-BETA*deriv[a])
  add(1+i,10+i,-diags(n0));add(1+i,1+i,-diags(nu+sig))
  if i==2:add(1+i,0,-diags(eq))
  for j in range(3):add(1+i,1+j,L[i,j]*I)
 for k,(i,j) in enumerate(PAIRS):
  row=4+k
  for a in [0,2]:
   if i==j:add(row,1+a,-BETA*deriv[a])
   if j==a:add(row,1+i,-BETA*deriv[a])
   if i==a:add(row,1+j,-BETA*deriv[a])
  if i==2:add(row,1+j,-diags(eq))
  if j==2:add(row,1+i,-diags(eq))
  for a in range(3):add(row,MI[a,j],L[i,a]*I);add(row,MI[i,a],L[j,a]*I)
  add(row,row,-diags(nu+sig))
  if i==j:add(row,0,diags(nu+sig))
 curl=[[Z,-D,Z],[D,Z,-1j*kd*I],[Z,1j*kd*I,Z]]
 for i in range(3):
  add(10+i,1+i,I);add(10+i,10+i,-diags(sig))
  for j in range(3):add(10+i,13+j,curl[i][j]);add(13+i,10+j,-curl[i][j])
 mat=bmat(A,format='csc');source=np.zeros((16,n),complex)
 iz=np.argmin(abs(zsi-114850));source[10,iz]=1.;source[11,iz]=.11986+1.01797j
 # Interleaving variables keeps the direct factorization narrow-band.
 op=-1j*eye(16*n,format='csc')-mat;perm=np.arange(16*n).reshape(16,n).T.ravel()
 sol=np.empty(16*n,complex);sol[perm]=spsolve(op[perm,:][:,perm],source.ravel()[perm],permc_spec='NATURAL')
 sol=sol.reshape(16,n);res=np.linalg.norm(op@sol.ravel()-source.ravel())/np.linalg.norm(source)
 assert res<1e-8,res
 # Equilibrium ions satisfy Poisson, including the tiny ambipolar correction.
 ni=n0+BETA*(D@eq)
 assert ni.min()>0
 return dict(z=z,n0=n0,eq_ez=eq,nu=nu,sponge=sig,beta=BETA,gyro=gyro,dx=dx,nx=nx),sol,source,dict(z_m=zsi,ion_density_m3=ni*NREF,linear_residual=res,kx_per_m=kx/LENGTH,kx_discrete_per_m=kd/LENGTH,resonance_m=zr,nu_ei_s=ei,nu_en_s=en)

def run(args):
 out=ROOT/'output';out.mkdir(exist_ok=True)
 cfg,sol,source,meta=setup(args.dz,args.nx)
 amplitude=args.amplitude/np.max(abs(sol[0])/cfg['n0'])
 sol*=amplitude;source*=amplitude
 if args.unforced:sol[:]=0;source[:]=0
 dt=.3*min(np.min(np.diff(cfg['z'])),cfg['dx'])
 steps=int(np.ceil(args.cycles*2*np.pi/dt));dt=args.cycles*2*np.pi/steps
 cfg.update(dt=dt,steps=steps,stride=max(1,steps//16),initial=np.stack((sol.real,sol.imag),axis=-1),source=np.stack((source.real,source.imag),axis=-1),linear=args.linear,initial_pressure_lift=not args.linear,output=str(out/(args.label+'.h5')),omega_si=OMEGA,length_si=LENGTH,density_si=NREF,electric_si=EREF,altitude_origin_m=0.)
 config=out/(args.label+'.json');config.write_text(json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in cfg.items()}))
 print('RUN',args.label,'nz',len(cfg['z']),'nx',args.nx,'steps',steps,'linear residual',meta['linear_residual'],flush=True)
 subprocess.run([str(ROOT/'target/release/nonlinear-conversion'),str(config)],check=True)
 with h5py.File(cfg['output'],'a') as h:
  h.attrs['cargo_lock_sha256']=hashlib.sha256((ROOT/'Cargo.lock').read_bytes()).hexdigest()
  h.attrs['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  h.attrs['executable_sha256']=hashlib.sha256((ROOT/'target/release/nonlinear-conversion').read_bytes()).hexdigest()
  for input_path in sorted(DATA.glob('*.h5')):
   h.attrs['input_sha256_'+input_path.name]=hashlib.sha256(input_path.read_bytes()).hexdigest()
  h['initial_phasor_normalized']=sol;h['source_phasor_normalized']=source
  for k,v in meta.items():h[k]=v
  h.attrs['pressure_closure']='Full symmetric pressure tensor; zero heat flux'
  h.attrs['collision_operator']='Number-conserving BGK, fixed 300 K stationary bath; rate nu_ei+nu_en'
  h.attrs['initialization']='Harmonic linearized density, velocity and central pressure; full runs add n*u*u to raw second moments. This is not a nonlinear steady state.'
  h.attrs['initial_pressure_lift']=not args.linear
  h.attrs['requested_initial_peak_dn_over_n']=args.amplitude
  h.attrs['spatial_discretization']='Fourth order periodic x; centered conservative dual-cell z; RK4; nonuniform z'
 config.unlink()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dz',type=float,default=.04);p.add_argument('--nx',type=int,default=8);p.add_argument('--cycles',type=float,default=1);p.add_argument('--amplitude',type=float,default=1e-4);p.add_argument('--linear',action='store_true');p.add_argument('--unforced',action='store_true');p.add_argument('--label',default='nonlinear_weak');run(p.parse_args())
