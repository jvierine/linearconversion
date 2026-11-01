"""Physical RF heating, outgoing-port loss, and incident O/X comparisons.
Discrete cold FDTD CW solutions, same observed Es and collisional current.
"""
from pathlib import Path
import itertools,json
import numpy as np,h5py
from numpy.polynomial import Polynomial as Poly
from scipy.constants import c,e,m_e,epsilon_0
import coupled_arecibo as a
ROOT=Path(__file__).resolve().parents[2]
with h5py.File(ROOT/'reference/coupled/minimum_incident.h5') as h:
 ANGLE=float(h.attrs['angle_deg']);OPT=h['source_polarization'][:]

def modes(s,ref,lo):
 # Local collisional eigenmodes at the homogeneous incident port. Root labels
 # continue the positive-flux O (larger kz) and X branches at these angles.
 om=s['omega'];gx,gy,gz=s['gyro']/om
 cross=np.array([[0,gz,-gy],[-gz,0,gx],[gy,-gx,0]])
 M=(s['physical_nu'][0,lo]/om-1j)*np.eye(3)+cross
 eps=np.eye(3)+1j*(s['wp'][0,lo]/om)**2*np.linalg.inv(M)
 q=s['kx']/om;mat=[[Poly([eps[i,j]]) for j in range(3)] for i in range(3)]
 mat[0][0]+=Poly([0,0,-1]);mat[1][1]+=Poly([-q*q,0,-1]);mat[2][2]+=Poly([-q*q]);mat[0][2]+=Poly([0,q]);mat[2][0]+=Poly([0,q])
 det=Poly([0])
 for perm in itertools.permutations(range(3)):
  term=Poly([(-1)**sum(perm[i]>perm[j] for i in range(3) for j in range(i+1,3))])
  for i in range(3):term*=mat[i][perm[i]]
  det+=term
 solutions=[]
 for nz in det.trim(tol=1e-13).roots():
  D=np.array([[mat[i][j](nz) for j in range(3)] for i in range(3)])
  _,_,vh=np.linalg.svd(D);E=vh[-1].conj();H=np.cross([q,0,nz],E)
  if np.cross(E,H.conj())[2].real<=0:continue
  v=np.linalg.solve(ref['initial'][:2,lo,:],E[:2])
  actual=ref['initial'][:,:,0]*v[0]+ref['initial'][:,:,1]*v[1]
  # Account for Yee half-cell staggering when checking launch polarization.
  expected=np.r_[E[:2],E[2]*np.exp(1j*nz*om*s['dz_cells'][lo]/2),H[:2]*np.exp(1j*nz*om*s['dz_cells'][lo]/2),H[2]]
  error=np.linalg.norm(actual[:6,lo]-expected)/np.linalg.norm(expected)
  solutions.append((nz.real,v,error))
 assert len(solutions)==2
 solutions.sort(key=lambda item:item[0],reverse=True)
 return {'O':solutions[0][1:],'X':solutions[1][1:]}

def analyze(g,s,ref,v,mode_error=0):
 state=np.einsum('qnp,p->qn',s['initial'],v);inc=np.einsum('qnp,p->qn',ref['initial'],v)
 z=s['z'];w=s['dz_cells'];dual=s['dual'];zf=z+w/2;lo=np.argmin(abs(zf-115100));hi=np.argmin(abs(zf-120800));mask=(z>zf[lo])&(z<zf[hi])
 budget=a.hf_port_budget(state,inc,lo,hi);pin=budget['incident_EM_flux_W_m2']
 # Positive physical collisional heating only; numerical absorbing layers excluded.
 ch=np.cos(s['omega']*s['dt']/2)
 cells=.5*epsilon_0*c*ch*np.sum(s['physical_nu']*abs(state[6:9])**2*np.array([dual,dual,w]),axis=0)/pin
 A=cells[mask].sum();res=budget['hf_nonreturn_fraction']-A
 assert abs(res)<3e-4,(g.name,res)
 stats=dict(budget,absorbed_heat_fraction=A,flux_minus_heat=res,angle_deg=np.rad2deg(np.arcsin(s['kx']/s['omega'])),center_m=s['center'],mode_boundary_relative_error=mode_error,nu_res_s1=float(np.interp(s['center'],z,s['physical_nu'][0])*c),finest_dz_m=float(w.min()),peak_Ez_over_incident=float(abs(state[2]).max()/np.linalg.norm(inc[:3,lo])))
 for width in [1.,5.,30.]:stats[f'incident_fraction_absorbed_within_{width:g}m']=float(cells[mask&(abs(z-s['center'])<width)].sum())
 g.attrs.update(stats)
 for k,val in dict(z_relative_m=z-s['center'],absorption_per_m=cells/dual*mask,cumulative_absorbed_fraction=np.cumsum(cells*mask),source_polarization=v).items():g.create_dataset(k,data=val,compression='gzip')
 print(g.name,{k:stats[k] for k in ['absorbed_heat_fraction','reflected_EM_fraction','flux_minus_heat','mode_boundary_relative_error','incident_fraction_absorbed_within_1m','incident_fraction_absorbed_within_30m']},flush=True)

def main():
 out=ROOT/'reference/coupled/hf_absorption_profile.h5'
 with h5py.File(out,'w') as h:
  h.attrs.update(frequency_hz=a.F,scope='Cold magnetized collisional FDTD, fixed observed density; physical heating excludes sponge. Cumulative integral is net deposition below height, not one-way incident attenuation. O/X input is matched to local collisional eigenpolarization.',fdtd_revision='346135fe3ee5eba8e4df89fc7c3e5678fa24f69c')
  h['provenance/source']=Path(__file__).read_text()
  for name,angle,dz,scale in [('event',ANGLE,.0025,1.),('event_coarse',ANGLE,.005,1.),('normal',0.,.005,1.),('oblique25',25.,.005,1.),('half_nu',ANGLE,.0025,.5),('double_nu',ANGLE,.0025,2.)]:
   print('Solving',name,flush=True)
   s=a.setup(dz=dz,cap=.5,angle=angle,thermal=False,collision_scale=scale,scenario={'ordering':'COLAMD'})
   r=a.setup(dz=dz,cap=.5,angle=angle,thermal=False,collision_scale=scale,scenario={'ordering':'COLAMD'},reference=True)
   lo=np.argmin(abs(s['z']+s['dz_cells']/2-115100));mo=modes(s,r,lo)
   for label in ['O','X']:
    v,err=mo[label];assert err<.002,(label,err)
    analyze(h.create_group(name+'/'+label),s,r,v,err)
   if name in ['event','event_coarse','half_nu','double_nu']:analyze(h.create_group(name+'/optimized'),s,r,OPT)
   h.flush()
 print(out,flush=True)
if __name__=='__main__':main()
