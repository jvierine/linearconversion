"""Electric-field cut from the event cold-plasma FDTD CW solution."""
from pathlib import Path
import sys, h5py, numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import coupled_arecibo as a
root=Path(__file__).resolve().parents[2]
with h5py.File(root/'reference/coupled/minimum_incident.h5') as h:
    angle=float(h.attrs['angle_deg']);v=h['source_polarization'][:]
s=a.setup(dz=.0025,cap=.5,thermal=False,angle=angle,scenario={'ordering':'COLAMD'})
r=a.setup(dz=.0025,cap=.5,thermal=False,angle=angle,scenario={'ordering':'COLAMD'},reference=True)
vf=np.einsum('qnp,p->qn',s['initial'],v);inc=np.einsum('qnp,p->qn',r['initial'],v)
z=s['z'];zf=z+s['dz_cells']/2;lo=np.argmin(abs(zf-115100))
# Interpolate staggered transverse components to the Ez height before |E|.
E=np.array([np.interp(zf,z,vf[q].real)+1j*np.interp(zf,z,vf[q].imag) for q in range(2)]+[vf[2]])
norm=np.linalg.norm(inc[:3,lo]);m=(zf>s['center']-400)&(zf<s['center']+100)
with h5py.File(root/'reference/coupled/fdtd_reflection_field.h5','w') as h:
    h['height_relative_m']=zf[m]-s['center'];h['E_complex_over_incident']=E[:,m]/norm
    h['total_magnitude_over_incident']=np.linalg.norm(E[:,m],axis=0)/norm
    h['vertical_magnitude_over_incident']=abs(E[2,m])/norm
    h.attrs.update(resonance_height_m=s['center'],frequency_hz=a.F,vacuum_incidence_angle_deg=angle,finest_dz_m=.0025,method='Steady solution of the discrete cold-plasma FDTD recurrence; magnetized and collisional; no pressure evolution',source=Path(__file__).read_text())
print('Saved event electric-field cut',flush=True)
