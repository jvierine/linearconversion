"""Neutral priors and ion-composition assumptions for all archived altitudes.

Native measurements are not averaged; only smooth environmental model fields
are tabulated coarsely and interpolated. MSIS is not used to constrain topside
plasma temperatures. IRI is an empirical composition assumption, not a fit.
"""
from pathlib import Path
import datetime,hashlib
import numpy as np,h5py,pymsis,ppigrf,iri2016
root=Path(__file__).parent/'data/eregion_fit'
with h5py.File(root/'all_de_inputs.h5') as h:tt=h['time_unix_s'][:]
t=np.linspace(tt[0],tt[-1],42);z=np.arange(0.,1500.,.3);zm=np.arange(0.,1000.1,2.)
dates=(np.repeat(t,len(zm))*1e6).astype('datetime64[us]')
p=pymsis.calculate(dates,np.full(len(dates),-66.7528),np.full(len(dates),18.3442),np.tile(zm,len(t)),version=2.1).reshape(len(t),len(zm),11)
b=ppigrf.igrf(-66.7528,18.3442,z,datetime.datetime(2018,5,4,17));B=np.sqrt(sum(x*x for x in b))[0]*1e-9;angle=np.rad2deg(np.arccos(abs(b[2][0])*1e-9/B))
with h5py.File(root/'full_altitude_priors.h5','w') as h:
 h['time_unix_s']=t;h['range_km']=z;h['B_T']=B;h['angle_k_B_deg']=angle;h['msis/height_km']=zm
 for key,idx in [('Tn_K',10),('N2_m3',1),('O2_m3',2),('O_m3',3)]:h['msis/'+key]=np.nan_to_num(p[:,:,idx])
 iri=[]
 for i,tv in enumerate(t):
  x=iri2016.IRI(datetime.datetime.fromtimestamp(tv,datetime.timezone.utc),[90,1500,1.5],18.3442,-66.7528)
  keys=['Ti','Te','nO+','nH+','nHe+','nNO+','nO2+','nN+'];iri.append(np.array([x[k].values for k in keys]).T)
  if i%10==0:print('IRI',i,len(t),flush=True)
 h['iri/height_km']=x.alt_km.values;h['iri/values']=np.array(iri);h['iri/ion_mass_u']=[16.,1.,4.,30.,32.,14.]
 h.attrs['iri_columns']=','.join(keys);h.attrs['composition']='IRI2016 fixed ion fractions, NO+ below 90 km; no inferred composition or negative ions'
 h.attrs['msis']='MSIS2.1 neutral fields tabulated to 1000 km; collisions set to zero above 1000 km'
 h.attrs['iri']='https://irimodel.org/; installed iri2016 Python/Fortran package; environmental priors only'
 h.attrs['provenance']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
print('saved',root/'full_altitude_priors.h5',flush=True)
