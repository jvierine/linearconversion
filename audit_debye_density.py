"""Finite-Debye-length audit of the existing D-region density operator.
No density fits are modified. Analytic curves are unmagnetized Maxwellian
single-positive-ion, no dust/negative ions. CSP scan uses event MSIS neutrals,
B, aspect, receiver and archived lags, setting Te=Ti=Tn to isolate screening.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import hashlib
import h5py,numpy as np
from scipy.constants import epsilon_0,k,e,c
from full_altitude_model import FullModel
ROOT=Path(__file__).resolve().parent
K=4*np.pi*430e6/c
ne=np.logspace(7,11,61);T=200.;x=K*K*epsilon_0*k*T/(ne*e*e)
with h5py.File(ROOT/'data/eregion_fit/all_de_inputs.h5') as h:lags=h['csp/lags_us'][:]
m=FullModel('csp',lags)
fine=FullModel('csp',lags,df=15.625,tail_step=250.,tail_max=4e6)
curves=[];rows=[];check=[]
with h5py.File(ROOT/'data/eregion_fit/all_de_priors.h5') as h:
 for target in [60,70,80,90]:
  j=np.argmin(abs(h['range_km'][:]-target));tn=h['Tn_K'][100,j];ns=[h[q][100,j] for q in ['N2_m3','O2_m3','O_m3']]
  B=h['B_T'][j];angle=h['angle_k_B_deg'][j]
  def val(model,n):return model(tn,tn,0,n,B,angle,ns,[0,0,0,1,0,0])[0].real
  limit=val(m,1e17);limit2=val(m,1e18)
  assert abs(limit2/limit-1)<1e-6
  response=np.array([val(m,n) for n in ne]);curves.append(response/limit)
  rows.append([h['range_km'][j],tn,B,angle,*ns,limit])
  for n in [1e8,1e9]:
   a,b=val(m,n),val(fine,n);check.append([target,n,a,b,abs(a/b-1)])
  print('scan',target,'km done',flush=True)
with h5py.File(ROOT/'data/eregion_fit/full_altitude_fits.h5') as h:
 z=h['height_km'][:];take=(z>=50)&(z<94.5);pars=h['results'][:,take,:4]
 d=np.sqrt(epsilon_0*k*pars[:,:,1]/(pars[:,:,0]*e*e));ncrit=epsilon_0*k*pars[:,:,1]/e**2*K*K
 geom=np.percentile(K*d,[10,50,90],axis=0)
with h5py.File(ROOT/'data/debye_density_audit.h5','w') as h:
 h['ne_m3']=ne;h['ideal_ion_relative_power']=2/((1+x)*(2+x));h['ideal_all_frequency_relative_power']=2*(1+x)/(2+x)
 h['csp_relative_power']=curves;h['csp_conditions']=rows;h['csp_conditions'].attrs['columns']='height_km,Tn_K,B_T,angle_k_B_deg,N2_m3,O2_m3,O_m3,collective_short_lag_response'
 h['frequency_convergence']=check;h['frequency_convergence'].attrs['columns']='height_km,ne_m3,original_short_lag_response,refined_short_lag_response,relative_difference'
 h['fit_height_km']=z[take];h['fitted_k_lambda_D_percentiles']=geom;h['fitted_k_lambda_D_percentiles'].attrs['percentiles']=[10,50,90]
 h.attrs.update(radar_frequency_hz=430e6,k_B_m1=K,illustrative_Te_K=T,critical_density_m3=K*K*epsilon_0*k*T/e**2,Debye_length_at_transition_m=1/K,
  purpose='Audit finite-screening effects already retained by full_altitude_model.py. No correction applied a second time to fitted density.',
  physics='Ideal ion factor2/[(1+x)(2+x)] and all-frequency factor2(1+x)/(2+x), x=(k lambda_De)^2; Te=Ti, single positive ion. Ratios relative to collective high-density response. CSP uses same finite-lag operator, magnetization, neutral collisions, receiver response as retrieval.',
  limitations='No negative ions, charged dust or heavy-cluster-ion sensitivity. Low-altitude fitted kDebye diagnostics are conditional and weak-signal fits are not independent density measurements.')
 for name in ['audit_debye_density.py','full_altitude_model.py','fit_eregion.py','fit_full_altitude.py']:
  h['provenance/'+name]= (ROOT/name).read_text()
 for name in ['all_de_priors.h5','all_de_inputs.h5','full_altitude_fits.h5']:
  path=ROOT/'data/eregion_fit'/name;h['provenance'].attrs[name]=str(path)+' sha256='+hashlib.sha256(path.read_bytes()).hexdigest()
 assert np.max(np.array(check)[:,-1])<.005
print('Max CSP refinement difference',np.max(np.array(check)[:,-1]))
for n in [1e8,1e9,1e10]:
 xx=K*K*epsilon_0*k*T/(n*e*e);print('ideal',n,'Debye m',np.sqrt(xx)/K,'kDebye',np.sqrt(xx),'ion correction',(1+xx)*(2+xx)/2)
