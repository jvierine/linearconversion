"""Frequency convergence and ACF observation-operator accuracy checks."""
import h5py,numpy as np
from full_altitude_model import FullModel,spectrum
from fit_eregion import response,K
from scipy.constants import m_e
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
root='data/eregion_fit/'
with h5py.File(root+'all_de_inputs.h5') as h:lags=h['csp/lags_us'][:]
fast=FullModel('clp');wide=FullModel('clp',df=125.,fmax=4000000.)
checks=[]
for ne,ti,te,frac in [(1e11,300,300,[0,0,0,1,0,0]),(1e12,1000,2000,[1,0,0,0,0,0]),(1e10,2000,3000,[.5,.5,0,0,0,0]),(1e8,3000,4000,[.1,.9,0,0,0,0]),(1e9,400,600,[.5,.5,0,0,0,0]),(1e9,200,200,[0,0,0,1,0,0]),(1e10,300,500,[0,0,0,1,0,0]),(1e11,100,60,[0,0,0,1,0,0])]:
 p=(ti,te,100.,ne,3.5e-5,47.,[0,0,0],frac)
 x=fast(*p);y=wide.kernel@spectrum(wide.f,ti,te,ne,3.5e-5,47.,[0,0,0],frac,100.)
 exact=response(2*np.pi*fast.f-K*100,ne,te,m_e,0,3.5e-5,47)
 nodes=response(2*np.pi*fast.node_f-K*100,ne,te,m_e,0,3.5e-5,47)
 ceerr=np.max(abs(fast.interp@nodes[0]-exact[0]))/np.max(abs(exact[0]))
 err=np.max(abs(x-y))/max(abs(y[0]),1e-4)
 checks.append([0,ne,ti,te,err,ceerr]);print('CLP',ne,ti,te,err,ceerr,flush=True)
a=FullModel('csp',lags);b=FullModel('csp',lags,df=15.625,tail_step=250.,tail_max=4000000.)
with h5py.File(root+'all_de_priors.h5') as h:
 for z,ne in [(50,1e8),(60,1e8),(80,1e9),(90,1e10),(94,1e11)]:
  j=np.argmin(abs(h['range_km'][:]-z));tn=h['Tn_K'][100,j];ns=[h[k][100,j] for k in ['N2_m3','O2_m3','O_m3']];p=(tn,tn,10.,ne,h['B_T'][j],h['angle_k_B_deg'][j],ns,[0,0,0,1,0,0]);x=a(*p);y=b(*p);err=np.max(abs(x-y))/abs(y[0]);checks.append([1,ne,tn,tn,err,0]);print('CSP',z,err,flush=True)
with h5py.File(root+'full_altitude_model_checks.h5','w') as h:
 h['checks']=checks;h.attrs['columns']='mode_0_clp_1_csp,ne,Ti,Te,max_ACF_error_over_first_lag,chebyshev_susceptibility_relative_error'
 h.attrs['reference']='CLP direct full spectrum +/-4MHz at125Hz; CSP doubled tail bandwidth and doubled frequency resolution'
assert np.max(np.array(checks)[:,4])<.005
