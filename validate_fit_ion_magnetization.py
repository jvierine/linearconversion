"""Bound the unmagnetized-ion approximation at the actual radar geometry."""
import numpy as np,h5py
from scipy.constants import m_e,atomic_mass,e
from full_altitude_model import spectrum,MASSES,FullModel
from fit_eregion import K
from isr.kinetic import Species,density_spectrum
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)
f=np.arange(-128000.,128000.1,250.);m=FullModel('clp');rows=[]
for ne,ti,te,B,frac in [(1e11,200,200,3.5e-5,[0,0,0,1,0,0]),(1e11,300,500,3.5e-5,[0,0,0,1,0,0]),(4e11,800,2000,3.2e-5,[.95,.01,0,.02,.01,.01]),(1e10,2000,2500,2.7e-5,[.5,.45,.03,0,0,.02]),(1e9,3000,4000,2e-5,[.05,.9,.03,0,0,.02])]:
 electrons=[Species('e',ne,te,m_e,-e)];ions=[Species(str(mm),ne*ff,ti,mm*atomic_mass,e) for mm,ff in zip(MASSES,frac) if ff>0]
 full=density_spectrum(f,K,electrons,ions,B=B,theta_deg=47)['total'];approx=spectrum(f,ti,te,ne,B,47,[0,0,0],frac)
 a=m.kernel@full;b=m.kernel@approx;err=np.max(abs(a-b))/max(abs(a[0]),1e-4);rows.append([ne,ti,te,B,err]);print(rows[-1],flush=True)
with h5py.File('data/eregion_fit/ion_magnetization_checks.h5','w') as h:
 h['checks']=rows;h.attrs['columns']='ne_m3,Ti_K,Te_K,B_T,max_ACF_difference_over_first_lag';h.attrs['angle_k_B_deg']=47.;h.attrs['f_r_MHz']=430.;h.attrs['nu_en_s_1']=0.;h.attrs['nu_ei_s_1']=0.
