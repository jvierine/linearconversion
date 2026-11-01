"""Numerical checks and ion-mass sensitivity for the measured ACF fits."""
import numpy as np
import h5py
from scipy.constants import m_e,e,atomic_mass
import fit_eregion as fit
from isr.kinetic import Species,density_spectrum
from threadpoolctl import threadpool_limits

def main():
 with h5py.File(fit.ROOT/'plasma_fits.h5') as h:p=h['parameters'][:];seed=h['density_seed_m3'][:];r=h['range_km'][:]
 with h5py.File(fit.ROOT/'compact_inputs.h5') as h:y=h['acf/mean'][:];var=h['acf/component_variance'][:]
 with h5py.File(fit.ROOT/'density_calibration.h5') as h:ne=h['ne_m3'][:]
 with h5py.File(fit.ROOT/'priors.h5') as h:
  tn=h['Tn_K'][:];B=h['B_T'][:];theta=h['angle_k_B_deg'][:];n2=h['N2_m3'][:];o2=h['O2_m3'][:];o=h['O_m3'][:]
 checks=[];sensitivity=[]
 with threadpool_limits(limits=1):
  for i in [5,15,20,22,30,38]:
   for j in [np.argmax(ne[i]),np.argmin(abs(r-105)),np.argmin(abs(r-133.5))]:
    amp,ti,te,v=p[i,j];prior=(tn[i,j],B[j],theta[j],n2[i,j],o2[i,j],o[i,j])
    m=fit.model(ti,te,v,seed[i,j],*prior[1:])
    nue,nui=fit.neutral_rates(te,*prior[3:])
    shifted=fit.F-fit.K*v/(2*np.pi)
    full=density_spectrum(shifted,fit.K,[Species('e',seed[i,j],te,m_e,-e,nue)],
                          [Species('NO+',seed[i,j],ti,30*atomic_mass,e,nui)],B=B[j],theta_deg=theta[j])
    exact=fit.KERNEL@full['total']
    magerr=np.max(abs(exact-m))/np.max(abs(m))
    oldf,oldkernel=fit.F,fit.KERNEL
    fit.F=np.arange(-128000.,128000.1,125.)
    fit.KERNEL=np.array([np.exp(2j*np.pi*t[:,None]*fit.F[None,:]).mean(axis=0)*125 for t in fit.LAGS])
    finer=fit.model(ti,te,v,seed[i,j],*prior[1:])
    fit.F,fit.KERNEL=oldf,oldkernel
    griderr=np.max(abs(finer-m))/np.max(abs(m))
    refitted,_,_,chi,_=fit.fit_one(y[i,j],var[i,j],prior,ne[i,j])
    frac=(refitted-p[i,j])/np.array([amp,ti,te,1.])
    checks.append(np.r_[i,j,magerr,griderr,frac])
    print('check',i,r[j],magerr,griderr,frac,flush=True)
   j=np.argmax(ne[i]);prior=(tn[i,j],B[j],theta[j],n2[i,j],o2[i,j],o[i,j])
   metal,se,pred,chi,ok=fit.fit_one(y[i,j],var[i,j],prior,ne[i,j],mass=56.)
   sensitivity.append(np.r_[i,j,metal,se,chi])
 with h5py.File(fit.ROOT/'fit_validation.h5','w') as h:
  h['numerical_checks']=checks;h['mass56_sensitivity']=sensitivity
  h.attrs['numerical_columns']='time_index,range_index,full_ion_magnetization_max_fraction,frequency_grid_max_fraction,calibrated_seed_amplitude_fraction,Ti_fraction,Te_fraction,velocity_difference_m_s'
  h.attrs['sensitivity_columns']='time_index,range_index,amplitude,Ti,Te,v,sigma_amp,sigma_Ti,sigma_Te,sigma_v,chi2'
  h.attrs['mass56_assumption']='mass-only sensitivity; same neutral rates retained, not a fully specified Fe+ collision model'
 print('max errors',np.max(np.array(checks)[:,2:4],axis=0),flush=True)

if __name__=='__main__':main()
