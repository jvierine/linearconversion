"""Density calibration from matched thermal sidebands at long IPP.

Do not calibrate on the narrow HF-enhanced line or on short-IPP folded returns.
Archive parameters, accepted points and calibration scatter in HDF5.
"""
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d
from scipy.constants import atomic_mass,m_e
from fit_eregion import ROOT,response,neutral_rates

def peak(f,y):
 # Retain the broad thermal peak; narrow channels do not set its center.
 smooth=gaussian_filter1d(y,2.)
 valid=(f>2.5)&(f<5.0);ix=np.where(valid)[0];j=ix[np.argmax(smooth[ix])]
 if j==ix[0] or j==ix[-1]:return np.nan,0.
 den=smooth[j-1]-2*smooth[j]+smooth[j+1]
 delta=.5*(smooth[j-1]-smooth[j+1])/den if den<0 else 0.
 return f[j]+np.clip(delta,-1,1)*(f[1]-f[0]),smooth[j]

def main():
 with h5py.File(ROOT/'plasma_fits.h5') as h:p=h['parameters'][:];r=h['range_km'][:];t=h['time_unix_s'][:];chi=h['reduced_chi2'][:]
 with h5py.File(ROOT/'priors.h5') as h:
  tn=h['Tn_K'][:];B=h['B_T'][:];theta=h['angle_k_B_deg'][:];n2=h['N2_m3'][:];o2=h['O2_m3'][:];o=h['O_m3'][:]
 with h5py.File(ROOT/'longipp.h5') as h:q=h['binned_excess'][:];pr=h['range_km'][:];f=h['frequency_MHz'][:];counts=h['n_records'][:]
 rows=[]
 # Separated heights on the smooth background above the thin sporadic layer.
 # Adjacent 0.3-km samples are not independent calibration measurements.
 for i in range(len(t)-1):
  for height in [129.,133.5,138.]:
   j=np.argmin(abs(r-height));k=np.argmin(abs(pr-height))
   fm,am=peak(f,q[i,k,:,0]);fp,ap=peak(f,q[i,k,:,1])
   if min(am,ap)<.04 or abs(fm-fp)>.08 or chi[i,j]>3 or not np.isfinite(p[i,j]).all():continue
   freq=(fm+fp)*.5*1e6;ti,te=p[i,j,1:3];v=p[i,j,3]
   nue,nui=neutral_rates(te,n2[i,j],o2[i,j],o[i,j])
   # Average absolute sidebands removes the leading common bulk Doppler shift.
   ce,_=response(np.array([2*np.pi*freq]),1.,te,m_e,nue,B[j],theta[j])
   ci,_=response(np.array([2*np.pi*freq]),1.,ti,30*atomic_mass,nui)
   ne=-1./(ce[0]+ci[0]).real
   gain=ne/(p[i,j,0]*r[j]**2)
   rows.append([i,j,t[i],r[j],fm,fp,am,ap,te,ti,ne,p[i,j,0],gain])
 rows=np.array(rows);logs=np.log(rows[:,-1]);med=np.median(logs);mad=1.4826*np.median(abs(logs-med))
 keep=abs(logs-med)<3*max(mad,.01)
 gain=np.exp(np.median(logs[keep]));scatter=np.std(logs[keep],ddof=1)
 ne=p[:,:,0]*r[None,:]**2*gain
 with h5py.File(ROOT/'density_calibration.h5','w') as h:
  h['points']=rows;h['accepted']=keep;h['ne_m3']=ne;h['gain']=gain;h['log_gain_scatter']=scatter;h['time_unix_s']=t;h['range_km']=r
  h.attrs['columns']='time_index,range_index,unix_s,range_km,negative_abs_MHz,positive_MHz,negative_peak_excess,positive_peak_excess,Te_K,Ti_K,ne_from_PL_m3,acf_amplitude,gain'
  h.attrs['calibration']='ne=amplitude*range_km^2*gain; exact magnetized electron real dielectric root at averaged thermal sidebands'
  h.attrs['selection']='3250 us IPP; smooth 129,133.5,138 km background; both sidebands above .04 excess; frequency agreement within .08 MHz; ACF chi2<3; robust 3-MAD gain gate'
  h.attrs['uncertainty']='log_gain_scatter is observed point scatter, not an independent-sample standard error; absolute pulse-ambiguity systematic not included'
 print('points',len(rows),'accepted',keep.sum(),'gain',gain,'scatter',scatter,flush=True)
 for i in [15,20,21,22,23,30,38]:
  j=np.nanargmax(ne[i]);print('peak',i,r[j],ne[i,j],p[i,j,1:])

if __name__=='__main__':main()
