"""Fit normalized44us response convolved with Gaussian line broadening.
The fitted area, rather than the tallest sample, supplies received line power.
This Gaussian is a descriptive spectral-width model, not a thermal inference.
"""
from pathlib import Path
import numpy as np,h5py
from scipy.optimize import least_squares
from numpy.polynomial.hermite import hermgauss
ROOT=Path(__file__).resolve().parents[2]
xg,wg=hermgauss(32);wg=wg/np.sqrt(np.pi)
def kernel(x,fc,s):return (np.sinc((x[:,None]-fc-np.sqrt(2)*s*xg)*44)**2)@wg

def fit(x,y,fc,A,half):
 use=abs(x-fc)<half;x=x[use];y=y[use]
 def mod(v):return v[0]*kernel(x,v[1],v[2])+v[3]+v[4]*(x-fc)
 results=[]
 for sig in [.0001,.008,.018]:
  r=least_squares(lambda v:mod(v)-y,[max(A,.01),fc,sig,0,0],bounds=([0,fc-.01,0,-1e5,-1e6],[1e7,fc+.01,.04,1e5,1e6]),x_scale=[max(A,.1),.005,.01,1,10],max_nfev=150)
  results.append(r)
 r=min(results,key=lambda v:sum(v.fun**2));return r.x,np.std(r.fun)
def main():
 with h5py.File(ROOT/'data/plasma_line_rcs_inputs.h5') as src,h5py.File(ROOT/'data/plasma_line_rcs.h5','a') as h:
  cat=h['catalog/rows'][:];h['catalog/rows_fixed44']=cat
  results=[]
  for row in cat:
   j,i=int(row[0]),int(row[1]);g=h[f'band{j}'];iz=int(g['selected_range_index'][i]);f=src[f'band{j}/frequency_mhz'][:];q=src[f'band{j}/excess'][i,iz];fc=float(row[3]);sens=[]
   for half in [.060,.075,.100]:
    v,rms=fit(f,q,fc,float(row[5]),half);sens.append([half,*v,rms])
   v=np.array(sens)[1,1:6];other=(j+2)%4;fo=src[f'band{other}/frequency_mhz'][:];qo=src[f'band{other}/excess'][i,iz];mask=abs(fo+v[1])<.075;X=np.c_[kernel(fo[mask],-v[1],v[2]),np.ones(sum(mask)),fo[mask]+v[1]];co=np.linalg.lstsq(X,qo[mask],rcond=None)[0]
   newA=v[0]-co[0];factor=row[8]/row[7];row[3]=v[1];row[5]=v[0];row[6]=co[0];row[7]=newA;row[8]=newA*factor;row[9]=10*np.log10(row[8]);row[10]=row[8]*g['values'][i,12];row[13]=sens[1][-1]
   results.append(sens)
   print(j,i,'width_kHz',v[2]*1000,'area_SNR_Hz',newA/44e-6,'RCS',row[8],flush=True)
  h['catalog/rows'][...]=cat;h['catalog/core_width_window_fits']=results
  h['catalog/core_width_window_fits'].attrs['columns']='half_window_mhz,area_times44us,center_mhz,Gaussian_sigma_mhz,background,slope_per_mhz,rms'
  h['catalog'].attrs['final_area_method']='Convolution of unit-area44us sinc-squared response and Gaussian intrinsic spectrum; area parameterA/tau in Hz. Matched opposite-sideband area subtracted at same height. Gaussian width is descriptive; no claim of thermal origin. Detection uses conservative fixed44us screen, while final RCS uses fitted area.'
  h['catalog'].attrs['final_columns']='band_id,record_index,time_unix_s,signed_frequency_mhz,height_km,selected_area_times_tau,opposite_area_times_tau,paired_area_times_tau,rcs_m2,rcs_dBsm,rcs_gaussian_m2,ion_multiplier,six_noise_rcs_m2,fit_rms'
  summary=[]
  for j in np.unique(cat[:,0]).astype(int):
   a=cat[cat[:,0]==j];r=a[np.argmax(a[:,8])];summary.append(r[[0,1,2,3,4,7,8,9,10,11]])
  h['catalog/peak_summary'][...]=summary
  h['provenance/integrated_core_source']=Path(__file__).read_text()
  print('PEAKS',np.array(summary),flush=True)

if __name__=='__main__':main()
