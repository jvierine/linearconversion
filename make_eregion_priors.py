"""Evaluate MSIS 2.1 and IGRF at Arecibo and the observation times/heights."""
import datetime
import numpy as np
import h5py
import pymsis
from pymsis.utils import get_f107_ap
import ppigrf
from fit_eregion import ROOT

def main():
 with h5py.File(ROOT/'compact_inputs.h5') as h:t=h['acf/time_unix_s'][:];r=h['acf/range_km'][:]
 tt=np.repeat(t,len(r));rr=np.tile(r,len(t));dates=(tt*1e6).astype('datetime64[us]');n=len(tt)
 v=pymsis.calculate(dates,np.full(n,-66.7528),np.full(n,18.3442),rr,version=2.1).reshape(len(t),len(r),11)
 b=ppigrf.igrf(-66.7528,18.3442,r,datetime.datetime(2018,5,4,17))
 f107,f107a,ap=get_f107_ap((t*1e6).astype('datetime64[us]'))
 B=np.sqrt(sum(x*x for x in b))[0]*1e-9;theta=np.rad2deg(np.arccos(abs(b[2][0])*1e-9/B))
 with h5py.File(ROOT/'priors.h5','w') as h:
  for name,value in [('time_unix_s',t),('range_km',r),('msis_output',v),('Tn_K',v[:,:,10]),('N2_m3',v[:,:,1]),('O2_m3',v[:,:,2]),('O_m3',v[:,:,3]),('B_T',B),('angle_k_B_deg',theta)]:h[name]=value
  h.attrs['model']='MSIS 2.1; IGRF14 coefficients evaluated at 2018-05-04 17:00 UTC';h.attrs['lat_deg']=18.3442;h.attrs['lon_deg']=-66.7528
  h['f107_previous_day']=f107;h['f107a_81day']=f107a;h['ap_history']=ap
  h.attrs['solar_geomagnetic_drivers']='historical drivers retrieved by pymsis; arrays retained for each time bin'
if __name__=='__main__':main()
