"""Fit all usable archived gates at native time/range sampling.

Composition and weak-signal temperature estimates are conditional on model
priors. No SNR mask deletes cells. All data products are HDF5. Provenance and
instrument limitations are stored as metadata, not rendered on figures.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,time,hashlib,multiprocessing as mp
import numpy as np,h5py
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from full_altitude_model import FullModel

D={}
def prepare(root):
 global D
 with h5py.File(root/'all_altitude_inputs.h5') as h:
  D['t']=h['time_unix_s'][:];D['edges']=h['time_edges_unix_s'][:];D['z']=h['range_km'][:]
  for mode in ['clp','csp']:
   D[mode]=h[mode+'/mean'][:];D[mode+'_lags']=h[mode+'/lags_us'][:]
   D[mode+'_native_time']=h[mode+('/record_time_unix_s' if mode=='clp' else '/record_mean_time_unix_s')][:]
   a=D[mode]
   # Empirical component variance from neighboring cycles, without smoothing
   # or averaging the ACF that is fitted.
   diff=abs(np.diff(a,axis=0))**2/4.
   var=np.empty(a.shape,float)
   floor=np.median(diff[:,D['z']>1350],axis=(0,1))
   for i in range(len(a)):var[i]=np.maximum(np.median(diff[max(0,i-2):min(len(diff),i+2)],axis=0),floor)
   D[mode+'_var']=var
 with h5py.File(root/'density_calibration.h5') as h:D['gain']=float(h['gain'][()]);D['gain_scatter']=float(h['log_gain_scatter'][()])
 z=D['z'];t=D['t'];zz,tt=np.meshgrid(z,t)
 with h5py.File(root/'full_altitude_priors.h5') as h:
  D['B']=h['B_T'][:];D['angle']=h['angle_k_B_deg'][:];pt=h['time_unix_s'][:]
  for key in ['Tn_K','N2_m3','O2_m3','O_m3']:
   grid=RegularGridInterpolator((pt,h['msis/height_km'][:]),h['msis/'+key][:],bounds_error=False,fill_value=None)
   a=grid(np.c_[tt.ravel(),np.clip(zz.ravel(),0,1000)]).reshape(tt.shape)
   if key!='Tn_K':a[:,z>1000]=0.
   D[key]=a
  v=h['iri/values'][:];iz=h['iri/height_km'][:]
  out=RegularGridInterpolator((pt,iz),v,bounds_error=False,fill_value=None)(np.c_[tt.ravel(),np.clip(zz.ravel(),90,1500)]).reshape(tt.shape+(8,))
  frac=np.maximum(out[:,:,2:],0);frac/=np.maximum(frac.sum(axis=2,keepdims=True),1.)
  blend=np.clip((z-140)/40,0,1)[None,:,None];frac*=blend;frac[:,:,3]+=1-blend[:,:,0]
  D['fractions']=frac
  D['ti0']=np.where(zz<150,D['Tn_K'],np.maximum(out[:,:,0],D['Tn_K']))
  D['te0']=np.where(zz<150,D['Tn_K'],np.maximum(out[:,:,1],D['Tn_K']))
  D['sigma_ti']=np.where(z<150,.25,np.log(3.))
  D['sigma_te']=np.full(len(z),np.log(3.))
 # Each worker inherits read-only arrays through fork on Linux.

def init():
 global MODELS
 MODELS={'csp':FullModel('csp',D['csp_lags']),'clp':FullModel('clp')}

def cell(i,j,max_nfev=60,start=None):
 z=D['z'][j];mode='csp' if z<94.5 else 'clp';y=D[mode][i,j];variance=D[mode+'_var'][i,j]
 scale=max(y[0].real,np.sqrt(variance[0]),1e-10);yn=y/scale
 err=np.maximum(np.sqrt(variance)/scale,.02)
 ti0=D['ti0'][i,j];te0=D['te0'][i,j];sti=D['sigma_ti'][j];ste=D['sigma_te'][j]
 B=D['B'][j];angle=D['angle'][j];ns=[D[k][i,j] for k in ['N2_m3','O2_m3','O_m3']];frac=D['fractions'][i,j]
 factor=scale*z*z*D['gain'];fun=MODELS[mode]
 def calc(p):
  ne=max(p[0]*factor,1e5);ti=ti0*np.exp(p[1]);te=te0*np.exp(p[2]);v=100*p[3]
  return p[0]*fun(ti,te,v,ne,B,angle,ns,frac)
 def residual(p):
  d=(calc(p)-yn)/err
  return np.r_[d.real,d.imag,p[1]/sti,p[2]/ste,p[3]/3.]
 guess=[max(2*yn[0].real,.02),0.,0.,0.]
 bounds=([0.,np.log(max(60/ti0,.2)),np.log(max(60/te0,.2)),-10.],[100.,np.log(min(10000/ti0,8.)),np.log(min(15000/te0,12.)),10.])
 if start is not None:
  guess=np.clip([start[0]/factor,np.log(start[2]/ti0),np.log(start[1]/te0),start[3]/100.],bounds[0],bounds[1])
 fit=least_squares(residual,guess,bounds=bounds,x_scale=[2.,.5,.5,1.],max_nfev=max_nfev,ftol=1e-5,xtol=1e-5,gtol=1e-5)
 p=fit.x;ne=p[0]*factor;ti=ti0*np.exp(p[1]);te=te0*np.exp(p[2]);v=100*p[3]
 normal=fit.jac.T@fit.jac
 cov=np.linalg.pinv(normal,rcond=1e-20)
 sig=np.sqrt(np.maximum(np.diag(cov),0))*[factor,ti,te,100.]
 if np.linalg.cond(normal)>1e18:sig[0]=np.inf
 chi=np.sum(residual(p)[:-3]**2)/max(1,2*len(y)-4)
 info=np.clip(1-np.diag(cov)[[1,2,3]]/[sti*sti,ste*ste,9.],0,1)
 atbound=np.any(abs(fit.active_mask)>0)
 return np.r_[ne,te,ti,v,sig[[0,2,1,3]],chi,fit.success,fit.nfev,info[[1,0,2]],atbound,y[0].real/np.sqrt(variance[0])]

def job(task):
 i,js=task;out=[]
 for j in js:
  try:out.append(cell(i,j))
  except Exception as exc:
   raise RuntimeError(f'fit time={i} range={D["z"][j]}: {exc}') from exc
 return i,js,np.array(out)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).parent/'data/eregion_fit');p.add_argument('--workers',type=int,default=8);p.add_argument('--preview',action='store_true');p.add_argument('--output',default='full_altitude_fits.h5');a=p.parse_args();prepare(a.root)
 js=np.flatnonzero(D['z']>=35.);start=time.monotonic()
 if a.preview:
  selected=[35.1,50.1,60,75,80.1,90,99,117.9,130,150,180,200,250,300,400,500,700,900,1100,1300,1490]
  jj=[int(np.argmin(abs(D['z']-z))) for z in selected];tasks=[(i,jj) for i in [0,50,100,150,200]]
 else:tasks=[(i,js[s:s+32].tolist()) for i in range(len(D['t'])) for s in range(0,len(js),32)]
 with h5py.File(a.root/a.output,'w') as h:
  h['time_unix_s']=D['t'];h['time_edges_unix_s']=D['edges'];h['height_km']=D['z'][js]
  h['clp_native_time_unix_s']=D['clp_native_time'];h['csp_native_time_unix_s']=D['csp_native_time']
  shape=(len(D['t']),len(js));ds=h.create_dataset('results',shape+(16,),dtype='f8',fillvalue=np.nan,compression='gzip')
  h['ion_fractions']=D['fractions'][:,js].astype('f4');h['ion_mass_u']=[16,1,4,30,32,14]
  h['Ti_prior_K']=D['ti0'][:,js].astype('f4');h['Te_prior_K']=D['te0'][:,js].astype('f4');h['B_T']=D['B'][js];h['angle_k_B_deg']=D['angle'][js]
  h['density_gain']=D['gain'];h['density_gain_log_scatter']=D['gain_scatter']
  h.attrs['columns']='ne_m3,Te_K,Ti_K,apparent_velocity_m_s,sigma_ne,sigma_Te,sigma_Ti,sigma_v,reduced_chi2,success,nfev,Te_prior_variance_reduction,Ti_prior_variance_reduction,v_prior_variance_reduction,at_bound,first_lag_snr'
  h.attrs['resolution']='native 300 m range sampling and ~80.4 s cycle cadence; CLP integration ~10 s per cycle, CSP seven integrations; no spatial or temporal averaging of fitted ACFs'
  h.attrs['coverage']='all archived range samples from first unblanked short-pulse gate at 35.1 km through 1499.7 km; no SNR mask'
  h.attrs['composition']='NO+ 30 u below 140 km, blend to IRI2016 fractions by 180 km; IRI O+,H+,He+,NO+,O2+,N+ above. Fixed composition, not measured. No negative ions, charged dust, or sporadic-E metal-ion inference.'
  h.attrs['prior']='MSIS2.1 Ti log sigma .25 below150km; Te log sigma ln3. Above150km IRI2016 Ti/Te with log sigma ln3. Velocity 0 +/-300 m/s weak regularization. Prior variance reductions retained.'
  h.attrs['limitations']='Native range sampling is not independent physical resolution; missing lag/range PSF and equal archived lag-bin weights. Neutral BGK only; Coulomb collisions absent. Velocity sign unverified. Topside weak-signal cells may be dominated by priors. Density formal sigma is infinite if the local Jacobian cannot constrain amplitude; bounds make Gaussian errors approximate.'
  h.attrs['calibration']='matched broad thermal plasma-line sidebands, excluding narrow HF enhancements; gain scatter is not a standard error'
  h.attrs['provenance']='fit_full_altitude.py; full_altitude_model.py; '+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  count=0
  with mp.get_context('fork').Pool(a.workers,initializer=init) as pool:
   for i,jj,result in pool.imap_unordered(job,tasks,chunksize=1):
    indices=np.array(jj)-js[0];ds[i,indices]=result;count+=len(jj)
    if a.preview or count%2048<32:
     h.flush();print('fitted',count,'of',sum(len(q[1]) for q in tasks),'seconds',round(time.monotonic()-start),'time',i,'range',D['z'][jj[0]],'median',np.median(result[:,:4],axis=0),flush=True)
  h.attrs['completed']=True
 print('finished',a.root/a.output,'seconds',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
