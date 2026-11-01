"""Read Seid archives; retain ACF statistics and both plasma-line sidebands.

Provenance belongs in HDF5, never rendered on scientific figures.
"""
from pathlib import Path
import numpy as np
import h5py
from scipy.ndimage import uniform_filter1d

ROOT=Path('/scratch/data/juha/arecibo/eregion_pl/processing')
INPUT=Path('/tmp/arecibo_eregion_fit_inputs_20261029.h5')
OUTPUT=Path('/tmp/arecibo_fit_compact_20261029.h5')

def main():
 with h5py.File(INPUT) as h:
  groups=[h['clp'][n] for n in sorted(h['clp'])]
  r=groups[0]['ranges_km'][:]
  use=(r>=100)&(r<=139.5); noise=r>=650
  t=np.array([g['t0'][()] for g in groups]);edges=np.arange(t.min()-5,t.max()+405,400.)
  nb=len(edges)-1; nr=use.sum();nl=43
  mean=np.full((nb,nr,nl),np.nan+0j);variance=np.full(mean.shape,np.nan);nobs=np.zeros(nb,int)
  offrange=np.zeros((nb,nl),complex)
  for b in range(nb):
   inds=np.where((t>=edges[b])&(t<edges[b+1]))[0]
   if not len(inds):continue
   a=np.array([groups[i]['acf'][:] for i in inds]);nobs[b]=len(inds)
   # Preserve distant-range returns for a bias sensitivity check, without
   # assuming the topside ionosphere is signal-free and subtracting it.
   bias=np.mean(a[:,noise,:],axis=1,keepdims=True)
   offrange[b]=bias.mean(axis=0)[0]
   x=a[:,use,:]
   mean[b]=np.mean(x,axis=0)
   # Real/imag component variance, with an off-range noise floor.
   floor=np.mean(np.var(a[:,noise,:],axis=1).real,axis=0)/(2*len(inds))
   if len(inds)>1:
    variance[b]=np.maximum(np.var(x,axis=0,ddof=1).real/(2*len(inds)),floor[None,:])
   else:variance[b]=floor
  lag=groups[0]['lags'][:]*1e-6  # archive stores microseconds
 files=sorted((ROOT/'results4').glob('spl-*.h5'))
 times=[]; peakneg=[];peakpos=[];heightneg=[];heightpos=[]; sources=[]
 with h5py.File(files[0]) as h:
  freq=h['freqs'][:];pr=h['range'][:]
  ip=np.where((freq>=2)&(freq<=7))[0]; im=np.array([np.argmin(abs(freq+freq[j])) for j in ip])
  ri=np.where((pr>=100)&(pr<=139.5))[0]
  # Average adjacent 0.15-km gates to match 0.3-km sampling, keeping z increasing.
  ri=ri[:len(ri)//2*2];rr=pr[ri].reshape(-1,2).mean(axis=1)
  sums=np.zeros((nb,len(rr),len(ip),2));counts=np.zeros(nb,int)
 for idx,p in enumerate(files):
  with h5py.File(p) as h:s=h['PL'][:];tt=float(h['t0'][()])
  bg=np.median(s[pr<85],axis=0)
  q=s/bg[None,:]-1
  neg=q[:,im];pos=q[:,ip]
  take=(pr>=100)&(pr<=130)
  an=uniform_filter1d(neg,size=3,axis=0,mode='nearest')[take]
  ap=uniform_filter1d(pos,size=3,axis=0,mode='nearest')[take]
  peakneg.append(an.max(axis=0));peakpos.append(ap.max(axis=0))
  heightneg.append(pr[take][an.argmax(axis=0)]);heightpos.append(pr[take][ap.argmax(axis=0)])
  times.append(tt);sources.append(str(p))
  b=np.searchsorted(edges,tt,side='right')-1
  if 0<=b<nb:
   sums[b,:,:,0]+=neg[ri].reshape(len(rr),2,len(ip)).mean(axis=1)
   sums[b,:,:,1]+=pos[ri].reshape(len(rr),2,len(ip)).mean(axis=1)
   counts[b]+=1
  if idx%200==0:print('PL',idx,len(files),flush=True)
 with h5py.File(OUTPUT,'w') as h:
  h.attrs['source_host']='seid.uit.no';h.attrs['source_root']=str(ROOT)
  h.attrs['provenance']='prepare_eregion_fit.py'
  h.attrs['acf_bin_seconds']=400.;h.attrs['pl_background_range_km']='79.95 to <85'
  h.attrs['enhancement_range_km']='100 to 130; max of 3-gate (0.45 km) range average'
  h['acf/time_unix_s']=(edges[:-1]+edges[1:])/2;h['acf/time_edges_unix_s']=edges
  h['acf/range_km']=r[use];h['acf/lags_s']=lag;h['acf/n_records']=nobs
  h['acf/mean']=mean;h['acf/component_variance']=variance
  h['acf/distant_range_mean']=offrange
  h['plasma/time_unix_s']=times;h['plasma/frequency_MHz']=freq[ip]
  h['plasma/source_files']=np.array(sources,dtype=h5py.string_dtype())
  for name,x in [('negative_excess',peakneg),('positive_excess',peakpos),('negative_peak_range_km',heightneg),('positive_peak_range_km',heightpos)]:
   h.create_dataset('plasma/'+name,data=np.array(x,dtype=np.float32),compression='gzip')
  h['plasma/binned_range_km']=rr;h['plasma/n_records']=counts
  h.create_dataset('plasma/binned_excess',data=(sums/np.maximum(counts[:,None,None,None],1)).astype(np.float32),compression='gzip')
 print('output',OUTPUT,OUTPUT.stat().st_size,flush=True)

if __name__=='__main__':main()
