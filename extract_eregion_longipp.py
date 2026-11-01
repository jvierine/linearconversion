"""Isolate long-IPP plasma lines to audit folded F-region returns."""
from pathlib import Path
import numpy as np
import h5py
ROOT=Path('/scratch/data/juha/arecibo/eregion_pl/processing')
def main():
 with h5py.File(ROOT/'intervals.h5') as h:it=h['txs'][:]/1e6;ipp=h['ipp'][:]
 with h5py.File('/tmp/arecibo_fit_compact_20261029.h5') as h:
  edges=h['acf/time_edges_unix_s'][:];freq=h['plasma/frequency_MHz'][:];rr=h['plasma/binned_range_km'][:]
 sums=np.zeros((len(edges)-1,len(rr),len(freq),2));counts=np.zeros(len(edges)-1,int)
 diagnostic=np.zeros((7,len(rr),len(freq),2));nc=np.zeros(7,int);modes=np.array([1000,1250,1500,1750,2250,2750,3250])
 for p in sorted((ROOT/'results4').glob('spl-*.h5')):
  tt=int(p.stem.split('-')[-1]);j=np.argmin(abs(it-tt));mode=ipp[j]
  if abs(it[j]-tt)>2:raise ValueError((tt,it[j]))
  b=np.searchsorted(edges,tt,side='right')-1
  if not 0<=b<len(counts):continue
  if mode!=3250 and b!=22:continue
  with h5py.File(p) as h:s=h['PL'][:];f=h['freqs'][:];r=h['range'][:]
  ip=np.array([np.argmin(abs(f-v)) for v in freq]);im=np.array([np.argmin(abs(f+v)) for v in freq]);ri=np.where((r>=100)&(r<=139.5))[0];ri=ri[:len(rr)*2]
  q=s/np.median(s[r<85],axis=0)[None,:]-1
  pair=np.stack([q[ri][:,im],q[ri][:,ip]],axis=-1).reshape(len(rr),2,len(freq),2).mean(axis=1)
  if mode==3250:sums[b]+=pair;counts[b]+=1
  if b==22:
   mi=np.where(modes==mode)[0][0];diagnostic[mi]+=pair;nc[mi]+=1
 with h5py.File('/tmp/arecibo_longipp_20261029.h5','w') as h:
  h['binned_excess']=np.array(sums/np.maximum(counts[:,None,None,None],1),dtype=np.float32);h['n_records']=counts
  h['range_km']=rr;h['frequency_MHz']=freq;h['ipp_us']=3250
  h['diagnostic_by_ipp']=np.array(diagnostic/np.maximum(nc[:,None,None,None],1),dtype=np.float32);h['diagnostic_n_records']=nc;h['diagnostic_modes_us']=modes
 print('saved',counts.sum(),nc,flush=True)
if __name__=='__main__':main()
