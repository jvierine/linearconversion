"""Retain every archived range gate, separately for both pulse modes.

Run in conda base on Seid. No source files are modified. Cycle averages retain
the native ~80 s cadence, actual times, and the lag-specific sampling counts.
"""
from pathlib import Path
import numpy as np
import h5py

ROOT = Path('/scratch/data/juha/arecibo/eregion_pl/processing/results3')
OUT = '/tmp/arecibo_allgates_exact_cycles_20261029.h5'

def main():
    cf = sorted(ROOT.glob('clp_acf-*.h5'))
    sf = sorted(ROOT.glob('csp_acf-*.h5'))
    with h5py.File(cf[0]) as h:
        ranges = h['ranges'][:]
        clags = h['lags'][:]
    times = []
    for p in cf:
        with h5py.File(p) as h: times.append(float(h['t0'][()]))
    times = np.asarray(times)
    # A cycle consists of seven CSP records followed by one CLP record.
    with h5py.File(sf[0]) as h: start = float(h['t0'][()])
    with h5py.File(sf[-1]) as h: end = float(h['t1'][()])
    with h5py.File(cf[-1]) as h: end = max(end, float(h['t1'][()]))
    # The archived sequence contains seven short-pulse integrations and one
    # long-pulse integration per cycle. Use actual cycle boundaries, not a
    # guessed offset from the long-pulse midpoint.
    assert len(sf)==7*len(cf), 'Inspect incomplete or changed pulse sequence'
    short_times=[];short_ends=[]
    for p in sf:
        with h5py.File(p) as h:
            short_times.append(float(h['t0'][()]));short_ends.append(float(h['t1'][()]))
    short_times=np.array(short_times);short_ends=np.array(short_ends)
    edges=np.r_[short_times[::7],end]
    assert np.all((times>=edges[:-1])&(times<edges[1:])), 'CLP outside CSP cycle'
    sl = []
    for p in sf[:14]:
        with h5py.File(p) as h: sl.extend(h['lags'][:5])
    sl = np.unique(sl)
    shape = (len(cf),len(ranges),len(sl))
    sums = np.zeros(shape,complex); sq = np.zeros(shape)
    counts = np.zeros((len(cf),len(sl)),int)
    st = np.zeros(len(cf)); sn = np.zeros(len(cf))
    paths = []
    for i,p in enumerate(sf):
        with h5py.File(p) as h:
            t = float(h['t0'][()]); a = h['acf'][:,:5]; l = h['lags'][:5]
        b = i//7
        for j,lag in enumerate(l):
            k = int(np.searchsorted(sl,lag))
            sums[b,:,k] += a[:,j]; sq[b,:,k] += abs(a[:,j])**2
            counts[b,k] += 1
        st[b] += t; sn[b] += 1; paths.append(str(p))
        if i%200 == 0: print('CSP',i,len(sf),flush=True)
    with h5py.File(OUT,'w') as h:
        h.attrs['source_root']=str(ROOT); h.attrs['source_host']='seid.uit.no'
        h.attrs['provenance']='extract_eregion_allgates.py'
        h.attrs['csp_lags']='first five columns only; sixth is unpopulated. Zero bin 4..39 us; nonzero lag j*IPP minus 0..39 us.'
        h['range_km']=ranges; h['time_edges_unix_s']=edges
        h['time_unix_s']=(edges[:-1]+edges[1:])/2
        h['csp/record_mean_time_unix_s']=st/sn
        h['csp/record_start_unix_s']=short_times
        h['csp/record_end_unix_s']=short_ends
        h['csp/lags_us']=sl; h['csp/count']=counts
        mean=sums/np.maximum(counts[:,None,:],1)
        var=np.maximum(sq/np.maximum(counts[:,None,:],1)-abs(mean)**2,0)
        h.create_dataset('csp/mean',data=mean,compression='gzip')
        h.create_dataset('csp/within_cycle_component_variance',data=var/2,compression='gzip')
        h['csp/source_files']=np.asarray(paths,dtype=h5py.string_dtype())
        ds=h.create_dataset('clp/mean',(len(cf),len(ranges),len(clags)),dtype='complex128',compression='gzip')
        h['clp/lags_us']=clags; h['clp/record_time_unix_s']=times
        clp_end=[]
        h['clp/source_files']=np.asarray([str(p) for p in cf],dtype=h5py.string_dtype())
        for i,p in enumerate(cf):
            with h5py.File(p) as f:
                ds[i]=f['acf'][:];clp_end.append(float(f['t1'][()]))
        h['clp/record_end_unix_s']=clp_end
        print('saved',OUT,ds.shape,counts.min(),counts.max(),flush=True)

if __name__=='__main__': main()
