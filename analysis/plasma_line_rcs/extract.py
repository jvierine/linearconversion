"""Read all native spectra on Seid; retain signed powers and exact frequency bins.
No deconvolved/display amplitudes; no source archive modifications.
"""
from pathlib import Path
import h5py,numpy as np,hashlib
ROOT=Path('/scratch/data/juha/arecibo/eregion_pl/processing')
OUT=Path('/tmp/arecibo_pl_rcs_inputs_20261031.h5')
files=sorted((ROOT/'results4').glob('spl-*.h5'))
with h5py.File(files[0]) as h:f=-h['freqs'][:];z=h['range'][:]
centers=[4.51,5.34,-4.51,-5.34]
with h5py.File(OUT,'w') as out:
 out.attrs.update(source_host='seid.uit.no',source_root=str(ROOT),frequency_convention='minus archive; positive display branch identified as upshifted in original event analysis',normalization='per-frequency median of native power below85km; signed power/background-1',source=Path(__file__).read_text())
 out['height_km']=z;out['frequency_mhz']=f;out['source_files']=np.array([str(p) for p in files],dtype=h5py.string_dtype())
 out['time_unix_s']=np.zeros(len(files));out['source_sha256']=np.array(['']*len(files),dtype=h5py.string_dtype())
 masks=[]
 for j,c in enumerate(centers):
  idx=np.flatnonzero(abs(f-c)<.11);masks.append(idx);g=out.create_group('band'+str(j));g.attrs['center_mhz']=c;g['frequency_mhz']=f[idx];g.create_dataset('excess',(len(files),len(z),len(idx)),dtype='f4',compression='gzip',chunks=(1,len(z),len(idx)))
 out.create_dataset('peak_height_spectra',(len(files),2,len(f)),dtype='f4',compression='gzip')
 out.create_dataset('peak_height_km',(len(files),2))
 out.create_dataset('max_excess',(len(files),len(f)),dtype='f4',compression='gzip')
 ionidx=np.flatnonzero((z>=125)&(z<=135));out['ion_height_km']=z[ionidx]
 out.create_dataset('ion_integrated_snr_hz',(len(files),len(ionidx),3),dtype='f8',compression='gzip')
 out['ion_half_bandwidth_hz']=[1.5e6,2e6,2.5e6]
 with h5py.File(ROOT/'intervals.h5') as h:tx=h['txs'][:]/1e6;ipp=h['ipp'][:]
 out['ipp_us']=np.zeros(len(files))
 for i,p in enumerate(files):
  with h5py.File(p) as h:a=h['PL'][:];t=float(h['t0'][()])
  bg=np.median(a[z<85],axis=0);q=a/bg-1;out['time_unix_s'][i]=t;out['source_sha256'][i]=hashlib.sha256(p.read_bytes()).hexdigest()
  it=np.argmin(abs(tx-t));out['ipp_us'][i]=ipp[it]
  for j,idx in enumerate(masks):out['band'+str(j)+'/excess'][i]=q[:,idx]
  valid=(z>=100)&(z<=130);out['max_excess'][i]=q[valid].max(axis=0)
  for j,c in enumerate(centers[:2]):
   peak=np.max(q[:,abs(f-c)<.025],axis=1);iz=np.argmax(np.where(valid,peak,-np.inf));out['peak_height_spectra'][i,j]=q[iz];out['peak_height_km'][i,j]=z[iz]
  df=abs(np.diff(f)[0])*1e6
  for j,w in enumerate([1.5,2.,2.5]):out['ion_integrated_snr_hz'][i,:,j]=q[ionidx][:,abs(f)<w].sum(axis=1)*df
  if i%100==0:print(i,len(files),flush=True)
print(OUT,OUT.stat().st_size,flush=True)
