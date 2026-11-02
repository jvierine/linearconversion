"""Extract full-cadence native plasma-line maps for a display-only movie."""
from pathlib import Path
import argparse,hashlib
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
files=sorted(a.root.glob('spl-*.h5'))
with h5py.File(files[0]) as h:z=h['range'][:];f=-h['freqs'][:]
zi=np.flatnonzero((z>=100)&(z<=140));fi=np.flatnonzero((f>=1.6)&(f<=6.9));fi=fi[np.argsort(f[fi])]
with h5py.File(a.output,'w') as out:
 out['height_km']=z[zi];out['frequency_mhz']=f[fi]
 out.create_dataset('snr',(len(files),len(zi),len(fi)),dtype='f4',compression='gzip',compression_opts=2,chunks=(1,len(zi),len(fi)))
 out['time_unix_s']=np.zeros(len(files));out['source_file']=np.array([x.name for x in files],dtype=h5py.string_dtype());out['source_sha256']=np.array(['']*len(files),dtype=h5py.string_dtype())
 out.attrs.update(normalization='P/median(P[z<85km],axis=range)-1',source_root=str(a.root))
 out['provenance/script']=Path(__file__).read_text()
 for i,path in enumerate(files):
  with h5py.File(path) as h:power=h['PL'][:];t=float(h['t0'][()])
  noise=np.median(power[z<85],axis=0)
  out['snr'][i]=(power[np.ix_(zi,fi)]/noise[fi]-1).astype('f4')
  out['time_unix_s'][i]=t;out['source_sha256'][i]=hashlib.sha256(path.read_bytes()).hexdigest()
  if i%100==0:print(i,len(files),flush=True)
print(a.output,flush=True)
