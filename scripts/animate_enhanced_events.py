"""Prepare native event spectra and assemble a one-second-per-record GIF.
Figures are rendered with Rustplotlib. Processing and selection provenance
are retained in HDF5, not overprinted on scientific figures.
"""
from pathlib import Path
import argparse, hashlib, subprocess
from datetime import datetime, timezone, timedelta
import h5py
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'media/enhanced_events';OUT.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'data/enhanced_events_animation.h5'

def prepare():
 with h5py.File(ROOT/'data/plasma_line_rcs.h5') as cat:
  rows=cat['catalog/rows'][:];rows=rows[np.isin(rows[:,0],[0,1])]
  records=np.unique(rows[:,1].astype(int));records=records[np.argsort(cat['time_unix_s'][records])]
  with h5py.File(DATA,'w') as out:
   out.attrs.update(frame_duration_ms=1000,selection='All catalogue detections in main upshifted 4.51 and 5.34 MHz bands; one frame per distinct acquisition record. Opposite and unclassified candidates excluded.',snr_definition='P/median(P[z<85km],axis=range)-1, as in event catalogue',display='10log10(max(SNR,10**(-1.5))); fixed -15..20 dB; no smoothing',source_catalog_sha256=hashlib.sha256((ROOT/'data/plasma_line_rcs.h5').read_bytes()).hexdigest())
   out['provenance/script']=Path(__file__).read_text()
   for j,i in enumerate(records):
    name=Path(cat['source_files'][i].decode()).name;p=ROOT/'evidence/event_animation'/name
    sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==cat['source_sha256'][i].decode()
    with h5py.File(p) as src:
     power=src['PL'][:];f=-src['freqs'][:];z=src['range'][:];t=float(src['t0'][()])
    fi=np.flatnonzero((f>=2)&(f<=6.5));fi=fi[np.argsort(f[fi])];zi=np.flatnonzero((z>=100)&(z<=140))
    noise=np.median(power[z<85],axis=0);snr=power[np.ix_(zi,fi)]/noise[fi]-1
    g=out.create_group(f'frames/{j:03d}');g['frequency_mhz']=f[fi];g['height_km']=z[zi];g.create_dataset('snr_db',data=10*np.log10(np.maximum(snr,10**(-1.5))),compression='gzip');g['events']=rows[rows[:,1]==i][:,[0,3,4]]
    utc=datetime.fromtimestamp(t,timezone.utc);ast=utc-timedelta(hours=4)
    g.attrs.update(label=f'4 May 2018: {utc:%H:%M:%S} UTC / {ast:%H:%M:%S} AST',frame_number=j+1,time_unix_s=t,record_index=int(i),source_file=name,source_sha256=sha)
   out.attrs.update(frame_count=len(records),detection_count=len(rows))
 print(DATA,flush=True)

def assemble():
 paths=sorted(OUT.glob('frame_*.png'));frames=[]
 for p in paths:
  im=Image.open(p).convert('RGB');im.thumbnail((1100,1000),Image.Resampling.LANCZOS);frames.append(im)
 with h5py.File(DATA) as h:assert len(frames)==h.attrs['frame_count']
 samples=[frames[i].resize((220,180)) for i in np.linspace(0,len(frames)-1,8).astype(int)]
 tile=Image.new('RGB',(880,360),'white')
 for i,im in enumerate(samples):tile.paste(im,((i%4)*220,(i//4)*180))
 palette=tile.quantize(colors=256)
 quant=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
 gif=OUT/'enhanced_plasma_line_events_1fps.gif'
 quant[0].save(gif,save_all=True,append_images=quant[1:],duration=1000,loop=0,disposal=2,optimize=False)
 with Image.open(gif) as test:
  assert test.n_frames==len(frames)
  for i in range(test.n_frames):test.seek(i);assert test.info['duration']==1000
 frames[len(frames)//2].save(OUT/'preview.png')
 print(gif, gif.stat().st_size, 'bytes;',len(frames),'frames',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','assemble']);args=p.parse_args()
 prepare() if args.mode=='prepare' else assemble()
