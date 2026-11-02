"""Fast full-cadence scan with slowed detections and held episode maxima.
Rustplotlib supplies the axes, typography and color map. Pixel compositing
caches those static elements; it does not interpolate between observations.
"""
from pathlib import Path
import subprocess,hashlib,json
from datetime import datetime,timezone,timedelta
import h5py,numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image,ImageFont,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'media/enhanced_events'
source=ROOT/'data/plasma_line_movie_native.h5';template=OUT/'movie_template.png'
base=Image.open(template).convert('RGB');rgb=np.asarray(base)
dark=(rgb.max(axis=2)<110);ys=np.flatnonzero(dark.mean(axis=1)>.5);top,bottom=int(ys[0]),int(ys[-1])+1
xs=np.flatnonzero(dark[(top+bottom)//2]);runs=np.split(xs,np.flatnonzero(np.diff(xs)>1)+1);xs=max(runs,key=len);left,right=int(xs[0])+1,int(xs[-1])
# Read the exact Rustplotlib ember lookup directly from its rendered color bar.
nonwhite=(rgb.min(axis=2)<230)
barrows=np.flatnonzero(nonwhite[bottom+40:].sum(axis=1)>rgb.shape[1]*.6)+bottom+40;ybar=int(np.median(barrows));xx=np.flatnonzero(nonwhite[ybar]);lut=rgb[ybar,xx[0]+2:xx[-1]-1]
size=(1280,round(base.height*1280/base.width/2)*2);sx=size[0]/base.width;sy=size[1]/base.height
box=tuple(round(v*s) for v,s in zip((left,top,right,bottom),(sx,sy,sx,sy)));base=base.resize(size,Image.Resampling.LANCZOS)
header=80;canvas=Image.new("RGB",(size[0],size[1]+header),"white");canvas.paste(base,(0,header));base=canvas;size=base.size;box=(box[0],box[1]+header,box[2],box[3]+header)
fontpath=subprocess.check_output(['fc-match','-f','%{file}','STIX Two Text']).decode();font=ImageFont.truetype(fontpath,23);small=ImageFont.truetype(fontpath,17)
with h5py.File(ROOT/'data/plasma_line_rcs.h5') as h:
 rows=h['catalog/rows'][:];rows=rows[np.isin(rows[:,0],[0,1])];times=h['time_unix_s'][:]
ids=np.unique(rows[:,1].astype(int));groups=np.split(ids,np.flatnonzero(np.diff(times[ids])>90)+1)
holds=[]
for group in groups:
 sub=rows[np.isin(rows[:,1],group)];holds.append(int(sub[np.argmax(sub[:,8]),1]))
fps=30;duration=np.ones(len(times),dtype=int)
near=np.min(abs(times[:,None]-times[ids][None,:]),axis=1)<=30;duration[near]=6;duration[ids]=15;duration[holds]=90
out=OUT/'plasma_line_events_fast_scan_highpass.mp4'
cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{size[0]}x{size[1]}','-framerate',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
with h5py.File(source) as h:
 assert len(times)==len(h['time_unix_s']);assert np.allclose(times,h['time_unix_s'][:],atol=.01,rtol=0)
 z=h['height_km'][:];f=h['frequency_mhz'][:];fi=np.flatnonzero((f>=2)&(f<=6.5));sigma=.1/np.median(np.diff(f))
 for i,t in enumerate(times):
  q=h['snr'][i];filtered=q-gaussian_filter1d(q,sigma,axis=1,mode='nearest');qdb=10*np.log10(np.maximum(filtered[:,fi],10**(-1.5)))
  ix=np.clip(np.rint((qdb+15)/35*(len(lut)-1)),0,len(lut)-1).astype(int)
  im=base.copy();im.paste(Image.fromarray(lut[ix][::-1]).resize((box[2]-box[0],box[3]-box[1]),Image.Resampling.NEAREST),box[:2]);d=ImageDraw.Draw(im)
  utc=datetime.fromtimestamp(t,timezone.utc);ast=utc-timedelta(hours=4)
  d.text((size[0]/2,22),f'4 May 2018: {utc:%H:%M:%S} UTC / {ast:%H:%M:%S} AST',font=font,fill='black',anchor='mm')
  mode='Pause: enhancement' if i in holds else ('Enhancement' if i in ids else ('Slow scan' if near[i] else 'Fast scan'))
  d.text((size[0]/2,51),mode+'  |  Frequency high-pass: 100 kHz Gaussian background',font=small,fill='#555555',anchor='mm')
  def xy(freq,alt):return (box[0]+(freq-2)/4.5*(box[2]-box[0]),box[3]-(alt-100)/40*(box[3]-box[1]))
  for row in rows[rows[:,1]==i]:
   f0=abs(row[3]);z0=row[4];start=xy(f0+.32,z0-4);tip=xy(f0+.035,z0-.45);v=np.array(tip)-start;v=v/np.linalg.norm(v);perp=np.array([-v[1],v[0]])
   a=np.array(tip)-12*v+5*perp;b=np.array(tip)-12*v-5*perp
   d.line([start,tip],fill='white',width=2);d.line([tuple(a),tip,tuple(b)],fill='white',width=2)
   d.text(xy(f0+.32,z0-5),('4.51' if row[0]==0 else '5.34')+' MHz',font=small,fill='white',anchor='mm')
  raw=im.tobytes()
  for _ in range(duration[i]):proc.stdin.write(raw)
  if i in holds:im.save(OUT/f'movie_pause_{i:04d}.png')
  if i%200==0:print('Encoded',i,'/',len(times),flush=True)
proc.stdin.close();assert proc.wait()==0
with h5py.File(ROOT/'data/plasma_line_movie_timing.h5','w') as h:
 h['time_unix_s']=times;h['video_frames_per_record']=duration;h['episode_peak_record_index']=holds;h['detection_record_index']=ids
 h.attrs.update(video_fps=fps,gaussian_sigma_mhz=.1,filter_axis='frequency only; q-gaussian(q), applied to linear excess/noise before log display',display_floor_db=-15.,display_ceiling_db=20.,quiet_records_per_second=30,context_records_per_second=5,detection_records_per_second=2,peak_hold_s=3,raw_data_unchanged=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
 h['provenance/script']=Path(__file__).read_text()
print(out,'duration',duration.sum()/fps,'seconds; holds',holds,flush=True)
