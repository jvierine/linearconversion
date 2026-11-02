"""Fast full-cadence scan with slowed detections and held episode maxima.
Rustplotlib supplies the color map and font convention. Axes, timestamps
and color bar are inset over the data; observations are never interpolated.
"""
from pathlib import Path
import subprocess,hashlib,argparse
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
size=(1280,960);box=(0,0,*size)
fontpath=subprocess.check_output(['fc-match','-f','%{file}','STIX Two Text']).decode()
font=ImageFont.truetype(fontpath,24);small=ImageFont.truetype(fontpath,20)
floor,ceiling=-25.,10.;sigma_mhz=.10;subtraction=.85
parser=argparse.ArgumentParser();parser.add_argument('--preview',type=int);args=parser.parse_args()
with h5py.File(ROOT/'data/plasma_line_rcs.h5') as h:
 rows=h['catalog/rows'][:];rows=rows[np.isin(rows[:,0],[0,1])];times=h['time_unix_s'][:]
ids=np.unique(rows[:,1].astype(int));groups=np.split(ids,np.flatnonzero(np.diff(times[ids])>90)+1)
holds=[]
for group in groups:
 sub=rows[np.isin(rows[:,1],group)];holds.append(int(sub[np.argmax(sub[:,8]),1]))
fps=30;duration=np.ones(len(times),dtype=int)
near=np.min(abs(times[:,None]-times[ids][None,:]),axis=1)<=30;duration[near]=6;duration[ids]=15;duration[holds]=90
out=OUT/'plasma_line_events_noise_visible.mp4'
cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{size[0]}x{size[1]}','-framerate',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','27','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
proc=None if args.preview is not None else subprocess.Popen(cmd,stdin=subprocess.PIPE)
with h5py.File(source) as h:
 assert len(times)==len(h['time_unix_s']);assert np.allclose(times,h['time_unix_s'][:],atol=.01,rtol=0)
 z=h['height_km'][:];f=h['frequency_mhz'][:];fi=np.flatnonzero((f>=2)&(f<=6.5));sigma=sigma_mhz/np.median(np.diff(f))
 df=np.median(np.diff(f));dz=np.median(np.diff(z));fmin=f[fi[0]]-df/2;fmax=f[fi[-1]]+df/2;zmin=z[0]-dz/2;zmax=z[-1]+dz/2
 for i,t in enumerate(times):
  if args.preview is not None and i!=args.preview:continue
  q=h['snr'][i];filtered=q-subtraction*gaussian_filter1d(q,sigma,axis=1,mode='nearest');qdb=10*np.log10(np.maximum(filtered[:,fi],10**(floor/10)))
  ix=np.clip(np.rint((qdb-floor)/(ceiling-floor)*(len(lut)-1)),0,len(lut)-1).astype(int)
  im=Image.fromarray(lut[ix][::-1]).resize(size,Image.Resampling.NEAREST);d=ImageDraw.Draw(im)
  utc=datetime.fromtimestamp(t,timezone.utc);ast=utc-timedelta(hours=4)
  mode='Pause: enhancement' if i in holds else ('Enhancement' if i in ids else ('Slow scan' if near[i] else 'Fast scan'))
  def text(pos,value,which=small,anchor='la'):
   d.text(pos,value,font=which,fill='white',anchor=anchor,stroke_width=1,stroke_fill='#171429')
  text((75,12),f'{utc:%Y-%m-%d   %H:%M:%S} UTC',font)
  text((75,43),f'{ast:%H:%M:%S} AST   |   {mode}')
  def xy(freq,alt):return (box[0]+(freq-fmin)/(fmax-fmin)*(box[2]-box[0]),box[3]-(alt-zmin)/(zmax-zmin)*(box[3]-box[1]))
  # Inset ticks and labels keep the whole frame available for spectra.
  for zz in [100,110,120,130,140]:
   yy=xy(2,zz)[1];d.line([(0,min(959,yy)),(10,min(959,yy))],fill='white',width=2)
   text((15,np.clip(yy,13,913)),str(zz),anchor='lm')
  alt=Image.new('RGBA',(180,28));ad=ImageDraw.Draw(alt);ad.text((0,0),'Altitude (km)',font=small,fill='white',stroke_width=1,stroke_fill='#171429');alt=alt.rotate(90,expand=True);im.paste(alt,(60,370),alt)
  for ff in [2,3,4,5,6]:
   xx=xy(ff,100)[0];d.line([(xx,959),(xx,949)],fill='white',width=2);text((max(20,xx),936),str(ff),anchor='mm')
  text((640,900),'Upshifted frequency (MHz)',anchor='mm')
  text((1260,21),'Filtered excess / noise (dB)',anchor='ra')
  bar=np.repeat(lut[np.linspace(len(lut)-1,0,200).astype(int)][:,None,:],14,axis=1)
  im.paste(Image.fromarray(bar),(1246,60));d=ImageDraw.Draw(im)
  for value in [-25,-20,-10,0,10]:
   yy=60+(ceiling-value)/(ceiling-floor)*199;d.line([(1240,yy),(1246,yy)],fill='white',width=1);text((1235,yy),str(value),anchor='rm')
  for row in rows[rows[:,1]==i]:
   f0=abs(row[3]);z0=row[4];start=xy(f0+.32,z0-4);tip=xy(f0+.035,z0-.45);v=np.array(tip)-start;v=v/np.linalg.norm(v);perp=np.array([-v[1],v[0]])
   a=np.array(tip)-12*v+5*perp;b=np.array(tip)-12*v-5*perp
   d.line([start,tip],fill='white',width=2);d.line([tuple(a),tip,tuple(b)],fill='white',width=2)
   d.text(xy(f0+.32,z0-5),('4.51' if row[0]==0 else '5.34')+' MHz',font=small,fill='white',anchor='mm')
  if args.preview is not None:
   im.save(OUT/f'compact_preview_{i:04d}.png');continue
  raw=im.tobytes()
  for _ in range(duration[i]):proc.stdin.write(raw)
  if i in holds:im.save(OUT/f'compact_pause_{i:04d}.png')
  if i%200==0:print('Encoded',i,'/',len(times),flush=True)
if args.preview is not None:raise SystemExit(0)
proc.stdin.close();assert proc.wait()==0
with h5py.File(ROOT/'data/plasma_line_movie_compact_timing.h5','w') as h:
 h['time_unix_s']=times;h['video_frames_per_record']=duration;h['episode_peak_record_index']=holds;h['detection_record_index']=ids
 h.attrs.update(video_fps=fps,gaussian_sigma_mhz=sigma_mhz,background_subtraction_fraction=subtraction,filter_axis='frequency only; q-0.85*gaussian(q), applied to linear excess/noise before log display',display_floor_db=floor,display_ceiling_db=ceiling,quiet_records_per_second=30,context_records_per_second=5,detection_records_per_second=2,peak_hold_s=3,raw_data_unchanged=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
 h['provenance/script']=Path(__file__).read_text()
print(out,'duration',duration.sum()/fps,'seconds; holds',holds,flush=True)
