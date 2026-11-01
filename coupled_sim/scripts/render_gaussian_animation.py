"""Animate scientific fields using Rustplotlib axes, fonts and transfer functions.

Only the field raster and the physical-time counter vary between frames. The
Rustplotlib palettes and asinh mapping are identical to the inset colorbars.
"""
from pathlib import Path
import subprocess,json,argparse
import numpy as np,h5py
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/gaussian_pulse';MEDIA=ROOT/'media/conversion';MEDIA.mkdir(exist_ok=True)
FONT=ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf',25)
PIX=4.8
CASES={'overview':[('Ez','overview',1e3),('density','closeup',1e-6)],
       'fields':[('Ez','closeup',1e3),('density','closeup',1e-6)],
       'flux':[('EM_flux','overview',1e12),('pressure_flux','wave_path',1e15)],
       'current':[('Jx','closeup',1e6),('Jz','closeup',1e6)]}
def rounded(value):
 exponent=10**np.floor(np.log10(max(value,1e-30)))
 return np.ceil(value/exponent)*exponent

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--preview",action="store_true");parser.add_argument('--phase',action='store_true');args=parser.parse_args()
 subprocess.run(['cargo','build','--release','--locked','--manifest-path',str(ROOT/'plotter/Cargo.toml'),'--bin','gaussian_movie_background'],check=True)
 with h5py.File(OUT/('phase_movie.h5' if args.phase else 'movie.h5')) as h:
  times=h['time_s'][:];angle=float(h.attrs['angle_deg']);f0=h.attrs['carrier_hz'];kx=h.attrs['kx_m1'];bottom=h.attrs['bottom_resonance_m'];top=h.attrs['top_resonance_m'];zall=h['height_m'][:]
  for case,panels in CASES.items():
   if args.phase and case!='fields':continue
   arrays=[];maxima=[];zs=[]
   for name,view,unit in panels:
    begin,end=h['views/'+view][:];arr=h[name][:,begin:end]*unit
    arrays.append(arr);zs.append(zall[begin:end]);maxima.append(rounded(np.max(abs(arr))))
   subprocess.run([str(ROOT/'plotter/target/release/gaussian_movie_background'),str(OUT),case,*map(str,maxima)],check=True,cwd=ROOT)
   base=Image.open(OUT/f'{case}_background.png').convert('RGB');assert base.size==(1920,1080),base.size
   boxes=[];patches=[];palettes=[];locations=[]
   for j,(_,view,_) in enumerate(panels):
    left=25 if j==0 else 224
    box=tuple(round(v*PIX) for v in [left,12,left+170,206]);boxes.append(box)
    patchbox=tuple(round(v*PIX) for v in [left+98,14,left+169,43]);patches.append((patchbox,base.crop(patchbox)))
    palettes.append(np.fromfile(OUT/f'{case}_palette_{j}.rgb',np.uint8).reshape(-1,3))
    height=box[3]-box[1];width=box[2]-box[0]
    if view=='overview':zz=np.linspace(120800,115100,height)
    elif view=='wave_path':zz=bottom+np.linspace(40,-160,height)
    else:zz=bottom+np.linspace(5,-20,height)
    xx=np.linspace(-100,100,width)
    locations.append((zz,np.exp(1j*kx*xx)))
   output_case='phase_resolved' if args.phase else case
   dest=MEDIA/f'gaussian_10us_{output_case}.mp4'
   cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
   proc=None if args.preview else subprocess.Popen(cmd,stdin=subprocess.PIPE)
   peak=int(np.argmax(np.max(abs(arrays[1]),axis=1)))
   poster=int(np.argmin(abs(times-80e-6))) if case in ['overview','flux'] else peak
   for frame,t in enumerate(times):
    if args.preview and frame not in [0,peak,poster,len(times)-1]:continue
    image=base.copy()
    for j,(name,view,_) in enumerate(panels):
     zz,xphase=locations[j];vertical=np.interp(zz,zs[j],arrays[j][frame])
     if case=='flux':values=np.broadcast_to(vertical[:,None],(len(zz),len(xphase)))
     else:values=np.real(vertical[:,None]*xphase[None,:]*np.exp(-2j*np.pi*f0*t))
     stretch=300 if view=='overview' else 30
     normalized=.5+.5*np.arcsinh(values/(maxima[j]/stretch))/np.arcsinh(stretch)
     raster=palettes[j][np.rint(np.clip(normalized,0,1)*65535).astype(np.uint16)]
     image.paste(Image.fromarray(raster),boxes[j][:2]);patchbox,patch=patches[j];image.paste(patch,patchbox[:2])
    draw=ImageDraw.Draw(image)
    draw.text((960,5),f'5.34 MHz   |   10 μs envelope   |   {angle:.2f}° incidence   |   t = {t*1e6:.2f} μs',font=FONT,fill='black',anchor='mt')
    for j,(_,view,_) in enumerate(panels):
     box=boxes[j]
     if view=='overview':
      for altitude,label in [(bottom,'Bottomside'),(top,'Topside')]:
       yy=box[1]+(120800-altitude)/(120800-115100)*(box[3]-box[1])
       for xx in range(box[0],box[2],20):draw.line((xx,yy,min(xx+10,box[2]),yy),fill=(80,80,80),width=1)
       draw.text((box[0]+8,yy-3),label,font=FONT,fill='black',stroke_width=1,stroke_fill='white',anchor='lb')
     else:
      label='Bottomside close-up' if view=='closeup' else 'Downward wave path'
      draw.text((box[0]+10,box[1]+10),label,font=FONT,fill='black',stroke_width=1,stroke_fill='white')
    note='Carrier phase resolved; time is physical' if args.phase else 'Cycle-averaged energy flux; positive upward' if case=='flux' else 'Carrier shown stroboscopically; time is physical'
    draw.text((960,1073),note,font=FONT,fill=(60,60,60),anchor='mb')
    if proc is not None:proc.stdin.write(image.tobytes())
    if frame in [0,peak,poster,len(times)-1]:image.save(OUT/f'{case}_{frame:04d}.png')
    if frame==poster:image.save(MEDIA/f'gaussian_10us_{output_case}.png')
    if frame%200==0:print('render',case,frame,'/',len(times),flush=True)
   if args.preview:continue
   proc.stdin.close();assert proc.wait()==0
   probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(dest)]))
   assert int(probe['streams'][0]['nb_frames'])==len(times)
   assert abs(float(probe['format']['duration'])-len(times)/30)<.05
   print('VIDEO',dest,probe['format']['duration'],flush=True)
if __name__=='__main__':main()
