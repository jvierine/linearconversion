"""One MP4 cycles through three two-panel views of the same pulse."""
from pathlib import Path
import subprocess,json
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/gaussian_pulse';MEDIA=ROOT/'media/conversion'
CASES=[('overview','1/3 · Pulsed field and density: phase mixing  |  Stroboscopic carrier'),
       ('flux','2/3 · Electromagnetic and plasma-wave energy flux  |  Positive upward'),
       ('current','3/3 · Horizontal and vertical current  |  Carrier sampled stroboscopically')]
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf',21)
cmd=['ffmpeg','-y','-loglevel','error']
for case,_ in CASES:cmd+=['-i',str(MEDIA/f'gaussian_10us_{case}.mp4')]
for case,label in CASES:
    image=Image.new('RGB',(1920,25),'white');draw=ImageDraw.Draw(image)
    draw.text((960,23),label,font=font,fill='black',anchor='mb')
    path=OUT/f'{case}_footer.png';image.save(path);cmd+=['-loop','1','-i',str(path)]
filters=';'.join(f'[{j}:v]drawbox=x=700:y=1049:w=520:h=31:color=white:t=fill[clean{j}];[clean{j}][{j+3}:v]overlay=0:1055:shortest=1[v{j}]' for j in range(3))
filters+=';[v0][v1][v2]concat=n=3:v=1:a=0[v]'
output=MEDIA/'gaussian_10us_probes.mp4'
cmd+=['-filter_complex_threads','2','-filter_complex',filters,'-map','[v]','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
subprocess.run(cmd,check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(output)]))
assert probe['streams'][0]['width']==1920 and probe['streams'][0]['height']==1080
assert int(probe['streams'][0]['nb_frames'])==3*986
print(output,probe['format']['duration'],flush=True)
