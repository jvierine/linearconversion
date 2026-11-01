"""Join startup diagnostics and the article's resolved steady phase into one MP4."""
from pathlib import Path
import json
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/gaussian_pulse'
MEDIA = ROOT/'media/conversion'
CASES = [
    ('continuous_startup', '1/5 · Incoming electric field and density · Switch-on; carrier sampled stroboscopically'),
    ('continuous_startup_fields', '2/5 · Local electric field and density · Switch-on; carrier sampled stroboscopically'),
    ('continuous_startup_flux', '3/5 · Electromagnetic and plasma-wave energy flux · Positive upward'),
    ('continuous_startup_current', '4/5 · Horizontal and vertical current · Switch-on; carrier sampled stroboscopically'),
    ('steady_article', '5/5 · Article steady-state phase · 48 frames per carrier cycle; no skipped cycles'),
]
font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Times New Roman.ttf', 20)
command = ['ffmpeg', '-y', '-loglevel', 'error']
frames = 0
chapters = [';FFMETADATA1']
elapsed_ms = 0
for name, title in CASES:
    path = MEDIA/f'{name}.mp4'
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_frames,duration','-of','json',str(path)]))['streams'][0]
    assert (probe['width'],probe['height']) == (1920,1080)
    count = int(probe['nb_frames'])
    frames += count
    end_ms = elapsed_ms + round(count/30*1000)
    chapters += ['[CHAPTER]', 'TIMEBASE=1/1000', f'START={elapsed_ms}', f'END={end_ms}', f'title={title}']
    elapsed_ms = end_ms
    command += ['-i',str(path)]
for name, title in CASES:
    footer = Image.new('RGB',(1920,23),'white')
    draw = ImageDraw.Draw(footer)
    assert draw.textlength(title,font=font) < 1880
    draw.text((960,22),title,font=font,fill='black',anchor='mb')
    path = OUT/f'{name}_section_footer.png'
    footer.save(path)
    command += ['-loop','1','-i',str(path)]
metadata = OUT/'continuous_chapters.ffmeta'
metadata.write_text('\n'.join(chapters)+'\n')
command += ['-i',str(metadata)]
n = len(CASES)
filters = ';'.join(f'[{i}:v][{i+n}:v]overlay=0:1057:shortest=1[v{i}]' for i in range(n))
filters += ';'+''.join(f'[v{i}]' for i in range(n))+f'concat=n={n}:v=1:a=0[v]'
output = MEDIA/'continuous_all_views.mp4'
command += ['-filter_complex_threads','2','-filter_complex',filters,'-map','[v]',
    '-map_metadata',str(2*n),'-map_chapters',str(2*n),'-an','-c:v','libx264',
    '-preset','fast','-threads','4','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
subprocess.run(command,check=True)
probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_chapters',
    '-show_format','-of','json',str(output)]))
assert int(probe['streams'][0]['nb_frames']) == frames
assert len(probe['chapters']) == n
assert abs(float(probe['format']['duration'])-frames/30) < .05
print(output, probe['format']['duration'], 'seconds;', frames, 'frames;', n, 'views', flush=True)
