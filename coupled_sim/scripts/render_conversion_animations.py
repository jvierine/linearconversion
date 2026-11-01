"""Rustplotlib frames and H.264/GIF movies of native coupled FDTD/fluid data."""
from pathlib import Path
import argparse,subprocess,json,shutil
import h5py
from prepare_conversion_animation import prepare
from lmode_animation import ROOT,OUT

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['topside_optimum','bottomside_optimum','topside_spitze']);args=p.parse_args()
    cases=[('topside_optimum',1,'Whole-layer simulation: topside-favoring incidence'),
           ('bottomside_optimum',-1,'Whole-layer simulation: bottomside-favoring incidence'),
           ('topside_spitze',1,'Whole-layer simulation: Spitze transmission window')]
    subprocess.run(['cargo','build','--release','--locked','--manifest-path',str(ROOT/'plotter/Cargo.toml'),'--bin','conversion_animation'],check=True)
    for label,side,title in cases:
        if args.case and args.case!=label:continue
        prepare(label,side)
        folder=OUT/(label+'_frames')
        subprocess.run([str(ROOT/'plotter/target/release/conversion_animation'),str(OUT/(label+'_display.h5')),str(folder),title],check=True,cwd=ROOT)
        one=OUT/(label+'_period.mp4');final=OUT/(label+'.mp4')
        subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','24','-i',str(folder/'frame_%03d.png'),'-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(one)],check=True)
        # Three repeats of the periodic CW solution. The displayed time resets
        # at each period; no turn-on, growth or moving envelope is fabricated.
        subprocess.run(['ffmpeg','-y','-loglevel','error','-stream_loop','2','-i',str(one),'-c','copy','-movflags','+faststart',str(final)],check=True)
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(one),'-filter_complex','fps=18,scale=1100:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer','-loop','0',str(OUT/(label+'.gif'))],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(final)]))
        assert abs(float(probe['format']['duration'])-12)<.1
        portable=ROOT/'reference/coupled/conversion_animation';portable.mkdir(parents=True,exist_ok=True)
        media=ROOT/'media/conversion';media.mkdir(parents=True,exist_ok=True)
        shutil.copy2(OUT/(label+'_display.h5'),portable/(label+'.h5'))
        shutil.copy2(final,media/(label+'.mp4'))
        shutil.copy2(folder/'frame_000.png',media/(label+'.png'))
        with h5py.File(portable/(label+'.h5'),'a') as h:
            for key,path in [('rust_driver','coupled_sim/src/main.rs'),('coupled_timestep','coupled_sim/src/lib.rs'),('Cargo.lock','coupled_sim/Cargo.lock'),('renderer','plotter/src/bin/conversion_animation.rs')]:
                h['provenance/'+key]=(ROOT/path).read_text()
        for filename in ['scan.h5','validation.h5']:
            if (OUT/filename).exists():shutil.copy2(OUT/filename,portable/filename)
        print('VIDEO',final,'duration',probe['format']['duration'],flush=True)

if __name__=='__main__':main()
