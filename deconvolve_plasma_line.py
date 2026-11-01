"""Empirical 2-D range/frequency deconvolution of archived linear plasma-line power.

The 5.34-MHz feature is treated as unresolved to estimate an effective PSF.
4--5 MHz NEVER enters estimation. Missing kernel samples are continued using
joint delay/Doppler reflection, an assumption for a power ambiguity response.
This is not a recovered transmit waveform or independent width measurement.
Nonnegative L1-regularized least squares uses the signed baseline-subtracted
power, not the high-pass display or clipped/log-transformed data.
"""
from pathlib import Path
import hashlib
import h5py
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

ROOT=Path(__file__).resolve().parent
TRAIN=1525453430

def load(epoch):
    path=ROOT/f'evidence/seid_event/results4/spl-{epoch}.h5'
    with h5py.File(path) as h:
        p=h['PL'][:]; f=-h['freqs'][:]; z=h['range'][:]
    ix=np.where((f>=2)&(f<=8))[0][::-1]
    f=f[ix];p=p[:,ix]
    off=(z<105)|(z>130)
    bg=np.median(p[off],axis=0)
    sigma=1.4826*np.median(abs(p[off]-bg),axis=0)
    return f,z,p,bg,sigma,path

def estimate(threshold=3.):
    f,z,p,bg,noise,path=load(TRAIN)
    a=p-bg
    roi=(z[:,None]>116)&(z[:,None]<119)&(f[None,:]>5.2)&(f[None,:]<5.5)
    iz,jf=np.unravel_index(np.argmax(np.where(roi,a,-np.inf)),a.shape)
    # Peak location for missing-wing continuation; do not shift native samples.
    def vertex(v,i):
        return .5*(v[i-1]-v[i+1])/(v[i-1]-2*v[i]+v[i+1])
    cz=iz+vertex(a[:,jf],iz);cf=jf+vertex(a[iz],jf)
    rz,rf=44,102 # +/-6.6 km and +/-1.245 MHz: measured echo ambiguity support
    zz,ff=np.meshgrid(np.arange(iz-rz,iz+rz+1),np.arange(jf-rf,jf+rf+1),indexing='ij')
    valid=(f[ff]>5.) # strict exclusion of entire 4--5 MHz region
    mirrored=np.stack([2*cz-zz,2*cf-ff])
    raw=a[zz,ff]
    continuation=map_coordinates(a,mirrored,order=1,mode='constant',cval=0.)
    mirrored_f=np.interp(mirrored[1],np.arange(len(f)),f)
    assert np.all(mirrored_f[~valid]>5.)
    empirical=np.where(valid,raw,continuation)
    sig=np.where(valid,noise[ff],np.interp(mirrored[1],np.arange(len(f)),noise))
    # Threshold at measured off-source fluctuations; no spatial smoothing.
    kernel=np.maximum(empirical-threshold*sig,0.)
    kernel/=kernel.sum()
    return kernel,valid,dict(center_height_km=z[iz],center_frequency_mhz=f[jf],
        interpolated_center_height_km=np.interp(cz,np.arange(len(z)),z),
        interpolated_center_frequency_mhz=np.interp(cf,np.arange(len(f)),f),
        threshold_sigma=threshold),raw,sig

def invert(a,k,sigma,iterations=500):
    # ||K||_2 <= ||k||_1 = 1, hence unit step is safe for FISTA.
    adj=k[::-1,::-1]
    lam=3.*sigma*np.sqrt(np.sum(k*k))
    # Sparse matched-response initialization resolves the bright point response
    # before iterative refinement; otherwise its huge sidelobe integral makes
    # a generic positive-image start converge unnecessarily slowly.
    x=np.zeros_like(a);res=a.copy()
    norm=fftconvolve(np.ones_like(a),adj*adj,mode='same')
    for _ in range(30):
        corr=fftconvolve(res,adj,mode='same')-lam
        score=corr/np.sqrt(np.maximum(norm,1e-30))
        ij=np.unravel_index(np.argmax(score),a.shape)
        if score[ij]<=3*sigma: break
        amount=corr[ij]/norm[ij]
        x[ij]+=amount
        rz,rf=np.array(k.shape)//2;i,j=ij
        loz,hiz=max(0,i-rz),min(a.shape[0],i+rz+1)
        lof,hif=max(0,j-rf),min(a.shape[1],j+rf+1)
        res[loz:hiz,lof:hif]-=amount*k[loz-i+rz:hiz-i+rz,lof-j+rf:hif-j+rf]
    y=x.copy();t=1.;hist=[];snap={}
    for it in range(1,iterations+1):
        r=fftconvolve(y,k,mode='same')-a
        nxt=np.maximum(y-fftconvolve(r,adj,mode='same')-lam,0.)
        tn=(1+np.sqrt(1+4*t*t))/2
        y=nxt+(t-1)/tn*(nxt-x);x=nxt;t=tn
        if it in [100,250,500]:
            residual=a-fftconvolve(x,k,mode='same')
            hist.append([it,np.linalg.norm(residual),x.sum()])
            snap[it]=x.copy()
    return x,a-fftconvolve(x,k,mode='same'),np.array(hist),snap,lam

def restoring_core(kernel):
    """Gaussian fitted to the three native samples across each core axis.

    Peak is preserved, not unit-integral normalized: this is a conservative
    sidelobe-cleaned display, not a super-resolved estimate of intrinsic width.
    """
    iz,jf=np.array(kernel.shape)//2
    offsets=[];widths=[]
    for v,i in [(kernel[:,jf],iz),(kernel[iz],jf)]:
        aa,bb,_=np.polyfit([-1.,0.,1.],np.log(v[i-1:i+2]),2)
        offsets.append(-bb/(2*aa));widths.append(np.sqrt(-.5/aa))
    z,f=np.meshgrid(np.arange(kernel.shape[0])-iz,np.arange(kernel.shape[1])-jf,indexing='ij')
    core=np.exp(-.5*((z-offsets[0])/widths[0])**2-.5*((f-offsets[1])/widths[1])**2)
    core*=kernel[iz,jf]/core[iz,jf]
    return core,np.array(widths)*[.15,25/2048]

def run():
    kernel,mask,meta,raw,kn=estimate()
    out=ROOT/'data/plasma_line_deconvolution.h5'
    temporary=out.with_suffix('.partial.h5')
    core,core_widths=restoring_core(kernel)
    with h5py.File(temporary,'w') as h:
        h['psf/restoring_core']=core
        h['psf/restoring_sigma_km_mhz']=core_widths
        h['psf/kernel']=kernel;h['psf/measured_mask']=mask
        h['psf/empirical_signed_power']=raw;h['psf/noise_scale']=kn
        h['psf/delta_height_km']=np.arange(-44,45)*.15
        h['psf/delta_frequency_mhz']=np.arange(-102,103)*25/2048
        h['psf'].attrs.update(meta)
        h.attrs.update(training_epoch=TRAIN,excluded_training_frequency_mhz=[4.,5.],
            method='nonnegative L1 regularized least squares, FISTA, 500 iterations; zero padded linear convolution',
            psf_status='empirical effective response; assumes 5.34 MHz enhancement unresolved and locally shift invariant; masked wing inferred using joint reflection',
            caveat='source width and PSF are not identifiable from one feature; recovered widths and absolute integrated power are not calibrated; independent time is a transfer check, not known-truth validation',
            restoration='Gaussian restoring core fitted to three native samples along each axis, same peak response, plus signed residual; no claim of resolved intrinsic source width',
            baseline='per-frequency median over ranges below 105 or above 130 km; signed residual fitted',
            provenance='deconvolve_plasma_line.py; plotter/src/bin/plasma_deconvolution.rs',
            reflection_reference='https://doi.org/10.1049/rsn2.70005')
        for epoch in [TRAIN,1525453540,1525460596]:
            f,z,p,bg,sigma,path=load(epoch)
            unit=1e14;a=(p-bg)/unit
            noise=float(np.median(sigma[(f>4)&(f<6.5)])/unit)
            x,res,hist,snap,lam=invert(a,kernel,noise)
            g=h.create_group(str(epoch))
            for name,v in dict(frequency_mhz=f,height_km=z,raw_power=p,background_power=bg,
                noise_sigma_power=sigma,observed_excess=a,recovered=x,reconvolved=a-res,
                residual=res,restored=fftconvolve(x,core,mode='same')+res,iteration_diagnostics=hist,recovered_100=snap[100],recovered_250=snap[250]).items():
                g.create_dataset(name,data=v,compression='gzip')
            g.attrs.update(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                power_unit=unit,noise_scale=noise,lambda_l1=lam,independent_of_psf_fit=(epoch!=TRAIN))
            # Exclude both source cores when measuring sidelobe region power.
            center=117.75 if epoch<1525460000 else 114.15
            side=(abs(z[:,None]-center)>.6)&(abs(z[:,None]-center)<6)&(abs(f[None,:]-5.34)<.22)
            core_mask=(abs(z[:,None]-center)<.6)&(abs(f[None,:]-5.34)<.10)
            secondary=(abs(z[:,None]-center)<.8)&(f[None,:]>4.40)&(f[None,:]<4.65)
            g.attrs['sidelobe_region_rms_before']=np.sqrt(np.mean(a[side]**2))
            g.attrs['sidelobe_region_rms_residual']=np.sqrt(np.mean(res[side]**2))
            g.attrs['secondary_recovered_sum']=x[secondary].sum()
            g.attrs['secondary_core_relative_rms']=np.linalg.norm(res[secondary])/max(np.linalg.norm(a[secondary]),1e-30)
            g.attrs['core_reconvolution_relative_rms']=np.linalg.norm(res[core_mask])/np.linalg.norm(a[core_mask])
            print(epoch,dict(g.attrs),flush=True)
        # Kernel threshold sensitivity on training frame, retained explicitly.
        g=h[str(TRAIN)];a=g['observed_excess'][:];noise=g.attrs['noise_scale']
        for th in [2.,4.]:
            kk,*_=estimate(th);xx,rr,_,_,_=invert(a,kk,noise,iterations=250)
            sg=h.create_group(f'sensitivity/threshold_{int(th)}');sg['kernel']=kk;sg['recovered']=xx;sg['residual']=rr
    temporary.replace(out)
    print(out)

if __name__=='__main__':run()
