"""Direct spectral synthesis at 24 frames per carrier cycle, without strobing."""
from pathlib import Path
import numpy as np,h5py
from scipy.constants import epsilon_0,k as kb
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/gaussian_pulse'
with h5py.File(OUT/'movie.h5') as movie:
    begin,end=movie['views/closeup'][:]
    tpeak=movie['time_s'][int(np.argmax(np.max(abs(movie['density'][:,begin:end]),axis=1)))]
    zclose=movie['height_m'][begin:end]
with h5py.File(OUT/'response_dz0.002.h5') as h:
    meta=dict(h.attrs);z=h['height_m'][:];bo=slice(*h['views/bottomside'][:])
    indices=np.arange(len(z))[bo][(z[bo]>=zclose.min()-1e-8)&(z[bo]<=zclose.max()+1e-8)]
    assert len(indices)==len(zclose)
    ne=h['ne_m3'][indices]
    df=float(meta['frequency_spacing_hz']);f0=float(meta['carrier_hz'])
    offsets=np.arange(-160000.,160000.+df/2,df)
    spectra=np.stack([h[f'response/{off:+.3f}'][:][[2,9]][:,indices] for off in offsets])
    sigma=meta['field_envelope_FWHM_s']/np.sqrt(8*np.log(2));t0=meta['source_peak_time_s']
    weights=df*sigma*np.sqrt(2*np.pi)*np.exp(-.5*(2*np.pi*offsets*sigma)**2)*np.exp(2j*np.pi*offsets*t0)
    times=tpeak-2.5e-6+np.arange(640)/(24*f0)
    signals=(np.exp(-2j*np.pi*times[:,None]*offsets[None,:])*weights[None,:])@spectra.reshape(len(offsets),-1)
    signals=signals.reshape(len(times),2,-1)
    with h5py.File(OUT/'phase_movie.h5','w') as out:
        out.attrs.update(meta);out.attrs['time_sampling']='24 frames per actual carrier cycle; no stroboscopic time skipping'
        out['time_s']=times;out['height_m']=zclose;out['views/closeup']=[0,len(zclose)]
        out['Ez']=signals[:,0].astype('complex64')
        out['density']=(signals[:,1]*np.sqrt(epsilon_0*ne/(3*kb*300))).astype('complex64')
        out['provenance/source']=Path(__file__).read_text()
        print('PHASE CLIP',times[[0,-1]]*1e6,'us',flush=True)
