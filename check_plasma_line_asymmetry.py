"""Paired native spectra; preserve signed excess and archived frequency sign.

The display convention f_display=-f_archive follows recovered event plots.
RF sign is adopted from the original manuscript, not independently established
by these power arrays. Off-range scatter is an empirical sensitivity scale,
not a confidence interval: neighboring range gates are correlated.
"""
from pathlib import Path
import hashlib
import h5py
import numpy as np
from scipy.integrate import trapezoid
from scipy.constants import e, m_e, epsilon_0
from estimate_hf_power import beam, KR

ROOT=Path(__file__).resolve().parent
F0=5.34002855

def main():
    out=ROOT/'output/fluid/plasma_line_asymmetry.h5'
    with h5py.File(out,'w') as h:
        h.attrs.update(provenance='check_plasma_line_asymmetry.py',
            sign_convention='display=-archive; enhanced positive-display branch identified as upshifted by original manuscript; RF sign not independently verified',
            sensitivity='standard deviation of fitted amplitudes at off-range gates; correlated gates; not a formal confidence bound',
            pulse_duration_s=44e-6,template_frequency_mhz=F0)
        for epoch in [1525453430,1525453540,1525460596]:
            path=ROOT/f'evidence/seid_event/results4/spl-{epoch}.h5'
            with h5py.File(path) as src:
                f=-src['freqs'][:];z=src['range'][:];p=src['PL'][:]
            off=(z<105)|(z>130)
            bg=np.median(p[off],axis=0);q=p/bg-1
            roi=(z[:,None]>110)&(z[:,None]<120)&(f[None,:]>5.2)&(f[None,:]<5.5)
            iz,j=np.unravel_index(np.argmax(np.where(roi,q,-np.inf)),q.shape)
            g=h.create_group(str(epoch));g['height_km']=z
            g.attrs.update(height_km=z[iz],source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for name,sign in [('enhanced',1),('opposite',-1)]:
                idx=np.where(abs(f-sign*F0)<.075)[0]
                idx=idx[np.argsort(sign*f[idx])]
                ff=sign*f[idx];template=np.sinc((ff-F0)*44)**2
                X=np.array([template,np.ones(len(template))]).T
                coef=np.linalg.lstsq(X,q[:,idx].T,rcond=None)[0].T
                b=g.create_group(name);b['frequency_magnitude_mhz']=ff
                b['signed_excess_snr']=q[iz,idx];b['fit']=X@coef[iz]
                b['amplitude_by_height']=coef[:,0]
                b.attrs.update(amplitude=coef[iz,0],offset=coef[iz,1],
                    off_range_amplitude_mean=np.mean(coef[off,0]),
                    off_range_amplitude_std=np.std(coef[off,0]))
            amp=g['enhanced'].attrs['amplitude'];sig=g['opposite'].attrs['off_range_amplitude_std']
            g.attrs['three_sigma_sensitivity_relative_to_enhanced_db']=10*np.log10(3*sig/amp)
            print(epoch,dict(g.attrs),dict(g['opposite'].attrs))
        # Existing driven solution: compare conjugate density components with
        # the same radar operator. Approximate kB=2kr for both sidebands here.
        with h5py.File(ROOT/'output/fluid/fdtd_extended_kz.h5') as src:
            z=src['setup/z'][:];ne=src['setup/ne'][:]
            dn=src['full/dn_phasor_m3'][:];s=dict(src['setup'].attrs)
        dz=z-s['center'];_,a,b,gouy=beam(z)
        J=a/(a-1j*b)*np.exp(-s['kx']**2/(4*(a-1j*b)))*np.exp(-2j*gouy)
        powers=[]
        for lo,hi in [(-80.,20.),(-100.,30.),(-110.,40.)]:
            win=np.sin(.5*np.pi*np.clip(np.minimum(dz-lo,hi-dz)/10,0,1))**2
            op=np.exp(2j*KR*dz)*J*win
            powers.append([abs(trapezoid(.5*d*op,z))**2 for d in [dn,dn.conj()]])
        g=h.create_group('model');g['window_comparison_power']=powers
        g.attrs['opposite_over_enhanced_power_db']=10*np.log10(powers[-1][1]/powers[-1][0])
        keep=(dz>-8)&(dz<4)
        g['height_relative_m']=dz[keep]
        g['local_collisionless_kz_squared_m2']=(s['D']-ne[keep]*e**2/(m_e*epsilon_0))/s['cs2']
        g.attrs.update(bragg_height_relative_m=s['match']-s['center'],
            caveat='1D vertical warm-fluid local branch, collisional driven solution; no kinetic damping; weak opposite overlap can be sensitive to source and window; not an exactly zero return')
        print('model',dict(g.attrs),powers)
    print(out)

if __name__=='__main__':main()
