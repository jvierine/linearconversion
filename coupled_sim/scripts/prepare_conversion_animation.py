"""Prepare display arrays from native coupled E/q recordings, retaining SI data.

Visualization only. Density is q*sqrt(eps0*n0/(m*cs^2)); no fitted/artificial
upgoing wave or envelope. Each movie displays one native HF period, slowed down.
The single Fourier-x mode reconstructs x dependence exactly within this model.
"""
from pathlib import Path
import argparse,json
import numpy as np,h5py
from scipy.constants import epsilon_0,m_e,c
from scipy.optimize import brentq
import coupled_arecibo as a
from lmode_animation import OUT,BOTTOM,TOP,ROOT

def interp(z,x,y):return np.interp(z,x,y.real)+1j*np.interp(z,x,y.imag)

def prepare(label,side):
    with h5py.File(OUT/(label+'.h5')) as h:
        attrs=dict(h.attrs);z=h['z_m'][:];zf=h['z_faces_m'][:];ne=h['ne_m3'][:]
        den=h['density_phasor_m3'][:];ep=h['E_phasor_V_m'][:];inc=h['incident_state'][:]
        kx=float(h['kx_rad_m'][()]);tt=h['native/time_s'][:]
        actual=h['native/E_q_real'][:]+1j*h['native/E_q_imag'][:]
        actual[:,3]*=np.sqrt(epsilon_0*ne/(m_e*a.CS**2))[None,:]
        # All cases have the same local incident electric-field magnitude, 1mV/m
        # peak at 115.1 km. The scan itself instead compares unit vertical flux.
        zi=115100.
        Einc=np.linalg.norm([interp(zi,z,inc[0]),interp(zi,z,inc[1]),interp(zi,zf,inc[2])])
        scale=1e-3/Einc
        r=TOP if side>0 else BOTTOM
        offsets=np.linspace(-3,16,1901) if side>0 else np.linspace(-16,3,1901)
        zz=r+offsets;nn=np.interp(zz,z,ne)
        d=interp(zz,z,den)*scale/nn*1e6
        ez=interp(zz,zf,ep[2])/Einc
        dt=np.array([interp(zz,z,row[3])*scale/nn*1e6 for row in actual])
        et=np.array([interp(zz,zf,row[2])/Einc for row in actual])
        # Show a continuous local phase; fit d phase over adjacent nodes at
        # Bragg scale. Only use the outgoing branch outside the conversion core.
        phase=np.unwrap(np.angle(d));kz=np.gradient(phase,zz)
        kb=4*np.pi*430e6/c
        check=(offsets*side>1)&(offsets*side<9)
        inds=np.where(check[:-1]&check[1:]&((abs(kz[:-1])-kb)*(abs(kz[1:])-kb)<0))[0]
        if len(inds):
            i=inds[np.argmin(abs(offsets[inds]))]
            bragg=float(np.interp(kb,np.sort(abs(kz[i:i+2])),offsets[i:i+2][np.argsort(abs(kz[i:i+2]))]))
        else:bragg=float('nan')
        x=np.linspace(-60,60,121)
        ds=np.max(abs(d));es=np.max(abs(ez))
        out=OUT/(label+'_display.h5')
        with h5py.File(out,'w') as f:
            f.attrs.update(attrs)
            f.attrs.update(incident_E_peak_V_m=.001,display_field_scale=scale,
                propagation_direction=side,bragg_offset_m=bragg,delta_ne_max_ppm=ds,Ez_max_ratio=es,
                scope='Native coupled CW recording, x dependence from single Fourier mode; one HF period slowed for display; no Landau damping',
                local_resonance_m=r)
            datasets=dict(z_relative_m=offsets,x_m=x,time_s=tt,
                density_real_ppm=dt.real,density_imag_ppm=dt.imag,
                Ez_real_ratio=et.real,Ez_imag_ratio=et.imag,
                density_envelope_ppm=abs(d),Ez_envelope_ratio=abs(ez),
                kz_rad_m=kz,wavelength_m=2*np.pi/np.maximum(abs(kz),1e-12),
                profile_height_km=np.linspace(117.5,118.8,1301),
                profile_ne_1e11=a.PROFILE(np.linspace(117500,118800,1301))/1e11,
                kx_rad_m=kx,phase_flux_fraction=attrs['upward_pressure_fraction' if side>0 else 'downward_pressure_fraction'])
            for key,value in datasets.items():f.create_dataset(key,data=value,compression='gzip' if np.ndim(value) else None)
            f['provenance/source']=Path(__file__).read_text()
        print(out,'Bragg offset',bragg,'peak ppm',ds,'peak Ez/Einc',es,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--side',type=int,choices=[-1,1],default=1);args=p.parse_args();prepare(args.label,args.side)
