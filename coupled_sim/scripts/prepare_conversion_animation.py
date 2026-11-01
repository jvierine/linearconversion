"""Native E/density movies with both conversion layers and signed energy flux.

Only visualization/resampling: no imposed upward wave. EM flux is the
cycle-averaged E cross B / mu0, collocated on Yee faces. Incident/reflected
separation is meaningful at outer ports; interior curves are signed NET flux.
Plasma-wave pressure flux is shown separately from electromagnetic Poynting flux.
"""
from pathlib import Path
import argparse
import numpy as np
import h5py
from scipy.constants import epsilon_0,m_e,c
import coupled_arecibo as a
from lmode_animation import OUT,BOTTOM,TOP

def interp(z,x,y):
    return np.interp(z,x,y.real)+1j*np.interp(z,x,y.imag)

def put(group,name,value):
    group.create_dataset(name,data=value,compression='gzip' if np.ndim(value) else None)

def prepare(label,side):
    with h5py.File(OUT/(label+'.h5')) as h:
        attrs=dict(h.attrs);z=h['z_m'][:];zf=h['z_faces_m'][:];ne=h['ne_m3'][:]
        den=h['density_phasor_m3'][:];state=h['state_phasor'][:];ep=state[:3];inc=h['incident_state'][:]
        kx=float(h['kx_rad_m'][()]);tt=h['native/time_s'][:]
        actual=h['native/E_q_real'][:]+1j*h['native/E_q_imag'][:]
        actual[:,3]*=np.sqrt(epsilon_0*ne/(m_e*a.CS**2))[None,:]
        Einc=np.linalg.norm([interp(115100,z,inc[0]),interp(115100,z,inc[1]),interp(115100,zf,inc[2])])
        scale=1e-3/Einc
        # H variables in the solver are c B. Use E at the same Yee faces.
        exf=(ep[0]+np.roll(ep[0],-1))/2
        eyf=(ep[1]+np.roll(ep[1],-1))/2
        em=.5*epsilon_0*c*np.real(exf*state[4].conj()-eyf*state[3].conj())
        np.testing.assert_allclose(em,h['EM_flux_W_m2'][:],rtol=1e-11,atol=1e-12)
        pf=h['pressure_flux_W_m2'][:]
        pin=attrs['incident_EM_flux_W_m2']
        zz=np.linspace(117500,118800,5201)
        out=OUT/(label+'_display.h5')
        with h5py.File(out,'w') as f:
            f.attrs.update(attrs)
            f.attrs.update(incident_E_peak_V_m=.001,display_field_scale=scale,
                propagation_direction=side,local_resonance_m=TOP if side>0 else BOTTOM,
                scope='Native coupled CW recording; both sides of the same solved layer shown; no Landau damping',
                flux_definition='Period-average signed vertical EM Poynting flux; positive upward. Separate incident/reflected/transmitted fluxes at the outer ports only. Pressure energy flux is not Poynting flux.')
            for name,val in dict(x_m=np.linspace(-60,60,121),time_s=tt,kx_rad_m=kx,
                profile_height_km=zz/1000,profile_ne_1e11=a.PROFILE(zz)/1e11,
                overview_net_EM_flux_percent=np.interp(zz,zf,em)/pin*100,
                phase_flux_fraction=attrs['upward_pressure_fraction' if side>0 else 'downward_pressure_fraction']).items():put(f,name,val)
            for direction,key,r in [(1,'topside',TOP),(-1,'bottomside',BOTTOM)]:
                g=f.create_group(key)
                offsets=np.linspace(-3,16,1901) if direction>0 else np.linspace(-16,3,1901)
                local=r+offsets;nn=np.interp(local,z,ne)
                d=interp(local,z,den)*scale/nn*1e6
                ez=interp(local,zf,ep[2])/Einc
                dt=np.array([interp(local,z,row[3])*scale/nn*1e6 for row in actual])
                et=np.array([interp(local,zf,row[2])/Einc for row in actual])
                kz=np.gradient(np.unwrap(np.angle(d)),local)
                kb=4*np.pi*430e6/c
                check=(offsets*direction>1)&(offsets*direction<9)
                inds=np.where(check[:-1]&check[1:]&((abs(kz[:-1])-kb)*(abs(kz[1:])-kb)<0))[0]
                bragg=float('nan')
                if len(inds) and max(abs(d))>.01:
                    i=inds[np.argmin(abs(offsets[inds]))];order=np.argsort(abs(kz[i:i+2]))
                    bragg=float(np.interp(kb,abs(kz[i:i+2])[order],offsets[i:i+2][order]))
                g.attrs.update(local_resonance_m=r,bragg_offset_m=bragg,
                    delta_ne_max_ppm=float(max(abs(d))),Ez_max_ratio=float(max(abs(ez))))
                values=dict(z_relative_m=offsets,density_real_ppm=dt.real,density_imag_ppm=dt.imag,
                    Ez_real_ratio=et.real,Ez_imag_ratio=et.imag,
                    density_envelope_ppm=abs(d),Ez_envelope_ratio=abs(ez),
                    kz_rad_m=kz,wavelength_m=2*np.pi/np.maximum(abs(kz),1e-12),
                    net_EM_flux_percent=np.interp(local,zf,em)/pin*100,
                    plasma_wave_flux_percent=np.interp(local,zf,pf)/pin*100)
                for name,val in values.items():put(g,name,val)
                # Preserve the original selected-side interface without copying
                # arrays: old keys are HDF5 hard links into the selected group.
                if direction==side:
                    f.attrs.update(dict(g.attrs))
                    for name in values:f[name]=g[name]
            f['provenance/source']=Path(__file__).read_text()
        print('Prepared both sides and signed flux:',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--side',type=int,choices=[-1,1],default=1);args=p.parse_args();prepare(args.label,args.side)
