"""Standalone figure arrays from the saved kinetic-pressure full-wave solutions."""
import h5py
import numpy as np

def prepare():
    with h5py.File('data/landau_bragg_study.h5','a') as s:
        for epoch in s:
            for freq in s[epoch]:
                g=s[epoch+'/'+freq];ident=g.attrs['full_wave_solution_id']
                res=g.attrs['resonance_height_m'];zb=g.attrs['bragg_height_m']
                with h5py.File('data/conversion_runs/'+ident+'.h5') as h:
                    z=h['z_m'][:]+h.attrs['origin_height_m'];d=h['density_fraction_peak'][:];E=h['E_peak_v_m'][:]
                    zp=np.linspace(res-100,res+10,4401);zm=np.linspace(zb-5,zb+5,4001)
                    arrays=dict(wave_height_relative_resonance_m=zp-res,
                        wave_density_real_percent=100*np.interp(zp,z,d.real),
                        wave_density_amplitude_percent=100*np.interp(zp,z,abs(d)),
                        micro_height_relative_bragg_m=zm-zb,micro_density_real_percent=100*np.interp(zm,z,d.real),
                        micro_density_amplitude_percent=100*np.interp(zm,z,abs(d)),
                        wave_Ex_amplitude_v_m=np.interp(zp,z,abs(E[:,0])),wave_Ez_amplitude_v_m=np.interp(zp,z,abs(E[:,2])))
                    for name,val in arrays.items():
                        if name in g:del g[name]
                        g[name]=val
                    g.attrs['density_amplitude_at_bragg']=abs(np.interp(zb,z,d))
                    if 'coarse_solution_id' in g.attrs:
                        with h5py.File('data/conversion_runs/'+g.attrs['coarse_solution_id']+'.h5') as cc:
                            zc=cc['z_m'][:]+cc.attrs['origin_height_m'];dc=cc['density_fraction_peak'][:]
                            a=np.interp(zm,z,abs(d));b=np.interp(zm,zc,abs(dc))
                            g.attrs['mesh_peak_envelope_relative_change']=np.max(abs(a-b))/np.max(a)
                amp=g['amplitude_survival'][:];zr=g['height_relative_resonance_m'][:]
                g.attrs['collisionless_amplitude_efold_depth_m']=-np.interp(np.exp(-1),amp,zr)
                g.attrs['collisionless_half_power_depth_m']=-np.interp(np.sqrt(.5),amp,zr)
        s.attrs['figure_provenance']='prepare_landau_figures.py; plotter/src/bin/landau_conversion.rs'
if __name__=='__main__':prepare()
