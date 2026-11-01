"""Keep artificial steep F cases archived; compare with the observed F profile."""
from pathlib import Path
import h5py,numpy as np
from scipy.interpolate import CubicSpline
from scipy.constants import e,epsilon_0,c
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/coupled/gradient_study'
with h5py.File(ROOT/'reference/gradient_summary.h5') as old,h5py.File(OUT/'observed_f_summary.h5','w') as h:
    ref=old.attrs['reference_bragg_column_per_sqrt_flux']
    h.attrs.update(reference_bragg_column_per_sqrt_flux=ref,display_floor_dB=-120.,
        interpretation='Event-conditioned F-region comparison; no claim about all F-region profiles. Polarization optimized per angle. Local Bragg projection, not absolute radar SNR.')
    for key in ['E_Ln_km','E_response_dB','measured_point','measured_angles','measured_angular_dB','E5_angles','E5_angular_dB']:
        h['plot/'+key]=old['plot/'+key][:]
    for sigma in [5,10,15]:
        vals=[];refinements=[]
        for p in sorted(OUT.glob(f'Fobserved{sigma}p_*.h5')):
            with h5py.File(p) as inp:
                a=dict(inp.attrs);db=20*np.log10(max(a['bragg_column_per_sqrt_flux'],1e-300)/ref)
                a['relative_response_dB']=db
                z=inp['z_m'][:];dz=inp['dz_cells_m'][:];dual=(dz+np.roll(dz,1))/2
                E=inp['E_phasor_V_m'][:];dn=inp['density_phasor_m3'][:]
                kx=2*np.pi*a['frequency_Hz']/c*np.sin(np.deg2rad(a['angle_deg']))
                div=1j*kx*E[0]+(E[2]-np.roll(E[2],1))/dual
                mask=(z>a['center_m']-a['bragg_distance_m']-30)&(z<a['center_m']+30)
                gauss=np.linalg.norm((div+e/epsilon_0*dn)[mask])/max(np.linalg.norm((e/epsilon_0*dn)[mask]),1e-250)
                a['gauss_relative_residual']=gauss
                assert gauss<1e-6 and a['harmonic_residual']<1e-7 and abs(a['balance'])<1e-3,(p.name,gauss,a['balance'])
                h.create_group('runs/'+p.stem).attrs.update(a)
                if a['dz_m']==.00375 and a['cap_m']==2:
                    vals.append([a['angle_deg'],db])
                    if a['angle_deg']==16.75:
                        h.create_group(f'cases/Fobserved{sigma}').attrs.update(a)
                else:refinements.append([a['angle_deg'],a['dz_m'],a['cap_m'],db])
        vals=np.array(sorted(vals));h[f'angles/{sigma}km']=vals
        h[f'refinement/{sigma}km']=np.array(refinements).reshape(-1,4)
        print('sigma',sigma,'maximum sampled',vals[np.argmax(vals[:,1])])
    rows=[h[f'cases/Fobserved{s}'].attrs for s in [5,10,15]]
    h['plot/F_Ln_km']=[a['Ln_m']/1000 for a in rows]
    h['plot/F_response_dB']=[max(-120,a['relative_response_dB']) for a in rows]
    v=h['angles/10km'][:];h['plot/Fobserved_angles']=v[:,0];h['plot/Fobserved_angular_dB']=np.maximum(-120,v[:,1])
    with h5py.File(ROOT/'data/fregion_observed_background.h5') as bg:
        z=bg['height_m'][:];use=(z>=190000)&(z<=270000);h['plot/F_altitude_km']=z[use]/1000
        h['plot/F_native_ne']=np.median(bg['native_ne_Te_Ti'][:,:,0],axis=0)[use]/1e11
        for sigma in [5,10,15]:
            a=bg[f'smoothing_{sigma}km'];ne=a['ne_m3'][:];p=CubicSpline(z,ne)
            h[f'plot/F_ne_{sigma}km']=ne[use]/1e11
            h[f'plot/F_Ln_{sigma}km']=ne[use]/p(z[use],1)/1000
        h.attrs['F_center_km']=bg['smoothing_10km'].attrs['center_m']/1000
    h['provenance/summary_source']=Path(__file__).read_text()
