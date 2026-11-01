"""Transient Gaussian source via the broadband discrete Maxwell-fluid resolvent.

The frequency response of the existing Yee/fluid recurrence is inverted,
including frequency-dependent propagation, conversion and damping. This is
not a travelling envelope imposed on one CW solution. The same dt, grid, kx,
source polarization and coefficients apply to every frequency. The carrier
incidence is optimal; the narrowband pulse has a fixed horizontal wavenumber.
"""
from pathlib import Path
import argparse,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import h5py,numpy as np
from scipy.constants import c,epsilon_0,m_e,e
from scipy.optimize import brentq
import coupled_arecibo as a
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/gaussian_pulse';OUT.mkdir(parents=True,exist_ok=True)

def init_worker(cfg,pol,angle,dz,allz,z):
    global TASK
    TASK=(cfg,pol,angle,dz,allz,z)

def frequency_response(offset):
    cfg,pol,angle,dz,allz,z=TASK
    cfg=dict(cfg,frequency_hz=5.34e6+offset)
    s=a.setup(angle=angle,dz=dz,cap=.5,scenario=cfg)
    assert np.array_equal(s['z'],z)
    zf=z+s['dz_cells']/2
    state=np.einsum('qnp,p->qn',s['initial'],pol)
    sampled=[np.interp(allz,zf if k in [2,3,4,8] else z,state[k]) for k in range(10)]
    return offset,np.asarray(sampled,dtype='complex64'),s['harmonic_residual']

def main():
    p=argparse.ArgumentParser();p.add_argument('--df',type=float,default=250.);p.add_argument('--dz',type=float,default=.002);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    with h5py.File(ROOT/'reference/coupled/minimum_incident.h5') as h:
        angle=float(h.attrs['angle_deg']);power=float(h.attrs['incident_vertical_flux_W_m2']);pol=h['source_polarization'][:]
    f0=5.34e6;omega=2*np.pi*f0;gyro=e*a.P['magnetic_t']/m_e
    tilt=np.deg2rad(a.P['field_angle_deg']);density=omega**2*(omega**2-gyro**2)/(omega**2-(gyro*np.cos(tilt))**2)*m_e*epsilon_0/e**2
    bottom=brentq(lambda z:a.PROFILE(z)-density,117000,118100)
    top=brentq(lambda z:a.PROFILE(z)-density,118100,119000)
    cfg=dict(ordering='COLAMD',center=bottom,frequency_hz=f0,
        fine_intervals=[(-180.,80.),(top-bottom-80.,top-bottom+180.)])
    base=a.setup(angle=angle,dz=args.dz,cap=.5,scenario=cfg)
    cfg.update(fixed_dt_m=base['dt'],fixed_kx_m1=base['kx'])
    ref=a.setup(angle=angle,dz=args.dz,cap=.5,scenario=cfg,reference=True)
    idx=np.argmin(abs(ref['z']+ref['dz_cells']/2-115100.))
    incident=np.einsum('qnp,p->qn',ref['initial'],pol)
    pol*=np.sqrt(power/a.electromagnetic_flux(incident)[idx])
    z=base['z'];zf=z+base['dz_cells']/2
    displays={'overview':np.arange(115100.,120800.1,2.),
              'bottomside':bottom+np.arange(-160.,40.001,.025),
              'topside':top+np.arange(-40.,160.001,.025)}
    allz=np.concatenate(list(displays.values()))
    sigma=10e-6/(2*np.sqrt(2*np.log(2)));t0=60e-6
    path=OUT/f'response_dz{args.dz:g}.h5'
    with h5py.File(path,'a') as h:
        h.attrs.update(carrier_hz=f0,angle_deg=angle,field_envelope_FWHM_s=10e-6,source_peak_time_s=t0,
            peak_incident_vertical_flux_W_m2=power,kx_m1=base['kx'],dt_s=base['dt']/c,
            bottom_resonance_m=bottom,top_resonance_m=top,dz_m=args.dz,
            method='Broadband inverse Fourier response of the coupled discrete Yee/electron-fluid recurrence; fixed kx and source polarization; not a modulated CW movie',
            model='Linear fluid pressure, electron-ion and electron-neutral collisions, fixed ions; no kinetic Landau damping',
            fields='Ex Ey Ez cBx cBy cBz normalized_jx normalized_jy normalized_jz normalized_density; interpolated onto common heights')
        if 'height_m' not in h:
            h['height_m']=allz;h['ne_m3']=np.interp(allz,z,base['ne'][0]);h['source_polarization']=pol
            offset=0
            for name,zview in displays.items():
                h['views/'+name]=[offset,offset+len(zview)];offset+=len(zview)
            h['provenance/driver']=Path(__file__).read_text()
            h['provenance/operator']=Path(a.__file__).read_text()
        frequencies=np.arange(-160000.,160000.+args.df/2,args.df)
        start=time.monotonic()
        pending=iter([v for v in frequencies if f'response/{v:+.3f}' not in h]);count=0
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
             initializer=init_worker,initargs=(cfg,pol,angle,args.dz,allz,z)) as pool:
            futures={pool.submit(frequency_response,v) for v in [next(pending,None) for _ in range(args.workers*2)] if v is not None}
            while futures:
                done,futures=wait(futures,return_when=FIRST_COMPLETED)
                for future in done:
                    offset,sampled,residual=future.result()
                    ds=h.create_dataset(f'response/{offset:+.3f}',data=sampled,compression='lzf')
                    ds.attrs['residual']=residual;count+=1
                    offset=next(pending,None)
                    if offset is not None:futures.add(pool.submit(frequency_response,offset))
                h.flush()
                if count%20<len(done):print('spectral pulse',count,'/',len(frequencies),'elapsed',round(time.monotonic()-start,1),'s',flush=True)
        h.attrs['frequency_spacing_hz']=args.df
        h.attrs['period_s']=1/args.df
        h.attrs['spectral_edge_amplitude']=np.exp(-.5*(2*np.pi*160000*sigma)**2)
        h.attrs['complete']=True
        print('DONE',path,flush=True)
if __name__=='__main__':main()
