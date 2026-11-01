"""Local inspection service for solved warm-fluid Maxwell conversion fields.

All scientific results are saved as HDF5. JSON is the browser transport only.
Animation reconstructs Re[a(z) exp(i kx x - i omega t)]; it is not a transient
turn-on calculation. Incident reference uses the uniform lower reservoir.
The default equilibrium is a smooth fit to calibrated 2018 density observations.
"""
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from threading import Lock
from functools import lru_cache
import hashlib, json, traceback
from urllib.parse import urlparse,parse_qs
import h5py
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0, atomic_mass
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from scipy.optimize import brentq
from isr.observed_layer import fit_observed_layer, scattering_profile, save_profile
from isr.conversion import Parameters, layer, optimal_polarization, scatter, matching_ratio, modes, components, physical_density
from isr.kinetic import Species, density_spectrum
from isr.landau import closure_profile,bragg_match
from isr.rays import solution_ray,electromagnetic_ray

ROOT=Path(__file__).resolve().parent
LOCK=Lock()
CACHE={}
DEFAULT=dict(frequency_mhz=5.33,scale_m=10.,angle_deg=-20.,temperature_k=300.,
             ion_temperature_k=300.,collision_s=1000.,magnetic_ut=35.025,field_angle_deg=46.815,dz_m=.0025,polarization_mode=-1.,profile_mode=1.,profile_time_unix_s=1525460596.,landau_mode=1.)
MODEL_VERSION='smooth-calibrated-full-es-v4-kinetic-pressure'
RAY_VERSION='muldrew-1993-es-ray-v1'


@lru_cache(maxsize=4)
def observed_layer(time_unix_s=1525460596.):
    result=fit_observed_layer(time_unix_s)
    save_profile(result,ROOT/'data/smooth_observed_layer.h5')
    return result


def bottomside_match(z, density, target):
    crossings=np.where((density[:-1]<target)&(density[1:]>=target))[0]
    if not len(crossings):raise ValueError('No bottomside Bragg match in the fitted profile at this HF frequency. The density is not rescaled to force a match.')
    j=crossings[0]
    return float(np.interp(target,density[j:j+2],z[j:j+2]))


def complex_pairs(a):
    return np.stack([np.asarray(a).real,np.asarray(a).imag],axis=-1).tolist()


def coherent_gate(z, ne, fractional_density, center, sigma, k):
    """Weighted fractional radar amplitudes; density is a peak phasor.

Each positive/negative-frequency line has integrated power |A|^2/4.
Returns A_up, A_down, effective electron column (multiply by aperture area).
An ideal coherent Gaussian gate is used, not the actual radar ambiguity.
"""
    w=np.exp(-.5*((z-center)/sigma)**2)
    den=np.trapezoid(w*ne,z)
    phase=np.exp(1j*k*(z-center))
    up=np.trapezoid(w*ne*fractional_density*phase,z)/den
    down=np.trapezoid(w*ne*np.conj(fractional_density)*phase,z)/den
    column=den**2/np.trapezoid(w*w*ne,z)
    return up,down,column


def validated(settings):
    p=DEFAULT|settings
    limits={'landau_mode':(0.,1.),'frequency_mhz':(4.,6.),'scale_m':(3.,40.),'angle_deg':(-30.,30.),
        'temperature_k':(200.,2500.),'ion_temperature_k':(150.,1500.),
        'collision_s':(0.,30000.),'magnetic_ut':(0.,60.),'field_angle_deg':(0.,80.),
        'dz_m':(.00125,.005),'polarization_mode':(-1.,1.),'profile_mode':(0.,1.),'profile_time_unix_s':(1525444500.,1525460800.)}
    if set(p)!=set(DEFAULT):raise ValueError('Unknown model parameter')
    for key,(lo,hi) in limits.items():
        p[key]=float(p[key])
        if not np.isfinite(p[key]) or not lo<=p[key]<=hi:raise ValueError(f'{key}: allowed range {lo} to {hi}')
    if p['polarization_mode'] not in (-1.,0.,1.):raise ValueError('Choose optimal, mode 1, or mode 2 polarization')
    if p['landau_mode'] not in (0.,1.):raise ValueError('Landau mode must be 0 or 1')
    if p['profile_mode'] not in (0.,1.):raise ValueError('Choose measured or illustrative density')
    if not p['profile_mode'] and 12*p['scale_m']/p['dz_m']>200000:raise ValueError('Choose a coarser mesh or smaller gradient scale (200,000-cell limit)')
    return p


def solve(settings):
    p=validated(settings)
    fingerprint=observed_layer(p['profile_time_unix_s'])['source_sha256'] if p['profile_mode'] else ''
    key=hashlib.sha256((MODEL_VERSION+fingerprint+json.dumps(p,sort_keys=True)).encode()).hexdigest()[:16]
    if key in CACHE:return CACHE[key]
    q=np.sin(np.deg2rad(p['angle_deg']))
    kB=4*np.pi*430e6/c
    delta=np.arcsin(q*2*np.pi*p['frequency_mhz']*1e6/c/kB)
    par=Parameters(frequency_hz=p['frequency_mhz']*1e6,temperature_k=p['temperature_k'],
        magnetic_t=p['magnetic_ut']*1e-6,magnetic_tilt_deg=p['field_angle_deg']-np.rad2deg(delta),q=q)
    origin=0.;profile_time='Illustrative';source_height=np.array([]);source_ne=np.array([])
    if p['profile_mode']:
        fitted=observed_layer(p['profile_time_unix_s']);source_height=fitted['source_height_m'];source_ne=fitted['source_ne_m3']
        # Include both sides and the whole peak; buffers sit five fitted widths
        # away from the layer center. No artificial overdense continuation.
        origin=fitted['es_center_m']-5*fitted['es_sigma_m'];end=fitted['es_center_m']+5*fitted['es_sigma_m']
        count=round((end-origin)/p['dz_m'])+1
        if count>2500001:raise ValueError('Smooth measured profile: use a 2.5 or 5 mm mesh (2,500,000-cell limit)')
        z=np.linspace(0,end-origin,count)
        ne,dn=scattering_profile(z+origin,fitted['coefficients'],fitted['knots_km'])
        ratio=ne*e**2/(epsilon_0*m_e*par.omega**2);grad=dn/ne
        distance=np.minimum(z,z[-1]-z);u=np.clip(distance/400.,0,1)
        nu=p['collision_s']*(10*u**3-15*u**4+6*u**5);nu[[0,-1]]=0.
        profile_time=fitted['time']
    else:
        z=np.linspace(-6*p['scale_m'],6*p['scale_m'],round(12*p['scale_m']/p['dz_m'])+1)
        ratio,grad,nu=layer(z,p['scale_m'],collision_s=p['collision_s'])
    pm,=matching_ratio([q*par.k0,0.,-np.sqrt(kB*kB-(q*par.k0)**2)],par)
    pressure=np.ones_like(z,dtype=complex);kinetic_table=None
    if p['landau_mode']:
        pressure,kinetic_table=closure_profile(z,ratio,par,taper_m=400. if p['profile_mode'] else p['scale_m'])
        pm,alpha_bragg=bragg_match(par,kB)
    else:alpha_bragg=0.
    zm=bottomside_match(z,ratio,pm)
    Y2=par.gyro@par.gyro;ca=np.cos(np.deg2rad(par.magnetic_tilt_deg))
    resonance_ratio=(1-Y2)/(1-Y2*ca*ca)
    zres=bottomside_match(z,ratio,resonance_ratio)
    zcritical=bottomside_match(z,ratio,1.)
    if p['polarization_mode']==-1:
        r=optimal_polarization(z,ratio,grad,nu,par,target_z_m=zm,pressure_response=pressure)
    else:
        branch=int(p['polarization_mode'])
        r=scatter(z,ratio,grad,nu,par,branch=branch,pressure_response=pressure)
        r['incoming_weights']=np.eye(2,dtype=complex)[branch]
    eunit=np.sqrt(2e-6/(epsilon_0*c)) # reference incident normal flux 1 microW/m^2
    ne=ratio*par.omega**2*epsilon_0*m_e/e**2
    E=r['E']*eunit
    density=physical_density(r['h'],par,1e-6)
    mu,v,direction,prop=modes(ratio[0],par)
    em=sorted([j for j in range(6) if direction[j]==1 and prop[j] and abs(mu[j])<5],key=lambda j:abs(mu[j]))
    incident=sum(a*np.exp(mu[j]*par.k0*(z-z[0]))[:,None]*components(v[:,j],ratio[0],0.,par)[0]
                 for a,j in zip(r['incoming_weights'],em))*eunit
    phasek=np.imag(np.conj(density)*np.gradient(density,z))/np.maximum(abs(density)**2,1e-100)
    # All raw samples remain in HDF5; the broad map is explicitly spatially filtered.
    overview_z=np.linspace(z[0],z[-1],1601)
    micro_z=np.linspace(zm-4,zm+4,4097)
    def interp(a,grid):return np.interp(grid,z,a)
    smoothing=max(.25,(z[-1]-z[0])/1000)
    # Filter only the overview, once; scientific samples and microscope are raw.
    radius=int(np.ceil(4*smoothing/p['dz_m']))
    kernel=np.exp(-.5*(np.arange(-radius,radius+1)*p['dz_m']/smoothing)**2)
    kernel/=kernel.sum()
    E_overview=fftconvolve(np.pad(E,((radius,radius),(0,0)),mode='edge'),kernel[:,None],mode='valid',axes=0)
    def sample(grid,smooth=False):
        ee=E_overview if smooth else E
        return dict(z=grid.tolist(),ne=interp(ne,grid).tolist(),p=interp(ratio,grid).tolist(),
            gradient=interp(grad,grid).tolist(),dne_dz=interp(ne*grad,grid).tolist(),
            collision=interp(nu,grid).tolist(),flux=interp(r['flux'],grid).tolist(),
            density=complex_pairs(interp(density,grid)),phase_k=interp(phasek,grid).tolist(),
            E=[complex_pairs(interp(ee[:,j],grid)) for j in range(3)],
            incident=[complex_pairs(interp(incident[:,j],grid)) for j in range(3)])
    nem=pm*par.omega**2*epsilon_0*m_e/e**2
    electrons=[Species('e',nem,p['temperature_k'],m_e,-e,p['collision_s'])]
    ions=[Species('molecular ion',nem,p['ion_temperature_k'],30*atomic_mass,e)]
    ion_f=np.linspace(-12000,12000,801)
    plasma_offset=np.linspace(-30000,30000,1201)
    def spec(f):return density_spectrum(f,kB,electrons,ions,B=par.magnetic_t,theta_deg=p['field_angle_deg'])['total'].tolist()
    output=dict(id=key,parameters=p,kx=q*par.k0,kB=kB,delta_deg=float(np.rad2deg(delta)),
        origin_height_m=float(origin),profile_time=profile_time,overview_smoothing_m=smoothing,
        profile_description=fitted['model'] if p['profile_mode'] else 'Illustrative tanh ramp',
        model_version=MODEL_VERSION,incident_reference_extent_m=min(400.,(z[-1]-z[0])/4),
        polarization_objective='local density amplitude at bottomside radar match',
        match_z_m=float(zm),match_ne_m3=float(nem),reference_flux_w_m2=1e-6,
        resonance_z_m=float(zres),critical_z_m=float(zcritical),resonance_to_bragg_m=float(zres-zm),critical_to_bragg_m=float(zcritical-zm),bragg_landau_alpha_m1=float(alpha_bragg),
        overview=sample(overview_z,True),micro=sample(micro_z),
        budget={k:r[k] for k in ['conversion','reflection','transmission','absorption','absorption_collision','absorption_landau','energy_error']},
        polarization=complex_pairs(r['incoming_weights']),
        thermal=dict(ion_frequency_hz=ion_f.tolist(),ion=spec(ion_f),
                     offset_hz=plasma_offset.tolist(),plasma=spec(par.frequency_hz+plasma_offset)),
        max_fractional_density=float(np.max(abs(density))))
    ray=solution_ray(z,ratio,nu,par,kinetic_table,zm) if kinetic_table is not None else None
    output['ray_geometry']=None if ray is None else {k:np.asarray(v).tolist() for k,v in ray.items()}
    output['ray_version']=RAY_VERSION
    path=ROOT/'data/conversion_runs'/f'{key}.h5'
    with h5py.File(path,'w') as f:
        f.attrs.update(p,model='Linear electron-fluid Maxwell with optional local Maxwellian kinetic complex pressure; fixed ions, no nonlinear instability',
            provenance='conversion_lab.py; isr/conversion.py; isr/kinetic.py',
            event_status='No heater operated during the event. Density is smoothly fitted to calibrated data when profile_mode=1; pump and thermodynamic parameters remain assumed.',
            model_version=MODEL_VERSION,density_source_sha256=fingerprint,
            origin_height_m=origin,profile_time=profile_time,profile_interpolation='Cubic smoothing spline with preserved Es shape; C2 coordinate taper in outer 400 m to collisionless uniform reservoirs' if p['profile_mode'] else 'tanh ramp',
            polarization_objective=output['polarization_objective'],
            reference_incident_normal_flux_w_m2=1e-6,match_z_m=zm,resonance_z_m=zres,critical_z_m=zcritical,bragg_landau_alpha_m1=alpha_bragg,
            resonance_to_bragg_m=zres-zm,critical_to_bragg_m=zcritical-zm,kinetic_limitation='Local longitudinal spatial-root pressure closure calibrated to downward branch; not full Vlasov electromagnetic tensor; collisionless loss includes n=0 Landau and gyroharmonic resonances',radar_zenith_deg=0.,layer_tilt_deg=np.rad2deg(delta))
        if kinetic_table is not None:
            for name,value in kinetic_table.items():f.create_dataset('kinetic_branch/'+name,data=value)
        for name,a in {'pressure_response':pressure,'z_m':z,'ne_m3':ne,'dlogne_dz_m1':grad,'collision_s':nu,'E_peak_v_m':E,
            'incident_reference_E_peak_v_m':incident,'density_fraction_peak':density,'phase_kz_m1':phasek,
            'normalized_energy_flux':r['flux'],'state':r['state'],'incoming_weights':r['incoming_weights'],
            'B_wave_peak_t':r['b']*eunit/c,'electron_velocity_peak_m_s':r['u']*e*eunit/(m_e*par.omega),
            'electron_current_peak_a_m2':-e*ne[:,None]*r['u']*e*eunit/(m_e*par.omega)}.items():
            f.create_dataset(name,data=a,compression='gzip')
        for name,a in output['thermal'].items():f.create_dataset('thermal/'+name,data=a)
        f.create_dataset('measured_profile/height_above_radar_m',data=source_height)
        f.create_dataset('measured_profile/ne_m3',data=source_ne)
        if p['profile_mode']:
            for name in ['coefficients','knots_km','source_sigma_m3','source_valid','model_height_m','model_ne_m3','model_dne_dz_m4','background_ne_m3','es_ne_m3']:
                f.create_dataset('measured_profile/'+name,data=fitted[name])
            f['measured_profile'].attrs.update(source=fitted['source_path'],calibration_gain=fitted['calibration_gain'],calibration_log_scatter=fitted['calibration_log_scatter'],limitation=fitted['limitation'])
        f.attrs.update(output['budget'])
        if ray is not None:
            save_ray(f,ray)
    CACHE[key]=output
    return output


def save_ray(h,ray):
    if 'ray_geometry' in h:del h['ray_geometry']
    g=h.create_group('ray_geometry')
    g.attrs.update(version=RAY_VERSION,reference='https://doi.org/10.1029/92RS00273',
        interpretation='Weak-loss electrostatic ray after k reaches 5/m; x origin is onset of diagnostic, not transmitter. Full-wave solution determines conversion and launch. Phase vector and group velocity are distinct.',
        damping='Collisionless kinetic spatial loss plus separately evaluated collisional spatial loss; power exp(-2 integral alpha |dz|).')
    for k,v in ray.items():g[k]=v


def upgrade_ray(item):
    """Add ray diagnostics to saved field solutions without re-solving them."""
    if item.get('ray_version')==RAY_VERSION:return item
    with h5py.File(ROOT/'data/conversion_runs'/f"{item['id']}.h5",'r+') as h:
        a=dict(h.attrs)
        if 'kinetic_branch' not in h:return item
        par=Parameters(frequency_hz=a['frequency_mhz']*1e6,temperature_k=a['temperature_k'],
            magnetic_t=a['magnetic_ut']*1e-6,magnetic_tilt_deg=a['field_angle_deg']-a['layer_tilt_deg'],q=np.sin(np.deg2rad(a['angle_deg'])))
        table={k:h['kinetic_branch/'+k][()] for k in h['kinetic_branch']}
        ratio=h['ne_m3'][:]*e**2/(epsilon_0*m_e*par.omega**2)
        ray=solution_ray(h['z_m'][:],ratio,h['collision_s'][:],par,table,a['match_z_m'])
        if ray is not None:save_ray(h,ray)
    item['ray_geometry']=None if ray is None else {k:np.asarray(v).tolist() for k,v in ray.items()}
    item['ray_version']=RAY_VERSION
    return item


def window(key,center):
    if len(key)!=16 or any(c not in '0123456789abcdef' for c in key):raise ValueError('Invalid solution identifier')
    if not np.isfinite(center):raise ValueError('Invalid microscope position')
    with h5py.File(ROOT/'data/conversion_runs'/f'{key}.h5') as f:
        z=f['z_m'][:];center=np.clip(center,z[0]+4,z[-1]-4);grid=np.linspace(center-4,center+4,4097)
        def real(name):return np.interp(grid,z,f[name][:]).tolist()
        def comp(name):return complex_pairs(np.interp(grid,z,f[name][:]))
        ee=f['E_peak_v_m'][:]
        return dict(z=grid.tolist(),ne=real('ne_m3'),gradient=real('dlogne_dz_m1'),
            collision=real('collision_s'),flux=real('normalized_energy_flux'),
            density=comp('density_fraction_peak'),phase_k=real('phase_kz_m1'),
            E=[complex_pairs(np.interp(grid,z,ee[:,j])) for j in range(3)])


@lru_cache(maxsize=12)
def hf_ray_diagnostic(key):
    """Incoming local EM rays and polarization for the current solved profile."""
    if len(key)!=16 or any(c not in '0123456789abcdef' for c in key):raise ValueError('Invalid solution identifier')
    from scipy.interpolate import PchipInterpolator
    from isr.observed_layer import profile
    path=ROOT/'data/conversion_runs'/f'{key}.h5'
    with h5py.File(path) as h:
        a=dict(h.attrs);z=h['z_m'][:];ne=h['ne_m3'][:]
        par=Parameters(frequency_hz=a['frequency_mhz']*1e6,temperature_k=a['temperature_k'],
            magnetic_t=a['magnetic_ut']*1e-6,magnetic_tilt_deg=a['field_angle_deg']-a['layer_tilt_deg'],q=np.sin(np.deg2rad(a['angle_deg'])))
        scale=e**2/(epsilon_0*m_e*par.omega**2)
        origin=a['origin_height_m']
        if a['profile_mode']:
            coeff=h['measured_profile/coefficients'][:];knots=h['measured_profile/knots_km'][:]
            start=float(knots[0]*1000-origin)
            def pfun(zz):
                n,dn,*_=profile(zz+origin,coeff,knots)
                return float(n*scale),float(dn*scale)
        else:
            spline=PchipInterpolator(z[::max(1,len(z)//10000)],ne[::max(1,len(z)//10000)]*scale)
            start=float(z[0])
            def pfun(zz):return float(spline(zz)),float(spline.derivative()(zz))
    mu,_,direction,prop=modes(pfun(start)[0],par)
    em=sorted([j for j in range(6) if direction[j]==1 and prop[j] and abs(mu[j])<5],key=lambda j:abs(mu[j]))
    result=[]
    for j in em:
        result.append(electromagnetic_ray(start,mu[j].imag*par.k0,pfun,par,max_step_s=2e-7))
    with h5py.File(path,'a') as h:
        if 'incoming_rays' in h:del h['incoming_rays']
        g=h.create_group('incoming_rays')
        g.attrs.update(reference='https://doi.org/10.1029/92RS00273',
            interpretation='Cold local Maxwell Hamilton rays including reflection and polarization; full-wave fields handle conversion and evanescent penetration. x=0 at the lower start of each ray.')
        for i,r in enumerate(result):
            gg=g.create_group(str(i))
            for k,v in r.items():
                if isinstance(v,str):gg.attrs[k]=v
                else:gg[k]=v
    return dict(origin_height_m=origin,rays=[{k:(complex_pairs(v) if k=='electric_polarization' else v if isinstance(v,str) else np.asarray(v).tolist()) for k,v in r.items()} for r in result])


def radar_diagnostic(key,frequency_mhz,zenith_deg):
    if frequency_mhz not in (224.,430.,930.) or not -30<=zenith_deg<=30:raise ValueError('Unsupported radar geometry')
    if len(key)!=16 or any(c not in '0123456789abcdef' for c in key):raise ValueError('Invalid solution identifier')
    with h5py.File(ROOT/'data/conversion_runs'/f'{key}.h5') as f:
        a=dict(f.attrs);z=f['z_m'][:];ne=f['ne_m3'][:]
        q=np.sin(np.deg2rad(a['angle_deg']));delta=np.deg2rad(a['layer_tilt_deg']);angle=np.deg2rad(zenith_deg)-delta
        k=4*np.pi*frequency_mhz*1e6/c;qx=-k*np.sin(angle);kdown=k*np.cos(angle)
        par=Parameters(frequency_hz=a['frequency_mhz']*1e6,temperature_k=a['temperature_k'],magnetic_t=a['magnetic_ut']*1e-6,magnetic_tilt_deg=a['field_angle_deg']-a['layer_tilt_deg'],q=q)
        ratios=matching_ratio([qx,0,-kdown],par)
        match_ratio=ratios[-1]
        if a.get('landau_mode',0):
            radar_par=Parameters(frequency_hz=par.frequency_hz,temperature_k=par.temperature_k,magnetic_t=par.magnetic_t,magnetic_tilt_deg=par.magnetic_tilt_deg,q=qx/par.k0)
            match_ratio,_=bragg_match(radar_par,k)
        target=match_ratio*par.omega**2*epsilon_0*m_e/e**2
        try:zm=bottomside_match(z,ne,target);exists=True
        except ValueError:zm=float(a['match_z_m']);exists=False
        electrons=[Species('e',target,a['temperature_k'],m_e,-e,a['collision_s'])]
        ions=[Species('ion',target,a['ion_temperature_k'],30*atomic_mass,e)]
        fi=np.linspace(-12000,12000,801);fo=np.linspace(-30000,30000,1201)
        def spectrum(f):return density_spectrum(f,k,electrons,ions,B=par.magnetic_t,theta_deg=abs(a['field_angle_deg']-zenith_deg))['total'].tolist()
        thermal=dict(ion_frequency_hz=fi.tolist(),ion=spectrum(fi),offset_hz=fo.tolist(),plasma=spectrum(par.frequency_hz+fo))
    return dict(k=k,kdown=kdown,qx=qx,match_z_m=zm,match_ne_m3=float(target),match_exists=exists,
                zenith_deg=zenith_deg,frequency_mhz=frequency_mhz,thermal=thermal,micro=window(key,zm))


def measurement_context(time_unix_s=1525460596.):
    f=observed_layer(time_unix_s)
    return dict(range_km=(f['source_height_m']/1000).tolist(),ne_m3=f['source_ne_m3'].tolist(),
        smooth_range_km=(f['model_height_m']/1000).tolist(),smooth_ne_m3=f['model_ne_m3'].tolist(),
        background_ne_m3=f['background_ne_m3'].tolist(),time=f['time'],range_spacing_m=300.,
        es_sigma_m=f['es_sigma_m'],residual_rms=f['residual_rms'],calibration_log_scatter=f['calibration_log_scatter'],
        status='Plasma-line-calibrated observations; smooth background with observed Es peak and flanks preserved. No deconvolution.')


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT),**kwargs)
    def send_json(self,obj,code=200):
        data=json.dumps(obj,allow_nan=False,separators=(',',':')).encode()
        self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if self.path.startswith('/api/hf-ray?'):
            try:
                q=parse_qs(urlparse(self.path).query)
                return self.send_json(hf_ray_diagnostic(q['id'][0]))
            except Exception as exc:return self.send_json({'error':str(exc)},400)
        if self.path.startswith('/api/measurement'):
            try:
                query=parse_qs(urlparse(self.path).query)
                return self.send_json(measurement_context(float(query.get('time',[1525460596.])[0])))
            except Exception as exc:return self.send_json({'error':str(exc)},400)
        if self.path.startswith('/api/window?'):
            try:
                q=parse_qs(urlparse(self.path).query)
                return self.send_json(window(q['id'][0],float(q['center'][0])))
            except Exception as exc:return self.send_json({'error':str(exc)},400)
        if self.path.startswith('/api/radar?'):
            try:
                q=parse_qs(urlparse(self.path).query)
                return self.send_json(radar_diagnostic(q['id'][0],float(q['frequency'][0]),float(q['zenith'][0])))
            except Exception as exc:return self.send_json({'error':str(exc)},400)
        return super().do_GET()
    def do_POST(self):
        if self.path!='/api/solve':return self.send_error(404)
        try:
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<10000:raise ValueError('Invalid request length')
            settings=json.loads(self.rfile.read(length))
            with LOCK:result=solve(settings)
            self.send_json(result)
        except Exception as exc:
            traceback.print_exc();self.send_json({'error':str(exc)},400)


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--port',type=int,default=8770);args=a.parse_args()
    for saved in (ROOT/'data').glob('landau_conversion_*.json'):
        item=json.loads(saved.read_text())
        if item.get('model_version')==MODEL_VERSION and (ROOT/'data/conversion_runs'/f"{item['id']}.h5").exists():CACHE[item['id']]=upgrade_ray(item)
    print(f'Conversion lab: http://127.0.0.1:{args.port}/web/conversion/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
