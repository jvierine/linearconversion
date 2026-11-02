"""Translate the measured Es profile; hold geometry/Te/polarization fixed.

The existing coupled Yee/fluid CW solver is reused. MSIS2.1 neutral densities
are interpolated in time and log density from the archived experiment prior.
Electron-ion collisions use the same Coulomb formula as the event model.
All products are HDF5. No atmospheric-model electron temperature is inferred.
"""
from pathlib import Path
import argparse,gc,hashlib,sys,time
import h5py,numpy as np
from scipy.constants import c,epsilon_0,m_e,k as kb,e
import coupled_arecibo as a
from minimum_incident_power import metric,TARGET,CAL,TAU,TSYS
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from isr.coherent import sideband_amplitude
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reference/coupled/altitude_sweep';OUT.mkdir(parents=True,exist_ok=True)
PRIOR=ROOT/'data/eregion_fit/full_altitude_priors.h5'
with h5py.File(ROOT/'reference/coupled/minimum_incident.h5') as h:
    BASE=dict(h.attrs)
CENTER=float(BASE['center_m']);ANGLE=float(BASE['angle_deg']);FLUX=float(BASE['incident_vertical_flux_W_m2'])
NRES=float(a.PROFILE(CENTER));LN=float(NRES/a.PROFILE.derivative()(CENTER));EPOCH=1525453430.301006
with h5py.File(PRIOR) as h:
    heights=h['msis/height_km'][:]*1000.;tt=h['time_unix_s'][:]
    it=int(np.clip(np.searchsorted(tt,EPOCH)-1,0,len(tt)-2));u=(EPOCH-tt[it])/(tt[it+1]-tt[it])
    neutrals={name:(1-u)*h['msis/'+name][it]+u*h['msis/'+name][it+1] for name in ['N2_m3','O2_m3','O_m3','Tn_K']}

def components(z,ne):
    tev=kb*a.TE/e;ncm=np.maximum(np.asarray(ne)*1e-6,1e-30)
    ei=2.91e-6*ncm*(23-np.log(np.sqrt(ncm)*tev**-1.5))/tev**1.5
    vals=[]
    for name in ['N2_m3','O2_m3','O_m3']:
        valid=neutrals[name]>0
        assert np.min(z)>=heights[valid].min() and np.max(z)<=heights[valid].max()
        vals.append(np.exp(np.interp(z,heights[valid],np.log(neutrals[name][valid]))))
    n2,o2,o=vals;t=a.TE
    en=2.33e-17*n2*(1-1.21e-4*t)*t+1.82e-16*o2*(1+.036*np.sqrt(t))*np.sqrt(t)+8.9e-17*o*(1+5.7e-4*t)*np.sqrt(t)
    return en,ei

def rates(z,ne):
    en,ei=components(z,ne);return en+ei

def configuration(center,control=False):
    shift=center-CENTER
    def profile(z):return a.PROFILE(np.asarray(z)-shift)
    def control_rates(z,ne):return rates(np.asarray(z)-shift,ne)
    return dict(profile=profile,center=center,banded_solver=True,
        bounds=(114000.+shift,122000.+shift),fine_bounds=(-160.,40.),
        sample_bounds=(115300.+shift,120800.+shift),reference_z=115300.+shift,
        source_z=114850.+shift,temperature=a.TE,rates=control_rates if control else rates)

def response(s):
    rel=s['z']-s['center'];n=s['initial'][9]*np.sqrt(epsilon_0*s['ne'][0]/(m_e*a.CS**2))[:,None]
    rows=[]
    for extra in [0.,20.,40.]:
        window=np.sin(.5*np.pi*np.clip(np.minimum(rel+100.+extra,30.-rel)/10.,0.,1.))**2
        rows.append([sideband_amplitude(s['z'],n[:,j],a.F,s['kx'],window,reference_range_m=s['center']) for j in range(2)])
    return np.asarray(rows),n

def baseline(dz=.005,cap=1.):
    cfg=configuration(CENTER)
    s=a.setup(dz=dz,cap=cap,angle=ANGLE,scenario=cfg)
    ref=a.setup(dz=dz,cap=cap,angle=ANGLE,scenario=cfg,reference=True)
    ix=np.argmin(abs(ref['z']+ref['dz_cells']/2-115100.));q=metric(ref['initial'],ix)
    resp,_=response(s);v=np.linalg.solve(q,resp[0].conj());v/=np.sqrt(np.vdot(v,q@v).real)
    incident=np.einsum('qnp,p->qn',ref['initial'],v)
    tangent=incident[:2,ix];tangent/=np.linalg.norm(tangent)
    power=abs(resp[0]@v)**2
    assert abs(power/BASE['radar_power_per_unit_flux_W']-1)<.005,(power,BASE['radar_power_per_unit_flux_W'])
    return tangent,s,ref,v

def run(center,tangent,dz=.005,cap=1.,control=False):
    path=OUT/f'{center/1000:.6f}km_dz{dz:g}_cap{cap:g}{"_control" if control else ""}.h5'
    if path.exists():
        with h5py.File(path) as cached:
            if 'global_energy_relative_error' in cached.attrs:return path
    start=time.monotonic();cfg=configuration(center,control)
    s=a.setup(dz=dz,cap=cap,angle=ANGLE,scenario=cfg)
    ref=a.setup(dz=dz,cap=cap,angle=ANGLE,scenario=cfg,reference=True)
    ix=np.argmin(abs(ref['z']+ref['dz_cells']/2-(115100.+center-CENTER)))
    v=np.linalg.solve(ref['initial'][:2,ix,:],tangent)
    inc=np.einsum('qnp,p->qn',ref['initial'],v);pin=a.electromagnetic_flux(inc)[ix];assert pin>0
    v/=np.sqrt(pin);inc/=np.sqrt(pin)
    np.testing.assert_allclose(a.electromagnetic_flux(inc)[ix],1.,rtol=1e-10)
    resp,n=response(s);amps=resp@v;power=abs(amps)**2
    state=np.einsum('qnp,p->qn',s['initial'],v)
    # Independent discrete energy balance, including the absorbing boundaries.
    weights=np.array([s['dual'],s['dual'],s['dz_cells']]);ch=np.cos(s['omega']*s['dt']/2)
    source=np.einsum('qnp,p->qn',s['source'],v)
    work=.5*epsilon_0*c*np.sum(np.real(state[:3].conj()*source[:3])*weights)
    coll=.5*epsilon_0*c*ch*np.sum(s['physical_nu']*abs(state[6:9])**2*weights)
    sponge=.5*epsilon_0*c*ch*(np.sum(s['sponge'][:3]*abs(state[:3])**2*weights)+np.sum(s['sponge'][3:]*abs(state[3:6])**2*np.array([s['dz_cells'],s['dz_cells'],s['dual']]))+np.sum(s['sponge'][:3]*abs(state[6:9])**2*weights)+np.sum(s['sponge'][0]*abs(state[9])**2*s['dual']))
    balance=float((work-coll-sponge)/work);assert abs(balance)<1e-7,balance
    en,ei=components(np.asarray(center),NRES)
    if control:en,ei=components(np.asarray(CENTER),NRES)
    snr=power*FLUX*CAL*TAU/(kb*TSYS)
    gain=10**(61.1/10);lam=c/430e6
    rcs=power*FLUX*(4*np.pi)**3*center**4/(1e6*gain**2*lam**2)
    rel=s['z']-center;keep=(rel>=-160)&(rel<=50)
    stats=dict(center_m=center,angle_deg=ANGLE,temperature_K=a.TE,Ln_m=LN,
        ne_res_m3=NRES,nu_en_res_s1=float(en),nu_ei_res_s1=float(ei),
        nu_total_res_s1=float(en+ei),dz_m=dz,cap_m=cap,reference_flux_W_m2=FLUX,
        power_per_unit_flux_W=float(power[0]),SNR=float(snr[0]),RCS_m2=float(rcs[0]),
        required_flux_W_m2=float(FLUX*TARGET/snr[0]),
        harmonic_residual=s['harmonic_residual'],pressure_adjoint_error=s['pressure_adjoint_error'],
        reference_harmonic_residual=ref['harmonic_residual'],cells=s['n'],global_energy_relative_error=balance,
        numerical_window_spread_db=float(np.ptp(10*np.log10(power))),control=control)
    with h5py.File(path,'w') as h:
        h.attrs.update(stats)
        for k,val in dict(window_power_per_flux_W=power,window_SNR=snr,window_RCS_m2=rcs,
            incident_tangential_polarization=tangent,source_polarization=v,
            height_relative_m=rel[keep],ne_m3=s['ne'][0,keep],density_phasor_per_sqrt_flux_m3=(n@v)[keep],
            Ez_per_sqrt_flux_V_m=state[2,keep],physical_nu_s1=s['physical_nu'][0,keep]*c).items():
            h.create_dataset(k,data=val,compression='gzip' if np.asarray(val).ndim else None)
        h['provenance/source']=Path(__file__).read_text()
    print(f'{center/1000:.3f} km dz={dz:g} control={control}: nu={float(en+ei):.5g}, RCS={10*np.log10(rcs[0]):.4f} dBm2, SNR={10*np.log10(snr[0]):.4f} dB, taper spread={stats["numerical_window_spread_db"]:.4g} dB, {time.monotonic()-start:.1f}s',flush=True)
    del s,ref;gc.collect();return path

def export(paths,checks,tangent):
    with h5py.File(ROOT/'data/altitude_sweep.h5','w') as h:
        h.attrs.update(reference_center_m=CENTER,angle_deg=ANGLE,temperature_K=a.TE,Ln_m=LN,
            incident_vertical_flux_W_m2=FLUX,frequency_hz=a.F,magnetic_T=float(a.P['magnetic_t']),
            field_angle_deg=float(a.P['field_angle_deg']),epoch_unix_s=EPOCH,
            latitude_deg=18.3442,longitude_deg=-66.7528,neutral_model='MSIS2.1',
            electron_ion_model='same Coulomb momentum-transfer formula as event calculation; determined by ne and Te, not MSIS',
            weak_response_scope='SNR below -80 dB is a numerical non-detection; exact tiny values depend on taper. Required flux omitted for these unresolved responses.',
            scope='Shifted measured density profile; fixed event B, angle, tangential incident polarization, Te and local upward flux. MSIS neutrals vary with absolute model altitude. Steady coupled Maxwell-fluid response; no Landau damping.',
            neutral_prior_sha256=hashlib.sha256(PRIOR.read_bytes()).hexdigest())
        h['incident_tangential_polarization']=tangent
        cols=['center_m','RCS_m2','SNR','required_flux_W_m2','nu_en_res_s1','nu_ei_res_s1','nu_total_res_s1','numerical_window_spread_db','harmonic_residual']
        h.attrs['table_columns']=','.join(cols)
        for label,pp in [('sweep',paths),('checks',checks)]:
            rows=[]
            for path in pp:
                with h5py.File(path) as s:
                    row=[s.attrs[k] for k in cols]
                    unresolved=s.attrs['SNR']<1e-8
                    if unresolved:row[3]=np.nan
                    rows.append(row);g=h.create_group(f'{label}/cases/{path.stem}');g.attrs.update(dict(s.attrs))
                    g.attrs['below_resolved_dynamic_range']=unresolved
                    if unresolved:g.attrs['required_flux_W_m2']=np.nan
                    for key in ['window_power_per_flux_W','window_SNR','window_RCS_m2','incident_tangential_polarization']:
                        s.copy(key,g)
                    # Selected full near-resonance fields; other cases are reproducible
                    # from the saved inputs and retained local full-case HDF5 files.
                    if label=='sweep' and any(abs(s.attrs['center_m']-x)<.01 for x in [90000.,98000.,110000.,CENTER,140000.]):
                        for key in ['height_relative_m','ne_m3','density_phasor_per_sqrt_flux_m3','Ez_per_sqrt_flux_V_m','physical_nu_s1']:
                            s.copy(key,g)
            h[f'{label}/table']=np.asarray(rows)
        z=np.arange(80000.,160001.,100.)
        en,ei=components(z,np.full_like(z,NRES))
        h['collision_profile/height_m']=z;h['collision_profile/nu_en_s1']=en
        h['collision_profile/nu_ei_s1']=ei;h['collision_profile/nu_total_s1']=en+ei
        h['collision_profile/Tn_K']=np.interp(z,heights,neutrals['Tn_K'])
        h['provenance/source']=Path(__file__).read_text()
        h['provenance/shared_solver_source']=Path(a.__file__).read_text()
    print(ROOT/'data/altitude_sweep.h5',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    tangent,s,ref,v=baseline();print('Baseline reproduced; fixed incident tangent:',tangent,flush=True)
    del s,ref;gc.collect()
    alt=np.array([90,98,CENTER/1000,140]) if args.pilot else np.unique(np.r_[np.arange(90,150.01,2.5),98.,CENTER/1000])
    paths=[run(float(z*1000),tangent) for z in alt]
    checks=[]
    for z in ([98,CENTER/1000] if args.pilot else [98,110,CENTER/1000,140]):
        checks.append(run(float(z*1000),tangent,dz=.0025,cap=.5))
    for z in ([98,140] if args.pilot else [90,98,110,CENTER/1000,130,140,150]):
        checks.append(run(float(z*1000),tangent,control=True))
    export(paths,checks,tangent)
if __name__=='__main__':main()
