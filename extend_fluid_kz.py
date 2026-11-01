"""Extended fixed-frequency warm-fluid solution and outgoing-wave diagnostic.

Provenance: drive_fluid_from_fdtd.py, recorded cold Maxwell field, calibrated
smooth density profile. No new kinetic damping is implied. The continuation
removes forcing BELOW -10 m and normalizes density to the full solution there.
This separates an outgoing wave from the surviving locally forced response.
"""
import hashlib
import numpy as np
import h5py
from scipy.integrate import cumulative_trapezoid
from drive_fluid_from_fdtd import (setup, harmonic, save_group, PROFILE, FIELD,
    ROOT, CubicHermiteSpline, solve_banded, e, m_e, epsilon_0, kb, c)


def continuation(s, r, top=-10.):
    j=int(np.argmin(abs(s['zf']-s['center']-top)))
    a=s['nu'][1:j]+s['sponge'][1:j]-1j*s['omega']
    rzz=(a*a+s['bz']**2)/(a*(a*a+s['bx']**2+s['bz']**2))
    lap=s['cs2']/(s['ne'][:j]*s['dz']**2)
    mat=np.zeros((3,j-1),complex)
    mat[1]=-(lap[:-1]+lap[1:])+1j*s['omega']/(s['nf'][1:j]*rzz)-e*e/(m_e*epsilon_0)
    mat[0,1:]=lap[1:-1];mat[2,:-1]=lap[1:-1]
    pt=r['velocity_phasor_m_s'][2,j]*s['nf'][j]
    rhs=np.zeros(j-1,complex);rhs[-1]=-lap[-1]*pt
    p=np.r_[0,solve_banded((1,1),mat,rhs),pt]
    eta=np.diff(p)/(1j*s['omega']*s['dz']*s['ne'][:j])
    # Total driven particle flux contains a smooth forced component. Match the
    # density phasor, not total particle flux, to define the diagnostic amplitude.
    scale=r['density_phasor'][j-1]/eta[-1]
    eta*=scale
    return dict(z=s['z'][:j]-s['center'],density_phasor=eta,
                kz_rad_m=np.gradient(np.unwrap(np.angle(eta)),s['dz']),
                boundary_height_m=float(s['zf'][j]-s['center']),
                density_normalization_complex=scale)


def main():
    cases=[]
    for n,lo,hi in [(120000,-160.,80.),(240000,-160.,80.),(320000,-200.,120.)]:
        s=setup(n,lo,hi);r=harmonic(s);q=continuation(s,r)
        cases.append((s,r,q))
    s,r,q=cases[1];z=s['z']-s['center'];bragg=s['match']-s['center']
    kr=4*np.pi*430e6/c;lim=np.sqrt(s['D']/s['cs2'])
    local=np.sqrt(np.maximum(0,(s['D']-s['ne']*e*e/(m_e*epsilon_0))/s['cs2']))
    checks={}
    for name,(ss,rr,qq) in zip(['coarse','fine','large'],cases):
        checks[name+'_bragg_amplitude']=float(np.interp(s['match'],ss['z'],abs(rr['density_phasor'])))
        checks[name+'_continuation_kz_at_minus100']=float(np.interp(-100,qq['z'],qq['kz_rad_m']))
        checks[name+'_continuation_amplitude_at_minus100']=float(np.interp(-100,qq['z'],abs(qq['density_phasor'])))
    for name in ['coarse','large']:
        for key in ['bragg_amplitude','continuation_kz_at_minus100','continuation_amplitude_at_minus100']:
            checks[name+'_'+key+'_relative_error']=abs(checks[name+'_'+key]/checks['fine_'+key]-1)
    assert abs(r['balance_relative'])<1e-8
    assert checks['coarse_bragg_amplitude_relative_error']<.002
    assert checks['coarse_continuation_kz_at_minus100_relative_error']<.01
    assert checks['coarse_continuation_amplitude_at_minus100_relative_error']<.02
    assert checks['large_continuation_amplitude_at_minus100_relative_error']<.01
    assert np.all(q['kz_rad_m'][(q['z']>-100)&(q['z']<-11)]<0)

    # Local dispersion across the observed density profile. This is a diagnostic
    # of the moment closure, not a surviving wave or a kinetic continuation.
    with h5py.File(PROFILE) as h:
        zz,nn,dd=(h[k][:] for k in ['model_height_m','model_ne_m3','model_dne_dz_m4'])
    density=CubicHermiteSpline(zz,nn,dd)
    zw=np.linspace(zz[0],s['center'],20001);nw=density(zw)
    kw=np.sqrt(np.maximum(0,(s['D']-nw*e*e/(m_e*epsilon_0))/s['cs2']))
    kd=kw*np.sqrt(epsilon_0*kb*s['te']/(nw*e*e))
    # Collision-only WKB attenuation, normalized at the Bragg height; excludes
    # geometrical/amplitude transport and is only evaluated within the HF grid.
    zd=np.linspace(bragg,-100.,10001)
    nd=density(s['center']+zd);nu=np.interp(s['center']+zd,s['zf'],s['nu'])
    a=nu-1j*s['omega'];R=(a*a+s['bz']**2)/(a*(a*a+s['bx']**2+s['bz']**2))
    kc=-np.sqrt((1j*s['omega']/R-nd*e*e/(m_e*epsilon_0))/s['cs2'])
    logamp=-cumulative_trapezoid(kc.imag,zd,initial=0.)
    atten=20/np.log(10)*logamp
    summary=dict(bragg_height_relative_m=bragg,bragg_altitude_m=s['match'],
        bragg_k_rad_m=kr,bragg_wavelength_m=2*np.pi/kr,
        formal_fluid_kmax_rad_m=lim,formal_fluid_min_wavelength_m=2*np.pi/lim,
        profile_max_k_rad_m=float(kw.max()),profile_bottom_height_m=float(zw[0]),
        collision_attenuation_db_at_minus100=float(atten[-1]),
        source_frequency_hz=s['frequency'],temperature_K=s['te'],
        dz_m=s['dz'],grid_nyquist_rad_m=np.pi/s['dz'])
    for db in [20,40]:
        d=float(np.interp(db,-atten,-zd))
        summary[f'collision_minus{db}db_height_relative_m']=-d
        summary[f'collision_minus{db}db_k_rad_m']=float(np.interp(-d,z,local))
    for d in [-4.052576365,-10,-20,-40,-60,-100]:
        kval=float(np.interp(d,z,local));print(f'z={d:.4f} m, |k|={kval:.5f} rad/m, lambda={2*np.pi/kval:.6f} m')
    print('SUMMARY',summary);print('CHECKS',checks)
    path=ROOT/'output/fluid/fdtd_extended_kz.h5'
    with h5py.File(path,'w') as h:
        h.attrs['generator']='extend_fluid_kz.py'
        h.attrs['source_sha256']=hashlib.sha256(FIELD.read_bytes()).hexdigest()
        h.attrs['profile_sha256']=hashlib.sha256(PROFILE.read_bytes()).hexdigest()
        h.attrs['limitations']='Fixed-frequency linear warm-fluid, one-way imposed Maxwell pump; collisions included, Landau damping absent. Formal n_e->0 bound is not a kinetic or detectability cutoff.'
        save_group(h.create_group('summary'),summary);save_group(h.create_group('checks'),checks)
        save_group(h.create_group('setup'),s);save_group(h.create_group('full'),r)
        save_group(h.create_group('outgoing'),q)
        h['outgoing'].attrs['definition']='Homogeneous continuation below -10 m, density phasor normalized to the full solution at the uppermost cell; continued local HF forcing omitted only in this diagnostic. Not an exact decomposition of the full driven solution.'
        plot=dict(z=z,k_local=local,lambda_local=2*np.pi/np.maximum(local,1e-30),
            amplitude=abs(r['density_phasor']),real=r['density_phasor'].real,
            k_phase=-r['kz_rad_m'],z_out=q['z'],amplitude_out=abs(q['density_phasor']),
            k_out=-q['kz_rad_m'],lambda_out=2*np.pi/np.maximum(-q['kz_rad_m'],1e-30),
            height_km=zw/1000,k_wide=kw,k_debye=kd,
            attenuation_z=zd,attenuation_db=atten)
        save_group(h.create_group('plot'),plot)


if __name__=='__main__':main()
