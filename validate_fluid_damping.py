"""Check browser damping diagnostics independently and retain scientific HDF5.

JS computes the displayed approximations; independent eigenvalues of the
homogeneous continuity/momentum/Poisson system validate the magnetized gamma.
No fitted damping coefficient and no kinetic validation is claimed.
"""
import json
import subprocess
from pathlib import Path
import h5py
import numpy as np
from scipy.constants import e,m_e,c,k as kb,epsilon_0
from scipy.interpolate import CubicHermiteSpline

ROOT=Path(__file__).resolve().parent
r=json.loads(subprocess.check_output(['node','fluid_browser/damping_report.mjs'],cwd=ROOT))
with h5py.File(ROOT/'data/smooth_observed_layer_1525453430.h5') as h:
    density=CubicHermiteSpline(h['model_height_m'][:],h['model_ne_m3'][:],h['model_dne_dz_m4'][:])
zr=r['settings']['center'];zm=r['bragg'];zc=r['critical'];sc=r['scales']
ln=float(density(zr)/abs(density(zr,1)))
wp_prime=abs(density(zr,1))*e*e/(epsilon_0*m_e)
checks_scales={
 'density_length_m':ln,'plasma_frequency_length_m':2*ln,
 'em_airy_m':float((c*c/(abs(density(zc,1))*e*e/(epsilon_0*m_e)))**(1/3)),
 'es_airy_m':float((3*kb*300/m_e/wp_prime)**(1/3)),
 'debye_m':float(np.sqrt(epsilon_0*kb*300/(density(zr)*e*e))),
 'bragg_wavelength_m':c/(2*430e6),
 'bragg_distance_m':zr-zm,
 'gyroradius_m':np.sqrt(kb*300/m_e)/(e*r['settings']['magnetic']*1e-6/m_e)}
for name,value in checks_scales.items():assert np.isclose(value,sc[name],rtol=1e-8), (name,value,sc[name])
assert sc['amplitude_damping_length_m']==2*sc['energy_damping_length_m']
assert np.isclose(sc['source_amplitude_fwhm_m']/sc['source_intensity_fwhm_m'],np.sqrt(2))
# A linear density profile must satisfy Delta z_B = k_B² ell_ES³ exactly.
w=2*np.pi*5.33e6;kbragg=4*np.pi*430e6/c
nearby_n=density(zr)-abs(density(zr,1))*sc['es_airy_m']**3*kbragg**2
assert np.isclose((r['branch']['D']-nearby_n*e*e/(epsilon_0*m_e))/r['branch']['cs2'],kbragg*kbragg,rtol=1e-8)
errors=[]
for check in r['checks']:
    angle=np.deg2rad(check['angle']);om=e*r['settings']['magnetic']*1e-6/m_e
    ox,oz=om*np.sin(angle),om*np.cos(angle)
    nu,k,D=check['nu'],check['k'],check['D']
    # Rescale eta as v = (drive frequency/k)*eta to condition eigenvalues.
    w=2*np.pi*r['settings']['frequency']*1e6
    matrix=np.array([[0,0,0,-1j*w],[0,-nu,-oz,0],
        [0,oz,-nu,-ox],[-1j*D/w,0,ox,-nu]],complex)
    modes=np.linalg.eigvals(matrix)
    mode=modes[np.argmax(modes.imag)]
    error=abs(-mode.real/check['expected_gamma']-1)
    assert error<1e-6, (check,error)
    errors.append(error)
    check['exact_eigenvalue_real_s1']=float(mode.real)
    check['exact_eigenvalue_imag_s1']=float(mode.imag)
    check['relative_gamma_error']=float(error)
out=ROOT/'data/fluid_damping_prior.h5'
with h5py.File(out,'w') as h:
    h.attrs.update(provenance='validate_fluid_damping.py; fluid_browser/damping_report.mjs; web/fluid/damping.mjs',
        settings_json=json.dumps(r['settings']),neutral_provenance_json=json.dumps(r['neutral_provenance']),
        assumptions='Maxwellian fluid momentum drag, singly charged ions, Te=300 K assumed; MSIS neutral prior, calibrated density. No Landau damping.',
        ray_limitation='Weak-damping local fluid ray estimate; turning region requires full-wave treatment; survival is not conversion efficiency or calibrated radar power.',
        maximum_eigenvalue_relative_error=max(errors),resonance_height_m=r['settings']['center'],bragg_height_m=r['bragg'],
        gamma_at_resonance_s1=r['center_rate']['total']*r['branch']['dragFactor'])
    for key,value in r['center_rate'].items():h.attrs['center_'+key]=value
    h['height_relative_m']=np.array(r['z'])-r['settings']['center']
    for key in ['ei','en','total']:h['nu_'+key+'_s1']=[row[key] for row in r['rates']]
    for i,scenario in enumerate(r['scenarios']):
        g=h.create_group('gradient_'+str(i));g.attrs['gradient_factor']=scenario['gradient']
        for key,value in scenario['match'].items():g.attrs['bragg_'+key]=value
        g['height_relative_m']=[x['z'] for x in scenario['rows']]
        g['time_s']=[x['time'] for x in scenario['rows']]
        g['energy_change_db']=[x['loss_db'] for x in scenario['rows']]
    h.attrs['eigenvalue_checks_json']=json.dumps(r['checks'])
    scale_group=h.create_group('scales')
    for name,value in sc.items():scale_group.attrs[name]=value
    scale_group.attrs['critical_height_m']=zc
    scale_group.attrs['definitions']='web/fluid/damping.mjs modelScales; density and plasma-frequency lengths differ by 2; amplitude and energy damping lengths differ by 2; EM Airy reference is unmagnetized normal incidence; ES scale is for the vertical warm-fluid branch'
print('At resonance:',r['center_rate'])
print('Amplitude e-fold time us:',1e6/(r['center_rate']['total']*r['branch']['dragFactor']))
for scenario in r['scenarios']:print('Gradient / Bragg ray:',scenario['gradient'],scenario['match'])
print('Maximum relative damping error against exact eigenvalues:',max(errors))
print(out)
print('Scale lengths:',json.dumps(sc,indent=2))
