"""Minimize incident upward HF flux for the measured coherent radar response.
Uses the existing coupled Yee/electron-fluid harmonic recurrence; no new physics.
Optimize the two complex source polarizations analytically at each angle.
"""
from pathlib import Path
import argparse,sys,hashlib
import h5py,numpy as np
from scipy.constants import c,epsilon_0,k as kb,m_e
from scipy.optimize import minimize_scalar
import coupled_arecibo as a
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from isr.coherent import sideband_amplitude

OUT=a.OUT/'minimum_incident';OUT.mkdir(parents=True,exist_ok=True)
with h5py.File(a.ROOT/'data/radar_calibration.h5') as h:
    TARGET=float(h.attrs['fitted_on_frequency_snr']);CAL=float(h.attrs['ion_line_measured_over_predicted'])
    TAU=float(h.attrs['pulse_duration_s']);TSYS=float(h.attrs['system_temperature_K'])

def metric(states,idx):
    def power(v): return a.electromagnetic_flux(np.einsum('qnp,p->qn',states,v))[idx]
    x,y=power([1,0]),power([0,1])
    cross=(power([1,1])-x-y)/2+1j*(x+y-power([1,1j]))/2
    return np.array([[x,cross],[cross.conjugate(),y]])

def solve(angle,dz=.005,cap=1.,keep=False):
    s=a.setup(dz=dz,angle=angle,cap=cap)
    ref=a.setup(dz=dz,angle=angle,cap=cap,reference=True)
    z=s['z'];d=z-s['center'];idx=np.argmin(abs(z+s['dz_cells']/2-115100.))
    Q=metric(ref['initial'],idx);assert np.linalg.eigvalsh(Q).min()>0
    n=s['initial'][9]*np.sqrt(epsilon_0*s['ne'][0]/(m_e*a.CS**2))[:,None]
    window=np.sin(.5*np.pi*np.clip(np.minimum(d+100,30-d)/10,0,1))**2
    response=np.array([sideband_amplitude(z,n[:,i],a.F,s['kx'],window,reference_range_m=s['center']) for i in range(2)])
    v=np.linalg.solve(Q,response.conj());v/=np.sqrt(np.vdot(v,Q@v).real)
    power=abs(response@v)**2;flux=TARGET*kb*TSYS/(CAL*TAU*power)
    # Cauchy-Schwarz optimum in the positive incident-flux metric.
    assert np.isclose(power,np.vdot(response.conj(),np.linalg.solve(Q,response.conj())).real,rtol=1e-10)
    state=np.einsum('qnp,p->qn',s['initial'],v);inc=np.einsum('qnp,p->qn',ref['initial'],v)
    assert abs(a.electromagnetic_flux(inc)[idx]-1)<1e-8
    rms=np.linalg.norm(inc[:3,idx])*np.sqrt(flux/2)
    # Free-space-equivalent field gives an unambiguous amplitude for the
    # vacuum-equivalent incidence parameter kx=(omega/c) sin(theta).
    equiv=np.sqrt(flux/(epsilon_0*c*np.cos(np.deg2rad(angle))))
    stats=dict(angle_deg=angle,dz_m=dz,cap_m=cap,incident_vertical_flux_W_m2=flux,
       incident_reference_E_rms_V_m=float(rms),vacuum_equivalent_E_rms_V_m=float(equiv),
       reference_plane_m=float(z[idx]),peak_Ez_V_m=float(abs(state[2,(d>-20)&(d<10)]).max()*np.sqrt(flux)),
       harmonic_residual=s['harmonic_residual'],radar_power_per_unit_flux_W=power,
       center_m=s['center'],frequency_hz=a.F,calibration=CAL,target_snr=TARGET)
    print(stats,flush=True)
    if keep:
      dn=n@v
      with h5py.File(OUT/'best_model.h5','w') as h:
       h.attrs.update(stats)
       h.attrs.update(convention='Re[phasor exp(i kx x-i omega t)]',normalization='one W/m2 upward incident EM flux; multiply phasors by sqrt(incident_vertical_flux_W_m2)',
         scope='Measured smooth profile; 300K; collisions; linear warm fluid; magnetic-meridian angles; source polarization optimized for coherent beam-filled upshifted radar return; no Landau damping',
         initialization='sustained harmonic solution; not a turn-on simulation',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
       for name,val in dict(z=z,dz_cells=s['dz_cells'],ne=s['ne'][0],density_phasor_m3=dn,E_phasor_V_m=state[:3],incident_reference_state=inc,
            state_phasor=state,kx_rad_m=s['kx'],source_polarization=v,gyro=s['gyro'],physical_nu=s['physical_nu']).items():h[name]=val
       h['provenance/analysis_source']=Path(__file__).read_text()
    return stats

def main():
    rows=[]
    def run(t,dz=.005,cap=1.,keep=False):
      stats=solve(float(t),dz,cap,keep);rows.append(stats)
      with h5py.File(OUT/'scan.h5','w') as h:
       for k in stats:h[k]=[r[k] for r in rows]
       h.attrs['angle_definition']='vacuum-equivalent angle in magnetic meridian: kx=(omega/c) sin(theta); Bx>0, Bz<0'
       h.attrs['polarization']='Analytic optimum of calibrated coherent radar power / incident EM flux at each angle'
       h['provenance/source']=Path(__file__).read_text()
      return stats
    coarse=[-25,-19.5,-16.75,-14,-10,0,5,10,12,14,16,16.75,18,19.5,25]
    for t in coarse: run(t)
    best=min(rows,key=lambda r:r['incident_vertical_flux_W_m2']);i=coarse.index(best['angle_deg'])
    assert 0<i<len(coarse)-1
    result=minimize_scalar(lambda t:np.log(run(t)['incident_vertical_flux_W_m2']),bounds=(coarse[i-1],coarse[i+1]),method='bounded',options={'xatol':.03})
    best=run(result.x,keep=True)
    refined=run(result.x,.0025,.5,keep=True)
    with h5py.File(OUT/'best_model.h5','a') as h:
      h.attrs['flux_mesh_relative_change']=abs(refined['incident_vertical_flux_W_m2']/best['incident_vertical_flux_W_m2']-1)
      h.attrs['scanned_angle_min_deg']=min(coarse);h.attrs['scanned_angle_max_deg']=max(coarse)
    print('FINAL',refined,flush=True)
if __name__=='__main__':main()
