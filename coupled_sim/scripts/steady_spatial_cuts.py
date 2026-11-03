"""Causal spatial field cuts for Figure 6; provenance stays in HDF5.
Reconstruct the response to the same rapid turn-on used in the startup memo.
The x dependence is exp(i kx x), exactly the model's single-kx geometry.
"""
from pathlib import Path
import hashlib
import h5py
import numpy as np
from scipy.constants import epsilon_0, k as kb
from plasma_line_startup import turn_on

ROOT = Path(__file__).resolve().parents[2]
source = ROOT/'output/gaussian_pulse/response_dz0.002_z60.h5'
t = .0015
sigma = 10e-6/np.sqrt(8*np.log(2))
center_time = 60e-6
with h5py.File(source) as h:
    offsets = np.array(sorted(map(float,h['response'].keys())))
    df = float(h.attrs['frequency_spacing_hz'])
    T = t+center_time
    om = 2*np.pi*offsets
    integral = np.full(len(om), complex(T))
    nz = om != 0
    integral[nz] = -np.expm1(-1j*om[nz]*T)/(1j*om[nz])
    weights = df*np.exp(-.5*(om*sigma)**2)*np.exp(1j*om*center_time)*integral
    # Check the continuous integral against the memo's sampled quadrature.
    with h5py.File(ROOT/'data/plasma_line_startup.h5') as start:
        H = start['received_sideband_transfer'][:]
        tt, AA = turn_on(offsets,H/H[len(H)//2],df,sigma)
        reference = np.interp(t,tt,AA)
        np.testing.assert_allclose(weights@(H/H[len(H)//2]),reference,rtol=2e-5,atol=2e-7)
    z = h['height_m'][:]; zr=float(h.attrs['bottom_resonance_m']); rel=z-zr
    use=np.flatnonzero((rel>=-80.1)&(rel<=60.1)); lo,hi=use[0],use[-1]+1
    z=z[lo:hi]; rel=z-zr; ne=h['ne_m3'][lo:hi]
    fields=np.zeros((2,len(z)),complex)
    for offset,weight in zip(offsets,weights):
        d=h[f'response/{offset:+.3f}']
        fields[0] += weight*d[2,lo:hi]
        fields[1] += weight*d[9,lo:hi]*np.sqrt(epsilon_0*ne/(3*kb*300))
    kx=float(h.attrs['kx_m1']); f0=float(h.attrs['carrier_hz'])
    cw=h['response/+0.000']
    e0=cw[2,lo:hi]; n0=cw[9,lo:hi]*np.sqrt(epsilon_0*ne/(3*kb*300))
    # Within the interaction region both fields must have settled by 1.5 ms.
    close=(rel>-12)&(rel<2)
    errors=[float(np.linalg.norm((v-v0)[close])/np.linalg.norm(v0[close])) for v,v0 in zip(fields,[e0,n0])]
    print("Broad-window settling errors", errors)
    bragg=(rel>-4.5)&(rel<-3.5)
    bragg_errors=[float(np.linalg.norm((v-v0)[bragg])/np.linalg.norm(v0[bragg])) for v,v0 in zip(fields,[e0,n0])]
    print("Bragg-window settling errors",bragg_errors)
    assert max(bragg_errors)<.05,bragg_errors
    attrs=dict(h.attrs)
with h5py.File(ROOT/'data/steady_spatial_cuts.h5','w') as out:
    out.attrs.update(attrs)
    out.attrs.update(snapshot_time_after_switch_on_midpoint_s=t,source_derivative_FWHM_s=10e-6,
                     source='Causal continuous illumination; fixed horizontal wavenumber; Re[a(z,t) exp(i kx x-i omega t)]',
                     relative_L2_Ez_settling_error=errors[0],relative_L2_density_settling_error=errors[1],
                     bragg_window_relative_L2_Ez_error=bragg_errors[0],bragg_window_relative_L2_density_error=bragg_errors[1])
    for name,field,limits,dz,unit in [('Ez',fields[0],[-80,60],.05,1e3),('density',fields[1],[-12,2],.01,1e-6)]:
        xe=np.linspace(-120,120,401); xc=(xe[:-1]+xe[1:])/2
        ze=np.arange(limits[0],limits[1]+dz/2,dz); zc=(ze[:-1]+ze[1:])/2
        a=np.interp(zc,rel,field)
        matrix=np.real(a[:,None]*np.exp(1j*kx*xc[None,:]-2j*np.pi*f0*t))*unit
        g=out.create_group(name);g['x_edges_m']=xe;g['z_edges_m']=ze
        g.create_dataset('instantaneous',data=matrix,compression='gzip')
        g['envelope']=abs(a)*unit
    with h5py.File(ROOT/'data/minimum_incident_plots.h5') as old:
        out.attrs['bragg_height_relative_m']=old.attrs['bragg_height_relative_m']
    out['height_relative_m']=rel
    out['Ez_envelope_V_m']=fields[0]
    out['density_envelope_m3']=fields[1]
    assert np.isfinite(fields).all()
    out['provenance/driver']=Path(__file__).read_text()
print('snapshot',t,'s; relative settling errors:',errors)
