"""Move both integration endpoints to audit weak-response spectral leakage."""
from pathlib import Path
import h5py,numpy as np
from scipy.constants import epsilon_0,m_e
import altitude_sweep as s
ROOT=Path(__file__).resolve().parents[2]
with h5py.File(ROOT/'data/altitude_sweep.h5') as h:tangent=h['incident_tangential_polarization'][:]
windows=[(-100.,30.),(-120.,50.),(-140.,80.),(-180.,100.)]
with h5py.File(ROOT/'data/altitude_sweep_endpoint_check.h5','w') as out:
    out['window_bounds_relative_m']=windows
    for center in [98000.,110000.,s.CENTER]:
        cfg=s.configuration(center);cfg['fine_bounds']=(-240.,120.)
        a=s.a.setup(dz=.005,cap=1.,angle=s.ANGLE,scenario=cfg)
        r=s.a.setup(dz=.005,cap=1.,angle=s.ANGLE,scenario=cfg,reference=True)
        ix=np.argmin(abs(r['z']+r['dz_cells']/2-(115100.+center-s.CENTER)))
        v=np.linalg.solve(r['initial'][:2,ix,:],tangent)
        inc=np.einsum('qnp,p->qn',r['initial'],v);v/=np.sqrt(s.a.electromagnetic_flux(inc)[ix])
        n=a['initial'][9]@v*np.sqrt(epsilon_0*a['ne'][0]/(m_e*s.a.CS**2))
        z=a['z'];rel=z-center;values=[]
        for lo,hi in windows:
            window=np.sin(.5*np.pi*np.clip(np.minimum(rel-lo,hi-rel)/10.,0.,1.))**2
            amp=s.sideband_amplitude(z,n,s.a.F,a['kx'],window,reference_range_m=center)
            values.append(abs(amp)**2*s.FLUX*s.CAL*s.TAU/(s.kb*s.TSYS))
        g=out.create_group(f'{center/1000:.6f}km');g['SNR']=values;g.attrs['center_m']=center
        db=10*np.log10(values);print(center/1000,db,'dB',flush=True)
        if center==98000.:assert db.max()<-100.
        else:assert np.ptp(db)<.001
    out.attrs['interpretation']='The extremely small 98 km residual is taper-dependent and establishes only non-detectability. Detectable/marginal responses are endpoint-stable.'
    out['provenance/source']=Path(__file__).read_text()
