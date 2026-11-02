"""Saved-result audits: original event, translation control, mesh and energy."""
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'data/altitude_sweep.h5'

def test_collisions_and_event_recovery():
    with h5py.File(PATH) as h,h5py.File(ROOT/'reference/coupled/minimum_incident.h5') as b:
        table=h['sweep/table'][:];event=table[np.argmin(abs(table[:,0]-h.attrs['reference_center_m']))]
        np.testing.assert_allclose(table[:,6],table[:,4]+table[:,5],rtol=1e-12)
        np.testing.assert_allclose(table[:,5],table[0,5],rtol=1e-12)
        assert np.all(np.diff(table[:,4])<0)
        np.testing.assert_allclose(event[2],b.attrs['target_snr'],rtol=.003)
        assert abs(h.attrs['angle_deg']-b.attrs['angle_deg'])<1e-12
        assert h.attrs['temperature_K']==300.
        assert 1000<h.attrs['Ln_m']<1060
        np.testing.assert_allclose(event[6],5772.27,rtol=1e-4)

def test_mesh_taper_and_discrete_energy():
    with h5py.File(PATH) as h:
        table=h['sweep/table'][:]
        assert np.max(table[:,7])<.3
        for group in ['sweep/cases','checks/cases']:
            for g in h[group].values():
                assert g.attrs['harmonic_residual']<1e-8
                assert abs(g.attrs['global_energy_relative_error'])<1e-7
                assert g.attrs['pressure_adjoint_error']<1e-10
        for g in h['checks/cases'].values():
            if not g.attrs['control']:
                coarse=table[np.argmin(abs(table[:,0]-g.attrs['center_m']))]
                assert abs(10*np.log10(g.attrs['RCS_m2']/coarse[1]))<.02

def test_damping_is_not_only_radar_geometry():
    with h5py.File(PATH) as h:
        rows=h['sweep/table'][:];lower=rows[np.argmin(abs(rows[:,0]-98000.))]
        for g in h['checks/cases'].values():
            if g.attrs['control'] and abs(g.attrs['center_m']-98000.)<.01:
                assert g.attrs['SNR']>50
                assert lower[2]/g.attrs['SNR']<1e-10
                break
        else:raise AssertionError('98 km fixed-collision control missing')

def test_independent_integration_endpoints():
    with h5py.File(ROOT/'data/altitude_sweep_endpoint_check.h5') as h:
        for key in ['98.000000km','110.000000km','117.945504km']:
            db=10*np.log10(h[key+'/SNR'][:])
            if key.startswith('98.'):assert db.max()<-100.
            else:assert np.ptp(db)<.001
