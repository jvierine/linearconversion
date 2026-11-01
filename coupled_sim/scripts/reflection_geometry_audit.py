"""Check reflection versus resonant absorption for fixed-background FDTD media.

The harmonic solve is the discrete Yee/current recurrence used by the native
FDTD control. Angle and source polarization are varied at fixed measured n0.
The zero-field normal-incidence case is a deliberately simpler reflection test.
"""
from pathlib import Path
import argparse
import time
import numpy as np
import h5py
import coupled_arecibo as a


def flux_form(basis, index):
    p0 = a.electromagnetic_flux(basis[:, :, 0])[index]
    p1 = a.electromagnetic_flux(basis[:, :, 1])[index]
    re = (a.electromagnetic_flux(basis[:, :, 0] + basis[:, :, 1])[index] - p0 - p1) / 2
    im = -(a.electromagnetic_flux(basis[:, :, 0] + 1j * basis[:, :, 1])[index] - p0 - p1) / 2
    return np.array([[p0, re + 1j * im], [re - 1j * im, p1]])


def main(dz=.0025, cap=.5, angles=None, output=None):
    with h5py.File(a.ROOT / 'data/illumination.h5') as h:
        v = h['source_polarization'][:]
    angles = [0, 5, 10, 16.75, 25, 40] if angles is None else angles
    cases = [('B0_zero_normal', 0, {'magnetic_t': 0}),
             ('B0_zero_normal_no_collisions', 0, {'magnetic_t': 0, 'collision_scale': 0}),
             *[(f'magnetized_{x:g}', x, {}) for x in angles]]
    with h5py.File(output or a.OUT / 'reflection_geometry_audit.h5', 'w') as out:
        out.attrs['complete'] = False
        out.attrs['source'] = Path(__file__).read_text()
        out.attrs['scope'] = 'Harmonic discrete fixed-background Maxwell-current solve; all source polarizations considered through 2x2 flux matrices'
        out.attrs['finest_dz_m'] = dz
        out.attrs['outer_grid_cap_m'] = cap
        for label, angle, scenario in cases:
            scenario['ordering'] = 'COLAMD'
            start = time.time()
            print('Solving', label, flush=True)
            scale = scenario.get('collision_scale', 1.)
            s = a.setup(dz=dz, cap=cap, angle=angle, thermal=False, scenario=scenario, collision_scale=scale)
            ref = a.setup(dz=dz, cap=cap, angle=angle, thermal=False,
                          reference=True, scenario=scenario, collision_scale=scale)
            zf = s['z'] + s['dz_cells'] / 2
            lo = np.argmin(abs(zf - 115100)); hi = np.argmin(abs(zf - 120800))
            incident = flux_form(ref['initial'], lo)
            reflected = -flux_form(s['initial'] - ref['initial'], lo)
            transmitted = flux_form(s['initial'], hi)
            # Whiten the positive incident-flux metric; avoid the local SciPy
            # generalized-Hermitian LAPACK wrapper, which segfaults on this host.
            whitening = np.linalg.inv(np.linalg.cholesky(incident))
            eig, u = np.linalg.eigh(whitening @ (reflected + transmitted) @ whitening.conj().T)
            vec = whitening.conj().T @ u
            np.testing.assert_allclose(vec.conj().T @ incident @ vec, np.eye(2), atol=1e-12)
            np.testing.assert_allclose((reflected + transmitted) @ vec, (incident @ vec) * eig,
                                       atol=1e-12 * np.linalg.norm(incident @ vec))
            evaluate = lambda p, q: float(np.real(p.conj() @ q @ p / (p.conj() @ incident @ p)))
            stats = dict(angle_deg=angle, B_T=scenario.get('magnetic_t', float(a.P['magnetic_t'])), collision_scale=scale,
                         R_x=evaluate(np.array([1, 0]), reflected),
                         R_y=evaluate(np.array([0, 1]), reflected),
                         R_paper_polarization=evaluate(v, reflected),
                         T_paper_polarization=evaluate(v, transmitted),
                         min_R_plus_T=eig[0], max_R_plus_T=eig[1],
                         harmonic_residual=s['harmonic_residual'])
            if scale == 0:
                # Finite outer ports and the launch-reference subtraction have
                # an independently measured ~1e-5 accounting error. Report the
                # lossless residual; do not interpret it as physical absorption.
                stats['lossless_port_residual'] = float(np.max(abs(1 - eig)))
                np.testing.assert_allclose(eig, 1., rtol=0, atol=2e-5)
            g = out.create_group(label); g.attrs.update(stats)
            g['incident_flux_form'] = incident
            g['reflected_flux_form'] = reflected
            g['transmitted_flux_form'] = transmitted
            g['polarization_eigenvectors'] = vec
            out.flush()
            print(label, stats, 'seconds', time.time() - start, flush=True)
        out.attrs['complete'] = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dz', type=float, default=.0025)
    parser.add_argument('--cap', type=float, default=.5)
    parser.add_argument('--angles', type=float, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.dz, args.cap, args.angles, args.output)
