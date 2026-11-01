"""Add solved angles to the manuscript sweep without changing its reference case."""
from pathlib import Path
import hashlib

import h5py
import numpy as np

from minimum_incident_power import solve

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reference/coupled/radar_angle_refinement.h5'


def main():
    with h5py.File(ROOT / 'reference/coupled/spitze_transmission.h5') as h:
        critical = float(h.attrs['spitze_angle_magnitude_deg'])
    with h5py.File(ROOT / 'reference/coupled/radar_sensitivity.h5') as h:
        existing_angles = h['angle/degrees'][:]
        flux_to_rcs = float(h.attrs['reference_flux_W_m2'] * h.attrs['rcs_factor'])
    angles = np.unique(np.r_[np.arange(-25., 26., 1.),
                            np.arange(-18., -15.99, .25),
                            np.arange(16., 18.01, .25), -critical, critical,
                            existing_angles[existing_angles <= 5.]])
    with h5py.File(OUTPUT, 'a') as out:
        out.attrs.update(angle_definition='Vacuum-equivalent magnetic-meridian angle; Bx>0, Bz<0',
                         scope='Same linear coupled Maxwell-fluid and coherent radar operator as the original sweep; optimized polarization at each angle')
        for angle in angles:
            key = f'angles/{angle:+.8f}'
            if key not in out:
                out.create_group(key).attrs.update(solve(float(angle)))
                out.flush()
        # Selected mesh checks at both critical geometries and in the former
        # sparsely sampled negative-angle interval.
        for angle in [-critical, -8., critical]:
            key = f'checks/{angle:+.8f}'
            if key not in out:
                fine = solve(float(angle), dz=.0025, cap=.5)
                coarse = dict(out[f'angles/{angle:+.8f}'].attrs)
                fine['relative_power_change'] = abs(fine['radar_power_per_unit_flux_W'] /
                                                     coarse['radar_power_per_unit_flux_W'] - 1.)
                out.create_group(key).attrs.update(fine)
                out.flush()
        for key in ['provenance/source', 'provenance/solver_sha256']:
            if key in out:
                del out[key]
        out['provenance/source'] = Path(__file__).read_text()
        out['provenance/solver_sha256'] = hashlib.sha256(
            Path(__file__).with_name('minimum_incident_power.py').read_bytes()).hexdigest()
        rows = [dict(g.attrs) for g in out['angles'].values()]
        # Weak coherent returns can be polluted by discretization error even
        # when the linear-system residual is tiny. Resolve the negative-angle
        # branch independently; accept <1 dB changes in the displayed range,
        # or two successive estimates below the -125 dB plotting boundary.
        for j, row in enumerate(rows):
            if row['angle_deg'] > 5.:
                continue
            previous = None
            for dz, cap in [(.00125, .25), (.000625, .125), (.0003125, .0625), (.00015625, .03125)]:
                key = f"convergence/{row['angle_deg']:+.8f}/{dz:g}_{cap:g}"
                if key not in out:
                    out.require_group(key).attrs.update(solve(float(row['angle_deg']), dz=dz, cap=cap))
                    out.flush()
                current = dict(out[key].attrs)
                value = 10*np.log10(current['radar_power_per_unit_flux_W']*flux_to_rcs)
                if previous is not None and (abs(value-previous) < 1. or max(value, previous) < -125.):
                    current['mesh_change_dB'] = abs(value-previous)
                    current['below_display_floor'] = max(value, previous) < -125.
                    rows[j] = current
                    out.require_group(f"accepted/{row['angle_deg']:+.8f}").attrs.update(current)
                    break
                previous = value
            else:
                raise RuntimeError(f"Angular response not resolved at {row['angle_deg']} degrees")
        out.flush()
    for path in [ROOT / 'data/radar_sensitivity.h5', ROOT / 'reference/coupled/radar_sensitivity.h5']:
        with h5py.File(path, 'a') as h:
            old = {float(a): (float(r), float(fl)) for a, r, fl in zip(
                h['angle/degrees'][:], h['angle/RCS_m2'][:], h['angle/required_flux_W_m2'][:])}
            flux = float(h.attrs['reference_flux_W_m2'])
            factor = float(h.attrs['rcs_factor'])
            for row in rows:
                angle = row['angle_deg']
                if angle not in old or angle <= 5.:
                    old[angle] = (row['radar_power_per_unit_flux_W'] * flux * factor,
                                  row['incident_vertical_flux_W_m2'])
            keys = sorted(old)
            for key, values in [('degrees', keys), ('RCS_m2', [old[k][0] for k in keys]),
                                ('required_flux_W_m2', [old[k][1] for k in keys])]:
                del h['angle/' + key]
                h['angle/' + key] = values
            h['angle'].attrs['refinement_sha256'] = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
            h['angle'].attrs['refinement_script'] = 'coupled_sim/scripts/refine_radar_angle_sweep.py'
            h['angle'].attrs['weak_return_mesh_criterion'] = 'Successive meshes within 1 dB, or both below -125 dB m2; angles <=5 degrees'
            # Keep the normal-incidence power curve consistent with the
            # refined angular solution; the optimized reference is unchanged.
            if 'power/RCS_0_m2' in h:
                eirp = h['power/EIRP_W'][:]
                h['power/RCS_0_m2'][...] = old[0.][0] * eirp / (4*np.pi*(125e3)**2*flux)
            best = int(np.argmax(h['angle/RCS_m2'][:]))
            print(path, 'points', len(keys), 'best angle', keys[best], flush=True)


if __name__ == '__main__':
    main()
