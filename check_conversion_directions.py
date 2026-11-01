"""Local phase and group directions at the radar Bragg match.

Differentiate the real longitudinal Maxwellian dielectric at fixed density,
B and temperature. This is a local weak-loss electrostatic diagnostic, not
a full electromagnetic ray trace through the conversion region.
Context: Muldrew (1993), doi:10.1029/92RS00273.
"""
import h5py
import numpy as np
from scipy.constants import c
from isr.conversion import Parameters
from isr.landau import bragg_match
from isr.rays import kinetic_group_velocity as group_velocity


if __name__ == '__main__':
    kR = 4*np.pi*430e6/c
    with h5py.File('data/conversion_directions.h5', 'w') as h:
        h.attrs.update(provenance='check_conversion_directions.py; isr/landau.py',
                       interpretation=__doc__, radar_hz=430e6,
                       coordinate='x tangent to layer, z upward normal; gyro axis in positive x,z quadrant')
        for freq in [4.51e6, 5.33e6]:
            q = np.sin(np.deg2rad(-20))
            tilt = np.arcsin(q*2*np.pi*freq/c/kR)
            par = Parameters(frequency_hz=freq, temperature_k=300,
                             magnetic_t=35.025e-6,
                             magnetic_tilt_deg=46.815-np.rad2deg(tilt), q=q)
            p, alpha = bragg_match(par, kR)
            kd = np.sqrt(kR*kR-(q*par.k0)**2)
            vg = group_velocity(par, p, kd)
            coarse = group_velocity(par, p, kd, step=.01)
            assert np.allclose(vg, coarse, rtol=2e-5)
            g = h.create_group(f'{freq/1e6:.2f}')
            g['wavevector_m1'] = [q*par.k0, -kd]
            g['group_velocity_m_s'] = vg
            g.attrs.update(frequency_hz=freq, temperature_k=300, magnetic_t=par.magnetic_t,
                           magnetic_tilt_deg=par.magnetic_tilt_deg,
                           density_ratio=p, spatial_damping_m1=alpha,
                           phase_angle_from_down_normal_deg=abs(np.rad2deg(tilt)),
                           group_angle_from_down_normal_deg=np.rad2deg(np.arctan2(abs(vg[0]), -vg[1])))
            print(freq, dict(g.attrs), 'vg=', vg)
