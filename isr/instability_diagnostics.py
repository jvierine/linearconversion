"""Diagnostic scales, not a kinetic stability solver or exclusion of OTSI.

The homogeneous threshold and optimistic strong-drive growth estimate follow
Djuth & Gonzales (1988), JGR 93, 196--208, doi:10.1029/JA093iA01p00196,
pp. 203--204. The source's field amplitude convention must be used consistently.
Threshold assumptions include a homogeneous, weakly ion-damped approximation;
finite gradients, oblique magnetized dynamics and actual pump fields require
an inhomogeneous stability calculation. Never compare total converted Langmuir
field to an EM dipole-pump threshold without checking the pump approximation.
"""
import numpy as np
from scipy.constants import e, m_e, epsilon_0, k as kb, atomic_mass


def otsi_homogeneous_field(ne_m3, te_k, ti_k, collision_s, angle_deg=0.):
    """Source-convention electric field amplitude (V/m), not ERP or flux.

angle is between daughter k and local pump E, not HF incidence or B tilt.
Aligned angle=0 gives the lowest angular threshold in this approximation.
nu is the electron collision rate in s^-1, not Hz times an implicit 2*pi.
"""
    wp=np.sqrt(np.asarray(ne_m3)*e*e/(epsilon_0*m_e))
    cosine=np.cos(np.deg2rad(angle_deg))
    return np.sqrt(4*np.asarray(ne_m3)*kb*(np.asarray(te_k)+ti_k)*collision_s
                   /(epsilon_0*wp))/np.abs(cosine)


def otsi_optimistic_growth(field_v_m, k_m1, frequency_hz, ion_mass_amu):
    """Optimized, undamped strong-drive amplitude rate gamma (s^-1).

Valid as a screening estimate under the source asymptotic assumptions,
not a universal kinetic upper bound. Power e-folding time is 1/(2 gamma).
"""
    return (np.asarray(k_m1)**2*e*e*np.asarray(field_v_m)**2/
            (8*m_e*ion_mass_amu*atomic_mass*2*np.pi*frequency_hz))**(1/3)


def acoustic_frequency(k_m1, te_k, ti_k, ion_mass_amu):
    """Weak-damping fluid estimate; ion composition and kinetic damping matter."""
    cs=np.sqrt(kb*(np.asarray(te_k)+3*ti_k)/(ion_mass_amu*atomic_mass))
    return np.asarray(k_m1)*cs/(2*np.pi)
