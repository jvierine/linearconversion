"""Check recorded-E magnetic reconstruction against exact discrete vacuum modes."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fdtd_only_control import magnetic_from_electric


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('polarization', [(1, 0), (0, 1), (1, 1j)])
def test_reconstruct_staggered_magnetic_field(direction, polarization):
    n = 64
    dz = .7
    dt = .25 * dz
    k = direction * 2 * np.pi / (n * dz)
    kd = 2 * np.sin(k * dz / 2) / dz
    omega = 2 / dt * np.arcsin(dt * abs(kd) / 2)
    node = np.exp(1j * k * np.arange(n) * dz)
    face = node * np.exp(1j * k * dz / 2)
    electric = np.array([polarization[0] * node, polarization[1] * node, node * 0])
    expected = np.array([-direction * polarization[1] * face,
                         direction * polarization[0] * face, face * 0])
    got = magnetic_from_electric(electric, np.full(n, dz), 0, dt, omega, np.zeros((6, n)))
    np.testing.assert_allclose(got, expected, atol=1e-13, rtol=1e-13)
