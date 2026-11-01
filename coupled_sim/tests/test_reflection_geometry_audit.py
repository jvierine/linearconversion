"""Polarization flux matrix must reproduce directly evaluated Poynting flux."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reflection_geometry_audit import flux_form
from coupled_arecibo import electromagnetic_flux


def test_complex_polarization_flux_form():
    rng = np.random.default_rng(321)
    basis = rng.normal(size=(10, 8, 2)) + 1j * rng.normal(size=(10, 8, 2))
    metric = flux_form(basis, 3)
    for _ in range(20):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        state = np.einsum('qnp,p->qn', basis, v)
        np.testing.assert_allclose(v.conj() @ metric @ v, electromagnetic_flux(state)[3], atol=1e-16)
