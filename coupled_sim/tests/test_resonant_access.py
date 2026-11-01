"""Independent local-dispersion checks for the ray-access audit."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from resonant_access_audit import cold_roots,Parameters,electromagnetic_dispersion


def test_unmagnetized_roots_match_scalar_dispersion():
    par=Parameters(magnetic_t=0,q=.25)
    for p in [.3,.8,1.1]:
        roots=cold_roots(p,par)
        np.testing.assert_allclose(roots**2,1-p-par.q**2,atol=5e-8)


def test_magnetized_polynomial_roots_satisfy_maxwell_matrix():
    par=Parameters(frequency_hz=5.34002855e6,magnetic_t=35.065e-6,
                   magnetic_tilt_deg=180-46.816,q=np.sin(np.deg2rad(14)))
    for p in [.3,.95,.98,1.01,1.1]:
        for nz in cold_roots(p,par):
            matrix=electromagnetic_dispersion([par.q*par.k0,0,nz*par.k0],p,par,cold=True)
            sv=np.linalg.svd(matrix,compute_uv=False)
            assert sv[-1]/max(sv[0],1)<1e-10
