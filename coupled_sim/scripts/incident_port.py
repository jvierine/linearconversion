"""Propagating electromagnetic launch channels at the homogeneous lower port.

A cold collisionless root identifies a propagating channel. Its collisional
continuation supplies the launch polarization. Evanescent source currents must
not be optimized as though they were incoming far-field radio waves.
"""
import itertools
import numpy as np
from numpy.polynomial import Polynomial


def eigenmodes(omega, gyro, wp, nu, kx):
    gx, gy, gz = np.asarray(gyro) / omega
    cross = np.array([[0, gz, -gy], [-gz, 0, gx], [gy, -gx, 0]])
    eps = np.eye(3) + 1j*(wp/omega)**2*np.linalg.inv(
        (nu/omega-1j)*np.eye(3)+cross)
    q = kx/omega
    mat = [[Polynomial([eps[i,j]]) for j in range(3)] for i in range(3)]
    mat[0][0] += Polynomial([0,0,-1])
    mat[1][1] += Polynomial([-q*q,0,-1])
    mat[2][2] += Polynomial([-q*q])
    mat[0][2] += Polynomial([0,q]); mat[2][0] += Polynomial([0,q])
    determinant = Polynomial([0])
    for perm in itertools.permutations(range(3)):
        term = Polynomial([(-1)**sum(perm[i]>perm[j] for i in range(3) for j in range(i+1,3))])
        for i in range(3):
            term *= mat[i][perm[i]]
        determinant += term
    result = []
    for nz in determinant.trim(tol=1e-13).roots():
        matrix = np.array([[mat[i][j](nz) for j in range(3)] for i in range(3)])
        _, _, vh = np.linalg.svd(matrix)
        field = vh[-1].conj()
        flux = np.cross(field, np.cross([q,0,nz],field).conj())[2].real
        result.append((nz, field, flux))
    return result


def incoming_channels(omega, gyro, wp, nu, kx):
    cold = [r for r in eigenmodes(omega,gyro,wp,0.,kx)
            if abs(r[0].imag)<1e-7 and r[2]>1e-8]
    collisional = eigenmodes(omega,gyro,wp,nu,kx)
    return [min(collisional,key=lambda r:abs(r[0]-root[0])) for root in cold]


def source_basis(s, reference, index):
    channels = incoming_channels(s['omega'],s['gyro'],s['wp'][0,index],
                                 s['physical_nu'][0,index],s['kx'])
    if len(channels)==2:
        return np.eye(2,dtype=complex)
    if not channels:
        raise ValueError('No propagating incident EM channel at lower reference plane')
    return np.linalg.solve(reference['initial'][:2,index,:], channels[0][1][:2])[:,None]
