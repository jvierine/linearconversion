"""Run the pinned standalone FDTD and coupled executables on identical backgrounds.

Both start from their saved discrete continuous-wave states, then advance for
one HF period. This tests the steady driven solution, not transmitter turn-on.
The standalone FDTD binary has no electron-fluid dependency or density equation.
Its dispersive plasma current is retained: freezing that current would remove
the plasma dielectric response rather than hold the background fixed.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e, k as kb

from coupled_arecibo import OUT, hf_port_budget
from model_inputs import ROOT, TARGET, REPO


def magnetic_from_electric(electric, widths, kx, dt, omega, sponge):
    """Recover the staggered harmonic H phasor from discrete Faraday's law."""
    df = (np.roll(electric, -1, axis=1) - electric) / widths
    curl = np.array([-df[1], df[0] - 1j * kx * electric[2],
                     1j * kx * electric[1]])
    wd = 2 * np.sin(omega * dt / 2) / dt
    return -curl / (-1j * wd + np.cos(omega * dt / 2) * sponge[3:6])


def main(cycles=1):
    assert cycles >= 1
    subprocess.run(['cargo', 'build', '--release', '--locked', '--manifest-path',
                    str(REPO / 'Cargo.toml'), '--bin', 'stratified',
                    '--target-dir', str(TARGET)], check=True)
    subprocess.run(['cargo', 'build', '--release', '--locked', '--manifest-path',
                    str(ROOT / 'Cargo.toml'), '--bin', 'arecibo-coupled'], check=True)
    paths = {'fdtd_only': OUT / 'cold_control.h5',
             'coupled': OUT / 'coupled.h5'}
    matched = ['z', 'dz_cells', 'ne', 'physical_nu', 'sponge', 'gyro', 'kx_rad_m']
    with h5py.File(paths['fdtd_only']) as a, h5py.File(paths['coupled']) as b:
        for key in matched:
            np.testing.assert_array_equal(a[key][()], b[key][()])
        for key in ['frequency_hz', 'angle_deg', 'dt_s', 'electron_temperature_K']:
            assert a.attrs[key] == b.attrs[key], key
        va, vb = a['source_polarization'][:], b['source_polarization'][:]
        np.testing.assert_allclose(va / va[0], vb / vb[0], rtol=1e-13)
    dest = OUT / 'fdtd_only_comparison.h5'
    with h5py.File(dest, 'w') as out:
        out.attrs['description'] = 'Standalone fixed-background FDTD versus coupled Maxwell-electron-fluid; matched grid, n0, B0, collision rates, illumination and ports'
        out.attrs['initialization'] = 'Discrete continuous-wave state; not switch-on or nonlinear background evolution'
        out.attrs['cycles'] = cycles
        out.attrs['driver_source'] = Path(__file__).read_text()
        out.attrs['fdtd_source'] = (REPO / 'src/bin/stratified.rs').read_text()
        out.attrs['matched_arrays'] = ','.join(matched)
        for name, path in paths.items():
            with h5py.File(path) as h:
                z = h['z'][:]; widths = h['dz_cells'][:]; n = len(z)
                dt = h.attrs['dt_s'] * c
                omega = 2 * np.pi * h.attrs['frequency_hz'] / c
                kx = float(h['kx_rad_m'][()]); sponge = h['sponge'][:]
                state = h['state_phasor'][:]; incident = h['incident_reference_state'][:]
                ne = h['ne'][:]; v = h['source_polarization'][:]
                spp = round(2 * np.pi / (omega * dt)); steps = cycles * spp
                dual = (widths + np.roll(widths, 1)) / 2
                source = np.zeros((3, n), complex)
                ii = np.argmin(abs(z - 114850.))
                source[:2, ii] = v / dual[ii]
                pair = lambda x: np.stack([x.real, x.imag], axis=-1).tolist()
                raw = OUT / f'{name}_native.bin'; config = OUT / f'{name}_native.json'
                cfg = dict(n=n, dz=float(widths.min()), dz_cells=widths.tolist(),
                           dt=dt, omega=omega, kx=kx, gyro=h['gyro'][:].tolist(),
                           wp=(np.sqrt(ne * e**2 / (epsilon_0 * m_e)) / c).tolist(),
                           nu=(h['physical_nu'][:] + sponge[:3]).tolist(),
                           sponge=sponge.tolist(), initial=pair(state), source=pair(source),
                           steps=steps, record_start=0, record_stride=max(1, spp // 16),
                           output=str(raw))
                if name == 'coupled':
                    cfg.update(sqrt_ne=np.sqrt(ne).tolist(),
                               beta=float(np.sqrt(3 * kb * h.attrs['electron_temperature_K'] / m_e) / c))
                exe = TARGET / 'release' / ('stratified' if name == 'fdtd_only' else 'arecibo-coupled')
                serialized = json.dumps(cfg, separators=(',', ':'))
                report_path = OUT / f'{name}_native_report.json'
                if (raw.exists() and report_path.exists() and config.exists()
                        and config.read_text() == serialized):
                    print(f'Reusing completed {name} run with identical inputs', flush=True)
                    report = json.loads(report_path.read_text())
                else:
                    config.write_text(serialized)
                    print(f'Running {name}: {n} cells, {steps} Rust timesteps', flush=True)
                    result = subprocess.run([str(exe), str(config)], capture_output=True, text=True)
                    print(result.stdout, result.stderr, flush=True)
                    result.check_returncode(); report = json.loads(result.stdout)
                    report_path.write_text(json.dumps(report))
                fields = 3 if name == 'fdtd_only' else 4
                rows = np.fromfile(raw, dtype='<f8').reshape(-1, 1 + 2 * fields * n)
                times = rows[:, 0]
                values = rows[:, 1:].reshape(-1, fields, n, 2)
                frames = values[..., 0] + 1j * values[..., 1]
                phasors = frames * np.exp(1j * omega * c * times[:, None, None])
                native_e = phasors[:, :3].mean(axis=0)
                err = np.max(abs(phasors[:, :3] - state[:3])) / np.max(abs(state[:3]))
                assert err < 1e-5, err
                native_h = magnetic_from_electric(native_e, widths, kx, dt, omega, sponge)
                np.testing.assert_allclose(native_h, state[3:6], atol=1e-5*np.max(abs(state[3:6])), rtol=1e-5)
                reconstructed = np.concatenate([native_e, native_h], axis=0)
                zf = z + widths / 2
                lo = np.argmin(abs(zf - 115100.)); hi = np.argmin(abs(zf - 120800.))
                budget = hf_port_budget(reconstructed, incident[:6], lo, hi)
                expected = hf_port_budget(state, incident, lo, hi)
                for key in ['reflected_EM_fraction', 'transmitted_EM_fraction', 'hf_nonreturn_fraction']:
                    assert abs(budget[key] - expected[key]) < 1e-6, (key, budget[key], expected[key])
                g = out.create_group(name)
                g.attrs.update(budget)
                g.attrs['max_native_E_relative_error'] = err
                g.attrs['native_report'] = json.dumps(report)
                g.attrs['input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                g.attrs['executable_sha256'] = hashlib.sha256(exe.read_bytes()).hexdigest()
                g.attrs['magnetic_field_diagnostic'] = 'H phasor reconstructed from recorded complex E with discrete Faraday law; checked against saved H'
                g['z_m'] = z; g['time_s'] = times
                g.create_dataset('E_time_complex', data=frames[:, :3], compression='gzip')
                g.create_dataset('E_phasor', data=native_e, compression='gzip')
                g.create_dataset('H_phasor', data=native_h, compression='gzip')
                if name == 'coupled':
                    g.create_dataset('density_variable_time_complex', data=frames[:, 3], compression='gzip')
                out.flush()
                print(name, budget, 'max E error', err, flush=True)
                raw.unlink(); config.unlink(); report_path.unlink()
        out.attrs['coupled_minus_fdtd_loss_fraction'] = out['coupled'].attrs['hf_nonreturn_fraction'] - out['fdtd_only'].attrs['hf_nonreturn_fraction']
        print('Additional coupled HF loss:', out.attrs['coupled_minus_fdtd_loss_fraction'], flush=True)
    print(dest, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=1)
    main(parser.parse_args().cycles)
