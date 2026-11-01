"""Reproduce the archived density estimator; do not replace physical fit products."""
from pathlib import Path
import hashlib
import h5py
import numpy as np
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parent
source = ROOT / 'evidence/seid_event/ne_result.h5'
with h5py.File(ROOT / 'data/eregion_fit/all_de_inputs.h5') as h:
    lag = h['csp/lags_us'][:]
    x = np.log10(np.where(lag == 0, 22, lag))
    counts = h['csp/count'][:]
    acf = h['csp/mean'][:, :500].real
    z = h['range_km'][:500]
averaged = acf.copy()
averaged[:, 3:266] = (acf[:, 2:265] + acf[:, 3:266] + acf[:, 4:267]) / 3
reconstructed = np.zeros(acf.shape[:2])
for i in range(len(acf)):
    weight = np.sqrt(counts[i])
    for order in [1, 2]:
        select = np.arange(500) <= 266 if order == 1 else np.arange(500) > 266
        design = np.vander(x, order + 1)
        at_short_lag = np.vander([np.log10(22)], order + 1)[0]
        estimator = at_short_lag @ np.linalg.pinv(design * weight[:, None]) * weight
        reconstructed[i, select] = averaged[i, select] @ estimator
reconstructed *= 42859649122.80702 * z[None, :] ** 2
with h5py.File(source) as h:
    legacy = h['Ne_m3'][:]
    labels = h['unix_seconds'][:]
    np.testing.assert_allclose(z, h['range_km'][:])
error = np.linalg.norm(reconstructed - legacy) / np.linalg.norm(legacy)
assert error < 1e-12, error
with h5py.File(ROOT / 'data/eregion_fit/full_altitude_fits.h5') as h:
    times = h['time_unix_s'][:]
    time_edges = h['time_edges_unix_s'][:]
assert len(times) == len(legacy)
assert np.all((labels >= time_edges[:-1]) & (labels <= time_edges[1:]))
select = (z >= 49.8) & (z <= 140.1)
out = ROOT / 'data/density_reference.h5'
with h5py.File(out, 'w') as h:
    h['map/ne_m3'] = legacy[:, select].T
    h['map/height_km'] = z[select]
    h['map/time_edges_utc_hour'] = (time_edges - 1525392000) / 3600
    h['map/time_utc_hour'] = (times - 1525392000) / 3600
    h['archived_time_labels_unix_s'] = labels
    for key, name in [('rgb', 'plasma'), ('density_rgb', 'nipy_spectral')]:
        h['palette/' + key] = colormaps[name](np.linspace(0, 1, 256))[:, :3]
    h.attrs.update(
        estimator='Archived weighted polynomial ACF estimator, range squared scaling and fixed gain; apparent density, not finite-Debye physical inversion.',
        averaging='Seven short-pulse integrations; three adjacent 300 m gates below 79.8 km (indices 3..265); no display smoothing.',
        timing='Physical acquisition-cycle edges; archived labels denote the final record start and are preserved separately.',
        reconstruction_relative_error=error,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        provenance=Path(__file__).read_text(),
    )
print(out, 'relative reproduction error', error)
