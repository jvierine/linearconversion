import sys
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
with h5py.File(sys.argv[1]) as h:
    z = h["z"][:] - h.attrs["center_m"]
    dn = h["density_phasor_m3"][:]
    dn *= np.sqrt(h.attrs["incident_vertical_flux_W_m2"])
keep = (z > -15) & (z < 3)
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(z[keep], np.abs(dn[keep]) / 1e6)
ax.set(xlabel="Height relative to resonance (m)",
       ylabel=r"Density amplitude ($10^6$ m$^{-3}$)")
fig.tight_layout()
fig.savefig(Path(sys.argv[1]).with_name("density_amplitude.png"), dpi=180)
