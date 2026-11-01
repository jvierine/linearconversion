"""Verify the 2018 Arecibo subset against Seid hashes and inventory its HDF5 data.

This reads original data without modifying them. It verifies recovery, not
physical calibration or the proposed HF excitation mechanism.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "evidence/seid_event"
records = []
for line in (BASE / "remote_sha256.txt").read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    path = BASE / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == expected, name
    record = {"file": name, "bytes": path.stat().st_size, "sha256": digest}
    if path.suffix == ".h5":
        with h5py.File(path) as h:
            record["datasets"] = {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in h.items()
            }
            t0 = float(h["t0"][()])
            record["t0_unix_s"] = t0
            record["t0_utc"] = datetime.fromtimestamp(t0, timezone.utc).isoformat()
            for key in ("range", "ranges", "freqs", "lags"):
                if key in h:
                    a = h[key][()]
                    record[key + "_endpoints"] = [float(a[0]), float(a[-1])]
            for key in ("PL", "acf"):
                if key in h:
                    record[key + "_finite_fraction"] = float(np.isfinite(h[key][()]).mean())
    else:
        record["identical_to_dropbox_copy"] = (
            path.read_bytes() == (ROOT / "evidence/sporadic_e_figures" / path.name).read_bytes()
        )
        assert record["identical_to_dropbox_copy"]
    records.append(record)

groups = defaultdict(lambda: {"files": 0, "bytes": 0})
for line in (BASE / "remote_inventory.txt").read_text().splitlines():
    size, name = line.split("\t", 1)
    if Path(name).suffix == ".h5":
        group = name.split("/")[0] if "/" in name else "top_level"
        groups[group]["files"] += 1
        groups[group]["bytes"] += int(size)

result = {
    "verified_on": "2026-10-29",
    "host": "seid.uit.no",
    "account": "juha",
    "remote_root": "/scratch/data/juha/arecibo/eregion_pl/processing",
    "ssh_route": "ssh work-laptop 'ssh juha@seid.uit.no'",
    "local_root": str(BASE),
    "status": "Processed event spectra and correlations recovered; raw complex voltages not located",
    "remote_hdf5_groups": dict(groups),
    "local_subset": records,
    "historical_voltage_paths": ["/media/USRPJUHA/eregion/", "/data2/2018.05.04"],
    "historical_paths_status": "Not populated on Seid at verification; may refer to former attached storage",
    "timing": "Original stuffr.unix2date uses UTC; acquisition-clock accuracy not independently verified",
    "calibration": "Original density scale and spectral reversal are present in scripts; physical calibration and sideband sign remain unvalidated",
    "script_archive": "evidence/seid_event/processing_scripts.tar.gz",
    "plot_script": "lp_plot_compare.py",
}
(BASE / "recovery_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
for target in (ROOT / "evidence/inventory.json", ROOT / "arecibo_overleaf/data_inventory.json"):
    inventory = json.loads(target.read_text())
    inventory["event_numerical_data"] = result
    inventory["event_figures"]["status"] = "Seven PNGs matched byte-for-byte to Seid; corresponding spectra and correlations recovered locally"
    inventory["overleaf"]["status"] = "Event figures, model products, and recovered Seid data locations documented; raw voltages still unlocated"
    target.write_text(json.dumps(inventory, indent=2) + "\n")
print(f"Verified {len(records)} files against remote SHA-256; seven images match Dropbox exactly.")
print(json.dumps(dict(groups), indent=2))
