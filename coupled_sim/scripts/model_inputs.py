"""Portable archived event inputs; scientific provenance lives in HDF5."""
from pathlib import Path
import json,subprocess
import numpy as np
import h5py
from scipy.constants import k as kb,e
ROOT=Path(__file__).resolve().parents[1]
meta=json.loads(subprocess.check_output(['cargo','metadata','--locked','--format-version','1','--manifest-path',str(ROOT/'Cargo.toml')],text=True,cwd=ROOT))
packages={p['name']:Path(p['manifest_path']).parent for p in meta['packages']}
REPO=packages['meteor-fdtd'];FLUID=packages['electron-fluid'];TARGET=Path(meta['target_directory'])
with h5py.File(ROOT/'data/background_prior.h5') as h:
    P={k:h[k][()] for k in ['z','ne','dne','field_angle_deg','magnetic_t']}
    P['neutrals']={k:h['neutrals/'+k][:] for k in ['z','N2_m3','O2_m3','O_m3']}

def rates(z,ne):
    tev=kb*300/e; ncm=np.maximum(ne*1e-6,1e-30)
    ei=2.91e-6*ncm*(23-np.log(np.sqrt(ncm)*tev**-1.5))/tev**1.5
    p=P['neutrals']; n2,o2,o=[np.exp(np.interp(z,p['z'],np.log(p[name]))) for name in ['N2_m3','O2_m3','O_m3']]
    t=300.; en=2.33e-17*n2*(1-1.21e-4*t)*t+1.82e-16*o2*(1+.036*np.sqrt(t))*np.sqrt(t)+8.9e-17*o*(1+5.7e-4*t)*np.sqrt(t)
    return ei+en
