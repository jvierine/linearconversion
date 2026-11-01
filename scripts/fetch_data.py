"""Download versioned HDF5 inputs with GitHub CLI and verify SHA-256."""
from pathlib import Path
import hashlib,subprocess,tempfile,shutil
import h5py
ROOT=Path(__file__).resolve().parents[1]
with h5py.File(ROOT/'data/release_assets.h5') as h:
 for asset,g in h.items():
  dest=ROOT/g.attrs['destination'];expected=g.attrs['sha256']
  if dest.exists():
   if hashlib.sha256(dest.read_bytes()).hexdigest()!=expected:
    raise RuntimeError(f'{dest} differs from the archived input; refusing to overwrite it')
   print('Verified',dest.relative_to(ROOT));continue
  with tempfile.TemporaryDirectory() as d:
   subprocess.run(['gh','release','download',h.attrs['tag'],'--repo',h.attrs['repository'],'--pattern',asset,'--dir',d],check=True)
   p=Path(d)/asset
   if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Checksum failed: {asset}')
   dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(p),dest)
  print('Downloaded and verified',dest.relative_to(ROOT))
