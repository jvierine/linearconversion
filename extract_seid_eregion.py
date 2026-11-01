"""Read-only selection of archived Arecibo products for reproducible fitting.

Run with conda base on Seid. Originals are never modified. Numeric products
and source paths are retained in HDF5; no CSV is written.
"""
from pathlib import Path
import argparse
import hashlib
import h5py
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', default='/scratch/data/juha/arecibo/eregion_pl/processing')
    ap.add_argument('--output', required=True)
    args=ap.parse_args(); root=Path(args.root)
    targets=np.array([1525446606,1525449983,1525453430,1525453540,
                      1525459259,1525459359,1525460596])
    with h5py.File(args.output,'w') as out:
        out.attrs.update(source_root=str(root),source_host='seid.uit.no',
                         provenance='extract_seid_eregion.py',
                         purpose='Unmodified array selections; not calibrated or fitted')
        for kind in ['clp','csp']:
            files=sorted((root/'results3').glob(kind+'_acf-*.h5'))
            group=out.create_group(kind)
            for i,path in enumerate(files):
                with h5py.File(path,'r') as f:
                    g=group.create_group(path.stem)
                    r=f['ranges'][:]
                    keep=((r>=90)&(r<=150))|((r>=650)&(r<=730))
                    g['ranges_km']=r[keep]
                    g.create_dataset('acf',data=f['acf'][keep,:],compression='gzip')
                    for name in ['t0','t1','lags','n_ipp']:
                        if name in f:g[name]=f[name][()]
                    g.attrs['source_path']=str(path)
                    g.attrs['original_shape']=f['acf'].shape
            print(kind,len(files),flush=True)
        files=sorted((root/'results4').glob('spl-*.h5'))
        ts=np.array([int(p.stem.split('-')[-1]) for p in files])
        wanted=np.unique(np.r_[targets,np.arange(ts.min(),ts.max()+1,300)])
        indices=np.unique([np.argmin(abs(ts-t)) for t in wanted])
        group=out.create_group('plasma_line')
        for i in indices:
            path=files[i]
            with h5py.File(path,'r') as f:
                g=group.create_group(path.stem)
                # Both sidebands and all original range gates are retained.
                for key in f:
                    value=f[key][()]
                    g.create_dataset(key,data=value,compression='gzip' if np.ndim(value)>0 else None)
                g.attrs['source_path']=str(path)
                g.attrs['sha256_original']=hashlib.sha256(path.read_bytes()).hexdigest()
        print('plasma_line',len(indices),flush=True)
    print('output',args.output,Path(args.output).stat().st_size,flush=True)


if __name__=='__main__':main()
