"""Transfer only retried fit rows; verify the complete result array by SHA-256."""
from pathlib import Path
import argparse,hashlib
import h5py,numpy as np

p=argparse.ArgumentParser();p.add_argument('mode',choices=['export','apply']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
full=a.root/'full_altitude_fits.h5';patch=a.root/'full_altitude_refit_patch.h5'
if a.mode=='export':
    with h5py.File(full,'r+') as h:
        h.attrs['range_boundary_note']='The finite 10000-us receive window attenuates long lags near the upper range boundary. Gates above 1435.5 km are retained but flagged; these fits are not calibrated physical measurements at the window edge.'
        h.attrs['velocity_note']='Apparent line-of-sight phase velocity; receiver sign has not been independently calibrated.'
        h.attrs['fit_status']='All unblanked archived gates fitted; optimizer success, active bounds, formal errors and prior dependence must accompany interpretation.'
        if 'range_boundary_affected' not in h:
            h['range_boundary_affected']=h['height_km'][:]>1435.5
        result=h['results'][:]
        with h5py.File(patch,'w') as o:
            indices=h['iteration_limit_refit/indices'][:]
            o['indices']=indices;o['results']=result[indices[:,0],indices[:,1]]
            h.copy('iteration_limit_refit',o)
            h.copy('range_boundary_affected',o)
            for k,v in h.attrs.items():o.attrs[k]=v
            o.attrs['result_array_sha256']=hashlib.sha256(result.tobytes()).hexdigest()
        print('exported',len(indices),'rows; remaining unconverged',int(np.count_nonzero(result[:,:,9]==0)))
else:
    with h5py.File(full,'r+') as h,h5py.File(patch) as q:
        result=h['results'][:];indices=q['indices'][:]
        result[indices[:,0],indices[:,1]]=q['results'][:]
        digest=hashlib.sha256(result.tobytes()).hexdigest()
        assert digest==q.attrs['result_array_sha256'], 'Result mismatch; do not overwrite'
        h['results'][:]=result
        for key in ['iteration_limit_refit','range_boundary_affected']:
            if key in h:del h[key]
            q.copy(key,h)
        for k,v in q.attrs.items():h.attrs[k]=v
        print('Verified full result array:',digest)
