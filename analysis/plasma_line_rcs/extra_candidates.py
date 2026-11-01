"""Audit isolated lower-altitude narrow features outside the two Es bands.
Not classified as Es plasma waves: origin and true/ambiguous range unverified.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from integrated_core import fit,kernel
import h5py,numpy as np
from scipy.constants import c,k
with h5py.File(ROOT/'data/plasma_line_rcs.h5','a') as out:
 t=1525448646
 with h5py.File(ROOT/f'evidence/seid_event/results4/spl-{t}.h5') as h:
  f=-h['freqs'][:];z=h['range'][:];p=h['PL'][:];time=float(h['t0'][()]);q=p/np.median(p[z<85],axis=0)-1
 i=np.argmin(abs(out['time_unix_s'][:]-time));cal=out['ion_check/per_record_multiplier'][i];rows=[]
 for c0 in [2.104,-2.405]:
  mask=(z>=100)&(z<=130);fi=abs(f-c0)<.03;loc=np.unravel_index(np.argmax(q[mask][:,fi]),q[mask][:,fi].shape);iz=np.flatnonzero(mask)[loc[0]]
  v,rms=fit(f,q[iz],c0,float(q[iz,fi].max()),.075);fm=abs(f+v[1])<.075;X=np.c_[kernel(f[fm],-v[1],v[2]),np.ones(sum(fm)),f[fm]+v[1]];opp=np.linalg.lstsq(X,q[iz,fm],rcond=None)[0][0]
  A=v[0]-opp;R=z[iz]*1000;pr=A*k*150/44e-6;rcs=pr*(4*np.pi)**3*R**4/(1e6*10**(2*6.11)*(c/430e6)**2)/cal
  rows.append([time,v[1],z[iz],A,rcs,10*np.log10(rcs),rms]);print(rows[-1],flush=True)
 g=out.create_group('unclassified_narrow_features');g['rows']=rows;g.attrs['columns']='time_unix_s,signed_frequency_mhz,apparent_height_km,paired_area_times44us,conditional_rcs_m2,conditional_rcs_dBsm,fit_rms';g.attrs['status']='Isolated102km narrow returns, not established sporadic-E plasma-line enhancements. RCS assumes apparent range is true scattering range; not valid if radio interference or range alias.'
 out['provenance/extra_candidate_source']=Path(__file__).read_text()
 cal=out['ion_check/per_record_multiplier'][:];valid=(cal>.05)&(cal<1);out['ion_check/calibration_valid']=valid
 assert np.all(valid[out['catalog/rows'][:,1].astype(int)])
 out.attrs['invalid_calibration_records']=int(sum(~valid));out['provenance/integrated_core_source'][()]=Path(__file__).with_name('integrated_core.py').read_text()
