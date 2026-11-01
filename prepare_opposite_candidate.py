"""Native opposite-sideband diagnostic and catalogue overlay inputs.
Archive power is unchanged; display normalization is P/median(P below85km).
The 102km features are retained as unclassified, not confirmed plasma waves.
"""
from pathlib import Path
import h5py,numpy as np,hashlib
ROOT=Path(__file__).resolve().parent
with h5py.File(ROOT/'data/plasma_line_rcs.h5') as cat,h5py.File(ROOT/'data/plasma_line_rcs_inputs.h5') as inp,h5py.File(ROOT/'evidence/seid_event/results4/spl-1525454063.h5') as raw,h5py.File(ROOT/'data/opposite_candidate.h5','w') as o:
 rows=cat['catalog/rows'][:];r=rows[rows[:,0]==3][0];i=int(r[1]);z=raw['range'][:];f=-raw['freqs'][:];p=raw['PL'][:];q=p/np.median(p[z<85],axis=0)-1
 iz=np.argmin(abs(z-r[4]));selz=(z>=115)&(z<=121);self=(f>=-5.45)&(f<=-5.23);fi=np.flatnonzero(self);fi=fi[np.argsort(f[fi])]
 o['map/frequency_mhz']=f[fi];o['map/height_km']=z[selz];o['map/power_over_noise_db']=10*np.log10(np.maximum(1+q[selz][:,fi],1e-9))
 o['candidate_row']=r;o.attrs.update(time_unix_s=r[2],height_km=r[4],frequency_mhz=r[3],rcs_m2=r[8],source=Path(__file__).read_text(),native_sha256=hashlib.sha256((ROOT/'evidence/seid_event/results4/spl-1525454063.h5').read_bytes()).hexdigest())
 f3=-inp['band3/frequency_mhz'][:];order=np.argsort(f3);o['core/frequency_magnitude_mhz']=f3[order]
 o['core/earlier']=inp['band3/excess'][i-1,iz][order];o['core/candidate']=inp['band3/excess'][i,iz][order];o['core/later']=inp['band3/excess'][i+1,iz][order]
 f1=inp['band1/frequency_mhz'][:];order=np.argsort(f1);o['core/positive_frequency_mhz']=f1[order];o['core/positive']=inp['band1/excess'][i,iz][order]
 o['core/times_unix_s']=inp['time_unix_s'][i-1:i+2];o['core/ion_calibration']=cat['ion_check/per_record_multiplier'][i-1:i+2]
 o['enhancements']=rows;o['unclassified']=cat['unclassified_narrow_features/rows'][:]
 o.attrs['classification']='49 main returns,1 isolated opposite candidate; two lower-altitude features remain unclassified. Frequency signs follow archived-display convention, not independently verified RF IQ orientation.'
print('candidate data prepared')
