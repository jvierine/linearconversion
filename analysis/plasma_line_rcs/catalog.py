"""Paired-sideband excess, per-integration ion calibration and event catalogue."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parent))
import h5py,numpy as np
from scipy.optimize import least_squares
from estimate import fit_core,CENTERS
from datetime import datetime,timezone

def utc(t):return datetime.fromtimestamp(t,timezone.utc).strftime('%H:%M:%S')
with h5py.File(ROOT/'data/plasma_line_rcs_inputs.h5') as inp,h5py.File(ROOT/'data/plasma_line_rcs.h5','a') as out:
 t=out['time_unix_s'][:];r=out['ion_check/height_m'][:];p=out['ion_check/predicted_integrated_snr_hz'][:];m=out['ion_check/measured_integrated_snr_hz'][:]
 ratio=m[:,:,1]/p;cal=np.median(ratio,axis=1);low=np.median(ratio[:,r<130000],axis=1);high=np.median(ratio[:,r>=130000],axis=1)
 for k,v in {'per_record_multiplier':cal,'split_height_validation_ratio':high/low,'finite_band_integral_ratios':np.median(m/m[:,:,1,None],axis=1)}.items():out['ion_check/'+k]=v
 out.attrs['final_rcs_calibration']='Per-native-integration median observed/predicted ion-line power125-135km.125-130 versus130-135km split and1.5/2/2.5MHz half-bandwidth integrals retained. Earlier fixed-reference columns retained explicitly; final catalogue uses per-record multiplier.'
 out.attrs['paired_excess_definition']='Signed fitted amplitude in selected sideband minus fitted amplitude at same height and opposite signed frequency. Assumes the ordinary thermal contribution is symmetric. Opposite-sideband narrow candidates are retained separately; no mechanism assigned.'
 rows=[];summary=[]
 for j in range(4):
  g=out['band'+str(j)];v=g['values'][:];other=(j+2)%4;of=inp[f'band{other}/frequency_mhz'][:];op=[]
  for i,iz in enumerate(g['selected_range_index'][:]):
   val,rms,*_=fit_core(of,inp[f'band{other}/excess'][i,int(iz)],CENTERS[other]);op.append([val[0],rms])
  op=np.array(op);D=v[:,0]-op[:,0];noise=np.sqrt(v[:,6]**2+out[f'band{other}/values'][:,6]**2)
  det=(v[:,7]>0)&(D>6*noise)
  # Conversion factor independent of fitted amplitude, robust at zero.
  R=v[:,2]*1000
  from scipy.constants import c,k
  factor=k*150/44e-6*(4*np.pi)**3*R**4/(1e6*10**(2*6.11)*(c/430e6)**2)/cal
  sigma=D*factor;nf=sigma*v[:,12];sense=6*noise*factor
  for key,a in dict(opposite_fit=op,paired_excess_snr=D,paired_noise_scale=noise,enhancement_detected=det,rcs_excess_m2=sigma,rcs_excess_gaussian_m2=nf,rcs_sensitivity_m2=sense,rcs_narrow_total_m2=v[:,0]*factor).items():g[key]=a
  ids=np.flatnonzero(det);print('FINAL BAND',j,'detections',len(ids),flush=True)
  for i in ids:
   rows.append([j,i,t[i],v[i,1],v[i,2],v[i,0],op[i,0],D[i],sigma[i],10*np.log10(sigma[i]),nf[i],cal[i],sense[i],v[i,5]])
  if len(ids):
   i=ids[np.argmax(sigma[ids])];summary.append([j,i,t[i],v[i,1],v[i,2],D[i],sigma[i],10*np.log10(sigma[i]),nf[i],cal[i]])
   print('FINAL PEAK',j,utc(t[i]),summary[-1],flush=True)
   # Fit-window and effective-duration sensitivities at each band's peak.
   iz=int(g['selected_range_index'][i]);f=inp[f'band{j}/frequency_mhz'][:];q=inp[f'band{j}/excess'][i,iz];fc=CENTERS[j];sens=[]
   for half in [.060,.075,.100]:
    use=abs(f-fc)<half;x=f[use];y=q[use]
    def model(a):return a[0]*np.sinc((x-a[1])*a[2])**2+a[3]+a[4]*(x-fc)
    a0=[v[i,0],v[i,1],44,v[i,3],v[i,4]]
    fit=least_squares(lambda a:model(a)-y,a0,bounds=([0,fc-.01,30,-1e5,-1e6],[1e6,fc+.01,65,1e5,1e6]),x_scale=[max(v[i,0],.1),.01,44,1,10])
    # Recovered integrated line power proportional to fitted peak / duration.
    sens.append([half,fit.x[0],fit.x[1],fit.x[2],np.std(fit.fun),fit.x[0]/fit.x[2]/(v[i,0]/44)])
   g['peak_window_duration_sensitivity']=sens;g.attrs['peak_index']=i
   print('SENSITIVITY',j,sens,flush=True)
 out['catalog/rows']=rows;out['catalog'].attrs['columns']='band_id,record_index,time_unix_s,signed_frequency_mhz,height_km,selected_amplitude,opposite_amplitude,paired_excess_amplitude,rcs_m2,rcs_dBsm,rcs_gaussian_m2,ion_multiplier,six_noise_rcs_m2,fit_rms'
 out['catalog/peak_summary']=summary
 out['catalog'].attrs['peak_columns']='band_id,record_index,time_unix_s,signed_frequency_mhz,height_km,paired_excess_amplitude,rcs_m2,rcs_dBsm,rcs_gaussian_m2,ion_multiplier'
 out['provenance/catalog_source']=Path(__file__).read_text()
 print('CHECK split-height ratio5,50,95%',np.percentile(high/low,[5,50,95]),flush=True)
 print('TOTAL',len(rows),flush=True)
