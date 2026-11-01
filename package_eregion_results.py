"""Package calibrated parameters, conditional uncertainties and plotting arrays."""
import hashlib
import h5py
import numpy as np
from fit_eregion import ROOT,neutral_rates

def main():
 with h5py.File(ROOT/'plasma_fits.h5') as h:
  p=h['parameters'][:];se=h['formal_sigma'][:];r=h['range_km'][:];t=h['time_unix_s'][:];pred=h['model_acf'][:];chi=h['reduced_chi2'][:]
 with h5py.File(ROOT/'density_calibration.h5') as h:ne=h['ne_m3'][:];gain=h['gain'][()];calerr=h['log_gain_scatter'][()];points=h['points'][:]
 with h5py.File(ROOT/'compact_inputs.h5') as h:y=h['acf/mean'][:];var=h['acf/component_variance'][:];lags=h['acf/lags_s'][:];edges=h['acf/time_edges_unix_s'][:];nobs=h['acf/n_records'][:]
 with h5py.File(ROOT/'priors.h5') as h:priors={k:h[k][:] for k in ['Tn_K','B_T','angle_k_B_deg','N2_m3','O2_m3','O_m3']}
 nue,nui=neutral_rates(p[:,:,2],priors['N2_m3'],priors['O2_m3'],priors['O_m3'])
 quality=(chi<3)&(nobs[:,None]>=4)&np.isfinite(p).all(axis=2)&(se[:,:,0]/p[:,:,0]<.3)
 bounds=(p[:,:,1]/priors['Tn_K']<.505)|(p[:,:,2]/priors['Tn_K']<.505)|(p[:,:,1]/priors['Tn_K']>2.97)|(p[:,:,2]/priors['Tn_K']>14.85)
 with h5py.File(ROOT/'plasma_parameters.h5','w') as h:
  for k,v in [('time_unix_s',t),('time_edges_unix_s',edges),('height_km',r),('ne_m3',ne),('ti_K',p[:,:,1]),('te_K',p[:,:,2]),('vi_m_s',p[:,:,3]),('ne_conditional_sigma_m3',ne*se[:,:,0]/p[:,:,0]),('ti_conditional_sigma_K',se[:,:,1]),('te_conditional_sigma_K',se[:,:,2]),('vi_conditional_sigma_m_s',se[:,:,3]),('reduced_chi2',chi),('nu_en_s_1',nue),('nu_in_s_1',nui)]:h[k]=v
  h['nu_ei_s_1']=np.zeros_like(ne);h['f_r_MHz']=430.;h['B_T']=priors['B_T'];h['angle_k_B_deg']=priors['angle_k_B_deg'];h['Tn_msis_K']=priors['Tn_K']
  h['quality_good']=quality&~bounds;h['temperature_bound_hit']=bounds;h['n_acf_records']=nobs
  h.attrs['quality_gate']='chi2<3, >=4 contributing records, amplitude relative formal error <0.3, finite solution, temperatures away from hard bounds'
  h['calibration_gain']=gain;h['calibration_log_scatter']=calerr
  h.attrs['composition']='NO+ 30 u; temperatures are composition-conditional. Mass 56 sensitivity fits are in fit_validation.h5.'
  h.attrs['collisions']='neutral number-conserving BGK only; nu_ei is fixed to zero in this model, not measured to be zero'
  h.attrs['uncertainties']='conditional diagonal-noise formal errors; common calibration scatter ~11% is separate; original lag ambiguity weights unavailable'
  h.attrs['velocity']='apparent along-k velocity; receiver Doppler sign not independently established'
  h.attrs['density_calibration']='thermal sidebands at 129,133.5,138 km; 3250 us IPP; enhanced HF line excluded'
  h.attrs['source_host']='seid.uit.no';h.attrs['source_root']='/scratch/data/juha/arecibo/eregion_pl/processing'
  for path in ['compact_inputs.h5','priors.h5','plasma_fits.h5','density_calibration.h5','fit_validation.h5']:
   h.attrs[path+'_sha256']=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
  # Default profile encompasses the strongest cluster of narrow enhancements.
  i=22;j=np.nanargmax(ne[i]);g=h.create_group('profile');g['time_index']=i;g['height_km']=r;g['ne_1e11_m3']=ne[i]/1e11
  g['ti_K']=p[i,:,1];g['te_K']=p[i,:,2];g['Tn_K']=priors['Tn_K'][i];g['vi_m_s']=p[i,:,3]
  pp=points[points[:,0]==i];g['plasma_line_density_1e11_m3']=pp[:,10]/1e11;g['plasma_line_height_km']=pp[:,3]
  g['acf_lag_us']=lags*1e6;g['acf_measured_real']=y[i,j].real/y[i,j,0].real;g['acf_measured_imag']=y[i,j].imag/y[i,j,0].real
  g['acf_model_real']=pred[i,j].real/y[i,j,0].real;g['acf_model_imag']=pred[i,j].imag/y[i,j,0].real;g['acf_sigma']=np.sqrt(var[i,j])/y[i,j,0].real
  g.attrs['time_start_unix_s']=edges[i];g.attrs['time_end_unix_s']=edges[i+1];g.attrs['acf_height_km']=r[j]
 print('packaged',ROOT/'plasma_parameters.h5')
if __name__=='__main__':main()
