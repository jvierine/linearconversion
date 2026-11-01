"""Equivalent monostatic RCS of the two persistent narrow enhancement bands.

Native signed spectra, no display/deconvolution amplitudes. At each epoch fit
one sinc-squared spectral core per band and sideband, with a local linear
background. Locate the range peak near the measured Es bottomside. Report
per-return, beam-weighted equivalent RCS, not a spatial integral of overlapping
ambiguity gates, and not intrinsic cross section outside the illuminated beam.
The finite-range Gaussian on-axis gain is retained as a separate correction.
"""
from pathlib import Path
import sys,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import h5py,numpy as np
from scipy.optimize import least_squares
from scipy.constants import c,k as kb,e,epsilon_0,physical_constants
from scipy.integrate import trapezoid
from estimate_hf_power import beam
PT=1e6;TSYS=150.;TAU=44e-6;LAM=c/430e6;GAIN=10**(61.1/10)
INPUT=ROOT/'data/plasma_line_rcs_inputs.h5';OUT=ROOT/'data/plasma_line_rcs.h5'
CENTERS=[4.5135,5.34003,-4.5135,-5.34003]

def fit_core(f,y,fc):
    use=abs(f-fc)<.075;x=f[use];yy=y[use]
    def model(v):return v[0]*np.sinc((x-v[1])*44)**2+v[2]+v[3]*(x-fc)
    X=np.c_[np.sinc((x-fc)*44)**2,np.ones(len(x)),x-fc];co=np.linalg.lstsq(X,yy,rcond=None)[0]
    r=least_squares(lambda v:model(v)-yy,[co[0],fc,co[1],co[2]],bounds=([-1e6,fc-.010,-1e6,-1e7],[1e6,fc+.010,1e6,1e7]),x_scale=[max(abs(co[0]),.1),.005,max(abs(co[1]),.1),10],ftol=1e-9,xtol=1e-9,gtol=1e-9)
    return r.x,float(np.sqrt(np.mean(r.fun**2))),x,yy,model(r.x)

def main():
 with h5py.File(INPUT) as inp,h5py.File(OUT,'w') as out:
  t=inp['time_unix_s'][:];z=inp['height_km'][:];N=len(t)
  with h5py.File(ROOT/'data/eregion_fit/density_native_csp.h5') as h:
   tz=h['t'][:];zz=h['z'][:];ne=h['ne_m3'][:];keep=(zz>110)&(zz<125)
   layer=zz[keep][np.argmax(ne[:,keep],axis=1)];es=np.interp(t,tz,layer)
  with h5py.File(ROOT/'output/fluid/hf_power_from_measured_pl.h5') as h:
   cal=float(h.attrs['ion_line_measured_over_predicted']);calmad=float(h.attrs['ion_line_ratio_mad'])
  out.attrs.update(radar_peak_power_W=PT,system_temperature_K=TSYS,radar_frequency_hz=430e6,pulse_duration_s=TAU,noise_equivalent_bandwidth_hz=1/TAU,nominal_gain_dBi=61.1,ion_line_power_multiplier=cal,reference_ion_multiplier_mad=calmad,
   rcs_definition='Equivalent on-axis monostatic RCS of the excess narrow line in one range response, not the sum of overlapping range or ambiguity replicas.',
   equation='Pr=A*kB*Tsys/tau; sigma_nominal=Pr*(4*pi)^3*R^4/(Pt*G^2*lambda^2); sigma_ion=sigma_nominal/Cion; finite_range_gain=8*R^2/w(R)^2',
   detection='A>6*off_range_MAD, A>3*core_fit_RMS, abs(df)<9.5kHz, positive line contrast A>0.5*abs(local baseline). Range search is Es peak minus3km to plus0.6km.',
   scope='Both signs of the persistent4.514 and5.340MHz bands across all1407 native spectra. Ordinary PL and ambiguity replicas are not distinct enhancements. Detection gates are operational, not formal completeness/FDR estimates.',
   limitations='Nominal antenna gain, event waveform/PSF and receiver calibration are incompletely known.44us rectangular effective spectral core. Peak range sampling retained; unresolved range-center offset not corrected. Finite-range Gaussian correction is model dependent. Negative fit amplitudes retained as diagnostics. No intrinsic width inferred. Frequency-dependent gains neglected across sidebands. Empirical scatter is not absolute calibration uncertainty.',
   source=Path(__file__).read_text(),input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),gain_reference='https://doi.org/10.5194/hgss-4-19-2013')
  out['time_unix_s']=t;out['es_peak_height_km']=es;out['ipp_us']=inp['ipp_us'][:];out['source_files']=inp['source_files'][:];out['source_sha256']=inp['source_sha256'][:]
  summaries=[]
  for j,fc in enumerate(CENTERS):
   src=inp['band'+str(j)];f=src['frequency_mhz'][:];use=abs(f-fc)<.075
   x=f[use];X=np.c_[np.sinc((x-fc)*44)**2,np.ones(len(x)),x-fc];pinv=np.linalg.pinv(X)
   values=np.zeros((N,14));modelrows=[];obsrows=[];amaps=[];selected=[]
   off=(z<100)|(z>130)
   for i in range(N):
    q=src['excess'][i].astype(float);coef=q[:,use]@pinv.T;a=coef[:,0]
    noise=1.4826*np.median(abs(a[off]-np.median(a[off])));search=(z>=es[i]-3)&(z<=es[i]+.6)
    iz=np.flatnonzero(search)[np.argmax(a[search])];v,rms,xx,yy,ym=fit_core(f,q[iz],fc)
    A,nu,b,slope=v;det=(A>6*noise)and(A>3*rms)and(abs(nu-fc)<.0095)and(A>.5*abs(b))
    R=z[iz]*1000;pr=A*kb*TSYS/TAU;sig=pr*(4*np.pi)**3*R**4/(PT*GAIN**2*LAM**2);w=beam(np.array([R]))[0][0];geff=8*R*R/w**2;nf=(GAIN/geff)**2
    values[i]=[A,nu,z[iz],b,slope,rms,noise,float(det),pr,sig,sig/cal,sig/cal*nf,nf,max(A,0)+6*max(noise,rms)]
    obsrows.append(yy);modelrows.append(ym);amaps.append(a);selected.append(iz)
   g=out.create_group('band'+str(j));g.attrs['center_mhz']=fc;g.attrs['columns']='amplitude_snr,frequency_mhz,height_km,baseline,slope_per_mhz,fit_rms,off_range_amplitude_mad,detected,received_power_nominal_W,rcs_nominal_far_m2,rcs_ion_far_m2,rcs_ion_gaussian_m2,gaussian_rcs_factor,amplitude_sensitivity_6sigma'
   g['values']=values;g['frequency_mhz']=x;g.create_dataset('observed',data=obsrows,compression='gzip');g.create_dataset('fitted',data=modelrows,compression='gzip');g.create_dataset('amplitude_by_height',data=amaps,compression='gzip');g['height_km']=z;g['selected_range_index']=selected
   ii=np.flatnonzero(values[:,7]);print('BAND',j,fc,'detections',len(ii),flush=True)
   if len(ii):
    i=ii[np.argmax(values[ii,10])];print('PEAK',int(t[i]),values[i],flush=True);summaries.append([j,i,t[i],*values[i]])
  out['peak_summary']=summaries
  # Matched-band opposite return at the enhanced branch's selected height;
  # do not maximize the noise independently when testing sideband asymmetry.
  for j in [0,1]:
   f=inp['band'+str(j+2)+'/frequency_mhz'][:];rows=[]
   for i,iz in enumerate(out['band'+str(j)+'/selected_range_index'][:]):
    v,rms,*_=fit_core(f,inp['band'+str(j+2)+'/excess'][i,int(iz)],CENTERS[j+2]);rows.append([v[0],rms])
   out['band'+str(j)+'/opposite_at_same_height']=rows
  # Independent full-experiment ion-power check with the existing density fits.
  with h5py.File(ROOT/'data/eregion_fit/full_altitude_fits.h5') as h:
   tt=h['time_unix_s'][:];rr=h['height_km'][:]*1000;keep=(rr>90000)&(rr<150000);rr=rr[keep];pars=h['results'][:,keep,:4]
  ri=inp['ion_height_km'][:]*1000;W=np.maximum(0,1-abs(ri[:,None]-rr[None,:])/(c*TAU/2));w=beam(rr)[0];re=physical_constants['classical electron radius'][0];pred=[]
  for ne,te,ti,vi in pars.transpose(0,2,1):
   ae=ne*e*e/(epsilon_0*kb*te*(4*np.pi/LAM)**2);ai=ae*te/ti;area=ae**2/((1+ae)*(1+ae+ai));weight=PT*re**2*LAM**2/(kb*TSYS)*ne*area/(np.pi*w*w);pred.append(trapezoid(W*weight,rr,axis=1))
  pred=np.array(pred);interp=np.array([np.interp(t,tt,pred[:,k]) for k in range(len(ri))]).T;measured=inp['ion_integrated_snr_hz'][:];ratio=np.median(measured[:,:,1]/interp,axis=1)
  out['ion_check/time_unix_s']=t;out['ion_check/ratio']=ratio;out['ion_check/height_m']=ri;out['ion_check/predicted_integrated_snr_hz']=interp;out['ion_check/measured_integrated_snr_hz']=measured
  for mode in np.unique(inp['ipp_us'][:]):
   v=ratio[inp['ipp_us'][:]==mode];print('ION',mode,np.percentile(v,[5,50,95]),flush=True)
 print(OUT)
if __name__=='__main__':main()
