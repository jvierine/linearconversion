"""Independent spectral-area recovery and final catalogue integrity checks."""
from pathlib import Path
import numpy as np,h5py
from scipy.signal import fftconvolve
from scipy.constants import c,k
from integrated_core import fit
ROOT=Path(__file__).resolve().parents[2]
# Construct broadened spectrum by dense-grid numerical convolution, independent
# of the Gauss-Hermite quadrature used in the fitted response.
x=np.linspace(-.5,.5,100001);dx=x[1]-x[0];sig=.008
pulse=44*np.sinc(44*x)**2
gaussian=np.exp(-.5*(x/sig)**2)/(np.sqrt(2*np.pi)*sig)
y=fftconvolve(pulse,gaussian,mode='same')*dx/44
native=np.arange(-6,7)*.01220703125
v,rms=fit(native,np.interp(native,x,y),0.,1.,.075)
assert abs(v[0]-1)<1e-3 and abs(v[2]-sig)<1e-4
with h5py.File(ROOT/'data/plasma_line_rcs.h5','a') as h:
 a=h['catalog/rows'][:]
 assert np.all(np.isfinite(a)) and np.all(a[:,8]>0)
 assert np.allclose(10*np.log10(a[:,8]),a[:,9])
 assert np.all(h['ion_check/calibration_valid'][:][a[:,1].astype(int)])
 # Recompute radar equation directly from final paired areas and calibration.
 R=a[:,4]*1000;P=k*150*a[:,7]/44e-6/a[:,11]
 rcs=(4*np.pi)**3*R**4*P/(1e6*10**(61.1/5)*(c/430e6)**2)
 assert np.allclose(rcs,a[:,8],rtol=1e-12,atol=0)
 h.attrs['synthetic_convolution_area_recovery']=v[0]
 h.attrs['synthetic_convolution_sigma_mhz']=v[2]
 h.attrs['synthetic_convolution_residual_rms']=rms
 h.attrs['rcs_validation']='Independent FFT-convolved synthetic core recovered within0.1%; all final RCS independently recomputed from spectral area and radar equation; all retained calibrations valid.'
 if 'validation_source' in h['provenance']:del h['provenance/validation_source']
 h['provenance/validation_source']=Path(__file__).read_text()
print('PASS: synthetic area ratio',v[0],'; Gaussian sigma MHz',v[2],'; final catalogue rows',len(a))
