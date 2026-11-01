"""Fixed-equivalent-RCS sensitivity benchmark; not a new plasma-wave simulation."""
from pathlib import Path
import hashlib
import h5py
import numpy as np
from scipy.constants import c, k as kb

ROOT = Path(__file__).resolve().parents[2]
EISCAT = 'https://eiscat.se/wp-content/uploads/2016/06/1998-2001-Multi-Annual-Report.pdf'
MH = 'https://apollo.haystack.mit.edu/isr/status/data/'
MHPOWER = 'https://www.haystack.mit.edu/the-millstone-hill-geospace-facility/'
MHTEMP = 'https://www.sgo.fi/Events/RS2012/l/04_Stromme_US_ISRs.pdf'
SY22 = 'https://doi.org/10.1029/2022JA030451'
SY24 = 'https://doi.org/10.1029/2024SW003963'
# Published representative hardware, not the guaranteed performance of a live experiment.
# SYISR's 5.5-dB allowance is the 2022 hard-target budget, shown as a scenario.
rows = [
 ('EISCAT UHF',931.,2.6,48.,100.,0.,EISCAT),
 ('EISCAT VHF',224.,3.,46.,300.,0.,EISCAT),
 ('EISCAT Svalbard (42 m)',500.,1.,45.,60.,0.,EISCAT),
 ('Millstone zenith',440.2,2.5,44.5,175.,0.,MH+' '+MHPOWER+' '+MHTEMP),
 ('Millstone MISA',440.2,2.5,43.5,175.,0.,MH+' '+MHPOWER+' '+MHTEMP),
 ('SYISR (2022)',440.,2.,43.,253.,5.5,SY22),
 ('SYISR expanded (2024)',440.,4.,46.,253.,5.5,SY24+' '+SY22),
]
source = ROOT/'data/plasma_line_rcs.h5'
with h5py.File(source) as h:
    i = np.argmin(abs(h['time_unix_s'][:]-1525453430.))
    sigma = float(h['band1/rcs_excess_m2'][i])
    R = float(h['band1/values'][i,2])*1000
    C = float(h['ion_check/per_record_multiplier'][i])
    observed = float(h['band1/values'][i,0])
tau=44e-6

def snr(sigma,f_mhz,p_mw,g_db,t,loss_db):
    return p_mw*1e6*10**(2*g_db/10)*(c/(f_mhz*1e6))**2*sigma*tau/((4*np.pi)**3*R**4*kb*t)*10**(-loss_db/10)

reference=snr(sigma,430.,1.,61.1,150.,-10*np.log10(C))
assert np.isclose(reference,observed,rtol=.001)
results=[]
for name,f,p,g,t,loss,url in rows:
    rho=snr(sigma,f,p,g,t,loss)
    # Independent on/off exponential powers, N in EACH average; conservative
    # Gaussian scattered voltage model, not phase-coherent pulse integration.
    n5=int(np.ceil(25*((1+rho)**2+1)/rho**2))
    ratio=(p*10**(2*(g-61.1)/10)*(430/f)**2*150/t*10**(-loss/10)/C)
    assert np.isclose(rho/reference,ratio,rtol=1e-12)
    results.append([f,p,g,t,loss,10*np.log10(rho),n5,rho])
    print(name,results[-1])
out=ROOT/'data/radar_detectability.h5'
with h5py.File(out,'w') as h:
    h.attrs.update(reference_snr=reference,reference_rcs_m2=sigma,range_m=R,window_s=tau,
      arecibo_calibration=C,scope='Fixed equivalent RCS, monostatic, equal range and window. Not beam-filled cross-radar wave-model prediction.',
      threshold_definition='rho/sqrt(((1+rho)^2+1)/N)>=5; N independent spectra per on/off mean',
      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    h['names']=np.array([r[0] for r in rows],dtype=h5py.string_dtype())
    h['sources']=np.array([r[-1] for r in rows],dtype=h5py.string_dtype())
    h['table']=results
    h['table'].attrs['columns']='frequency_MHz,peak_power_MW,gain_dBi,total_temperature_K,extra_loss_dB,snr_dB,N5,snr_linear'
    h['provenance/source']=Path(__file__).read_text()
# TeX is a presentation product; numerical scientific data are stored in HDF5.
lines=[r'\begin{tabular}{@{}lrrrrrrr@{}}',r'\toprule',
 r'Radar & MHz & MW & dBi & K & $L$ (dB) & SNR (dB) & $N_5$\\',r'\midrule']
for row,v in zip(rows,results):
 f,p,g,t,loss,sdb,n5,rho=v
 lines.append(f'{row[0]} & {f:g} & {p:g} & {g:g} & {t:g} & {loss:g} & {sdb:+.1f} & {n5:,}'+r'\\')
lines.extend([r'\bottomrule',r'\end{tabular}'])
(ROOT/'figures/radar_detectability_table.tex').write_text('\n'.join(lines)+'\n')
