"""D/E ACF inversion at the native cycle cadence with both pulse modes.

The fine frequency grid uses an analytic narrow-Lorentzian subtraction below
the collisional transition, retaining the full BGK residual. This prevents an
unresolved D-region line from being replaced by one huge frequency-grid bin.
Atomic O is absent in MSIS below its modeled lower boundary and contributes zero
to these collision rates. Ion composition remains conditional on NO+ (30 u).
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse
import multiprocessing as mp
import numpy as np
import h5py
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
from scipy.constants import k as kb,e,epsilon_0,m_e,atomic_mass
from fit_eregion import K,response,neutral_rates,model as clp_model,LAGS

ROOT=Path(__file__).parent/'data/eregion_fit'

class ShortPulseModel:
    def __init__(self,lags_us,df=31.25,fmax=64000.):
        self.f=np.arange(-fmax,fmax+df/2,df)
        # z(t) conj(z(t+nominal-fraction)): the nonzero CSP lag sign is
        # opposite to the positive within-pulse CLP lag convention.
        self.lags=[np.arange(4,40)*1e-6 if l==0 else (-l+np.arange(40))*1e-6 for l in lags_us]
        self.kernel=np.array([np.exp(2j*np.pi*t[:,None]*self.f).mean(axis=0)*df for t in self.lags])
    def __call__(self,ti,te,v,ne,B,theta,n2,o2,o):
        nue,nui=neutral_rates(te,n2,o2,o)
        w=2*np.pi*self.f-K*v
        ce,qe=response(w,ne,te,m_e,nue,B,theta)
        ci,qi=response(w,ne,ti,30*atomic_mass,nui)
        eps=1+ce+ci
        s=abs(ce/eps)**2*qi+abs((1+ci)/eps)**2*qe
        narrow=nui>8*K*np.sqrt(kb*ti/(30*atomic_mass))
        if narrow:
            ae=ne*e**2/(epsilon_0*kb*te*K*K)
            ai=ne*e**2/(epsilon_0*kb*ti*K*K)
            d=K*K*kb*ti/(30*atomic_mass*nui)
            gamma=d*(1+ai/(1+ae))
            weight=(ae/(1+ae))**2*d/gamma
            s=s-weight*2*gamma/(w*w+gamma*gamma)
            exact=np.array([np.mean(weight*np.exp(-gamma*abs(t)+1j*K*v*t)) for t in self.lags])
        else:exact=0.
        return self.kernel@s+exact

def prepare():
    import datetime,pymsis,ppigrf
    with h5py.File(ROOT/'all_de_inputs.h5') as h:
        t=h['time_unix_s'][:]; r=h['range_km'][:]
    z=np.tile(r,len(t));dates=(np.repeat(t,len(r))*1e6).astype('datetime64[us]')
    n=len(z)
    p=pymsis.calculate(dates,np.full(n,-66.7528),np.full(n,18.3442),z,version=2.1).reshape(len(t),len(r),11)
    b=ppigrf.igrf(-66.7528,18.3442,r,datetime.datetime(2018,5,4,17))
    B=np.sqrt(sum(x*x for x in b))[0]*1e-9; theta=np.rad2deg(np.arccos(abs(b[2][0])*1e-9/B))
    with h5py.File(ROOT/'all_de_priors.h5','w') as h:
        for k,v in [('time_unix_s',t),('range_km',r),('Tn_K',p[:,:,-1]),('N2_m3',p[:,:,1]),('O2_m3',p[:,:,2]),('O_m3',np.nan_to_num(p[:,:,3])),('B_T',B),('angle_k_B_deg',theta)]:h[k]=v
        h.attrs['model']='MSIS 2.1 and IGRF14 at Arecibo; missing low-altitude atomic O set to zero'

def init_worker():
    global INPUT,PRI,SHORT,GAIN
    INPUT=h5py.File(ROOT/'all_de_inputs.h5')
    PRI=h5py.File(ROOT/'all_de_priors.h5')
    SHORT=ShortPulseModel(INPUT['csp/lags_us'][:])
    with h5py.File(ROOT/'density_calibration.h5') as h:GAIN=float(h['gain'][()])

def fit_cell(ij):
    i,j=ij;r=float(INPUT['range_km'][j]);nt=len(INPUT['time_unix_s'])
    mode='csp' if r<94.5 else 'clp'
    y=INPUT[mode+'/mean'][i,j]
    # Local differences estimate noise without treating one record as exact.
    a=INPUT[mode+'/mean'][max(0,i-2):min(nt,i+3),j]
    var=np.mean(abs(np.diff(a,axis=0))**2,axis=0)/4.
    noise=np.median(abs(INPUT[mode+'/mean'][max(0,i-1):min(nt,i+2),:160].imag),axis=(0,1))/0.67449
    var=np.maximum(var,noise**2)
    tn=float(PRI['Tn_K'][i,j]);B=float(PRI['B_T'][j]);theta=float(PRI['angle_k_B_deg'][j])
    ns=tuple(float(PRI[k][i,j]) for k in ['N2_m3','O2_m3','O_m3'])
    scale=max(float(y[0].real),float(np.sqrt(var[0])),1e-9)
    err=np.maximum(np.sqrt(var)/scale,0.02);yn=y/scale
    fun=SHORT if mode=='csp' else clp_model
    def residual(p):
        amp=np.exp(p[0]);ti=tn*np.exp(p[1]);te=tn*np.exp(p[2]);v=p[3]
        ne=max(amp*scale*r*r*GAIN,1.)
        m=amp*fun(ti,te,v,ne,B,theta,*ns)
        d=(m-yn)/err
        return np.r_[d.real,d.imag,p[1]/.25,p[2]/np.log(3.)]
    lo=[-14,np.log(.5),np.log(.5),-300.];hi=[7,np.log(3.),np.log(15.),300.]
    fit=least_squares(residual,[np.log(2),0.,.05,0.],bounds=(lo,hi),max_nfev=50,x_scale='jac')
    amp,ti,te,v=np.exp(fit.x[0])*scale,tn*np.exp(fit.x[1]),tn*np.exp(fit.x[2]),fit.x[3]
    ne=max(amp*r*r*GAIN,1.)
    model=amp*fun(ti,te,v,ne,B,theta,*ns)
    cov=np.linalg.pinv(fit.jac.T@fit.jac)
    sigma=np.sqrt(np.maximum(np.diag(cov),0))*np.array([amp,ti,te,1.])
    chi=np.sum(abs((model-y)/(err*scale))**2)/max(1,2*len(y)-4)
    return i,j,np.array([amp*r*r*GAIN,te,ti,v]),np.array([sigma[0]*r*r*GAIN,sigma[2],sigma[1],sigma[3]]),chi,fit.success,fit.nfev,model

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--workers',type=int,default=8);ap.add_argument('--preview',action='store_true');args=ap.parse_args()
    if args.prepare:prepare();return
    with h5py.File(ROOT/'all_de_inputs.h5') as h:
        t=h['time_unix_s'][:];r=h['range_km'][:];edges=h['time_edges_unix_s'][:];nl=len(h['csp/lags_us'])
    js=np.where((r>=50)&(r<=150))[0]
    todo=[(i,int(j)) for i in range(len(t)) for j in js]
    if args.preview:todo=[(100,int(np.argmin(abs(r-z)))) for z in [50,60,70,80,85,90,94,95,99,100,118,130,140]]
    name='all_de_preview.h5' if args.preview else 'all_de_fits.h5'
    with h5py.File(ROOT/name,'w') as h:
        h['time_unix_s']=t;h['time_edges_unix_s']=edges;h['height_km']=r[js]
        shape=(len(t),len(js));p=h.create_dataset('parameters',shape+(4,),dtype='f8',fillvalue=np.nan)
        se=h.create_dataset('formal_sigma',shape+(4,),dtype='f8',fillvalue=np.nan)
        chi=h.create_dataset('reduced_chi2',shape,dtype='f8',fillvalue=np.nan)
        ok=h.create_dataset('fit_success',shape,dtype='bool');nf=h.create_dataset('nfev',shape,dtype='i4')
        m=h.create_dataset('model_acf',shape+(max(nl,43),),dtype='complex128',fillvalue=np.nan+0j,compression='gzip')
        h.attrs['parameter_order']='ne_m3, Te_K, Ti_K, apparent_velocity_m_s'
        h.attrs['composition']='NO+ 30 u, no negative ions or charged dust; low-altitude temperatures are conditional and may be prior dominated'
        h.attrs['sampling']='every D/E gate 50..150 km; native ~80 s cycles; no SNR threshold or quality-mask deletion'
        h.attrs['source_modes']='CSP below 94.5 km, CLP at or above 94.5 km; equal lag-bin weights; receiver sign not independently verified'
        h.attrs['provenance']='fit_eregion_full.py; original all-5000-gate extraction on Seid /tmp/arecibo_allgates_20261029.h5'
        with mp.Pool(args.workers,initializer=init_worker) as pool:
            for n,(i,j,params,sigma,c,success,nfev,model) in enumerate(pool.imap_unordered(fit_cell,todo,chunksize=1)):
                k=j-js[0];p[i,k]=params;se[i,k]=sigma;chi[i,k]=c;ok[i,k]=success;nf[i,k]=nfev;m[i,k,:len(model)]=model
                if args.preview or n%200==0:
                    h.flush();print(n+1,len(todo),i,r[j],params,c,nfev,flush=True)
        h.attrs['completed']=True

if __name__=='__main__':main()
