"""Native CSP density amplitude with separately pooled CLP spectral shape.

Do not smooth density, its measured short-lag amplitude, or the displayed map.
Shape pooling is explicit: 3 cycles x 5 gates, after normalizing each CLP ACF
by its own cycle's seven-record CSP short-lag power. The physical response
and independent plasma-line calibration are kept separate from display.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import argparse,time,hashlib,multiprocessing as mp
import h5py,numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from full_altitude_model import FullModel
from fit_eregion import response,neutral_rates
from scipy.constants import m_e,atomic_mass

ROOT=Path(__file__).resolve().parent
D={}

def prepare():
    packed=ROOT/'data/density_analysis_inputs.h5'
    if packed.exists():
        with h5py.File(packed) as h:
            D.update({k:h['inputs/'+k][()] for k in h['inputs']})
        return
    with h5py.File(ROOT/'data/eregion_fit/all_de_inputs.h5') as h:
        z=h['range_km'][:];keep=(z>=94.5)&(z<=140.1)
        D['z']=z[keep];D['t']=h['time_unix_s'][:]
        D['edges']=h['time_edges_unix_s'][:]
        D['clp']=h['clp/mean'][:,keep,:]
        D['power']=h['csp/mean'][:,keep,0].real
        D['csp_imag']=h['csp/mean'][:,keep,0].imag
        D['power_var']=h['csp/within_cycle_component_variance'][:,keep,0]/(h['csp/count'][:,0][:,None]-1)
        D['lags']=h['csp/lags_us'][:]
        assert np.all(D['power']>0)
    with h5py.File(ROOT/'data/eregion_fit/full_altitude_fits.h5') as h:
        zz=h['height_km'][:];keep=(zz>=94.5)&(zz<=140.1)
        D['old']=h['results'][:,keep,:]
        # Earlier compact extraction labelled cycles with CSP start-time means.
        # Use the corrected complete-cycle centers/edges from the native fit
        # archive, and verify that the actual CSP integrations still align.
        with h5py.File(ROOT/'data/eregion_fit/all_de_inputs.h5') as raw:
            np.testing.assert_allclose(h['csp_native_time_unix_s'][:],raw['csp/record_mean_time_unix_s'][:],rtol=0,atol=1e-5)
            np.testing.assert_allclose(h['clp_native_time_unix_s'][:],raw['clp/record_time_unix_s'][:],rtol=0,atol=1e-5)
        D['t']=h['time_unix_s'][:];D['edges']=h['time_edges_unix_s'][:]
        D['ti0']=h['Ti_prior_K'][:,keep];D['B']=h['B_T'][keep];D['theta']=h['angle_k_B_deg'][keep]
        D['gain']=h['density_gain'][()]
    tt,zz=np.meshgrid(D['t'],D['z'],indexing='ij')
    with h5py.File(ROOT/'data/eregion_fit/full_altitude_priors.h5') as h:
        for k in ['N2_m3','O2_m3','O_m3']:
            D[k]=RegularGridInterpolator((h['time_unix_s'][:],h['msis/height_km'][:]),h['msis/'+k][:],bounds_error=False,fill_value=None)(np.c_[tt.ravel(),zz.ravel()]).reshape(tt.shape)

def init():
    global MODEL
    MODEL=FullModel('clp')

def model(ti,te,v,ne,i,j):
    return MODEL(ti,te,v,ne,D['B'][j],D['theta'][j],[D[k][i,j] for k in ['N2_m3','O2_m3','O_m3']],np.array([0,0,0,1,0,0]))

def shape(task):
    i,j=task
    ht,hr=D['time_window']//2,D['range_window']//2
    sl=(slice(max(0,i-ht),min(len(D['t']),i+ht+1)),slice(max(0,j-hr),min(len(D['z']),j+hr+1)))
    normalized=D['clp'][sl]/D['power'][sl][...,None]
    a=normalized.reshape(-1,43);y=a.mean(axis=0)
    # Empirical weights; correlation makes formal errors conditional. A 2%
    # floor avoids treating a quiet pooled lag as an exact constraint.
    err=np.maximum(np.sqrt(np.mean(abs(a-y)**2,axis=0)/(2*len(a))),.02)
    tn=float(D['ti0'][i,j]);ne=float(2*D['power'][i,j]*D['gain']*D['z'][j]**2)
    def residual(p):
        ac=p[0]*model(tn*np.exp(p[1]),tn*np.exp(p[2]),100*p[3],ne,i,j)
        r=(ac-y)/err
        return np.r_[r.real,r.imag,p[1]/.25,p[2]/np.log(3),p[3]/3]
    fit=least_squares(residual,[2,0,0,0],bounds=([.05,np.log(.3),np.log(.3),-10],[10,np.log(5),np.log(8),10]),max_nfev=80,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    p=fit.x
    return i,j,[tn*np.exp(p[1]),tn*np.exp(p[2]),100*p[3],p[0],np.mean(residual(p)[:-3]**2),float(fit.success)]

def density_row(i):
    out=[]
    for j,z in enumerate(D['z']):
        ti,te,v=D['physical_shape'][i,j]
        measured=D['power'][i,j]*D['gain']*z*z
        ne=2*measured
        for _ in range(4):
            response=float((model(ti,te,v,ne,i,j)[:4]@np.array([6,10,10,10])/36).real)
            updated=measured/response
            if abs(updated/ne-1)<1e-5:ne=updated;break
            ne=updated
        out.append([ne,response])
    return i,np.array(out)

def calibrate():
    """Re-estimate the gain from ordinary PLs, with alternating-time validation."""
    if (ROOT/'data/density_analysis_inputs.h5').exists():
        with h5py.File(ROOT/'data/density_analysis_inputs.h5') as h:
            points=h['calibration/points'][:];edges=h['calibration/time_edges_unix_s'][:]
    else:
        with h5py.File(ROOT/'data/eregion_fit/density_calibration.h5') as h:points=h['points'][:]
        with h5py.File(ROOT/'data/eregion_fit/compact_inputs.h5') as h:edges=h['acf/time_edges_unix_s'][:]
    records=[]
    for row in points:
        b=int(row[0]);j=int(np.argmin(abs(D['z']-row[3])))
        use=(D['t']>=edges[b])&(D['t']<edges[b+1]);ids=np.flatnonzero(use)
        ti,te,v=D['physical_shape'][use,j].mean(axis=0)
        ns=[np.mean(D[k][use,j]) for k in ['N2_m3','O2_m3','O_m3']]
        nue,nui=neutral_rates(te,*ns);omega=np.array([2*np.pi*.5*(row[4]+row[5])*1e6])
        ce,_=response(omega,1.,te,m_e,nue,D['B'][j],D['theta'][j])
        ci,_=response(omega,1.,ti,30*atomic_mass,nui)
        ne=-1/(ce[0]+ci[0]).real
        r=np.mean([(model(*D['physical_shape'][i,j],ne,i,j)[:4]@np.array([6,10,10,10])/36).real for i in ids])
        gain=ne*r/(np.mean(D['power'][use,j])*D['z'][j]**2)
        records.append([b,row[2],D['z'][j],ne,gain,ti,te])
    records=np.array(records);logs=np.log(records[:,4]);med=np.median(logs);mad=1.4826*np.median(abs(logs-med));keep=abs(logs-med)<3*mad
    train=keep&(records[:,0].astype(int)%2==0);test=keep&~train
    train_gain=np.exp(np.median(logs[train]));ratios=train_gain/records[test,4]
    final_gain=np.exp(np.median(logs[keep]));D['gain']=final_gain
    print('PL calibration',final_gain,'held-out median ratio',np.median(ratios),'log scatter',np.std(np.log(ratios)),flush=True)
    return records,keep,test,ratios

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--time-window',type=int,default=3);p.add_argument('--range-window',type=int,default=5);p.add_argument('--output',default='density_native_csp.h5');a=p.parse_args()
    assert a.time_window>0 and a.range_window>0 and a.time_window%2 and a.range_window%2
    prepare();D['time_window']=a.time_window;D['range_window']=a.range_window
    it=np.unique(np.r_[np.arange(0,len(D['t']),a.time_window),len(D['t'])-1]);iz=np.unique(np.r_[np.arange(0,len(D['z']),a.range_window),len(D['z'])-1])
    shapes=np.zeros((len(it),len(iz),6));start=time.monotonic()
    ctx=mp.get_context('fork')
    with ctx.Pool(a.workers,initializer=init) as pool:
        for n,(i,j,value) in enumerate(pool.imap_unordered(shape,[(i,j) for i in it for j in iz])):
            shapes[np.searchsorted(it,i),np.searchsorted(iz,j)]=value
            if n%400==0:print('shape',n,'/',len(it)*len(iz),'seconds',time.monotonic()-start,flush=True)
    tt,zz=np.meshgrid(D['t'],D['z'],indexing='ij');coords=np.c_[tt.ravel(),zz.ravel()]
    physical=shapes[:,:,:3].copy();physical[:,:,:2]=np.log(physical[:,:,:2])
    D['physical_shape']=RegularGridInterpolator((D['t'][it],D['z'][iz]),physical)(coords).reshape(tt.shape+(3,));D['physical_shape'][:,:,:2]=np.exp(D['physical_shape'][:,:,:2])
    init();cal,accepted,heldout,ratios=calibrate()
    result=np.zeros(tt.shape+(2,))
    with ctx.Pool(a.workers,initializer=init) as pool:
        for i,row in pool.imap_unordered(density_row,range(len(D['t']))):
            result[i]=row
            if i%25==0:print('density row',i,'seconds',time.monotonic()-start,flush=True)
    (ROOT/'data/eregion_fit').mkdir(exist_ok=True,parents=True)
    with h5py.File(ROOT/'data/eregion_fit'/a.output,'w') as h:
        for k in ['t','z','edges','power','power_var','csp_imag','physical_shape','B','theta']:h[k]=D[k]
        h['ne_m3']=result[:,:,0];h['short_lag_response']=result[:,:,1];h['old_ne_m3']=D['old'][:,:,0]
        h['shape/time_unix_s']=D['t'][it];h['shape/height_km']=D['z'][iz];h['shape/parameters']=shapes
        h.attrs['shape_columns']='Ti_K,Te_K,v_m_s,relative_amplitude,normalized_residual_variance,optimizer_success'
        h['calibration/points']=cal;h['calibration/accepted']=accepted;h['calibration/heldout']=heldout;h['calibration/heldout_density_ratio']=ratios
        h['calibration/points'].attrs['columns']='calibration_time_bin,time_unix_s,height_km,ne_from_pl_m3,gain,Ti_K,Te_K'
        h.attrs['density_gain']=D['gain'];h.attrs['method']='Native seven-CSP short-lag amplitude; shape from amplitude-normalized CLP in explicitly saved neighborhoods; no density averaging.'
        h.attrs['calibration_log_scatter']=float(np.std(np.log(cal[accepted,4]),ddof=1))
        h.attrs['shape_time_cycles']=a.time_window;h.attrs['shape_range_gates']=a.range_window
        h['t'].attrs['units']='Unix seconds, complete acquisition-cycle center'
        h['z'].attrs['units']='km above radar'
        h['ne_m3'].attrs['units']='m^-3'
        h['power'].attrs['sampling']='mean of seven short-pulse integrations, archived 4-39 us lag bin'
        for path in [ROOT/'data/density_analysis_inputs.h5',ROOT/'data/eregion_fit/all_de_inputs.h5',ROOT/'data/eregion_fit/full_altitude_fits.h5']:
            if path.exists():h.attrs['input_sha256_'+path.stem]=hashlib.sha256(path.read_bytes()).hexdigest()
        h.attrs['limitations']='Shape resolution is coarser than density sampling; NO+ composition and MSIS Ti prior; shared shape errors are correlated; amplitude-conditional errors are not total density uncertainties.'
        h['provenance/source']=Path(__file__).read_text();h.attrs['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print('complete',time.monotonic()-start,flush=True)

if __name__=='__main__':main()
