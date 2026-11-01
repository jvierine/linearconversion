"""Physical inversion of the archived polynomial ACF amplitude estimator.

Density and ACF amplitude are not smoothed beyond the original three-gate
average below 79.8 km. Only nuisance spectral-shape parameters are pooled.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import time
import hashlib
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import median_filter, gaussian_filter
from scipy.constants import e, epsilon_0, k, m_e, atomic_mass
from full_altitude_model import FullModel
from fit_eregion import K, response, neutral_rates

ROOT = Path(__file__).resolve().parent
FRACTIONS = [0, 0, 0, 1, 0, 0]
ALPHA = 42859649122.80702


def coefficients(lags, counts, order):
    x = np.log10(np.where(lags == 0, 22, lags))
    w = np.sqrt(counts)
    return np.vander([np.log10(22)], order + 1)[0] @ np.linalg.pinv(np.vander(x, order + 1)*w[:, None])*w


def projected_response(model, coeff, densities, ti, te, v, B, angle, ns):
    """Vectorized equivalent of coeff @ FullModel(...).real for many ne."""
    n = np.asarray(densities)[:, None]
    nue, nui = neutral_rates(te, *ns)
    narrow = nui > 8*K*np.sqrt(k*ti/(30*atomic_mass))
    def spectra(f, subtract_narrow=False):
        w = 2*np.pi*f-K*v
        ce, qe = response(w, 1., te, m_e, nue, B, angle)
        ci, qi = response(w, 1., ti, 30*atomic_mass, nui)
        ce = n*ce; ci = n*ci
        den = 1+ce+ci
        s = abs(ce/den)**2*qi+abs((1+ci)/den)**2*qe
        if subtract_narrow and narrow:
            ae = n*e**2/(epsilon_0*k*te*K*K)
            ai = n*e**2/(epsilon_0*k*ti*K*K)
            d = K*K*k*ti/(30*atomic_mass*nui)
            gamma = d*(1+ai/(1+ae))
            weight = (ae/(1+ae))**2*d/gamma
            s -= weight*2*gamma/(w*w+gamma*gamma)
        return s
    out = spectra(model.f, True) @ (coeff @ model.kernel).real
    out += coeff[0]*(spectra(model.tail) @ model.tail_kernel.real)
    if narrow:
        ae = densities*e**2/(epsilon_0*k*te*K*K)
        ai = densities*e**2/(epsilon_0*k*ti*K*K)
        d = K*K*k*ti/(30*atomic_mass*nui)
        gamma = d*(1+ai/(1+ae))
        weight = (ae/(1+ae))**2*d/gamma
        for c, lag in zip(coeff, model.lags):
            out += c*weight*np.mean(np.exp(-gamma[:, None]*abs(lag))*np.cos(K*v*lag), axis=1)
    return out


def main():
    with h5py.File(ROOT/'data/density_reference.h5') as h:
        z = h['map/height_km'][:]
        t = h['map/time_utc_hour'][:]*3600+1525392000
        edges = h['map/time_edges_utc_hour'][:]
        apparent = h['map/ne_m3'][:].T
        palette = h['palette/density_rgb'][:]
    with h5py.File(ROOT/'data/eregion_fit/all_de_inputs.h5') as h:
        lags = h['csp/lags_us'][:]
        counts = h['csp/count'][:]
        assert np.all(counts == counts[0])
    coeffs = np.array([coefficients(lags, counts[0], q) for q in [1, 2]])
    with h5py.File(ROOT/'data/eregion_fit/all_de_priors.h5') as h:
        inds = np.array([np.argmin(abs(h['range_km'][:]-v)) for v in z])
        tn = h['Tn_K'][:, inds]
        neutrals = np.stack([h[q][:, inds] for q in ['N2_m3', 'O2_m3', 'O_m3']], axis=-1)
        B = h['B_T'][:][inds]; angle = h['angle_k_B_deg'][:][inds]
    with h5py.File(ROOT/'data/eregion_fit/density_native_csp.h5') as h:
        shape = h['physical_shape'][:]
        zs = h['z'][:]
        cal = h['calibration/points'][:]
        accepted = h['calibration/accepted'][:].astype(bool)
    # Smooth only separately estimated temperature/drift nuisance parameters.
    pooled = np.empty_like(shape)
    for q in range(3):
        a = np.log(shape[:, :, q]) if q < 2 else shape[:, :, q]
        a = gaussian_filter(median_filter(a, size=(5, 5), mode='nearest'), sigma=(5, 5), mode='nearest')
        pooled[:, :, q] = np.exp(a) if q < 2 else a
    conditions = np.stack([tn, tn, np.zeros_like(tn)], axis=-1)
    blend = np.clip((z-90)/10, 0, 1)
    for i in range(len(t)):
        for q in range(3):
            a = np.interp(z, zs, pooled[i, :, q])
            conditions[i, :, q] = (1-blend)*conditions[i, :, q]+blend*a
    ni = np.unique(np.r_[np.arange(0, len(t), 4), len(t)-1])
    old_nj = np.unique(np.r_[np.arange(0, len(z), 3), len(z)-1])
    # Resolve the two breakpoints in the independently prescribed shape blend.
    nj = np.unique(np.r_[old_nj, np.flatnonzero(((z>=89.4)&(z<=90.6))|((z>=99.3)&(z<=100.8)))])
    # Start one decade below the displayed density floor. At far lower density
    # the near-zero projected signal is sensitive to numerical cancellation.
    # Such samples are flagged unresolved, not extrapolated into detections.
    logn = np.arange(7., 14.0001, .05); densities = 10**logn
    model = FullModel('csp', lags)
    cache = ROOT/'output/density_render_audit/debye_operator_lookup.h5'
    import inspect
    key = hashlib.sha256(conditions.tobytes()+neutrals.tobytes()+logn.tobytes()+coeffs.tobytes()+ni.tobytes()+nj.tobytes()+inspect.getsource(projected_response).encode()).hexdigest()
    reuse = False
    if cache.exists():
        with h5py.File(cache) as h:
            if h.attrs.get('key') == key:
                curves = h['curves'][:]; collective = h['collective'][:]; reuse = True
    if not reuse:
        curves = np.empty((2, len(ni), len(nj), len(logn)))
        collective = np.empty((2, len(ni), len(nj)))
        available=set()
        old_key=hashlib.sha256(conditions.tobytes()+neutrals.tobytes()+logn.tobytes()+coeffs.tobytes()+ni.tobytes()+old_nj.tobytes()+inspect.getsource(projected_response).encode()).hexdigest()
        if cache.exists():
            with h5py.File(cache) as h:
                if h.attrs.get('key')==old_key:
                    old_curves=h['curves'][:];old_collective=h['collective'][:]
                    for b,j in enumerate(nj):
                        if j in old_nj:
                            bb=int(np.flatnonzero(old_nj==j)[0]);curves[:,:,b]=old_curves[:,:,bb];collective[:,:,b]=old_collective[:,:,bb];available.add(b)
        start = time.monotonic()
        from concurrent.futures import ThreadPoolExecutor
        def node(job):
            a,b=job;i,j=ni[a],nj[b]
            r=np.array([projected_response(model, coeff, np.r_[densities,1e18], *conditions[i,j], B[j], angle[j], neutrals[i,j]) for coeff in coeffs])
            return a,b,r
        with ThreadPoolExecutor(max_workers=4) as pool:
            for a,b,r in pool.map(node,((a,b) for a in range(len(ni)) for b in range(len(nj)) if b not in available)):
                curves[:,a,b]=r[:,:-1];collective[:,a,b]=r[:,-1]
                if b==max(set(range(len(nj)))-available):print('lookup',a+1,'/',len(ni),'elapsed seconds',round(time.monotonic()-start),flush=True)
        with h5py.File(cache,'w') as h:
            h['curves']=curves;h['collective']=collective;h.attrs['key']=key
    assert np.all(np.isfinite(curves))
    assert np.all(curves > 0), (curves.min(), 'estimator response not positive')
    grids=[RegularGridInterpolator((t[ni],z[nj],logn),np.log(curves[q]),bounds_error=True) for q in range(2)]
    limgrids=[RegularGridInterpolator((t[ni],z[nj]),np.log(collective[q])) for q in range(2)]
    # Same ordinary-PL calibration records; recompute gain for this estimator.
    with h5py.File(ROOT/'data/eregion_fit/compact_inputs.h5') as h:
        caledges=h['acf/time_edges_unix_s'][:]
    with h5py.File(ROOT/'data/eregion_fit/density_calibration.h5') as h:raw_cal=h['points'][:]
    np.testing.assert_array_equal(raw_cal[:,0],cal[:,0])
    np.testing.assert_allclose(raw_cal[:,3],cal[:,2])
    calrecords=[]
    for row,raw in zip(cal,raw_cal):
        b=int(row[0]);j=int(np.argmin(abs(z-row[2])));ids=np.flatnonzero((t>=caledges[b])&(t<caledges[b+1]))
        ti,te,_=conditions[ids,j].mean(axis=0);nue,nui=neutral_rates(te,*neutrals[ids,j].mean(axis=0))
        w=np.array([2*np.pi*.5*(raw[4]+raw[5])*1e6])
        ce,_=response(w,1.,te,m_e,nue,B[j],angle[j]);ci,_=response(w,1.,ti,30*atomic_mass,nui)
        npl=float(-1/(ce[0]+ci[0]).real)
        r=np.exp(grids[1](np.c_[t[ids],np.full(len(ids),z[j]),np.full(len(ids),np.log10(npl))]))
        measured=np.mean(apparent[ids,j]/ALPHA)
        calrecords.append([row[0],row[1],z[j],npl,npl*np.mean(r)/measured])
    calrecords=np.array(calrecords)
    train=accepted&(cal[:,0].astype(int)%2==0);heldout=accepted&~train
    gain=np.exp(np.median(np.log(calrecords[accepted,4])))
    train_gain=np.exp(np.median(np.log(calrecords[train,4])))
    prediction=[]
    for row in calrecords[heldout]:
        j=int(np.argmin(abs(z-row[2])));ids=np.flatnonzero((t>=caledges[int(row[0])])&(t<caledges[int(row[0])+1]))
        coords=np.array(np.meshgrid(t[ids], [z[j]], logn,indexing='ij')).reshape(3,-1).T
        r=np.exp(grids[1](coords)).reshape(len(ids),len(logn)).mean(axis=0)
        nfit=10**np.interp(np.log(train_gain*np.mean(apparent[ids,j]/ALPHA)),np.log(densities*r),logn)
        prediction.append(nfit/row[3])
    ne=np.full_like(apparent,np.nan);limit=np.full_like(apparent,np.nan);valid=apparent>0
    conditioning=np.full_like(apparent,np.nan)
    for i in range(len(t)):
        for j in range(len(z)):
            q=0 if z[j]<=79.8+1e-6 else 1
            r=np.exp(grids[q](np.c_[np.full(len(logn),t[i]),np.full(len(logn),z[j]),logn]))
            power=densities*r
            assert np.all(np.diff(power)>0), (i,j,'non-monotone inversion')
            measured=gain*apparent[i,j]/ALPHA
            if measured>power[0] and measured<power[-1]:
                ne[i,j]=10**np.interp(np.log(measured),np.log(power),logn)
                limit[i,j]=measured/np.exp(limgrids[q]([[t[i],z[j]]]))[0]
                conditioning[i,j]=np.interp(np.log10(ne[i,j]),logn,np.gradient(np.log(power),np.log(densities)))
            else:valid[i,j]=False
    # Full, independently evaluated operator comparisons, including grid refinement.
    fine=FullModel('csp',lags,df=15.625,tail_step=250.,tail_max=4e6)
    checks=[]
    for i in [13,77,167]:
        for target in [60.,70.,79.2,80.4,90.,98.,118.,130.]:
            j=int(np.argmin(abs(z-target)));q=0 if z[j]<=79.8+1e-6 else 1
            n=ne[i,j] if valid[i,j] else 1e8
            args=(*conditions[i,j],n,B[j],angle[j],neutrals[i,j],FRACTIONS)
            exact=float(coeffs[q]@model(*args).real)
            vector=projected_response(model,coeffs[q],np.array([n]),*conditions[i,j],B[j],angle[j],neutrals[i,j])[0]
            refined=float(coeffs[q]@fine(*args).real)
            lookup=np.exp(grids[q]([[t[i],z[j],np.log10(n)]]))[0]
            checks.append([t[i],z[j],n,exact,vector,refined,lookup,abs(vector/exact-1),abs(refined/exact-1),abs(lookup/exact-1)])
    checks=np.array(checks)
    print('checks max vector, refinement, interpolation:',checks[:,-3:].max(axis=0),flush=True)
    with h5py.File(ROOT/'output/density_render_audit/operator_checks.h5','w') as h:h['checks']=checks
    print('lookup checks above0.5percent',checks[checks[:,-1]>.005],flush=True)
    assert checks[:,-3].max()<1e-6
    assert checks[:,-2:].max()<.005
    out=ROOT/'data/density_reference_debye.h5'
    with h5py.File(out,'w') as h:
        h['map/ne_m3']=ne.T;h['map/height_km']=z;h['map/time_utc_hour']=(t-1525392000)/3600;h['map/time_edges_utc_hour']=edges
        h['map/valid']=valid.T;h['map/apparent_ne_m3']=apparent.T;h['map/collective_ne_m3']=limit.T;h['map/log_power_log_density_slope']=conditioning.T
        h['conditions/Ti_Te_velocity']=conditions;h['conditions/neutrals']=neutrals;h['conditions/B_T']=B;h['conditions/angle_k_B_deg']=angle
        h['palette/density_rgb']=palette
        with h5py.File(ROOT/'data/density_reference.h5') as old:h['palette/rgb']=old['palette/rgb'][:]
        h['estimator/coefficients']=coeffs;h['estimator/lags_us']=lags
        h['calibration/points']=calrecords;h['calibration/accepted']=accepted;h['calibration/heldout_density_ratio']=prediction
        h['calibration/points'].attrs['columns']='time_bin,time_unix_s,height_km,ne_from_ordinary_pl_m3,gain'
        h['calibration/heldout']=heldout;h['calibration/training_gain']=train_gain
        h['conditions/Ti_Te_velocity'].attrs['columns']='Ti_K,Te_K,apparent_velocity_m_s'
        h['conditions/neutrals'].attrs['columns']='N2_m3,O2_m3,O_m3'
        h['validation/operator_checks']=checks
        h['validation/operator_checks'].attrs['columns']='time,height,ne,full_model,vectorized,refined,lookup,relative_vector_error,relative_refinement_error,relative_lookup_error'
        h.attrs.update(density_gain=gain,heldout_median_ratio=float(np.median(prediction)),heldout_log_scatter=float(np.std(np.log(prediction))),
          method='Invert original weighted polynomial ACF estimator through full finite-Debye spectrum, receiver and lag response, including electron and ion terms.',
          averaging='Original seven-integration ACF and three-gate averaging below79.8km. No density or display smoothing.',
          shape='MSIS Te=Ti=Tn and zero drift below90km; blend90..100km to independently pooled CLP shape. Shape only: 5x5 median then Gaussian sigma5 cycles/gates.',
          limits='Conditional NO+ plasma. No negative ions, charged dust, heavy cluster/metal ions. Neutral BGK; no Coulomb operator. Nonpositive or out-of-grid estimator values flagged invalid and shown at display floor, not asserted positive detections.',
          modeled_density_range_m3=[densities[0],densities[-1]],provenance=Path(__file__).read_text())
        for name in ['full_altitude_model.py','fit_eregion.py','prepare_density_reference.py']:
            h['provenance_code/'+name]=(ROOT/name).read_text()
    print(out,'gain',gain,'held-out ratio',np.median(prediction),'log scatter',np.std(np.log(prediction)),flush=True)

if __name__=='__main__':main()
