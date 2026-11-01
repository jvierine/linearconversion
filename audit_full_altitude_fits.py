"""Inspect the completed native-resolution fit and retain representative ACFs.

Run next to the exact input product on Revontuli. No measured data are changed.
Scientific metadata retain source filenames; plotted artifacts omit them.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import h5py
import fit_full_altitude as fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    with h5py.File(root / 'full_altitude_fits.h5') as h:
        assert h.attrs.get('completed', False), 'fit is not complete'
        a = h['results'][:]
        z = h['height_km'][:]
        t = h['time_unix_s'][:]
        gain = h['density_gain'][()]
    assert np.isfinite(a[:, :, :4]).all(), 'unfitted parameter cells'
    report = dict(shape=list(a.shape), n_cells=int(a.shape[0]*a.shape[1]),
                  altitude_km=[float(z[0]), float(z[-1])],
                  median_cadence_s=float(np.median(np.diff(t))),
                  success_fraction=float(a[:,:,9].mean()),
                  failed=int(np.count_nonzero(a[:,:,9] == 0)),
                  at_bound_fraction=float(a[:,:,14].mean()))
    bands = []
    for lo, hi in [(35,80),(80,95),(95,110),(110,140),(140,180),
                   (180,300),(300,500),(500,800),(800,1100),(1100,1500)]:
        b = a[:,(z>=lo)&(z<hi)]
        row = dict(altitude_km=[lo,hi],
                   median_parameters=np.median(b[:,:,:4],axis=(0,1)).tolist(),
                   chi2_quantiles=np.quantile(b[:,:,8],[.1,.5,.9,.99]).tolist(),
                   failed=int(np.count_nonzero(b[:,:,9] == 0)),
                   fraction_information_over_half=(b[:,:,11:14]>.5).mean(axis=(0,1)).tolist(),
                   median_density_relative_error=float(np.median(b[:,:,4]/np.maximum(b[:,:,0],1))),
                   median_first_lag_snr=float(np.median(b[:,:,15])))
        bands.append(row)
    report['altitude_bands']=bands
    print(json.dumps(report, indent=2), flush=True)
    fit.prepare(root)
    fit.init()
    # Four times and twelve heights, including weak-signal and sporadic layers.
    times = [0, 65, 130, 200]
    heights = [50.1, 80.1, 90., 99., 117.9, 130., 180., 250., 400., 700., 1000., 1450.]
    with h5py.File(root / 'full_altitude_fit_audit.h5', 'w') as out:
        out.attrs['summary_json'] = json.dumps(report)
        out.attrs['provenance'] = 'audit_full_altitude_fits.py; full_altitude_fits.h5; all_altitude_inputs.h5'
        out.attrs['residual_definition'] = '(observed-model) / empirical component sigma; model floor 2 percent of local ACF scale'
        out.attrs['selection'] = 'Prespecified times and heights; no selection by fit quality'
        for i in times:
            for height in heights:
                j = int(np.argmin(abs(z-height)))
                rawj = int(np.argmin(abs(fit.D['z']-z[j])))
                ne,te,ti,v = a[i,j,:4]
                mode = 'csp' if z[j]<94.5 else 'clp'
                obs = fit.D[mode][i,rawj]
                var = fit.D[mode+'_var'][i,rawj]
                scale = max(obs[0].real, np.sqrt(var[0]), 1e-10)
                sigma = np.maximum(np.sqrt(var),.02*scale)
                model = fit.MODELS[mode]
                ns = [fit.D[k][i,rawj] for k in ['N2_m3','O2_m3','O_m3']]
                amp = ne / (z[j]**2*gain)
                pred = amp*model(ti,te,v,max(ne,1e5),fit.D['B'][rawj],
                                fit.D['angle'][rawj],ns,fit.D['fractions'][i,rawj])
                g = out.create_group(f'acf/t{i:03d}_h{j:04d}')
                g['observed'] = obs
                g['model'] = pred
                g['observed_real'] = obs.real
                g['observed_imag'] = obs.imag
                g['model_real'] = pred.real
                g['model_imag'] = pred.imag
                g['sigma_component'] = sigma
                g['lag_us'] = np.array([lag.mean()*1e6 for lag in model.lags])
                g['parameters'] = a[i,j]
                g.attrs['height_km'] = z[j]
                g.attrs['time_unix_s'] = t[i]
                g.attrs['mode'] = mode
        # Compare native results with the independent thermal plasma-line points.
        with h5py.File(root / 'density_calibration.h5') as cal:
            points=cal['points'][:]
            accepted=cal['accepted'][:]
        rows=[]
        for point,keep in zip(points,accepted):
            if not keep: continue
            j=int(np.argmin(abs(z-point[3])))
            ii=np.flatnonzero(abs(t-point[2])<=200.)
            if len(ii)==0: continue
            # Reconstruct density from the same measured frequencies using native
            # fitted temperatures and the same magnetized dielectric convention.
            k=int(ii[np.argmin(abs(t[ii]-point[2]))])
            rawj=int(np.argmin(abs(fit.D['z']-z[j])))
            te=float(np.median(a[ii,j,1]));ti=float(np.median(a[ii,j,2]))
            from fit_eregion import response, neutral_rates
            from scipy.constants import m_e,atomic_mass
            ns=[fit.D[q][k,rawj] for q in ['N2_m3','O2_m3','O_m3']]
            nue,nui=neutral_rates(te,*ns)
            w=np.array([2*np.pi*.5*(point[4]+point[5])*1e6])
            ce,_=response(w,1.,te,m_e,nue,fit.D['B'][rawj],fit.D['angle'][rawj])
            ci,_=response(w,1.,ti,30*atomic_mass,nui)
            ne_pl=-1./(ce[0]+ci[0]).real
            ne_fit=float(np.median(a[ii,j,0]))
            rows.append([point[2],z[j],ne_pl,ne_fit,ne_fit/ne_pl,te,ti,len(ii)])
        out['calibration_check']=np.array(rows)
        out['calibration_check'].attrs['columns']='unix_s,height_km,ne_thermal_pl_m3,median_ne_fit_m3,fit_over_pl,Te_K,Ti_K,n_cycles'
        ratio=np.array(rows)[:,4]
        print('Native-fit / thermal-PL density ratio quantiles:',np.quantile(ratio,[.1,.5,.9]),flush=True)
        out.attrs['calibration_ratio_quantiles_10_50_90']=np.quantile(ratio,[.1,.5,.9])


if __name__ == '__main__':
    main()
