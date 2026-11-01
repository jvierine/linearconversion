"""Smooth equilibrium fitted to calibrated ISR density, not sample interpolation.

Curvature regularization is strong in the noisy background and weak through Es.
Widths are apparent widths: the original range ambiguity function is unavailable.
No deconvolution or unobserved sharp edge is introduced. All heights are the
archived vertical radar range coordinate (radar height not added).
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import numpy as np
import h5py
from scipy.optimize import least_squares,lsq_linear
from scipy.interpolate import BSpline

SOURCE = Path(__file__).resolve().parents[1]/'data/eregion_fit/full_altitude_fits.h5'


def profile(height_m, coefficients, knots=None):
    """Return n0 and dn0/dz in SI; optional Gaussian locator decomposition."""
    if knots is not None:
        spline=BSpline(knots,coefficients,3)
        n=1e11*spline(np.asarray(height_m)/1000)
        dn=1e8*spline.derivative()(np.asarray(height_m)/1000)
        return n,dn,None,None
    b0,b1,b2,log_a,center,log_width = coefficients
    z=np.asarray(height_m)/1000
    x=(z-120)/15
    bg=1e11*np.exp(b0+b1*x+b2*x*x)
    width=np.exp(log_width)
    u=(z-center)/width
    es=1e11*np.exp(log_a-.5*u*u)
    derivative=bg*(b1+2*b2*x)/15000-es*u/(1000*width)
    return bg+es,derivative,bg,es


def fit_observed_layer(time_unix_s=1525460596., source=SOURCE):
    with h5py.File(source) as h:
        csp='ne_m3' in h and 'power_var' in h
        times=h['t' if csp else 'time_unix_s'][:]
        i=int(np.argmin(abs(times-time_unix_s)))
        height=h['z' if csp else 'height_km'][:]*1000
        keep=(height>=100000)&(height<=135000)
        height=height[keep]
        if csp:
            ne=h['ne_m3'][i,keep]
            sigma=ne*np.sqrt(h['power_var'][i,keep])/h['power'][i,keep]
            valid=np.isfinite(ne)&(ne>0)&np.isfinite(sigma)&(sigma>0)
            gain=float(h.attrs['density_gain']);scatter=float(h.attrs['calibration_log_scatter'])
            field=float(np.median(h['B'][:][keep]));angle=float(np.median(h['theta'][:][keep]))
        else:
            values=h['results'][i,keep];ne=values[:,0];sigma=values[:,4]
            valid=(values[:,9]>0)&np.isfinite(ne)&(ne>0)&np.isfinite(sigma)&(sigma>0)
            gain=float(h['density_gain'][()]); scatter=float(h['density_gain_log_scatter'][()])
            field=float(np.median(h['B_T'][:][keep]));angle=float(np.median(h['angle_k_B_deg'][:][keep]))
        # Prevent a few extremely small formal errors determining the entire fit.
        sigma=np.maximum(sigma,.05*ne)
    candidates=np.where((height>110000)&(height<125000)&valid)[0]
    peak=candidates[np.argmax(ne[candidates])]
    initial=[np.log(np.median(ne[valid])/1e11),.2,0.,np.log(ne[peak]/1e11),height[peak]/1000,np.log(.5)]
    fit=least_squares(lambda a:(profile(height[valid],a)[0]-ne[valid])/sigma[valid],
        initial,bounds=([-4,-3,-3,-5,110,np.log(.15)],[3,3,3,4,125,np.log(2.)]),
        loss='soft_l1',f_scale=1.5,xtol=1e-12,ftol=1e-12,gtol=1e-12)
    if not fit.success: raise RuntimeError(f'Density fit failed: {fit.message}')
    # The Gaussian only locates the layer and its extent. The final equilibrium
    # is a cubic smoothing spline, preserving the observed Es shape, including
    # asymmetry. Strong curvature suppression applies above and below the layer;
    # weaker suppression through Es preserves its measured peak and flanks.
    zz=height/1000; center=fit.x[4]; width=np.exp(fit.x[5])
    knots=np.r_[np.repeat(zz[0],4),np.arange(zz[0]+.3,zz[-1]-.01,.3),np.repeat(zz[-1],4)]
    basis=BSpline(knots,np.eye(len(knots)-4),3)
    design=basis(zz[valid]); yy=ne[valid]/1e11; sig=sigma[valid]/1e11
    grid=np.linspace(zz[0],zz[-1],1401)
    outer=.5*(1+np.tanh((np.abs(grid-center)-2.5*width)/.35))
    length=.12+1.08*outer
    if csp:
        # Retain the measured narrow peak while smoothing only its background.
        # The transition must lie beyond the resolved flanks, not inside them.
        outer=.5*(1+np.tanh((np.abs(grid-center)-max(3.5*width,1.5))/.25))
        length=.06+1.14*outer
    penalty=basis.derivative(2)(grid)*(length**2*np.sqrt((grid[1]-grid[0])/.3)/np.median(sig))[:,None]
    # Nondecreasing B-spline coefficients guarantee nonnegative derivative.
    # Apply this only below the Es layer: no spurious downward density rise at
    # the lower data boundary. Elsewhere retain the observed peak and flanks.
    greville=np.array([np.mean(knots[j+1:j+4]) for j in range(len(knots)-4)])
    monotone_count=int(np.sum(greville<center-2*width))
    transform=np.eye(len(greville))
    transform[:monotone_count,:monotone_count]=np.tril(np.ones((monotone_count,monotone_count)))
    weights=np.ones(len(yy))
    for _ in range(8):
        w=np.sqrt(weights)/sig
        system=np.vstack([design*w[:,None],penalty])@transform
        result=lsq_linear(system,np.r_[yy*w,np.zeros(len(grid))],bounds=(0,np.inf),method='bvls',tol=1e-10)
        if not result.success:raise RuntimeError('Constrained density smoothing did not converge')
        coefficients=transform@result.x
        residual=(design@coefficients-yy)/sig
        weights=1/np.sqrt(1+(residual/1.5)**2)
        if csp:
            # A resolved, repeatable high-SNR Es peak is signal, not an outlier
            # against the much smoother background. Keep its data weights.
            weights[np.abs(zz[valid]-center)<2.5*width]=1.
    dense=np.linspace(height[0],height[-1],7001)
    n,dn,_,_=profile(dense,coefficients,knots)
    if np.any(n<=0):raise ValueError('Smooth density is not positive')
    bg=profile(dense,fit.x)[2];es=n-bg
    with open(source,'rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(coefficients=coefficients,knots_km=knots,source_height_m=height,source_ne_m3=ne,
        source_sigma_m3=sigma,source_valid=valid,model_height_m=dense,
        model_ne_m3=n,model_dne_dz_m4=dn,background_ne_m3=bg,es_ne_m3=es,
        time_unix_s=float(times[i]),time=datetime.fromtimestamp(times[i],timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        source_time_index=i,calibration_gain=gain,calibration_log_scatter=scatter,
        magnetic_t=field,field_angle_deg=angle,es_center_m=float(fit.x[4]*1000),
        es_sigma_m=float(np.exp(fit.x[5])*1000),peak_ne_m3=float(n.max()),
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        source_path=str(source),source_sha256=digest,
        model='C2 cubic smoothing spline; monotonically increasing lower background, strong background smoothing, observed Es peak and flanks preserved',
        limitation='300 m sampling is not independent resolution; fitted width is apparent; unknown range PSF is not deconvolved; common calibration scatter is separate')


def scattering_profile(height_m, coefficients, knots=None, taper_m=400.):
    """Fit in the interior; C2 flat continuation into boundary reservoirs.

    Taper the coordinate derivative only in the outer buffers, away from Es.
    The physical density fit itself is retained separately for inspection.
    """
    height=np.asarray(height_m)
    lo,hi=height[0],height[-1]
    mapped=height.copy(); derivative=np.ones_like(height)
    # Integral of the quintic smoothstep, matched to the identity at u=1.
    for bottom in (True,False):
        distance=height-lo if bottom else hi-height
        mask=distance<taper_m; u=distance[mask]/taper_m
        integral=2.5*u**4-3*u**5+u**6
        offset=taper_m*(.5+integral)
        mapped[mask]=lo+offset if bottom else hi-offset
        derivative[mask]=10*u**3-15*u**4+6*u**5
    n,dn,_,_=profile(mapped,coefficients,knots)
    return n,dn*derivative


def save_profile(fit,path):
    with h5py.File(path,'w') as h:
        for key,value in fit.items():
            if isinstance(value,np.ndarray):h.create_dataset(key,data=value)
            else:h.attrs[key]=value
        h.attrs['coefficient_order']='cubic B-spline coefficients, density in 1e11 m^-3; knots_km supplied; Gaussian locator is not the final density model'
        h.attrs['provenance']='isr/observed_layer.py; '+fit['source_path']
