"""Direct-path source compatibility, not a transmitter probability density.

Cold magnetoionic group rays in a horizontally stratified measured profile.
The local conversion response uses the existing warm Maxwell/fluid solver,
extended by rotating B into each incidence plane. Ordinary-mode illumination.
No prior ionospheric hop; no ducting or horizontal density structure. The
coupling weights are per equal incident vertical flux below Es, not equal
ground transmitter power. Geographical assets and provenance are in HDF5.
"""
from pathlib import Path
import sys,datetime,hashlib,time,argparse
import h5py,numpy as np
from scipy.constants import e,m_e,epsilon_0,k as kb
from scipy.ndimage import median_filter,gaussian_filter1d
from scipy.interpolate import PchipInterpolator,LinearNDInterpolator
from scipy.optimize import minimize
from pyproj import Geod
import ppigrf
import coupled_arecibo as a
from minimum_incident_power import metric,TARGET,CAL,TAU,TSYS
from hf_absorption_profile import modes
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from isr.coherent import sideband_amplitude

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reference/coupled/source_location_map.h5'
LAT,LON=18.3442,-66.7528

def dielectric(ne,gyro):
    gx,gy,gz=gyro
    cross=np.array([[0,gz,-gy],[-gz,0,gx],[gy,-gx,0]])
    sus=1j*np.linalg.inv(-1j*np.eye(3)+cross)
    return np.eye(3)+np.asarray(ne)[...,None,None]*e**2/(epsilon_0*m_e*a.W**2)*sus

def ordinary_nz(eps,qx,qy=0.):
    """Berreman eigenproblem; select larger-nz positive-z Poynting branch."""
    eps=np.asarray(eps);shape=eps.shape[:-2]
    ez=np.stack([-eps[...,2,0],-eps[...,2,1],np.zeros(shape)+qy,
                 np.zeros(shape)-qx],axis=-1)/eps[...,2,2,None]
    hz=np.broadcast_to(np.array([-qy,qx,0.,0.]),ez.shape)
    mat=np.zeros(shape+(4,4),complex)
    mat[...,0,:]=qx*ez;mat[...,0,3]+=1
    mat[...,1,:]=qy*ez;mat[...,1,2]-=1
    mat[...,2,:]=qx*hz-eps[...,1,2,None]*ez
    mat[...,2,0]-=eps[...,1,0];mat[...,2,1]-=eps[...,1,1]
    mat[...,3,:]=qy*hz+eps[...,0,2,None]*ez
    mat[...,3,0]+=eps[...,0,0];mat[...,3,1]+=eps[...,0,1]
    val,vec=np.linalg.eig(mat)
    flux=(vec[...,0,:]*vec[...,3,:].conj()-vec[...,1,:]*vec[...,2,:].conj()).real
    valid=(flux>1e-9)&(abs(val.imag)<1e-6)
    chosen=np.argmax(np.where(valid,val.real,-np.inf),axis=-1)
    nz=np.take_along_axis(val,chosen[...,None],axis=-1)[...,0].real
    return np.where(valid.any(axis=-1),nz,np.nan)

def ray(z,ne,theta,phi,B,tilt):
    """Integrate dx/dz=-d(nz)/d(qx), dy/dz=-d(nz)/d(qy).

    The derivative is the group-ray direction, not the wave-normal direction.
    qx,qy are conserved because the profile is horizontally stratified.
    """
    ph=np.deg2rad(phi);ti=np.deg2rad(tilt)
    gyro=e*B/(m_e*a.W)*np.array([np.sin(ti)*np.cos(ph),np.sin(ti)*np.sin(ph),-np.cos(ti)])
    eps=dielectric(ne,gyro);q=np.sin(np.deg2rad(theta));h=1e-5
    nz=ordinary_nz(eps,q)
    dx=-(ordinary_nz(eps,q+h)-ordinary_nz(eps,q-h))/(2*h)
    dy=-(ordinary_nz(eps,q,h)-ordinary_nz(eps,q,-h))/(2*h)
    good=np.isfinite(nz+dx+dy)
    bad=np.flatnonzero(~good);n=int(bad[0]) if len(bad) else len(z)
    if n<2:return np.nan,np.nan,0.,n
    return np.trapezoid(dx[:n],z[:n]),np.trapezoid(dy[:n],z[:n]),z[n-1],n

def coupling(theta,phi,dz=.02,cap=2.):
    cfg={'field_azimuth_deg':phi}
    s=a.setup(dz=dz,cap=cap,angle=theta,scenario=cfg)
    r=a.setup(dz=dz,cap=cap,angle=theta,scenario=cfg,reference=True)
    z=s['z'];i=np.argmin(abs(z+s['dz_cells']/2-115100))
    v,err=modes(s,r,i)['O'];Q=metric(r['initial'],i)
    v/=np.sqrt(np.vdot(v,Q@v).real)
    n=s['initial'][9]@v*np.sqrt(epsilon_0*s['ne'][0]/(m_e*a.CS**2))
    d=z-s['center'];win=np.sin(.5*np.pi*np.clip(np.minimum(d+100,30-d)/10,0,1))**2
    power=abs(sideband_amplitude(z,n,a.F,s['kx'],win,reference_range_m=s['center']))**2
    flux=TARGET*kb*TSYS/(CAL*TAU*power)
    return flux,err,s['harmonic_residual'],s['center']

def inputs():
    # Store portable inputs once; future runs need no external data archive.
    inp=ROOT/'coupled_sim/data/source_map_background.h5'
    if inp.exists():
        with h5py.File(inp) as h:return {k:h[k][()] for k in h}
    raw=ROOT.parent/'data/density_reference_debye.h5'
    with h5py.File(raw) as h:
        z=h['map/height_km'][:]*1000;t=h['map/time_utc_hour'][:]
        i=np.argmin(abs(t-(17+3/60+50/3600)))
        ne=h['map/ne_m3'][:,i];valid=np.isfinite(ne)&(ne>0)
        log=np.interp(z,z[valid],np.log(ne[valid]))
        ne=np.exp(gaussian_filter1d(median_filter(log,size=3),1.))
    b=np.array([float(v[0]) for v in ppigrf.igrf(LON,LAT,118.,datetime.datetime(2018,5,4,17))])*1e-9
    res=dict(height_m=z,ne_m3=ne,B_ENU_T=b,
             declination_deg=np.degrees(np.arctan2(b[0],b[1])))
    with h5py.File(inp,'w') as h:
        for k,v in res.items():h[k]=v
        h.attrs.update(epoch_unix_s=1525453430.,lower_profile='Debye-corrected CSP density, nearest cycle; log median 3 gates and Gaussian sigma 1 gate. Smooth conversion profile above 102 km; blend 98-102 km.',source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),magnetic_model='IGRF14 at Arecibo, 118 km; constant vector during local planar tracing')
    return res

def profile(z,inp):
    zn=inp['height_m'];ne=inp['ne_m3']
    low=np.exp(np.interp(z,zn,np.log(ne)))
    low=np.where(z<zn[0],ne[0]*np.exp((z-zn[0])/5000),low)
    blend=np.clip((z-98000)/4000,0,1);blend=blend**2*(3-2*blend)
    return (1-blend)*low+blend*a.PROFILE(np.clip(z,100000,120000))

def refine():
    with h5py.File(OUT) as h:start=h['best'][:2]
    result=minimize(lambda v:np.log(coupling(*v)[0]),start,method='Nelder-Mead',
        bounds=[(10,18),(-25,25)],options={'initial_simplex':np.array([start,start+[.5,0],start+[0,5]]),
        'xatol':.025,'fatol':1e-5,'maxiter':120})
    assert result.success
    with h5py.File(OUT,'r+') as h:
        z=h['profile/height_m'][:];ne=h['profile/ne_m3'][:];decl=h.attrs['declination_deg']
        theta,phi=result.x;flux,err,res,center=coupling(theta,phi)
        x,y,top,n=ray(z,ne,theta,phi,float(a.P['magnetic_t']),float(a.P['field_angle_deg']))
        b=np.deg2rad(decl+phi);east=-(x*np.sin(b)-y*np.cos(b));north=-(x*np.cos(b)+y*np.sin(b))
        lon,lat,_=Geod(ellps='WGS84').fwd(LON,LAT,np.degrees(np.arctan2(east,north)),np.hypot(east,north))
        if 'grid_best' not in h:h['grid_best']=h['best'][:]
        del h['best'];h['best']=[theta,phi,lat,lon,east,north,flux,top,err,res]
        checks=[]
        for dz,cap in [(.02,2.),(.01,1.),(.005,.5),(.0025,.5)]:
            checks.append([dz,cap,*coupling(theta,phi,dz=dz,cap=cap)[:3]])
        del h['checks/mesh'];h['checks/mesh']=checks
        h['checks/mesh'].attrs['columns']='dz_m,cap_m,required_flux_W_m2,mode_error,harmonic_residual'
        h.attrs['refined_optimum']='Nelder-Mead in theta and phi, 0.025 degree termination tolerance; starting at grid optimum'
        del h['provenance/source'];h['provenance/source']=Path(__file__).read_text()
        print('REFINED',h['best'][:],checks,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--angles',type=float,default=1.);ap.add_argument('--azimuth-step',type=float,default=10.);args=ap.parse_args()
    inp=inputs();decl=float(inp['declination_deg']);B=float(a.P['magnetic_t']);tilt=float(a.P['field_angle_deg'])
    phis=np.arange(0,360,args.azimuth_step);angles=np.arange(0,31,args.angles)
    center=coupling(14,0)[3]
    z=np.unique(np.r_[np.arange(0,100000,200.),np.arange(100000,115000,50.),np.arange(115000,center-50,10.),np.arange(center-50,center-.1,.5),center-.1])
    ne=profile(z,inp);geo=Geod(ellps='WGS84')
    n=len(phis)*len(angles);rows=[];start=time.time()
    with h5py.File(OUT,'w') as h:
        h.attrs.update(frequency_hz=a.F,radar_latitude_deg=LAT,radar_longitude_deg=LON,declination_deg=decl,center_m=center,
          scope='Direct ground-to-Es ordinary-mode rays; planar horizontally stratified measured profile and constant local IGRF B. Full-wave collisional warm-fluid conversion response per equal incident vertical flux below Es. Not a transmitter probability; no prior hops, no Landau damping; no lower-path absorption or ray-tube focusing weights.',
          angle_convention='theta vacuum-equivalent from vertical; phi propagation azimuth relative to horizontal B, clockwise. Source is opposite the integrated upward group-ray displacement.',
          coupling_model='same warm Yee/continuity/momentum equations; field azimuth rotation added; ordinary-mode source matched at homogeneous lower port')
        h['profile/height_m']=z;h['profile/ne_m3']=ne;h['profile/B_ENU_T']=inp['B_ENU_T']
        h['provenance/source']=Path(__file__).read_text()
        for phi in phis:
            for theta in angles:
                flux,err,res,_=coupling(theta,phi)
                x,y,top,nn=ray(z,ne,theta,phi,B,tilt)
                bearing=np.deg2rad(decl+phi)
                east=-(x*np.sin(bearing)-y*np.cos(bearing));north=-(x*np.cos(bearing)+y*np.sin(bearing))
                lon,lat,_=geo.fwd(LON,LAT,np.degrees(np.arctan2(east,north)),np.hypot(east,north))
                rows.append([theta,phi,lat,lon,east,north,flux,top,err,res])
            print(f'{len(rows)}/{n}; {time.time()-start:.1f} s; phi={phi}',flush=True)
            if 'scan' in h:del h['scan']
            h['scan']=np.asarray(rows);h['scan'].attrs['columns']='theta_deg,phi_deg,source_lat_deg,source_lon_deg,source_east_m,source_north_m,required_vertical_flux_W_m2,ray_max_height_m,mode_error,harmonic_residual';h.flush()
        tab=np.asarray(rows);ok=tab[:,7]>center-1.;best=np.argmin(np.where(ok,tab[:,6],np.inf))
        h['best']=tab[best]
        checks=[]
        for dz,cap in [(.02,2.),(.01,1.),(.005,.5),(.0025,.5)]:
            checks.append([dz,cap,*coupling(*tab[best,:2],dz=dz,cap=cap)[:3]])
        h['checks/mesh']=checks
        h['checks/mesh'].attrs['columns']='dz_m,cap_m,required_flux_W_m2,mode_error,harmonic_residual'
        print('BEST',tab[best], 'CHECKS',checks,flush=True)
    refine()

if __name__=='__main__':main()
