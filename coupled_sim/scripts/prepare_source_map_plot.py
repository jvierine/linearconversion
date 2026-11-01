"""Geographic plotting arrays only; provenance retained in HDF5."""
from pathlib import Path
import h5py,numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import box,Point
from shapely.ops import unary_union
import cartopy.io.shapereader as shp
import source_location_map as s

def main():
    with h5py.File(s.OUT,'r+') as h:
        tab=h['scan'][:];center=h.attrs['center_m']
        # A cold ray may turn just below resonance while the full-wave field
        # still couples across the evanescent gap. Keep that response. Locate
        # its footprint at the last propagating point; do not turn a ray cutoff
        # into a false zero-conversion condition.
        ok=np.all(np.isfinite(tab),axis=1)
        vals=10*np.log10(h['best'][6]/tab[:,6])
        lon=np.linspace(-67.4,-65.5,761);lat=np.linspace(17.8,19.0,481)
        xx,yy=np.meshgrid((lon[:-1]+lon[1:])/2,(lat[:-1]+lat[1:])/2)
        points=tab[ok][:,[3,2]];color=vals[ok]
        pts,ix=np.unique(np.round(points,8),axis=0,return_index=True)
        interp=LinearNDInterpolator(pts,color[ix],fill_value=np.nan)
        power=interp(xx,yy)
        # Do not bridge large gaps left by inaccessible ordinary-mode rays.
        dist=cKDTree(pts).query(np.c_[xx.ravel(),yy.ravel()])[0].reshape(xx.shape)
        power[dist>.055]=np.nan
        power[power < -30]=np.nan
        if 'map' in h:del h['map']
        g=h.create_group('map');g['longitude_edges_deg']=lon;g['latitude_edges_deg']=lat
        g.create_dataset('relative_response_db',data=power,compression='gzip')
        g.attrs['normalization']='0 dB strongest scanned ordinary-mode response; equal incident vertical HF flux below Es. No transmitter spatial prior or probability normalization.'
        g.attrs['mask']='Uncolored areas are outside sampled rays, more than 0.055 degrees from a sample, or response below -30 dB. Not excluded transmitter locations.'
        g.attrs['ray_endpoint']='Resonance minus 0.1 m where accessible; otherwise last propagating point below resonance. Full-wave response retained across the evanescent gap. Endpoint shortfall is stored in the scan; at most 216 m in this scan.'
        clip=box(lon[0],lat[0],lon[-1],lat[-1]);land=[];coast=[]
        p=shp.natural_earth('10m','physical','land')
        for geom in shp.Reader(p).geometries():
            if geom.intersects(clip):
                cut=geom.intersection(clip)
                for part in ([cut] if cut.geom_type=='Polygon' else cut.geoms):
                    if part.geom_type=='Polygon':land.append(np.array(part.exterior.coords))
        for i,arr in enumerate(land):g[f'land/{i:04d}']=arr
        p=shp.natural_earth('10m','physical','coastline')
        for geom in shp.Reader(p).geometries():
            if geom.intersects(clip):
                cut=geom.intersection(clip)
                for part in ([cut] if cut.geom_type=='LineString' else cut.geoms):
                    if part.geom_type=='LineString':coast.append(np.array(part.coords))
        for i,arr in enumerate(coast):g[f'coast/{i:04d}']=arr
        g.attrs['geography']='Natural Earth 1:10m land/coastline/populated_places; municipal town coordinates from US Census 2020 Puerto Rico CDP table.'
        towns=[];names=[]
        for r in shp.Reader(shp.natural_earth('10m','cultural','populated_places')).records():
            if clip.contains(r.geometry) and r.attributes['NAME'] in ['Ponce','Mayagüez','Arecibo','San Juan']:
                towns.append([r.geometry.x,r.geometry.y]);names.append(r.attributes['NAME'])
        towns.append([-66.7235351,18.1637939]);names.append('Adjuntas')
        g['towns/lonlat']=towns;g['towns/names']=np.array(names,dtype=h5py.string_dtype())
        best=h['best'][:];h.attrs['best_source_on_land']=any(__import__('shapely').geometry.Polygon(x).contains(Point(best[3],best[2])) for x in land)
        for lim in [-3,-10,-20]:
            use=ok&(vals>=lim)
            print(lim,'dB N',use.sum(),'lat',tab[use,2].min(),tab[use,2].max(),'lon',tab[use,3].min(),tab[use,3].max(),'distance km',np.hypot(tab[use,4],tab[use,5]).min()/1e3,np.hypot(tab[use,4],tab[use,5]).max()/1e3)
        print('Best land',h.attrs['best_source_on_land'])
        if 'checks/ray_sensitivity' in h:del h['checks/ray_sensitivity']
        z=h['profile/height_m'][:];ne=h['profile/ne_m3'][:]
        checks=[]
        for fac in [.5,1.,1.5]:
            weight=np.clip((110000-z)/10000,0,1)
            x,y,top,n=s.ray(z,ne*(1+(fac-1)*weight),14,0,float(s.a.P['magnetic_t']),float(s.a.P['field_angle_deg']))
            checks.append([fac,x,y,top])
        h['checks/ray_sensitivity']=checks
        h['checks/ray_sensitivity'].attrs['columns']='lower_density_factor,dx_m,dy_m,last_height_m; 14 degree magnetic meridian ray; density multiplier tapers to unity by 110 km'
        del h['provenance/source'];h['provenance/source']=Path(s.__file__).read_text()
        if 'provenance/plot_preparation' in h:del h['provenance/plot_preparation']
        h['provenance/plot_preparation']=Path(__file__).read_text()

if __name__=='__main__':main()
