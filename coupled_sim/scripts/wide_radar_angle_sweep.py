"""Extend the scan to +/-80 degrees, excluding evanescent launch channels.

Fixed incident vertical flux and measured background, optimized propagating
polarization. An unavailable launch is not a computed zero radar return.
"""
from pathlib import Path
import hashlib
import h5py
import numpy as np
from scipy.constants import c,e,m_e,epsilon_0
import coupled_arecibo as a
from incident_port import incoming_channels
from minimum_incident_power import solve
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reference/coupled/radar_angle_wide.h5'
def channels(angle):
    tilt=np.deg2rad(a.P['field_angle_deg'])
    gyro=e*a.P['magnetic_t']/m_e/c*np.array([np.sin(tilt),0.,-np.cos(tilt)])
    ne=float(a.PROFILE(115300.))
    return len(incoming_channels(a.W/c,gyro,np.sqrt(ne*e*e/m_e/epsilon_0)/c,
        float(a.cold.rates(115300.,ne))/c,a.W/c*np.sin(np.deg2rad(angle))))
def main():
    with h5py.File(ROOT/'reference/coupled/radar_sensitivity.h5') as h:
        old={float(x):float(y) for x,y in zip(h['angle/degrees'][:],h['angle/RCS_m2'][:])}
        factor=float(h.attrs['reference_flux_W_m2']*h.attrs['rcs_factor'])
    angles=np.unique(np.r_[np.arange(-80.,81.,2.),np.arange(-54.,-45.),np.arange(46.,55.),list(old)])
    values=[];counts=[]
    with h5py.File(OUT,'a') as h:
        for angle in angles:
            count=channels(angle);counts.append(count)
            if count==0:values.append(np.nan);continue
            if -25.<=angle<=25. and angle in old:values.append(old[angle]);continue
            key=f'solves/{angle:+.8f}';previous=None;accepted=None
            for dz,cap in [(.005,1.),(.0025,.5),(.00125,.25),(.000625,.125),(.0003125,.0625)]:
                group=f'{key}/{dz:g}'
                if group not in h or 'radar_power_per_unit_flux_W' not in h[group].attrs:
                    h.require_group(group).attrs.update(solve(float(angle),dz=dz,cap=cap,propagating_only=True));h.flush()
                row=dict(h[group].attrs);db=10*np.log10(row['radar_power_per_unit_flux_W']*factor)
                if previous is not None and (abs(db-previous)<1 or max(db,previous)<-125):
                    accepted=row['radar_power_per_unit_flux_W']*factor
                    h[key].attrs.update(accepted_dz_m=dz,mesh_change_dB=abs(db-previous),below_display_floor=max(db,previous)<-125)
                    break
                previous=db
            if accepted is None:raise RuntimeError(f'Unresolved angle {angle}')
            values.append(accepted)
        for name,array in [('angle_deg',angles),('RCS_m2',values),('incident_channel_count',counts)]:
            if name in h:del h[name]
            h[name]=array
        lo,hi=0.,80.
        for _ in range(40):
            middle=(lo+hi)/2
            if channels(middle):lo=middle
            else:hi=middle
        h.attrs['launch_cutoff_magnitude_deg']=(lo+hi)/2
        h.attrs.update(angle_definition='Vacuum-equivalent magnetic-meridian angle from vertical; kx=(omega/c) sin(theta)',
            reference_plasma_frequency_hz=np.sqrt(float(a.PROFILE(115300.))*e*e/m_e/epsilon_0)/2/np.pi,
            reference_z_m=115100.,scope='Same measured profile and incident vertical flux. Optimized propagating incident polarization. NaN means no propagating launch channel, not a computed zero.',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        for name,source in [('source',Path(__file__)),('port',Path(__file__).with_name('incident_port.py'))]:
            if 'provenance/'+name in h:del h['provenance/'+name]
            h['provenance/'+name]=source.read_text()
        print('Completed',len(angles),'angles; propagating',sum(np.array(counts)>0),'best',angles[np.nanargmax(values)],flush=True)
if __name__=='__main__':main()
