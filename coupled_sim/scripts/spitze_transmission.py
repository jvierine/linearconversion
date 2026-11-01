"""Check both signed Spitze windows through the complete measured Es layer.

Positive x is along B's horizontal component, Bz<0. Upward access is at
negative incidence in this convention. Resolve both bottomside and topside
plasma resonances before interpreting power transmitted beyond the layer.
"""
from pathlib import Path
import h5py,numpy as np
from scipy.constants import e,m_e,epsilon_0
import es_fregion_electric_field as model

def main():
    a=model.a
    fce=e*float(a.P['magnetic_t'])/(2*np.pi*m_e);y=fce/a.F
    angle=np.rad2deg(np.arcsin(np.sqrt(y/(1+y))*np.sin(np.deg2rad(float(a.P['field_angle_deg'])))))
    z=np.linspace(115300,120800,20001)
    fpeak=np.sqrt(a.PROFILE(z).max()*e*e/(epsilon_0*m_e))/(2*np.pi)
    full=a.OUT/'spitze_transmission_resolved_full.h5'
    with h5py.File(full,'w') as h:
        h.attrs.update(spitze_angle_magnitude_deg=angle,frequency_hz=a.F,
            peak_plasma_frequency_hz=fpeak,gyrofrequency_hz=fce,
            field_aligned_L_max_plasma_frequency_hz=np.sqrt(a.F*(a.F+fce)),
            angle_convention='kx positive along horizontal component of B; Bz<0. Negative angle is the upward radio window.',
            scope='Coupled collisional fluid-Maxwell; ordinary-mode input. Both Es resonance crossings resolved. Outgoing EM includes all polarizations; no claim of field alignment everywhere, no Landau damping.')
        h['provenance/source']=Path(__file__).read_text()
        h['provenance/field_solver_driver']=Path(model.__file__).read_text()
        for name,theta,dz in [('downward_window',angle,.005),('upward_window',-angle,.005),('downward_window_fine',angle,.0025),('upward_window_fine',-angle,.0025)]:
            model.solve(h.create_group(name),'Es',theta,dz,.5,{'fine_bounds':(-150.,450.)})
            h.flush()
    out=model.ROOT/'reference/coupled/spitze_transmission.h5'
    model.export(full,out)
    print(out,flush=True)

if __name__=='__main__':main()
