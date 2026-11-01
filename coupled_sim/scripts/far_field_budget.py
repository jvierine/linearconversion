"""Audit HF non-return from saved outgoing fields, independently of heating.

No simulation state or old diagnostics are overwritten. All ports precede
the absorbing buffers. Port-height variation includes physical path loss in
the lossy launch medium; these are specified finite outer reference planes,
not an infinite-distance limit in an absorbing atmosphere.
"""
from pathlib import Path
import h5py,numpy as np
from scipy.constants import c,k as kb,m_e
from coupled_arecibo import OUT,flux,electromagnetic_flux,hf_port_budget

CASES=['recorded','coupled','cold_control','density_csp_sensitivity','density_csp_mesh_check']
def main():
    with h5py.File(OUT/'far_field_budget.h5','w') as out:
        out.attrs.update(definition='1 - outgoing propagating EM power / incident EM power; reflected plus transmitted for this stratified single-kx geometry',
            scope='Steady state; finite outer reference planes before absorbers. Side ports required for finite beams. Not a kinetic heating or conversion-efficiency measurement.',
            normalization='EM Poynting flux only. Pressure flux and volume heating are independent closure diagnostics.',
            source=Path(__file__).read_text())
        for name in CASES:
            path=OUT/(name+'.h5')
            if not path.exists():continue
            with h5py.File(path) as h:
                z=h['z'][:];dz=h['dz_cells'][:];zf=z+dz/2
                v=h['state_phasor'][:];inc=h['incident_reference_state'][:]
                lo=np.argmin(abs(zf-115100));hi=np.argmin(abs(zf-120800))
                s=dict(sqrt_ne=np.sqrt(h['ne'][:]),beta=np.sqrt(3*kb*h.attrs['electron_temperature_K']/m_e)/c if h.attrs['thermal_pressure_enabled'] else 0.)
                a=hf_port_budget(v,inc,lo,hi);pin=a['incident_EM_flux_W_m2']
                pf=flux(s,v)[1];pi=flux(s,inc)[1];pr=flux(s,v-inc)[1]
                q=h['collision_loss_per_cell_W_m2'][:]
                physical=(z>zf[lo])&(z<zf[hi]);heat=q[physical].sum()/pin
                escaping_pressure=(-pr[lo]+pf[hi])/pin
                incident_pressure=pi[lo]/pin
                em=electromagnetic_flux(v);ei=electromagnetic_flux(inc);er=electromagnetic_flux(v-inc)
                cross=(em[lo]-ei[lo]-er[lo]+pf[lo]-pi[lo]-pr[lo])/pin
                a.update(lower_plane_m=zf[lo],upper_plane_m=zf[hi],collision_heat_fraction=float(heat),
                    outgoing_pressure_fraction=float(escaping_pressure),incident_pressure_fraction=float(incident_pressure),
                    incident_reflected_interference_fraction=float(cross),
                    closure_residual=float(a['hf_nonreturn_fraction']-heat-escaping_pressure+incident_pressure),
                    discrete_flux_balance_residual=float(a['hf_nonreturn_fraction']-heat-escaping_pressure+incident_pressure+cross))
                assert abs(a['discrete_flux_balance_residual'])<5e-5
                assert abs(escaping_pressure)<1e-6
                g=out.create_group(name);g.attrs.update(a)
                samples=[]
                for lower in np.arange(115000.,115251.,50.):
                    il=np.argmin(abs(zf-lower))
                    for upper in [120800.,121000.,121200.]:
                        iu=np.argmin(abs(zf-upper))
                        assert np.max(h['sponge'][:,[il,iu]])==0
                        b=hf_port_budget(v,inc,il,iu)
                        samples.append([zf[il],zf[iu],b['reflected_EM_fraction'],b['transmitted_EM_fraction'],b['hf_nonreturn_fraction']])
                g['port_sweep']=samples
                g['port_sweep'].attrs['columns']='lower_m,upper_m,R_EM,T_EM,HF_nonreturn'
                a['port_sweep_nonreturn_span']=float(np.ptp(np.array(samples)[:,-1]));g.attrs.update(a)
                print(name,a)
        if 'coupled' in out and 'cold_control' in out:
            out.attrs['matched_grid_warm_minus_cold_nonreturn']=out['coupled'].attrs['hf_nonreturn_fraction']-out['cold_control'].attrs['hf_nonreturn_fraction']
        out['provenance/port_diagnostic_source']=Path(__file__).with_name('coupled_arecibo.py').read_text()
if __name__=='__main__':main()
