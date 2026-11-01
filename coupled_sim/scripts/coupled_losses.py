"""Separate physical collision channels and audit the coupled HF energy budget.
Inputs are the saved shared-Rust run; no absorber damping enters physical loss.
"""
from pathlib import Path
import h5py
import numpy as np
from scipy.constants import c,epsilon_0,e,k as kb
from coupled_arecibo import ROOT,OUT,P,hf_port_budget

def main():
    with h5py.File(OUT/'recorded.h5') as h:
        z=h['z'][:];width=h['dz_cells'][:];dual=(width+np.roll(width,1))/2
        ne=h['ne'][:];state=h['state_phasor'][:];center=h.attrs['center_m']
        positions=np.clip(np.array([z,z,z+width/2]),115300,120800)
        tev=kb*300/e;ncm=ne*1e-6
        ei=2.91e-6*ncm*(23-np.log(np.sqrt(ncm)*tev**-1.5))/tev**1.5
        p=P['neutrals'];n2,o2,o=[np.exp(np.interp(positions,p['z'],np.log(p[key]))) for key in ['N2_m3','O2_m3','O_m3']]
        t=300.;en=2.33e-17*n2*(1-1.21e-4*t)*t+1.82e-16*o2*(1+.036*np.sqrt(t))*np.sqrt(t)+8.9e-17*o*(1+5.7e-4*t)*np.sqrt(t)
        np.testing.assert_allclose(ei+en,h['physical_nu'][:]*c,rtol=1e-13)
        factor=.5*epsilon_0*np.cos(np.pi*h.attrs['frequency_hz']*h.attrs['dt_s'])*abs(state[6:9])**2*np.array([dual,dual,width])
        qi=(factor*ei).sum(axis=0);qn=(factor*en).sum(axis=0)
        np.testing.assert_allclose(qi+qn,h['collision_loss_per_cell_W_m2'][:],rtol=1e-13)
        zf=z+width/2;ia=np.argmin(abs(zf-115100));ib=np.argmin(abs(zf-120800))
        budget=hf_port_budget(state,h['incident_reference_state'][:],ia,ib)
        pin=budget['incident_EM_flux_W_m2'];qi/=pin;qn/=pin
        mask=(z>zf[ia])&(z<zf[ib]);q=(qi+qn)*mask
        total=q.sum();cum=np.cumsum(q)
        quantiles=np.interp(np.array([.025,.5,.975])*total,cum,z)-center
        net=(h['EM_flux_W_m2'][:]+h['pressure_flux_W_m2'][:])/pin
        attrs=dict(reflected_fraction=h.attrs['reflected_fraction'],transmitted_fraction=h.attrs['transmitted_fraction'],
            electron_ion_loss_fraction=qi[mask].sum(),electron_neutral_loss_fraction=qn[mask].sum(),
            total_collision_loss_fraction=total,lower_plane_m=zf[ia],upper_plane_m=zf[ib],center_m=center,
            balance_residual=h.attrs['boundary_balance_error'],peak_downward_pressure_fraction=-h['pressure_flux_W_m2'][:].min(),
            nu_ei_at_resonance_s1=np.interp(center,z,ei[0]),nu_en_at_resonance_s1=np.interp(center,z,en[0]))
        attrs.update(budget)
        data=dict(height_m=z-center,loss_ei_per_m=qi/dual,loss_en_per_m=qn/dual,
            cumulative_loss=cum,net_flux=net,EM_flux=h['EM_flux_W_m2'][:]/pin,pressure_flux=h['pressure_flux_W_m2'][:]/pin,
            loss_quantile_heights_m=quantiles)
    with h5py.File(OUT/'cold_control.h5') as h:
        attrs.update(cold_reflected_fraction=h.attrs['reflected_fraction'],cold_collision_loss_fraction=h.attrs['physical_loss_between_planes'])
    with h5py.File(OUT/'losses.h5','w') as h:
        h.attrs.update(attrs);h.attrs['provenance']=Path(__file__).read_text()
        h.attrs['normalization']='Incident EM Poynting flux at lower outer port; absorber losses excluded. Conditional model fractions.'
        h.attrs['interpretation']='HF loss is 1-R_EM-T_EM. Cumulative collision heating and local pressure transport are independent diagnostics, not its definition. Landau damping absent.'
        for k,v in data.items():h.create_dataset('plot/'+k,data=v,compression='gzip')
    print(attrs);print('Loss 2.5,50,97.5 percentile heights (m):',quantiles)
if __name__=='__main__':main()
