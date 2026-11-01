"""Preserve measured sampling limitations and illustrative instability scales.

No event ion-line upper limit is inferred here. The experimenter confirms that the ion-line ACFs
establish an unchanged ion line; this layout audit does not quantify its uncertainty.
"""
from pathlib import Path
import json
import numpy as np
import h5py
from scipy.constants import c,epsilon_0,m_e,e
from isr.instability_diagnostics import otsi_homogeneous_field,otsi_optimistic_growth,acoustic_frequency


def main():
    root=Path('evidence/seid_event/results4')
    report=dict(reported_ion_line='Ion line unchanged, established by ion-line ACFs; confirmed by experimenter',
                upper_limit_status='Not independently quantified; no OTSI/PDI exclusion yet',
                plasma_products=[],ion_acf_products=[],
                scope='Audit of the seven local paired products, not every file in the server archive')
    for path in sorted(root.glob('spl*.h5')):
        with h5py.File(path) as f:
            report['plasma_products'].append(dict(file=path.name,
                frequency_bin_spacing_hz=float(np.median(np.diff(f['freqs'][:]))*1e6),
                shape=list(f['PL'].shape)))
    for path in sorted(root.glob('csp*.h5')):
        with h5py.File(path) as f:
            acf=f['acf'][:];lags=f['lags'][:]
            active=np.flatnonzero(np.any(acf!=0,axis=0))
            dt=float(lags[1])*1e-6
            report['ion_acf_products'].append(dict(file=path.name,
                nominal_lags_us=lags.tolist(),populated_columns=active.tolist(),
                unused_final_column=bool(np.all(acf[:,-1]==0)),
                integration_s=float(f['t1'][()]-f['t0'][()]),
                nominal_uniform_lag_nyquist_hz=1/(2*dt),
                caveat='Fractional-lag averaging and pulse inversion also enter the response; not a six-bin resolved ion spectrum'))
    report['processing_caveat']='The original comparison PNG subtracts a smooth 2-D component, reverses frequency columns and displays only positive MHz offsets. It does not establish a near-zero ion-line detection limit.'
    Path('evidence/seid_event/mechanism_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    with h5py.File('data/instability_diagnostics.h5','w') as f:
        f.attrs.update(scope='Illustrative screens, not calibrated event thresholds or exclusions',
            reference='Djuth and Gonzales 1988, doi:10.1029/JA093iA01p00196, pp.203-204',
            provenance='audit_hf_mechanisms.py; isr/instability_diagnostics.py',
            field_convention='Source electric-field amplitude; use a consistent peak/RMS convention for any comparison',
            gamma_convention='Amplitude growth rate; power e-folding is 1/(2 gamma)',
            event_status='Ion line unchanged, established by ACFs; model-specific numerical limits and incident HF field not determined here')
        freq=5.33e6;ne=epsilon_0*m_e*(2*np.pi*freq)**2/e**2;kB=4*np.pi*430e6/c
        g=f.create_group('illustrative_threshold')
        rates=np.array([1e3,1e4,3e4]);g['electron_collision_s']=rates
        g.attrs.update(ne_m3=ne,te_k=300.,ti_k=300.,pump_frequency_hz=freq)
        for angle in [0,42]:g[f'field_angle_{angle}_v_m']=otsi_homogeneous_field(ne,300.,300.,rates,angle)
        g=f.create_group('acoustic_scale')
        mass=np.array([24.,30.,56.]);g['ion_mass_amu']=mass
        g.attrs.update(k_m1=kB,ti_k=300.)
        for te in [300,1000]:g[f'frequency_te_{te}_hz']=acoustic_frequency(kB,te,300,mass)
        g=f.create_group('published_checks')
        g.attrs['threshold_v_m']=otsi_homogeneous_field(3.2e11,230,230,18000)
        gamma=otsi_optimistic_growth(2.2,18.,5.1e6,56.)
        g.attrs['amplitude_growth_s']=gamma;g.attrs['power_efold_s']=1/(2*gamma)
        print('published checks',dict(g.attrs))
        print('illustrative thresholds',f['illustrative_threshold/field_angle_0_v_m'][:])
        print('acoustic',f['acoustic_scale/frequency_te_300_hz'][:],f['acoustic_scale/frequency_te_1000_hz'][:])


if __name__=='__main__':main()
