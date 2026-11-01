"""Physical audits of regenerated event fields; never substitute reference files."""
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_physics import audit, ROOT

OUT=ROOT/'output/coupled'
def require(name):
    p=OUT/name
    if not p.exists():pytest.skip('Generate event outputs with scripts/reproduce.py and the native validation recording first')
    return p

@pytest.mark.parametrize('name',['recorded','convergence','cold_control','validation_recorded'])
def test_event_fields_obey_equations_and_energy_balance(name):
    audit(require(name+'.h5'))

@pytest.mark.parametrize('dataset,component',[('state_phasor',2),('state_phasor',8),('density_phasor_m3',None)])
def test_audit_rejects_corrupted_simulation_fields(tmp_path,dataset,component):
    import shutil
    target=tmp_path/'corrupt.h5';shutil.copy2(require('recorded.h5'),target)
    with h5py.File(target,'r+') as h:
        if component is None: h[dataset][:]*=1.01
        else: h[dataset][component]=h[dataset][component]*1.01
    with pytest.raises(AssertionError):audit(target)


def test_native_recording_follows_its_harmonic_initial_state():
    from audit_physics import native_recording
    native_recording(require('validation_recorded.h5'))


def test_event_mesh_boundary_and_window_sensitivity():
    from audit_physics import convergence
    for name in ['baseline','fine','finest','convergence','outer_mesh','larger_buffer']:
        require(name+'.h5')
    convergence(OUT)
