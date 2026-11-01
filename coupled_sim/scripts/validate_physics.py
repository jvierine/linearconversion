"""Run fresh native tests and optionally audit regenerated event data; save HDF5."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys
import h5py
import numpy as np
from audit_physics import ROOT, audit, convergence, native_recording


def put(group,metrics):
    for key,value in metrics.items():
        if np.ndim(value):group[key]=value
        else:group.attrs[key]=value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--with-event',action='store_true',help='Require and audit the regenerated event/mesh outputs and validation_recorded.h5')
    args=parser.parse_args()
    out=ROOT/'output/coupled/physics_validation.h5';out.parent.mkdir(parents=True,exist_ok=True)
    result=subprocess.run(['cargo','test','--release','--locked','--test','physics','--','--nocapture'],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(result.stdout,end='')
    with h5py.File(out,'w') as h:
        h.attrs['passed']=False
        h.attrs['event_requested']=args.with_event
        h.attrs['scope']='Numerical equation/continuum verification of collisional warm fluid + Maxwell; no kinetic Landau validation or independent measurement-amplitude prediction'
        h['provenance/native_test_log']=result.stdout
        h['provenance/Cargo.lock']=(ROOT/'Cargo.lock').read_text()
        for path in ['src/lib.rs','src/main.rs','tests/physics.rs','scripts/audit_physics.py','scripts/validate_physics.py']:
            h['provenance/source/'+path]=(ROOT/path).read_text()
        for line in result.stdout.splitlines():
            if 'VALIDATION {' in line:
                record=json.loads(line.split('VALIDATION ',1)[1]);put(h.create_group('native/'+record['name']),record['metrics'])
        result.check_returncode()
        assert len(h['native'])==6,'Missing native test diagnostics'
        subprocess.run(['cargo','build','--release','--locked'],cwd=ROOT,check=True)
        analytic=subprocess.run([sys.executable,'scripts/validate_warm.py'],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        h['provenance/physical_mode_test_log']=analytic.stdout
        print(analytic.stdout,end='');analytic.check_returncode()
        with h5py.File(out.parent/'analytic_validation.h5') as physical:
            physical.copy('results',h,name='physical_mode_results')
        h['provenance/source/scripts/validate_warm.py']=(ROOT/'scripts/validate_warm.py').read_text()
        if args.with_event:
            cases=['recorded','convergence','cold_control','validation_recorded']
            for case in cases:
                path=out.parent/(case+'.h5')
                if not path.exists():raise FileNotFoundError(f'Regenerate {path}; archived reference data are not a substitute')
                metrics=audit(path);g=h.create_group('event/'+case);put(g,metrics)
                g.attrs['input_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                print(case,metrics,flush=True)
            put(h.create_group('event/native_recording'),native_recording(out.parent/'validation_recorded.h5'))
            put(h.create_group('event/mesh_convergence'),convergence(out.parent))
            # Negative controls deliberately corrupt E, J, and density to check
            # that the diagnostic actually rejects inconsistent field outputs.
            pytest=subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            h['provenance/event_test_log']=pytest.stdout;print(pytest.stdout,end='');pytest.check_returncode()
            assert 'skipped' not in pytest.stdout,'Requested event validation must not silently skip tests'
        h.attrs['passed']=True
    print(f'PASS: {out}')

if __name__=='__main__':main()
