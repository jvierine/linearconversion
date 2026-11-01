"""Reproduce the event-conditioned F-region comparison and resolution checks."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
runs=[]
for sigma,angles in [(10,[-25,-20,-18,-17,-16.75,-16.5,-16.25,-16,-15,-10,0,10,14,15,16,16.75,17,18,20,25]),
                     (5,[-17,-16.5,-16.25,-16,16,16.75]),
                     (15,[-17,-16.5,-16.25,-16,16,16.75])]:
    runs.extend((sigma,angle,.00375) for angle in angles)
runs.extend([(5,-16.5,.005),(10,-16.5,.005),(15,-16.5,.005),(10,16.75,.005)])
for sigma,angle,dz in runs:
    subprocess.run([sys.executable,str(ROOT/'scripts/fregion_comparison.py'),
        '--sigma-km',str(sigma),'--angle',str(angle),'--dz',str(dz),'--cap','2',
        '--phase-grid'],check=True,cwd=ROOT)
# The finer central-case check uses the bounded-bandwidth ordering to limit
# memory. Harmonic, Gauss-law and energy residuals still gate acceptance.
subprocess.run([sys.executable,str(ROOT/'scripts/fregion_comparison.py'),
    '--sigma-km','10','--angle','-16.5','--dz','.0025','--cap','2',
    '--phase-grid','--ordering','NATURAL'],check=True,cwd=ROOT)
