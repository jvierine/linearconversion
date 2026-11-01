"""Stage archived HDF5 figure inputs without running a new simulation."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
mapping={
 'reference/coupled/summary.h5':'output/coupled/summary.h5',
 'reference/coupled/density_comparison.h5':'output/coupled/density_comparison.h5',
 'reference/fluid/plasma_line_asymmetry.h5':'output/fluid/plasma_line_asymmetry.h5',
 'reference/fluid/hf_power_from_measured_pl.h5':'output/fluid/hf_power_from_measured_pl.h5',
 'coupled_sim/reference/losses.h5':'coupled_sim/output/coupled/losses.h5',
 'coupled_sim/reference/observed_fregion_summary.h5':'coupled_sim/output/coupled/gradient_study/observed_f_summary.h5',
}
for src,dst in mapping.items():
 p=ROOT/dst;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():shutil.copy2(ROOT/src,p)
(ROOT/'figures').mkdir(exist_ok=True)
print('Archived figure inputs ready; existing calculation outputs preserved.')
