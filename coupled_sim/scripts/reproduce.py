"""Reproduce the paper's coupled case and separate numerical sensitivities."""
import argparse
from coupled_arecibo import run
from summarize_coupled import main as summarize
from coupled_losses import main as losses
from far_field_budget import main as far_field

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cycles', type=int, default=1,
                   help='Rust periods for recorded case; 0 computes harmonic states only')
    a = p.parse_args()
    cases = [('baseline', .02, 2., 600.), ('fine', .01, 2., 600.),
             ('finest', .005, 2., 600.), ('convergence', .0025, 2., 600.),
             ('outer_mesh', .01, 1., 600.), ('larger_buffer', .01, 2., 900.),
             ('coupled', .0025, .5, 600.), ('recorded', .005, .5, 600.)]
    for name, dz, cap, buffer in cases:
        run(dz=dz, cap=cap, buffer=buffer, label=name,
            native_cycles=a.cycles if name == 'recorded' else 0)
    run(dz=.0025, cap=.5, label='cold_control', thermal=False)
    summarize()
    losses()
    far_field()
