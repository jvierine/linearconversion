"""Reproduce the controlled CW sweep and selected refinement checks.

Run from the project with conda run -n base python .../run_gradient_grid.py.
The shared solver equations and diagnostics are in gradient_comparison.py.
"""
from gradient_comparison import run
for case in ['measured','E0.5','E1','E2','E5','E10','E20','F5','F20','F40']:
    for angle in [0.,10.,14.,16.75,19.5,25.]:
        run(case,angle,dz=.005,cap=1.)
for case in ['measured','E1','E5','F20']:
    for angle in [-25.,-19.5,-16.75,-14.,-10.,15.,16.,17.,18.]:
        run(case,angle,.005,1.)
for case in ['measured','E1','E5','F5','F20']:
    run(case,16.75,.0025,.5,1.5)
run('F40',16.75,.00375,1.,1.)
run('F20',16.75,.005,.5,1.)
run('F20',16.75,.0025,1.,1.)
run('noEs',16.75,.005,.5)
