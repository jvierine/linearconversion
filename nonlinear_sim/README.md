# Total-density scalar model and tensor comparison

The simple total-density model is `src/scalar.rs`. It evolves **total** electron
density and momentum `[n_e, n_e u_x, n_e u_y, n_e u_z]`, coupled to all Maxwell
field components. No pressure tensor is evolved. The equations are

\[
\partial_t n_e+\nabla\cdot(n_e\mathbf u)=0,\qquad\mathbf J=-en_e\mathbf u,
\]
\[
\partial_t(n_e\mathbf u)+\nabla\cdot(n_e\mathbf u\mathbf u)
=-\frac{e}{m_e}n_e[\mathbf E+\mathbf u\times\mathbf B]
-c_e^2\nabla n_e-\nu n_e\mathbf u,
\qquad c_e^2=3k_BT_{\rm ref}/m_e.
\]

This is the nonlinear barotropic continuation of the paper's effective scalar
closure, not a nonlinear adiabatic temperature model. The effective pressure is
`p_eff=m_e c_e^2 n_e`, with constant `T_ref=300 K`. A fixed equilibrium electric
field supports the background gradient. The conservative implementation subtracts
equilibrium fluxes and forces before differentiation, so the layer remains
stationary without a wave. It stores total density; density differences are
computed only where needed for equilibrium subtraction or diagnostics. Physical
pressure anisotropy, temperature evolution and Landau damping remain absent.
Here the total field includes the fixed equilibrium field that supports the
measured density gradient. Subtracting the exactly balanced equilibrium terms
inside the force calculation does not linearize the evolved density or momentum.

From this repository root:

```sh
cargo build --release --manifest-path nonlinear_sim/Cargo.toml --bin scalar_total
cargo test --release --manifest-path nonlinear_sim/Cargo.toml --test scalar_total
conda run -n base python nonlinear_sim/scripts/run_scalar_total.py --label scalar_total
conda run -n base python nonlinear_sim/scripts/run_scalar_total.py --linear --label scalar_total_linear
conda run -n base python nonlinear_sim/scripts/run_scalar_total.py --start equilibrium --ramp-cycles 10 --cycles 120 --dz .08 --frames 120 --label scalar_switch_on
conda run -n base python nonlinear_sim/scripts/diagnose_scalar_total.py scalar_switch_on
```

`output/scalar_total.h5` contains `electron_density_m3[time,x,z]`, the equilibrium
density, ion density, positions, all evolved fields and momenta, and the initial
scalar harmonic solution. In this file, state entry zero is total density, not
a perturbation. The input harmonic solution is a weak-wave initial condition;
it does not prescribe the later nonlinear evolution. With `--start equilibrium`,
the actual initial density is exactly the background and all wave fields and
electron momenta are zero. The radio current source is switched on with a smooth
sine-squared envelope. Its charge is integrated consistently for the Gauss-law
check; it is separate from electron density.

`diagnose_scalar_total.py` reads the saved total density and then computes
`delta_density_x0_m3 = total_density_x0_m3 - equilibrium_density_m3` and the
horizontal Fourier components. These are diagnostic products, not evolved state
variables. Both total density and the extracted wave are retained. The integrated
source-sheet scale can be fixed with `--drive-scale` for grid comparisons; the
default `--amplitude` scales the harmonic reference, not the initial density in
an equilibrium-start run.

Five tests verify stationary nonuniform equilibrium, Bohm--Gross dispersion
with charge conservation, quadratic departure from the linear limit, and
wave creation from equilibrium by a ramped source. The fifth checks the analytic
collisional decay rate, including the damped oscillation frequency. On the
Arecibo profile, a one-period run starting at peak fractional amplitude 1e-4
differs from its matching linear calculation by 1.58e-6 of the density-wave
L2 amplitude over 117.930--117.960 km. Compact results are saved in
`reference/scalar_total_check.h5`. This is a one-grid, one-period consistency
check, not spatial convergence of the event RCS or a nonlinear steady state.
The paper's existing validated linear results remain the quantitative reference.

The 10-period total-density run remains positive and has a maximum normalized
Gauss residual of 4.2e-14. Its principal horizontal density-wave component
differs from the harmonic reference by 2.1e-8 in relative L2 norm near conversion.
This checks time evolution on one mesh. The separate fixed-drive harmonic grid
sweep in `reference/scalar_total_grid.h5` does **not** pass a 1% spatial convergence
criterion: the last 5-to-2.5 mm comparison changes the complex wave by about 10%,
or 3.8% after a fitted complex rescaling. The latter is only a shape diagnostic,
not evidence of absolute power convergence. Source placement, the outer EM mesh,
and the graded grid below the resolved wave region also require control. These
new scalar runs therefore do not replace the paper's RCS or energy-loss results.

The equilibrium-start test ran for 120 periods (22.47 microseconds) with total
density positive and normalized Gauss residual below 1.3e-13. It did **not**
establish a resolved converted wave: the near-resonance signal was only
6.35e-10 of the background, and its vertical phase alternated by almost pi
between adjacent cells. That is a grid-scale numerical artifact. The saved
switch-on figure is a diagnostic of this failure, not a physical wave-generation
result. The centered collocated spatial discretization and graded-grid transition
need correction and convergence checks before interpreting a switch-on run.
The initialized 10-period run and analytic tests establish the total-density
implementation's basic behavior, not the validity of this transient.

Render the equilibrium-start diagnostics from the repository root:

```sh
cargo run --release --manifest-path plotter/Cargo.toml --bin scalar_total_density
```

## Tensor comparison model

This model evolves electron density, momentum, the full symmetric pressure tensor,
and all six electromagnetic field components on a real horizontal–vertical grid.
It retains advection, the density-dependent current, and the wave magnetic force.
The reusable moment equations are in `jvierine/electron-fluid`; the Maxwell curl
operator is in `jvierine/fdtd`. `Cargo.lock` fixes the exact dependencies.

The physical equations are

\[
\partial_t n+\nabla\cdot(n\mathbf u)=0,
\]
\[
m_en(\partial_t+\mathbf u\cdot\nabla)\mathbf u
=-en(\mathbf E+\mathbf u\times\mathbf B)-\nabla\cdot\mathbf P
-m_en\nu\mathbf u,
\]
\[
\epsilon_0\partial_t\mathbf E=\nabla\times\mathbf B/\mu_0-\mathbf J,
\quad \partial_t\mathbf B=-\nabla\times\mathbf E,
\quad\mathbf J=-en\mathbf u.
\]

Pressure evolves through the second-moment equation. A Gaussian closure sets the
central third moment (heat flux tensor) to zero. It is **not a Vlasov solver**:
Landau damping and trapping are absent. Permittivity is represented by the current
response; an additional plasma dielectric update would double-count that response.

The collision operator is a number-conserving Bhatnagar–Gross–Krook (BGK) relaxation
toward a stationary 300 K bath. Its rate uses the existing electron–neutral plus
electron–ion estimates, but its pressure/energy relaxation is a modeling choice,
not an exact Coulomb collision operator. Do not use these runs to establish a new
absolute heating or absorption result without testing the collision closure.

## Run

From this directory:

```sh
cargo test --release
cargo build --release
conda run --no-capture-output -n base python scripts/run.py --cycles 2 --amplitude .0001 --label weak_lifted
conda run --no-capture-output -n base python scripts/run.py --cycles 2 --amplitude .0001 --linear --label weak_linear
conda run --no-capture-output -n base python scripts/run.py --cycles 2 --amplitude .001 --label medium_full
conda run --no-capture-output -n base python scripts/run.py --cycles 2 --amplitude .01 --label strong_full
conda run --no-capture-output -n base python scripts/run.py --cycles 2 --amplitude .0001 --dz .02 --nx 16 --label weak_refined_lifted
conda run --no-capture-output -n base python scripts/run.py --cycles 20 --amplitude .0001 --label weak_long
conda run -n base python scripts/diagnostics.py
```

HDF5 retains every field and moment at every saved horizontal and vertical point.
`state` is ordered `[time,x,z,variable]`. Variables are perturbations of
`[n,n ux,n uy,n uz,Mxx,Mxy,Mxz,Myy,Myz,Mzz,Ex,Ey,Ez,Hx,Hy,Hz]`, where
`Mij=n ui uj+Pij/m_e` and `H=cB`. Normalization scales are HDF5 attributes.
Time is in seconds; `z_m` is altitude in metres. Full snapshots are generated locally;
compact diagnostic arrays are versioned under `reference/`.

## Setup and interpretation

The smooth measured Arecibo density profile is sampled over 114–122 km, with the
existing constant continuation outside 115.3–120.8 km. Temperature is currently
300 K. The ambipolar equilibrium field balances the pressure gradient; stationary
ion density includes the tiny correction required by Gauss's law. Ions do not move.
The wave has frequency 5.34003 MHz, incidence 16.75 degrees, and periodic horizontal
extent equal to one incident horizontal wavelength. This is an illustrative
favorable geometry, not an identified transmitter.

An independent sparse solve of the **linearized ten-moment equations** supplies
the initial harmonic fields and the maintained electric-current source sheet.
The nonlinear initial raw second moment includes the quadratic bulk-flow term,
so central pressure remains physical. Amplitude is set by peak initial fractional
density, not inferred transmitter power. The subsequent Rust evolution is nonlinear.
A run started this way tests harmonic-state evolution; it is not a switch-on or
nonlinear steady-state calculation. The linear comparison uses the same pressure
closure, separating nonlinearity from differences with the older barotropic model.

Fourth-order horizontal differences, centered conservative vertical differences,
and shared RK4 stages preserve the discrete charge constraints. Absorbing buffers
are artificial and must not be counted as physical absorption. Their electric
conductivity has a corresponding artificial charge flux; source-sheet charge is
included in the Gauss audit. This centered method is intended for smooth waves;
it rejects nonpositive pressure/density and is not a shock-capturing method.

Tests check equilibrium, Langmuir and transverse-wave dispersion, Gauss's law,
energy conservation, analytic current damping, the weak-amplitude limit, physical
initial pressure, and fourth-order time convergence. These checks do not establish
convergence of the Arecibo radar cross section or far-field loss. The refined-grid
comparison is a normalized shape diagnostic, not absolute power convergence.

## Current results

The eight integration tests and three reusable-kernel tests pass. The 0.01%
initial-density case remains stable for 20 radio periods (3.75 microseconds).
Its normalized Gauss residual remains below 2e-14. This duration is short compared
with collisional relaxation and is not evidence of nonlinear saturation.

The generated mean density and second **horizontal spatial** harmonic scale
approximately quadratically with drive over initial amplitudes 0.01%, 0.1%, and
1%. The single harmonic present in the initial state cannot represent these terms;
the real-space nonlinear model can.

**The event calculation is not yet spatially converged to 1%.** Reducing vertical
spacing from 4 to 2 cm changes the locally normalized envelope by about 5.1% and
the phase-aligned complex shape by about 37.6%. Further refinement to 2.5 mm leaves
about 3.3% complex-shape difference from the previous grid. The latter sweep solves
the linear initial boundary problem, not a new nonlinear time evolution. Outer-grid
resolution, grading, and boundaries require independent checks as well. These
results therefore do not replace the manuscript's existing linear-model RCS or
far-field energy budget. `reference/grid_check.h5` records the failed 1% criterion.

The diagnostic plots show a downward phase progression with contracting wavelength,
and a crossing of the 430 MHz Bragg wavelength roughly 4 m below the selected
resonance. Treat that location as provisional while spatial convergence is resolved.

Render figures from the analysis-repository root:

```sh
cargo run --release --manifest-path plotter/Cargo.toml --bin nonlinear_validation
```
