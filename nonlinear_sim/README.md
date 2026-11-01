# Nonlinear electron-fluid / Maxwell model

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
