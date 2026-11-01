# Arecibo HF conversion

Coupled Rust Maxwell and electron-fluid model for the 4 May 2018 Arecibo
sporadic-E plasma-line event. The measured smooth density profile, neutral
prior, illumination polarization and radar-calibration products are included
as HDF5 inputs. Raw radar voltages are not included.

## Run

Requires Rust, an MPI development installation (required by the FDTD crate),
and Python with NumPy, SciPy and h5py. These repositories currently require
GitHub access. Authenticate Git with `gh auth setup-git` if necessary.

```sh
git clone --recurse-submodules https://github.com/jvierine/linearconversion.git
cd linearconversion/coupled_sim
conda run -n base python -m pip install -r requirements.txt
cargo build --release --locked
conda run -n base python scripts/validate_warm.py
conda run -n base python scripts/reproduce.py
conda run -n base python scripts/check_reference.py
```

## Physics validation

To recompute the denser incidence-angle sweep for Figure 7, run from this
directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run -n base python scripts/refine_radar_angle_sweep.py
```

This solves additional angles using the existing coupled harmonic equations,
checks selected angles on a finer mesh, and updates only the angle panel's
HDF5 arrays. It retains the reference case and the other sensitivity panels.

```sh
conda run -n base python scripts/validate_physics.py
```

This runs the **same coupled timestep as the executable**, against independent
equation residuals and continuum solutions. Tests cover Maxwell, momentum and
continuity, both Gauss constraints on a nonuniform magnetized grid, EM and
Langmuir dispersion, collisional amplitude decay at `nu/2`, and conservation of
wave energy plus collision heating. Refinement must give second-order errors.
These tests do not require the large event outputs.

After `scripts/reproduce.py`, validate the event as well:

```sh
cargo build --release --locked
conda run -n base python scripts/coupled_arecibo.py --dz .005 --cap .5 --cycles 1 --label validation_recorded
conda run -n base python scripts/validate_physics.py --with-event
```

The event audit independently evaluates the equations on the saved fields,
checks the downward wave's shortening and Bragg match, tests grid/buffer/radar
window sensitivity, and compares every recorded E/density frame with its
harmonic initial state. Corrupted E, current and density outputs must fail.
Missing event outputs are errors in this command, not silent passes.
Diagnostics and provenance are saved in `output/coupled/physics_validation.h5`.

The checked reference run gives a solved phase match at **4.05 m below
resonance**, versus **4.09 m** from local warm electrostatic dispersion.
The final mesh refinement changes peak density by **0.24%** and the coherent
radar power by **0.052%**. The archived validation report is
`reference/physics_validation.h5`.
This verifies the implemented collisional warm-fluid model; it does not test
Landau damping, missing nonlinear physics, or independently predict the fitted
measurement amplitude. The native event check starts in continuous-wave
equilibrium and is not a transmitter turn-on experiment.

The reproduction runs the measured case, a cold control, and separate mesh
and boundary checks. One HF period of the measured case is advanced by Rust;
it starts from the discrete continuous-wave solution, not transmitter turn-on.
Use `--cycles 0` for harmonic solutions only or a larger integer for longer
native recordings. Scientific outputs go to `output/coupled/*.h5`.

The earlier idealized E/F gradient sweep is archived for provenance. Its
artificial F5/F20/F40 cases are not the paper's F-region comparison; use the
observed-profile workflow below.

## Coupling

At each timestep the driver calls the FDTD magnetic-field update, the fluid
momentum/continuity update, and the FDTD electric-field update using the
returned electron current. There is one electron population and no additional
Poisson field. The reusable fluid API is independent of the FDTD library.

- [FDTD library](https://github.com/jvierine/fdtd): Maxwell Yee steps.
- [Electron-fluid library](https://github.com/jvierine/electron-fluid): momentum,
  continuity, pressure and drag; see its coupling contract for units/staggering.
- `Cargo.toml` pins both exact Git revisions; `Cargo.lock` pins dependencies.

## Interpretation

For the favorable specified illumination, the downward wave reaches the
430 MHz Bragg wavelength about 4.09 m below resonance. Matching the observed
line amplitude assumes coherent beam filling, 1 MW radar peak power, 150 K
system temperature, nominal antenna gain and ion-line calibration. It is a
conditional amplitude fit, not an independent transmitter-power measurement.

This is a linear magnetized warm-fluid calculation with electron-ion and
electron-neutral collisions, fixed ions, and one horizontal Fourier mode.
It does not include Landau damping, trapping, or nonlinear instability.
Total absorption is nearly the same in the matched cold control: it is not
an extra conversion efficiency. The paper now compares against the event's
observed smooth F-region background. The archived artificial F5/F20/F40
profiles are sensitivity cases, not evidence of ordinary F-region enhancement.

The `reference/` files retain the paper's diagnostic results for regression
comparison. Validation checks analytic electromagnetic and Langmuir modes,
charge/energy balance, and numerical sensitivity; it does not validate the
omitted physics. Input hashes and origins are in `data/manifest.h5`.

## HF loss

The control is **FDTD with a fixed background plasma**, including the usual
magnetized dispersive electron current and the same physical collisions.
It has no separate electron-fluid pressure or continuity evolution. The
coupled run replaces that current update with the momentum--continuity
solver. Both use the same measured background density; neither run changes
the equilibrium density through nonlinear heating or transport.

To compare the standalone executable from the pinned FDTD repository with
the coupled Rust executable, after generating the event outputs:

```sh
conda run --no-capture-output -n base python scripts/fdtd_only_control.py
conda run -n base python -m pytest tests/test_fdtd_only_control.py -q
```

The script checks identical backgrounds, grids, rates, timesteps and source
polarization, then advances both for one HF period from their continuous-wave
states. The FDTD executable does not call the electron-fluid library. Recorded
complex electric fields are checked against the harmonic solution; their
magnetic phasors are recovered from discrete Faraday's law to check the
outgoing-power budget. Results are saved to
`output/coupled/fdtd_only_comparison.h5`. This is a steady-state comparison,
not a pulse turn-on or slowly modified-background experiment.

The native check gives 99.58610068% HF loss for standalone FDTD and
99.58610113% for the coupled model. Their difference, 4.46e-9 of incident
power, is unresolved relative to the flux-accounting error. The compact
report is `reference/fdtd_only_comparison.h5`.

The large loss is specific to illumination. With the same plasma and
collisions, normal incidence reflects 91.98%. At 16.75 degrees, a
reflection-favoring polarization reflects 90.02%, whereas the paper's
excitation-favoring polarization reflects 0.414%. The FDTD control already
retains resonant electron-current absorption. Reproduce this comparison with:

```sh
conda run -n base python scripts/reflection_geometry_audit.py
```

Flux matrices and polarization eigenvectors are archived in
`reference/reflection_geometry_audit.h5`; these are harmonic solutions of
the discrete FDTD equations, not additional native time recordings.

HF loss is **1 minus all outgoing far-field EM power divided by incident
EM power**. In this stratified single-horizontal-wave-number model it is
`1 - R_EM - T_EM`, including both polarizations. A finite-beam model must
also include sideways radiation. Reflection is outgoing power, not loss.

```sh
conda run -n base python scripts/far_field_budget.py
conda run -n base python -m pytest tests/test_far_field_budget.py -q
```

The diagnostic subtracts incident fields before measuring reflection and
uses outer ports at 115.1 and 120.8 km, before the absorbing buffers.
Heating and pressure-wave flux are independent closure diagnostics. Local
field enhancement and the peak downward pressure flux do not define loss.
The report includes a port-height sweep; physical propagation beyond the
specified ports is outside its budget. For pulses, integrate the outgoing
power until stored energy has escaped or dissipated before claiming loss.

The reference gives 99.5846% HF non-return, and the CSP background gives
97.0911%. At matched 2.5 mm spacing the reference warm and cold cases both
give 99.5861%, with a difference below the numerical accounting error.
No additional warm-fluid HF loss is resolved by this comparison.

## Density retrieval

The E-region density uses seven short-pulse integrations per cycle. Only the
normalized long-pulse spectral shape is pooled (3 cycles x 5 gates); density
amplitude retains 300 m / 80.4 s sampling. The compact observed ACF inputs and
priors are bundled, with source hashes in HDF5 metadata.

```sh
conda run -n base python recover_density_information.py --workers 6
conda run -n base python scripts/density_sensitivity.py
```

The second command repeats the coupled calculation with the supplied smooth
CSP density and checks a finer grid. The earlier CLP-background case remains
the explicit reference; the two backgrounds give Bragg separations of about
4.09 m and 2.61 m. Calibration and shared-shape uncertainties remain relevant.

## Observed F-region comparison

Use the contemporaneous F-region density and its own gradient, not an Es
profile moved to F-region altitude. The bundled input contains three cycles
of native density/temperature estimates, MSIS neutrals and IGRF fields.
Gaussian vertical smoothing widths of 5, 10 and 15 km give local density
scale lengths of 63, 82 and 96 km. The local temperature is about 1580 K.

```sh
conda run -n base python scripts/prepare_fregion_background.py
conda run -n base python scripts/run_observed_fregion.py
conda run -n base python scripts/summarize_fregion_comparison.py
```

The sweep uses a wavelength-adapted grid and repeats selected cases at a
second resolution. It uses the same coupled equations and local Bragg
projection as the E-region control. Equal flux at the F-region lower boundary
does not include screening by the intervening sporadic-E layer. Results
remain conditional on the fluid closure and omit Landau damping.

## Pressure closure used for the paper

The production Rust and harmonic solvers retain the simple scalar force
`-cs^2 grad(delta_ne/ne)`, with `cs^2=3 kB Te/me` and constant 300 K.
The coefficient is the standard Bohm--Gross longitudinal thermal response
(Bohm & Gross 1949, doi:10.1103/PhysRev.75.1851); scalar gamma=3 thermal
models are also used by Gondarenko et al. (2009, doi:10.1029/2009GL038379).
The particular inhomogeneous force uses effective barotropic background
support, not an exact magnetized pressure tensor.

A separate harmonic audit restores the physical-background correction from
`p0=ne kB Te`, `delta_p=3 kB Te delta_ne` (Willes & Cairns 2001, Eqs. 5--6,
doi:10.1071/AS01051). It changes this case's coherent echo power by 0.271%,
consistent at 1 cm and 0.5 cm resolution. The unmagnetized kinetic frequency
at the Bragg scale differs from Bohm--Gross by 0.00055%. These checks do not
validate magnetized kinetic effects, Landau damping, or the collision closure.

```sh
conda run -n base python scripts/check_scalar_closure.py
```

Results are in `../reference/coupled/scalar_closure_check.h5`. The optional
`pressure_gradient_audit` setup flag is for this harmonic comparison only;
it does not change the production Rust evolution or its energy accounting.
The nonlinear tensor solver remains a separate experimental model.

### Minimum incident power for the observed 5.34 MHz line

`conda run -n base python coupled_sim/scripts/minimum_incident_power.py`
scans magnetic-meridian incidence and optimizes the complex source
polarization for the coherent radar return per unit upward incident EM flux.
The minimum within the -25 to +25 degree scan is near 13.96 degrees:
8.43e-12 W/m2, or 65.3 microvolt/m RMS in the incident wave at the lower
reference plane. The 2.5 mm grid changes the inferred flux by 0.054% from
the 5 mm grid. These are conditional beam-filled coherent-model results,
not a transmitter power or a global optimum over all geometries.

`conda run -n base python coupled_sim/scripts/plot_minimum_incident_data.py`
generates `data/minimum_incident_plots.h5`. The Rust figure binary is
`minimum_incident`; it reads that HDF5 product. The height-time image is the
real part of the sustained harmonic solution over three periods, not a
switch-on or growth simulation. `reference/coupled/minimum_incident.h5`
retains the solved fields and density covering the radar-response window.
Validate the saved solution with
`conda run -n base python coupled_sim/scripts/check_minimum_incident.py`.

The current optimum's HF power budget and incident-ray access are checked by
`conda run -n base python coupled_sim/scripts/resonant_access_audit.py` after
running the minimum-incident-power calculation. It writes
`coupled_sim/reference/resonant_access_audit.h5`. The 14-degree optimum absorbs
77.5% and reflects 22.5%; the older 16.75-degree reference absorbs 99.6%.
These are different illuminations. Cold local rays diagnose the turning heights;
they do not supply mode-conversion efficiencies or amplitudes. The full-wave
solution supplies the fields and energy budget. The `resonant_access` Rust
plotter renders this diagnostic. The independent local-dispersion checks run
with `conda run -n base python -m pytest coupled_sim/tests/test_resonant_access.py`.

## FDTD versus ray/WKB reflection

```sh
conda run -n base python coupled_sim/scripts/fdtd_ray_benchmark.py
cargo run --release --locked --manifest-path plotter/Cargo.toml --bin fdtd_ray_benchmark -- reference/coupled/fdtd_ray_benchmark.h5
conda run -n base python -m pytest coupled_sim/tests/test_fdtd_ray_benchmark.py -q
```

Run these from the repository root. If conda's MPI wrapper points to an
unavailable compiler, set `MPICC` to a working MPI wrapper before the first
command (on the development Mac: `/opt/homebrew/bin/mpicc`).
The scalar control uses the measured smooth Es density, B=0 and no collisions;
the event control restores B and collisions. Native Rust recordings start
from CW equilibrium, not switch-on. WKB field-magnitude errors are for the
scalar control only, not magnetized mode-conversion or finite-beam ray errors.
The HDF5 report includes mesh checks, executable hash and the complete driver.

## Where the HF power is absorbed

From the repository root:

```sh
conda run -n base python coupled_sim/scripts/hf_absorption_profile.py
cargo run --release --locked --manifest-path plotter/Cargo.toml --bin hf_absorption_profile
conda run -n base python -m pytest coupled_sim/tests/test_hf_absorption_profile.py -q
```

The fixed-background FDTD calculation integrates physical collisional heating
and independently measures `1-R-T` at the outer electromagnetic ports.
Numerical absorbing buffers are excluded. The cumulative curve is deposition
below each altitude, including incident and reflected fields; it is not a
one-way attenuation curve. O/X illumination matches the local collisional
mode polarization at the lower homogeneous reference plane.

For the observed profile, O-mode absorption is 8.0% at normal incidence,
76.3% at 13.956 degrees, and 13.3% at 25 degrees; X-mode absorption at
13.956 degrees is 11.2%. Angles are vacuum-equivalent angles from vertical.
The earlier optimized-polarization case absorbs 77.5%. At 13.956 degrees,
71.1% of incident O-mode power is deposited within 30 m of resonance.
Halving/doubling collisions gives 75.3%/78.1% O-mode absorption: the resonant
field strengthens/narrows as damping decreases. This is cold-plasma
collisional resonance absorption, not a demonstrated additional warm-fluid
or Landau loss. See Gondarenko et al. (2009), doi:10.1029/2009GL038379.

The native FDTD recurrence was checked separately by `fdtd_ray_benchmark.py`.
These profiles are its harmonic steady-state solutions. Heating and port loss
close within 0.03 percentage point for all tested illuminations; the residual
includes finite-port incident/reflected interference and discretization.
Results and source provenance are in `reference/coupled/hf_absorption_profile.h5`.

## Animated bottomside and topside conversion

The same coupled Rust Yee/electron-fluid timestep now has movies of both sides
of the measured smooth sporadic-E layer. These are **sustained continuous-wave
solutions**, initialized from the discrete driven solution, then
advanced for one radio period. They are not switch-on or growth simulations.
All six Maxwell components and the electron current/pressure/continuity remain
coupled; the displayed horizontal dependence is the model's single Fourier mode.

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 conda run -n base python coupled_sim/scripts/lmode_animation.py scan
OPENBLAS_NUM_THREADS=1 conda run -n base python coupled_sim/scripts/lmode_animation.py record
OPENBLAS_NUM_THREADS=1 conda run -n base python coupled_sim/scripts/check_conversion_animation.py
conda run -n base python coupled_sim/scripts/render_conversion_animations.py
```

Movies and the complete HDF5 fields are written to `output/lmode_animation/`.
Rendering requires the pinned rustplotlib crate and `ffmpeg`. The MP4 repeats
one period three times, slowed to 12 seconds. The GIF loops the same period.
No envelope growth or upward-moving wave is imposed by the plotter. Every movie
shows **both sides of the same full-layer solution**: blue A marks the topside
zoom and orange B marks the bottomside zoom. Matching boxes locate the zooms on
the overview density and electromagnetic-energy-flux profiles. Each zoom shows
the signed vertical electric field, a two-dimensional density perturbation,
and the signed pressure energy flux. Height increases upward in every panel.
Field and density color limits stay fixed in time. The Spitze movie uses
explicitly labeled expanded scales to reveal its weaker waves: electric-field
ratios of +/-10 and +/-5, and density ranges of +/-7 and +/-3 ppm, on the top
and bottom respectively. Compare axis values, not visual size, across movies.

The full-height electromagnetic curve is the period-averaged **net vertical
Poynting flux**, calculated from E cross B / mu0, positive upward and normalized
to incident upward power. Boundary labels separately give upward incident,
downward reflected, and upward transmitted power at the outer ports. We do not
interpret the interior net curve as a separately decomposed upgoing mode where
incident/reflected/converting fields overlap. The pressure-energy-flux panels
show the fluid transport separately; they are not electromagnetic Poynting flux.
Both flux diagnostics are steady period averages while E and density oscillate.

The incident wave is the **ordinary eigenmode at the lower port** in all cases.
Angles are signed, vacuum-equivalent angles from vertical in the magnetic
meridian; positive horizontal wave number is along the horizontal component of
B, with B directed downward. Thus opposite signs describe different directions.
The through-layer calculation tests the Z/L transmission route, rather than
assuming that the wave is purely field-aligned L mode everywhere.

For 5.340 MHz and this layer, the upward Spitze window is at -16.697 degrees.
It transmits 91.64% of the incident EM power but is not the strongest generator
of upward electron plasma waves. The O-input scan from -24 to +24 degrees gives
its largest **outward pressure energy flux averaged 3--6 m from resonance** at
-12.272 degrees on the topside and +16.466 degrees on the bottomside. This is
an explicit conversion metric after some collisional damping; it is not the
fraction of all HF energy absorbed or an optimization over all azimuths and
polarizations. The topside optimum transmits 23.64%, reflects 47.16%, and loses
29.20% of its EM power between the outer ports. The bottomside optimum transmits
negligibly, reflects 4.27%, and loses 95.73%. These losses include ordinary
collisional absorption and must not be called conversion efficiencies.

The numerical wave phase reaches the approximate 430 MHz Bragg wave number
4 pi f/c at 3.74 m above the topside resonance, and 3.98 m below the bottomside
resonance. Local magnetized warm-fluid dispersion predicts 3.76 and 4.04 m.
The movies use an illustrative incident peak field of **1 mV/m** at 115.1 km;
this is not an estimate of the event's unknown transmitter. Amplitudes scale
linearly in this model. All cases include electron-ion and electron-neutral
collisions, but **not Landau damping**.

The native evolution uses 5 mm cells near both conversion layers and 0.25 m
maximum spacing elsewhere. Halving both spacings changes the density envelope
by 0.42--0.43% over 1--8 m outward from resonance. The complex wave differs by
12--13.5%, owing to phase sensitivity, so these movies do not establish percent-
level convergence of phase or coherent radar power. Native evolution agrees
with its discrete driven solution to better than 1e-7; Gauss's law, outgoing
phase/pressure direction, wavelength contraction and collisional decay are
checked independently in the saved fields. Compact recordings, the angle scan
and check results are in `reference/coupled/conversion_animation/`.

## Wide incidence scan and Gaussian pulse

`conda run -n base python coupled_sim/scripts/wide_radar_angle_sweep.py`
extends the magnetic-meridian scan to ±80°. Polarization optimization admits
only propagating incoming modes. Beyond about ±54.2°, neither mode propagates
at this model's homogeneous lower port (plasma frequency 3.313 MHz). Those
angles are recorded as unavailable, not zero RCS. This is a boundary-access
limit for this background, not a universal incidence cutoff.

A 5.34 MHz Gaussian source pulse uses a 10 µs electric-field-envelope FWHM
and the radar-optimal carrier incidence/polarization. Its transient is obtained
by inverse Fourier synthesis of the frequency response of the same discrete
coupled Yee/fluid recurrence, with fixed grid, timestep, and horizontal
wavenumber. Each spectral component is solved independently; a moving envelope
is not imposed on a single continuous-wave solution. Electron-ion and
electron-neutral damping are included; Landau damping is not.

```sh
OPENBLAS_NUM_THREADS=1 conda run -n base python coupled_sim/scripts/gaussian_pulse.py --dz .002
OPENBLAS_NUM_THREADS=1 conda run -n base python coupled_sim/scripts/validate_gaussian_pulse.py --dz .002
OPENBLAS_NUM_THREADS=1 conda run -n base python coupled_sim/scripts/prepare_gaussian_animation.py --dz .002
conda run -n base python coupled_sim/scripts/render_gaussian_animation.py
conda run -n base python coupled_sim/scripts/combine_gaussian_probes.py
```

The single `media/conversion/gaussian_10us_probes.mp4` cycles through three
two-panel views: whole-layer electric field and bottomside density close-up
(−50 to +50 m from the carrier resonance; local electric-field and current
views retain −20 to +20 m),
vertical electromagnetic and plasma-wave energy fluxes, then horizontal and
vertical current. Each view replays the same pulse through 3 ms. Fluxes are carrier-cycle averages, positive
upward. The phase plots use stroboscopic carrier sampling, with the actual
physical time displayed. All color scales are fixed throughout a movie.
HDF5 arrays and numerical checks are retained under `output/gaussian_pulse/`.

The short pulse also produces phase mixing: oscillations excited across a
finite altitude band acquire different phases because the local plasma
frequency varies with height. In the local cold limit,
`k_z(t) = k_z(t_0) - (t-t_0) d omega_pe/dz`.
At 10 m below the carrier resonance, the saved transient gives
`d k_z/dt = -1.649e4 rad/(m s)`, versus `-1.672e4 rad/(m s)` from the
background plasma-frequency gradient (1.4% difference over 150–800 µs).
Consequently, wavelength contraction across the whole excited band is
expected for this pulse and must not be interpreted as a single-frequency
wave packet simply moving downward. Pressure energy flux measures wave
energy transport separately. The optional `prepare_phase_closeup.py` and
`render_gaussian_animation.py --phase` produce a short clip at 24 frames
per actual carrier cycle, without skipped cycles.

## Continuous illumination movies

`prepare_steady_animation.py` uses the article's saved minimum-power harmonic
solution, including its incident-field normalization and polarization.
`render_gaussian_animation.py --cases fields --input output/gaussian_pulse/steady_movie.h5 --stem steady_article`
shows eight periods at 48 frames per period. It displays the sustained phase
motion; it does not simulate arrival or growth from rest.

`prepare_gaussian_animation.py --continuous` instead reconstructs switch-on
from the broadband 5.340 MHz pulse responses. The Gaussian-envelope response
is integrated in the rotating frame to obtain the response to a smooth
error-function source turn-on. This is causal frequency-response synthesis
of the same linear discrete equations, not a long native Rust timestep run.
The initial wave state is zero, while the equilibrium electron density and
magnetic field remain present. The final field is checked against the carrier
harmonic response. The startup grid/carrier are those of the pulse calculation;
the exact article phase movie uses the article's saved grid and carrier.

`validate_continuous_startup.py` checks the zero initial wave state and final
electric-field/density convergence. Use `render_gaussian_animation.py --cases
overview --input output/gaussian_pulse/startup_movie.h5 --stem continuous_startup`
for the switch-on movie. Its source rises around 60 µs; the background density
and static magnetic field are present throughout. Long-duration carrier motion
is stroboscopic; the separate steady article movie resolves every cycle.

`media/conversion/continuous_all_views.mp4` cycles through five two-panel views:
whole-layer field/density, local field/density, vertical energy fluxes,
horizontal/vertical currents, and the article's resolved steady-state phase.
The first four views replay the same causal switch-on calculation. Their time
counter restarts for each replay; the last view uses a separate phase clock.
Chapter markers identify each view. Build it after the startup and steady
movies with:

```sh
for view in fields flux current; do
  conda run -n base python coupled_sim/scripts/render_gaussian_animation.py --cases "$view" --input output/gaussian_pulse/startup_movie.h5 --stem "continuous_startup_$view"
done
conda run -n base python coupled_sim/scripts/combine_continuous_views.py
```
