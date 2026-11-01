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
