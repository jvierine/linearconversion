# Linear conversion beneath sporadic E

Code and HDF5 inputs for the 4 May 2018 Arecibo observations: radar processing,
plasma-line analysis, scientific figures, and the coupled Maxwell–electron-fluid model.

- `radar/legacy_2018/`: original processing scripts recovered from Seid, unchanged (Python 2).
- Root Python scripts, `isr/`, and `analysis/plasma_line_rcs/`: density fitting,
  Debye correction, plasma-line deconvolution and radar cross sections.
- `plotter/`: Rust/rustplotlib figures.
- `coupled_sim/`: coupled solver, pinned FDTD/fluid dependencies, input profiles and physics tests.
- `nonlinear_sim/`: total-density scalar fluid/Maxwell solver and separate tensor comparison.
- `data/`, `evidence/`, and `reference/`: processed observations and compact model products.

## Reproduce the paper figures

Clone with `git clone --recurse-submodules https://github.com/jvierine/linearconversion.git`.

Requires conda base, Rust, HDF5, MPI, and a TeX installation. Private solver
dependencies require GitHub access. Run from this repository root:

```sh
conda run -n base python -m pip install -r requirements.txt
conda run -n base python scripts/fetch_data.py
conda run -n base python scripts/prepare_reference_figures.py
cargo run --release --locked --manifest-path plotter/Cargo.toml --bin opposite_candidate
cargo run --release --locked --manifest-path plotter/Cargo.toml --bin coupled
cargo run --release --locked --manifest-path plotter/Cargo.toml --bin coupled_losses
cargo run --release --locked --manifest-path plotter/Cargo.toml --bin gradient_comparison
cargo run --release --locked --manifest-path plotter/Cargo.toml --bin radar_sensitivity -- reference/coupled/radar_sensitivity.h5
cargo run --release --locked --manifest-path plotter/Cargo.toml --bin es_fregion_electric_field
```

Figures go to `figures/`. These commands redraw archived results. To rerun
simulation timesteps, use the commands in [coupled_sim/README.md](coupled_sim/README.md).
The model includes collisional damping; it does not include Landau damping.

To recompute the angle, polarization, power, and gradient RCS figure, run
`conda run -n base python coupled_sim/scripts/radar_sensitivity_sweeps.py`.
Check its grid, integration-window, and power-scaling results with
`conda run -n base python -m pytest coupled_sim/tests/test_radar_sensitivity.py -q`.

To recompute the measured sporadic-E/F-region field comparison and the two
signed Spitze-window power budgets:

```sh
conda run -n base python coupled_sim/scripts/es_fregion_electric_field.py
conda run -n base python coupled_sim/scripts/spitze_transmission.py
conda run -n base python -m pytest coupled_sim/tests/test_es_fregion_electric_field.py coupled_sim/tests/test_spitze_transmission.py -q
```

Compact HDF5 results are in `reference/coupled/`; reruns retain the complete
field arrays in `coupled_sim/output/coupled/`. Positive incidence follows the
horizontal component of the downward geomagnetic field; the upward radio
window is at negative incidence in this coordinate convention.

## Reproduce the density processing

```sh
conda run -n base python prepare_density_reference.py
conda run -n base python correct_reference_density.py
conda run -n base python audit_debye_density.py
```

The first command checks reproduction of the archived density estimator;
the second applies the finite-Debye response. The plotting code uses
`nipy_spectral` for electron density, UTC and AST axes, and distinct markers.

## Radar processing and experiment configuration

[radar/EXPERIMENT.md](radar/EXPERIMENT.md) lists the pulse sequence and what is
verified versus reconstructed. The legacy pipeline detects transmit pulses,
groups pulse modes, inverts lag products using the measured transmitted
waveform, and forms plasma-line spectra. It requires the original DigitalRF
voltages, Python 2 and the historical dependencies; it is not a Python 3 CLI.
Three large processed HDF5 inputs are stored in the `article-inputs-v1` GitHub
release and downloaded with SHA-256 verification by `scripts/fetch_data.py`.
Raw voltages are not included. The author recovered them on an external drive.
The processed archive is on Seid at
`/scratch/data/juha/arecibo/eregion_pl/processing/`.
Modern extraction scripts retain that source path explicitly.

Code/input checksums and original repository revisions are in `provenance.h5`.
The published paper is maintained separately on Overleaf.

## Independent ISR comparison

The validation-table repository is a pinned Git submodule. Fetch its original
Vierinen and Swoboda calculators, then run the independent spectrum check:

```sh
git submodule update --init --recursive
conda run -n base python scripts/fetch_validation_sources.py
conda run -n base python validate_fit_independent_codes.py
```

## Checks

```sh
conda run -n base python -m pytest tests -q
cargo test --locked --manifest-path coupled_sim/Cargo.toml --test physics
```

The original Seid scripts are an archival source snapshot, not ported Python 3
software. The current analysis scripts use conda base. Auxiliary exploratory
scripts can require intermediate products beyond the figure inputs bundled here;
the explicit workflows above are the supported entry points.

## Coherent radar echo

`isr/coherent.py` implements the retarded coherent-voltage integral in Kudeki's
*ECE 458 Lecture Notes on Applications of Radio Wave Propagation* (2026 revision),
Eqs. (5.49), (5.51), and Sect. 5.3.4. Fields are summed before squaring, with the
transmitted envelope, antenna response, and density propagation delay retained.
The notes are cited in the paper; the private attachment is not redistributed.

Reproduce the saved harmonic-model echo and its 25 MHz, 44 microsecond spectrum:

```sh
conda run -n base python coupled_sim/scripts/coherent_echo.py \
  --model reference/coupled/coherent_echo.h5 --output output/coherent_echo_replay.h5
conda run -n base python -m pytest tests/test_coherent.py -q
cargo run --release --manifest-path plotter/Cargo.toml --bin coherent_echo
```

The reference HDF5 contains the compact simulated density profile, both sideband
amplitudes, complex echo samples, spectrum, and measurement comparison. Amplitude
is fitted under the coherent beam-filling assumption. The effective rectangular
record models the principal spectral response, not the unknown full code/receiver
ambiguity function. This calculation uses the converged linear event model;
the experimental nonlinear solver's spatial-convergence limits remain unchanged.

## Direct-path HF source map

Scan ordinary-mode coupling over incidence and azimuth, then trace magnetoionic
group rays to the ground for the 5.34 MHz enhancement:

```sh
conda run -n base python coupled_sim/scripts/source_location_map.py
conda run -n base python coupled_sim/scripts/prepare_source_map_plot.py
cargo run --release --manifest-path plotter/Cargo.toml --bin source_location_map
conda run -n base python -m pytest coupled_sim/tests/test_source_location_map.py -q
```

The map compares equal incident vertical HF fluxes below the layer. It is not
a probability map or an equal-ground-transmitter-power comparison. The rays
assume horizontal stratification and no earlier ionospheric hop. The HDF5
product is `reference/coupled/source_location_map.h5`.

## Cross-radar detectability

```sh
conda run -n base python coupled_sim/scripts/radar_detectability.py
```

This compares published EISCAT, Millstone Hill, and SYISR hardware at the same
Arecibo-equivalent cross-section, range, and spectral window. Inputs, source
links, SNRs, and an explicit averaging criterion are in
`data/radar_detectability.h5`. It is a hardware-sensitivity benchmark, not a
new prediction of beam-filled plasma-wave scattering at each radar frequency.
