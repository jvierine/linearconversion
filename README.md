# Linear conversion beneath sporadic E

Code and HDF5 inputs for the 4 May 2018 Arecibo observations: radar processing,
plasma-line analysis, scientific figures, and the coupled Maxwell–electron-fluid model.

- `radar/legacy_2018/`: original processing scripts recovered from Seid, unchanged (Python 2).
- Root Python scripts, `isr/`, and `analysis/plasma_line_rcs/`: density fitting,
  Debye correction, plasma-line deconvolution and radar cross sections.
- `plotter/`: Rust/rustplotlib figures.
- `coupled_sim/`: coupled solver, pinned FDTD/fluid dependencies, input profiles and physics tests.
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
```

Figures go to `figures/`. These commands redraw archived results. To rerun
simulation timesteps, use the commands in [coupled_sim/README.md](coupled_sim/README.md).
The model includes collisional damping; it does not include Landau damping.

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
