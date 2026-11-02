# Altitude sensitivity of linear conversion

The measured smooth sporadic-E density profile is translated so its 5.340 MHz plasma-wave resonance lies at 27 heights between 90 and 150 km. The gradient scale length remains 1.028 km, Te remains 300 K, and the event magnetic field, incidence angle (13.9558 degrees), incident tangential polarization and local upward incident flux (8.433e-12 W/m2) are held fixed.

MSIS2.1 neutral N2, O2 and O densities are interpolated at the observation time and each new model altitude. Electron-neutral momentum-transfer rates use these densities and Te. Electron-ion collisions use the event model's Coulomb formula with local ne and Te; MSIS does not supply electron-ion collisions. At the same point within the shifted layer, ne and Te are unchanged, so nu_ei is unchanged. Tn is retained in HDF5 but is not substituted for Te.

The additional controls translate the event collision profile with the layer instead of recomputing the neutral collision rates. They isolate the small radar geometry change from the much larger collisional suppression. Illumination is normalized just below each shifted layer; this test does not include propagation from a ground transmitter.

`altitude_sweep.h5` stores the sweep, controls, mesh checks, collision curves and representative near-resonance density/E-field phasors. `table_columns` names the table fields. All unabridged local solves are retained under `reference/coupled/altitude_sweep/` and can be regenerated. `altitude_sweep_endpoint_check.h5` independently changes both integration endpoints. The tiny 98 km Fourier residual is taper-dependent; it establishes non-detectability, not an accurate cross section of order -240 dB m2. Values below -80 dB SNR are flagged and no required flux is assigned to them. The plotted RCS range stops at -160 dB m2.

The original event response is recovered within 0.3%; selected refinements change RCS by less than 0.02 dB. Tests also check the discrete energy balance and stability to moving both Fourier-integration endpoints for resolved responses.

```sh
conda run -n base python coupled_sim/scripts/altitude_sweep.py
conda run -n base python coupled_sim/scripts/check_altitude_sweep.py
conda run -n base python -m pytest -q coupled_sim/tests/test_altitude_sweep.py
cargo run --release --locked --manifest-path plotter/Cargo.toml --bin altitude_sweep
```

These are prescribed translated Es layers, not observations of dense layers at every altitude. The comparison uses the established linear collisional warm-fluid/Maxwell recurrence and excludes kinetic Landau damping.
