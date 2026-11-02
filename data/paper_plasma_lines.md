# Figure 3 plasma-line spectra

The HDF5 contains the native excess/noise spectra and filtered displays for 14:52:31 UTC (ordinary profile) and 17:03:50 UTC (enhancements), 4 May 2018. It also stores the measured timestamps, source hashes, background estimates, and the existing ion-line/model frequency overlays.

Display filter: subtract 85% of a Gaussian frequency background with standard deviation 100 kHz, in linear excess/noise units. There is no altitude or time smoothing. Both article panels use -15 to +5 dB; nonpositive and weaker residuals take the floor color. Do not use the filtered arrays for quantitative power or SNR inference.

Reproduce with the local native movie extraction available:

```sh
conda run -n base python scripts/prepare_paper_plasma_lines.py
cargo run --release --locked --manifest-path plotter/Cargo.toml --bin paper_plasma_lines
```

The committed HDF5 is sufficient to render the figure without the larger native extraction. The renderer uses Rustplotlib ArticleLayout at 177 mm manuscript width and 10 pt body font. Copy the generated PDF and include wrapper to the manuscript's figures/model directory.
