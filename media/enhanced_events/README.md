# Plasma-line event animations

- `enhanced_plasma_line_events_1fps.gif`: the 40 retained detection records, one per second.
- `plasma_line_events_fast_scan_highpass.mp4`: all 1,407 records in chronological order, at 30 output frames/s. Quiet intervals advance at 30 records/s, nearby intervals at 5 records/s, and detections at 2 records/s. The strongest RCS record in each of eight episodes is held for three seconds. Detections separated by more than 90 seconds start a new episode.

The MP4 subtracts a Gaussian-smoothed frequency background (standard deviation 100 kHz) from linear excess power/noise before logarithmic display. It does not filter across altitude or time. The fixed display range is -15 to 20 dB relative to noise; negative and weaker residuals use the lowest color. This display filter is not deconvolution or a quantitative power correction. The original spectra and catalogue remain unchanged.

The record timing and filter settings are in `data/plasma_line_movie_timing.h5`.

To reproduce from the native `results4` spectra:

```sh
conda run -n base python scripts/extract_event_movie.py --root /path/to/results4 --output data/plasma_line_movie_native.h5
cargo run --release --locked --manifest-path plotter/Cargo.toml --bin event_movie_template
conda run -n base python scripts/make_event_movie.py
```

The native movie input is a local 578 MiB extraction, not committed to Git. Axes and color map use Rustplotlib; static graphics are cached during frame composition. UTC and AST timestamps are actual observation times, while playback speed varies as described above.
