# Plasma-line event animations

- `enhanced_plasma_line_events_1fps.gif`: the 40 retained detection records, one per second.
- `plasma_line_events_compact_gentle_filter.mp4`: the current, full-frame version with inset axes, UTC/AST datetime, and colorbar.
- `plasma_line_events_fast_scan_highpass.mp4`: the previous version with stronger filtering and outside axes.

Both MP4s show all 1,407 records in chronological order, at 30 output frames/s. Quiet intervals advance at 30 records/s, nearby intervals at 5 records/s, and detections at 2 records/s. The strongest RCS record in each of eight episodes is held for three seconds. Detections separated by more than 90 seconds start a new episode.

The current MP4 subtracts 85% of a Gaussian-smoothed frequency background (standard deviation 100 kHz) from linear excess power/noise before logarithmic display. This gentler subtraction preserves more broad emission. The previous version subtracted the full background with a 100 kHz standard deviation. It does not filter across altitude or time. The current fixed display range is -20 to 10 dB relative to noise (previously -15 to 20 dB); negative and weaker residuals use the lowest color. This display filter is not deconvolution or a quantitative power correction. The original spectra and catalogue remain unchanged.

The record timing and filter settings are in `data/plasma_line_movie_compact_timing.h5` (current) and `data/plasma_line_movie_timing.h5` (previous).

To reproduce from the native `results4` spectra:

```sh
conda run -n base python scripts/extract_event_movie.py --root /path/to/results4 --output data/plasma_line_movie_native.h5
cargo run --release --locked --manifest-path plotter/Cargo.toml --bin event_movie_template
conda run -n base python scripts/make_event_movie.py
```

The native movie input is a local 578 MiB extraction, not committed to Git. The color map and font convention follow Rustplotlib; labels and the colorbar are inset during frame composition. UTC and AST timestamps are actual observation times, while playback speed varies as described above.
