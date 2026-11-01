# 4 May 2018 radar configuration

| Setting | Value | Evidence |
|---|---|---|
| Radar | 430 MHz, vertical, approximately 1 MW peak | Experimenter |
| Complex sampling | 25 MHz; nominal ±12.5 MHz baseband | `lp_an.py`, `lpi_fft_mpi_clp.py` |
| Short / long pulse | 44 / 440 microseconds | Transmit windows in the original decoder |
| Short-pulse IPPs | 1, 1.25, 1.5, 1.75, 2.25, 2.75, 3.25 ms | `find_modes.py`, `intervals.h5` |
| Long-pulse IPP | 10 ms | Same |
| Integration / cycle | Approximately 10 / 80 seconds | `intervals.h5` |
| Modulation | Pseudorandom phase coding | Experimenter recollection; measured-waveform decoder is compatible |
| Bit duration | Assumed 2 microseconds for both modes | Experimenter recollection, not independently recovered from metadata |
| Code length | 22 / 220 bits if 2-microsecond bits | Pulse duration divided by assumed bit duration |
| Feed | 430 MHz Gregorian feed | Identified by the experimenter on 1 November 2026 after checking the September 2017 destruction of the line feed; archived receiver channel is named `gr` |

The decoder uses recorded transmit leakage, not an explicit sequence of bits.
`lpi_fft_mpi_clp.py` decimates 25 MHz voltages to 1 MHz, forms transmit and
receive lag products, and inverts their Fourier-domain normal equations.
Its `rdec=2` is output/lag-product decimation, not proof of 2-microsecond bits.
The unused generic random-code generator in `stuffr.py` also does not identify
the transmitted code. A saved raw transmit pulse would independently establish
bit duration and binary versus polyphase modulation.

Original processing order:
`find_tx_e.py` → `find_modes.py` → `group_modes.py` →
`lpi_fft_mpi.py` / `lpi_fft_mpi_clp.py`; `lp_an.py` processes plasma lines.
Some helper functions and paths are historical leftovers: retain them as evidence,
not as specifications of settings that ran.

Sulzer (1986), *A radar technique for high range resolution incoherent scatter
autocorrelation function measurements utilizing the full average power of
klystron radars*, Radio Science 21, 1033–1040,
https://doi.org/10.1029/RS021i006p01033, describes the coded-long-pulse method.
This is distinct from his seven-frequency phase-modulation paper that year.

The 14 September 2017 proposal, “Collisions: BGK, Brownian, and Hard-Sphere
Approximations Resolved”, by Fentzke, Vierinen and Sulzer, requests a combined
coded-long-pulse and pulse-to-pulse experiment targeting 90–150 km, with
commensal D-region measurements. It does not specify baud duration or feed.

Snapshot retrieved from Seid on 1 November 2026; file hashes are retained in
`provenance.h5`. The original scripts have not been modernized or silently corrected.

Observatory damage report: https://researchinfrastructureoutreach.com/wp-content/uploads/2018/05/ImpactsHurricaneMaria.pdf
