// Data and extraction: nonlinear_sim/scripts/diagnose_scalar_total.py.
// All density-wave quantities are derived from saved total electron density.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
const BLUE: &str = "#245774";
const RED: &str = "#8b3940";
fn read(h: &File, n: &str) -> Result<Vec<f64>> {
    Ok(h.dataset(n)?.read_raw()?)
}
fn line(f: &mut Figure, p: usize, x: &[f64], y: &[f64], color: &str, dashed: bool) {
    let mut l = Line::new(x.iter().zip(y).map(|(&x, &y)| [x, y]).collect(), color);
    if dashed {
        l.dash = Some("4 3".into());
    }
    f.panels[p].push(Layer::Line(l));
}
fn main() -> Result<()> {
    let label = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "scalar_switch_on".into());
    let h = File::open(format!("nonlinear_sim/reference/{label}_diagnostics.h5"))?;
    let z = read(&h, "z_m")?;
    let neq = read(&h, "equilibrium_density_m3")?;
    let total = read(&h, "final_total_density_x0_full_profile_m3")?;
    let zr = h.attr("resonance_m")?.read_scalar::<f64>()?;
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 260.;
    f.height_mm = 120.;
    let ps = f.grid(1, 3)?;
    f.height_mm = 120.;
    for (i, &p) in ps.iter().enumerate() {
        f.panels[p].rect = Rect::new(24. + 86. * i as f64, 7., 59., 88.);
    }
    let p = ps[0];
    f.panels[p].axes = Some(Axes::new(
        [1., 4.5],
        [117.4, 118.5],
        r"$n_e$ ($10^{11}$ m$^{-3}$)",
        "Altitude (km)",
    ));
    let altitude = z.iter().map(|v| v / 1000.).collect::<Vec<_>>();
    line(
        &mut f,
        p,
        &neq.iter().map(|v| v / 1e11).collect::<Vec<_>>(),
        &altitude,
        RED,
        true,
    );
    line(
        &mut f,
        p,
        &total.iter().map(|v| v / 1e11).collect::<Vec<_>>(),
        &altitude,
        BLUE,
        false,
    );
    // Weak perturbations overlap at this scale; the adjacent panel resolves them.
    f.panels[p].legend = Some(Legend::new(vec![
        LegendEntry::line("Equilibrium", RED),
        LegendEntry::line("Total density", BLUE),
    ]));
    let height = read(&h, "height_relative_m")?;
    let nt = read(&h, "time_s")?.len();
    let nlocal = height.len();
    let delta = read(&h, "delta_density_x0_m3")?;
    let local_n = neq
        .iter()
        .zip(&z)
        .filter(|(_, z)| **z > zr - 10. && **z < zr + 5.)
        .map(|(n, _)| *n)
        .collect::<Vec<_>>();
    let scale = if read(&h, "peak_fractional_wave")?
        .iter()
        .copied()
        .fold(0., f64::max)
        < 1e-7
    {
        1e9
    } else {
        1e6
    };
    let unit = if scale == 1e9 { "ppb" } else { "ppm" };
    let frac = delta[(nt - 1) * nlocal..nt * nlocal]
        .iter()
        .zip(&local_n)
        .map(|(v, n)| scale * v / n)
        .collect::<Vec<_>>();
    let max = frac.iter().map(|v| v.abs()).fold(0., f64::max).max(1e-6) * 1.1;
    let p = ps[1];
    f.panels[p].axes = Some(Axes::new(
        [-max, max],
        [-10., 5.],
        &format!(r"$(n_e-n_{{\rm eq}})/n_{{\rm eq}}$ ({unit})"),
        "Height relative to resonance (m)",
    ));
    line(&mut f, p, &frac, &height, BLUE, false);
    let t = read(&h, "time_s")?
        .iter()
        .map(|v| v * 1e6)
        .collect::<Vec<_>>();
    let amp = read(&h, "peak_fractional_wave")?
        .iter()
        .map(|v| v * scale)
        .collect::<Vec<_>>();
    let p = ps[2];
    let ymax = amp.iter().copied().fold(0., f64::max) * 1.08;
    f.panels[p].axes = Some(Axes::new(
        [0., *t.last().unwrap()],
        [0., ymax],
        r"Time ($\mu$s)",
        &format!("Peak density-wave amplitude ({unit})"),
    ));
    line(&mut f, p, &t, &amp, BLUE, false);
    std::fs::create_dir_all("figures")?;
    f.save(format!("figures/{label}_total_density.png"))?;
    Ok(())
}
