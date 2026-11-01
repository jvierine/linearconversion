// Arrays and audit: coupled_sim/scripts/resonant_access_audit.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn rd(h: &File, k: &str) -> Result<Vec<f64>> {
    Ok(h.dataset(k)?.read_raw::<f64>()?)
}
// Use the library's color/dash cycle; legend keys copy the actual paths.
fn series(
    f: &mut Figure,
    p: usize,
    x: &[f64],
    y: &[f64],
    index: usize,
    label: &str,
) -> LegendEntry {
    let line = Line::series(x.iter().zip(y).map(|(&x, &y)| [x, y]).collect(), index);
    let key = line.legend(label);
    f.panels[p].push(Layer::Line(line));
    key
}
fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("coupled_sim/reference/resonant_access_audit.h5".into());
    let h = File::open(path)?;
    let center = h.attr("resonance_height_m")?.read_scalar::<f64>()?;
    let critical = h.attr("critical_height_m")?.read_scalar::<f64>()? - center;
    let ox = rd(&h, "ray_3/horizontal_m")?;
    let oz = rd(&h, "ray_3/height_m")?;
    let j = oz
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0;
    let offset = ox[j];
    let mut f = ArticleLayout::new(177., 10.)?.figure(1., 105.)?;
    let ps = f.grid(1, 2)?;
    f.height_mm = 105.;
    f.panels[ps[0]].rect = Rect::new(22., 7., 70., 84.);
    f.panels[ps[1]].rect = Rect::new(112., 7., 60., 84.);
    let mut ax = Axes::new(
        [-750., 800.],
        [-250., 60.],
        "Horizontal displacement (m)",
        "Height relative to resonance (m)",
    );
    ax.xticks = vec![-500., 0., 500.];
    ax.yticks = vec![-200., -100., 0.];
    f.panels[ps[0]].axes = Some(ax);
    let mut keys = vec![];
    for (index, root, label) in [(0, "ray_3", "O ray"), (1, "ray_2", "X ray")] {
        let x = rd(&h, &format!("{root}/horizontal_m"))?;
        let z = rd(&h, &format!("{root}/height_m"))?;
        let xx: Vec<_> = x.iter().map(|v| v - offset).collect();
        let zz: Vec<_> = z.iter().map(|v| v - center).collect();
        keys.push(series(&mut f, ps[0], &xx, &zz, index, label));
    }
    for (y, index) in [(0., 3), (critical, 4)] {
        series(&mut f, ps[0], &[-750., 800.], &[y, y], index, "");
    }
    let mut leg = Legend::new(keys);
    // Fixed clear corner, checked in the rendered manuscript.
    leg.avoid_data = false;
    leg.position = LegendPosition::LowerRight;
    f.panels[ps[0]].legend = Some(leg);
    let mut ax = Axes::new([0.01, 200.], [-250., 60.], r"$|E_z|/|E_{\rm inc}|$", "");
    ax.set_xscale(AxisScale::Log10)?;
    ax.yticks = vec![-200., -100., 0.];
    f.panels[ps[1]].axes = Some(ax);
    series(
        &mut f,
        ps[1],
        &rd(&h, "Ez_over_incident_peak")?,
        &rd(&h, "Ez_height_relative_m")?,
        0,
        "",
    );
    let mut keys = vec![];
    for (y, index, label) in [(0., 3, "Resonance"), (critical, 4, r"$f_{pe}=f_0$")] {
        keys.push(series(&mut f, ps[1], &[0.01, 200.], &[y, y], index, label));
    }
    let mut leg = Legend::new(keys);
    // Fixed clear corner, checked in the rendered manuscript.
    leg.avoid_data = false;
    leg.position = LegendPosition::LowerRight;
    f.panels[ps[1]].legend = Some(leg);
    f.labels.push(Label::new("(a)", 22., 3.));
    f.labels.push(Label::new("(b)", 112., 3.));
    for ext in ["png", "pdf"] {
        f.save(format!("figures/resonant_access.{ext}"))?;
    }
    Ok(())
}
