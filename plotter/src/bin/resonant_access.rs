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
    // The inset deliberately places its ticks inside the parent axes;
    // inspect the rendered page for overlap with actual ray curves.
    f.check_label_geometry = false;
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
        // Arrowheads follow the stored ray order (incident then reflected).
        // Size in page units, so unequal axis scales do not distort the heads.
        let turn = zz.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap().0;
        let target = if index == 0 { -100. } else { -220. };
        for range in [1..turn, turn + 1..zz.len() - 1] {
            let k = range.min_by(|&a, &b| (zz[a] - target).abs().total_cmp(&(zz[b] - target).abs())).unwrap();
            let (sx, sy) = (70. / 1550., 84. / 310.);
            let (dx, dy) = ((xx[k + 1] - xx[k - 1]) * sx, (zz[k + 1] - zz[k - 1]) * sy);
            let norm = dx.hypot(dy);
            let (ux, uy) = (dx / norm, dy / norm);
            let tip = [xx[k], zz[k]];
            let left = [tip[0] - (2.5 * ux + 1. * uy) / sx, tip[1] - (2.5 * uy - 1. * ux) / sy];
            let right = [tip[0] - (2.5 * ux - 1. * uy) / sx, tip[1] - (2.5 * uy + 1. * ux) / sy];
            let mut head = Line::series(vec![left, tip, right], index);
            head.dash = None;
            head.width_pt = 1.4;
            f.panels[ps[0]].push(Layer::Line(head));
        }
    }
    for (y, index) in [(0., 3), (critical, 4)] {
        series(&mut f, ps[0], &[-750., 800.], &[y, y], index, "");
    }
    // Bx>0, Bz<0. Use the event input and data-coordinate direction;
    // the unequal horizontal/vertical axis scales also apply to this vector.
    let background = File::open("coupled_sim/data/background_prior.h5")?;
    let tilt = background.dataset("field_angle_deg")?.read_scalar::<f64>()?.to_radians();
    let start = [85., 10.];
    let end = [start[0] + 90. * tilt.sin(), start[1] - 90. * tilt.cos()];
    let mut field = Line::new(vec![start, end], "#171717");
    field.width_pt = 1.2;
    f.panels[ps[0]].push(Layer::Line(field));
    // Construct the arrowhead in page units so it remains legible.
    let sx = 70. / 1550.;
    let sy = 84. / 310.;
    let dx = (end[0] - start[0]) * sx;
    let dy = (end[1] - start[1]) * sy;
    let length = dx.hypot(dy);
    let (ux, uy) = (dx / length, dy / length);
    let left = [end[0] - (2. * ux + 0.8 * uy) / sx,
                end[1] - (2. * uy - 0.8 * ux) / sy];
    let right = [end[0] - (2. * ux - 0.8 * uy) / sx,
                 end[1] - (2. * uy + 0.8 * ux) / sy];
    let mut head = Line::new(vec![left, end, right], "#171717");
    head.width_pt = 1.2;
    f.panels[ps[0]].push(Layer::Line(head));
    f.panels[ps[0]].label_at(r"$\mathbf B_0$", [285., -20.]);
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
    // Close-up uses the original ordered ray samples, not an interpolated cartoon.
    let mut zoom_axes = Axes::new([-30., 10.], [-3., 23.], "x (m)", "");
    zoom_axes.xticks = vec![-20., 0.];
    zoom_axes.yticks = vec![0., 20.];
    let zoom = f.panels.len();
    f.panels.push(Panel::new(Rect::new(65., 40., 24., 21.), zoom_axes));
    let zx: Vec<_> = ox.iter().map(|v| v - offset).collect();
    let zz: Vec<_> = oz.iter().map(|v| v - center).collect();
    series(&mut f, zoom, &zx, &zz, 0, "");
    for (y, index) in [(0., 3), (critical, 4)] {
        series(&mut f, zoom, &[-30., 10.], &[y, y], index, "");
    }
    // Incident O-ray polarization at its upward resonance crossing.
    let polarization = File::open("data/resonant_access_polarization.h5")?;
    let anchor = rd(&polarization, "anchor_xz_m")?;
    let direction = rd(&polarization, "unit_projection_xz")?;
    let tip = [anchor[0] + 7.5 * direction[0], anchor[1] + 7.5 * direction[1]];
    let mut electric = Line::series(vec![[anchor[0], anchor[1]], tip], 2);
    electric.dash = None;
    electric.width_pt = 1.3;
    f.panels[zoom].push(Layer::Line(electric));
    let (sx, sy) = (24. / 40., 21. / 26.);
    let (dx, dy) = ((tip[0] - anchor[0]) * sx, (tip[1] - anchor[1]) * sy);
    let norm = dx.hypot(dy);
    let (ux, uy) = (dx / norm, dy / norm);
    let left = [tip[0] - (1.3 * ux + 0.65 * uy) / sx, tip[1] - (1.3 * uy - 0.65 * ux) / sy];
    let right = [tip[0] - (1.3 * ux - 0.65 * uy) / sx, tip[1] - (1.3 * uy + 0.65 * ux) / sy];
    let mut head = Line::series(vec![left, tip, right], 2);
    head.dash = None;
    head.width_pt = 1.3;
    f.panels[zoom].push(Layer::Line(head));
    f.panels[zoom].label_at(r"$\mathbf E$", [-24., 9.]);
    // Leader links the actual turning point to the inset without covering B0.
    let mut leader = Line::new(vec![[5., critical + 4.], [600., 40.], [600., -60.]], "#777777");
    leader.width_pt = 0.6;
    f.panels[ps[0]].push(Layer::Line(leader));
    f.labels.push(Label::new("(a)", 22., 3.));
    f.labels.push(Label::new("(b)", 112., 3.));
    for ext in ["png", "pdf"] {
        f.save(format!("figures/resonant_access.{ext}"))?;
    }
    Ok(())
}
