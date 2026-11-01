// Arrays and provenance: coupled_sim/scripts/radar_sensitivity_sweeps.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn rd(h: &File, k: &str) -> Result<Vec<f64>> {
    Ok(h.dataset(k)?.read_raw::<f64>()?)
}
fn db(x: f64) -> f64 {
    10. * x.max(1e-40).log10()
}
fn edges(v: &[f64]) -> Vec<f64> {
    let mut e = vec![v[0] - (v[1] - v[0]) / 2.];
    e.extend(v.windows(2).map(|p| (p[0] + p[1]) / 2.));
    e.push(v[v.len() - 1] + (v[v.len() - 1] - v[v.len() - 2]) / 2.);
    e
}
fn line(f: &mut Figure, p: usize, x: &[f64], y: &[f64], index: usize, label: &str) -> LegendEntry {
    let l = Line::series(x.iter().zip(y).map(|(&x, &y)| [x, y]).collect(), index);
    let key = l.legend(label);
    f.panels[p].push(Layer::Line(l));
    key
}
fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("reference/coupled/radar_sensitivity.h5".into());
    let h = File::open(path)?;
    let obs = db(h.attr("reference_RCS_m2")?.read_scalar::<f64>()?);
    let mut f = ArticleLayout::new(177., 10.)?.figure(1., 156.)?;
    let ps = f.grid(2, 2)?;
    f.height_mm = 156.;
    for (i, &p) in ps.iter().enumerate() {
        f.panels[p].rect = Rect::new(
            if i % 2 == 0 { 22. } else { 112. },
            if i < 2 { 8. } else { 92. },
            60.,
            44.,
        );
        f.labels.push(Label::new(
            format!("({})", ['a', 'b', 'c', 'd'][i]),
            if i % 2 == 0 { 22. } else { 112. },
            if i < 2 { 3. } else { 87. },
        ));
    }
    let p = ps[0];
    let mut ax = Axes::new(
        [-25., 25.],
        [-125., -55.],
        r"Incidence angle ($^\circ$)",
        r"RCS (dB m$^2$)",
    );
    ax.xticks = vec![-20., 0., 20.];
    ax.yticks = vec![-120., -100., -80.];
    f.panels[p].axes = Some(ax);
    let model = line(
        &mut f,
        p,
        &rd(&h, "angle/degrees")?,
        &rd(&h, "angle/RCS_m2")?
            .into_iter()
            .map(db)
            .collect::<Vec<_>>(),
        0,
        "Model",
    );
    let observed = line(&mut f, p, &[-25., 25.], &[obs, obs], 4, "Observed");
    f.panels[p].legend = Some(Legend::new(vec![model, observed]));
    // Cold-plasma Spitze geometry from the same field and pump frequency.
    let spitze_file = File::open("reference/coupled/spitze_transmission.h5")?;
    let spitze = spitze_file.attr("spitze_angle_magnitude_deg")?.read_scalar::<f64>()?;
    for angle in [-spitze, spitze] {
        line(&mut f, p, &[angle, angle], &[-125., -55.], 3, "");
        let mut label = Label::new(
            if angle < 0. { r"$-16.7^\circ$" } else { r"$+16.7^\circ$" },
            if angle < 0. { 37. } else { 66. }, 48.);
        label.leader = Some([22. + (angle + 25.) / 50. * 60., 44.]);
        f.labels.push(label);
    }
    f.labels.push(Label::new("Spitze", 52., 31.));
    let p = ps[1];
    let mut ax = Axes::new(
        [0., 90.],
        [-180., 180.],
        r"$\tan^{-1}|E_y/E_x|$ ($^\circ$)",
        r"$\arg(E_y/E_x)$ ($^\circ$)",
    );
    ax.xticks = vec![0., 45., 90.];
    ax.yticks = vec![-180., 0., 180.];
    f.panels[p].axes = Some(ax);
    let sc = f.scale(Scale::new(
        [-125., -75.],
        Normalization::Linear,
        ColorMap::ember(),
    )?);
    f.panels[p].push(Layer::Pcolormesh(Field::new(
        edges(&rd(&h, "polarization/amplitude_angle_deg")?),
        edges(&rd(&h, "polarization/relative_phase_deg")?),
        rd(&h, "polarization/RCS_m2")?.into_iter().map(db).collect(),
        sc,
    )?));
    let opt = rd(&h, "polarization/optimum")?;
    for points in [
        vec![[opt[0] - 3., opt[1]], [opt[0] + 3., opt[1]]],
        vec![[opt[0], opt[1] - 15.], [opt[0], opt[1] + 15.]],
    ] {
        f.panels[p].push(Layer::Line(Line::new(points, "#ffffff")));
    }
    // O/X coordinates are derived from saved incident cold eigenvectors.
    // The complete edge is a single linear state: phase is undefined there.
    let pol = File::open("data/polarization_annotations.h5")?;
    let ordinary = rd(&pol, "O")?;
    let extraordinary = rd(&pol, "X")?;
    let xy = |a: f64, phase: f64| [112. + a / 90. * 60., 8. + (180. - phase) / 360. * 44.];
    for (point, text, position) in [
        ([ordinary[0], ordinary[1]], "O", [126., 15.]),
        ([extraordinary[0], extraordinary[1]], "X", [161., 45.]),
        ([45., 0.], r"$+45^\circ$ linear", [142., 35.5]),
        ([0., 0.], r"$E_x$ linear", [123., 26.]),
        ([90., 0.], r"$E_y$ linear", [161., 26.]),
    ] {
        let anchor = xy(point[0], point[1]);
        if point[0] > 0. && point[0] < 90. {
            let circle = (0..=48).map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 48.;
                [point[0] + 1.0 * a.cos(), point[1] + 5.5 * a.sin()]
            }).collect();
            f.panels[p].push(Layer::Line(Line::new(circle, "#171717")));
        }
        let mut label = Label::new(text, position[0], position[1]);
        label.leader = Some(anchor);
        f.labels.push(label);
    }
    let mut optimum_label = Label::new("Optimum", 160., 14.);
    optimum_label.leader = Some(xy(opt[0], opt[1]));
    f.labels.push(optimum_label);
    let mut cb = Colorbar::new(
        Rect::new(112., 70., 60., 2.5),
        sc,
        r"RCS (dB m$^2$)",
        vec![-120., -100., -80.],
    );
    cb.horizontal = true;
    f.colorbars.push(cb);
    f.auto_colorbar_layout = false;
    let p = ps[2];
    let mut ax = Axes::new([1e-3, 1e4], [-120., -40.], "HF EIRP (W)", r"RCS (dB m$^2$)");
    ax.set_xscale(AxisScale::Log10)?;
    ax.xticks = vec![1e-3, 1., 1e3];
    ax.xticklabels = Some(vec![r"$10^{-3}$".into(), "1".into(), r"$10^3$".into()]);
    ax.yticks = vec![-120., -100., -80., -60., -40.];
    f.panels[p].axes = Some(ax);
    let x = rd(&h, "power/EIRP_W")?;
    let mut keys = vec![];
    for (a, index) in [(0, 1), (14, 0), (25, 2)] {
        keys.push(line(
            &mut f,
            p,
            &x,
            &rd(&h, &format!("power/RCS_{a}_m2"))?
                .into_iter()
                .map(db)
                .collect::<Vec<_>>(),
            index,
            &format!(r"${a}^\circ$"),
        ));
    }
    line(&mut f, p, &[1e-3, 1e4], &[obs, obs], 4, "");
    f.panels[p].legend = Some(Legend::new(keys));
    let p = ps[3];
    let mut ax = Axes::new(
        [0.4, 5.1],
        [-200., -5.],
        r"Density scale length $L_n$ (km)",
        r"RCS (dB m$^2$)",
    );
    ax.xticks = vec![1., 2., 3., 4., 5.];
    ax.yticks = vec![-200., -160., -120., -80.];
    f.panels[p].axes = Some(ax);
    let tab = rd(&h, "gradient/table")?;
    let x: Vec<_> = tab.chunks(4).map(|r| r[0]).collect();
    let y: Vec<_> = tab.chunks(4).map(|r| db(r[1])).collect();
    let sweep = line(&mut f, p, &x, &y, 0, "Gradient sweep");
    line(&mut f, p, &[0.4, 5.1], &[obs, obs], 4, "");
    let ln = h.attr("reference_Ln_m")?.read_scalar::<f64>()? / 1000.;
    let measured = line(&mut f, p, &[ln, ln], &[-200., -5.], 3, r"Measured $L_n$");
    f.panels[p].legend = Some(Legend::new(vec![sweep, measured]));
    for (i, &p) in ps.iter().enumerate() {
        if let Some(l) = f.panels[p].legend.as_mut() {
            // Fixed clear corners, checked in the rendered manuscript.
            l.avoid_data = false;
            l.position = if i == 3 {
                LegendPosition::UpperRight
            } else {
                LegendPosition::UpperLeft
            };
        }
    }
    for ext in ["pdf", "png"] {
        f.save(format!("figures/radar_sensitivity.{ext}"))?;
    }
    Ok(())
}
