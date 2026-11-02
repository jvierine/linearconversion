// Scientific arrays and provenance: data/altitude_sweep.h5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn read(h: &File, k: &str) -> Result<Vec<f64>> {
    Ok(h.dataset(k)?.read_raw::<f64>()?)
}
fn line(
    f: &mut Figure,
    p: usize,
    points: Vec<[f64; 2]>,
    index: usize,
    label: &str,
    dash: bool,
) -> LegendEntry {
    let mut l = Line::series(points, index);
    if dash {
        l.dash = Some("4 2".into());
    }
    let key = l.legend(label);
    f.panels[p].push(Layer::Line(l));
    key
}
fn main() -> Result<()> {
    let h = File::open("data/altitude_sweep.h5")?;
    let table = read(&h, "sweep/table")?;
    let rows: Vec<_> = table.chunks_exact(9).collect();
    let event = h.attr("reference_center_m")?.read_scalar::<f64>()? / 1000.;
    let mut f = ArticleLayout::new(177., 10.)?.figure(1., 111.)?;
    let ps = f.grid(1, 2)?;
    f.height_mm = 111.;
    for (i, &p) in ps.iter().enumerate() {
        f.panels[p].rect = Rect::new(if i == 0 { 20. } else { 109. }, 9., 64., 75.);
        f.labels.push(Label::new(
            if i == 0 { "(a)" } else { "(b)" },
            if i == 0 { 20. } else { 109. },
            3.,
        ));
    }
    let p = ps[0];
    let mut ax = Axes::new(
        [-160., -55.],
        [90., 150.],
        r"RCS (dB m$^2$)",
        "Resonance altitude (km)",
    );
    ax.xticks = vec![-150., -125., -100., -75.];
    ax.yticks = vec![90., 105., 120., 135., 150.];
    f.panels[p].axes = Some(ax);
    let mut marker = Line::new(vec![[-160., event], [-55., event]], "#999999");
    marker.dash = Some("3 3".into());
    marker.width_pt = 0.6;
    f.panels[p].push(Layer::Line(marker));
    line(
        &mut f,
        p,
        rows.iter()
            .map(|r| [10. * r[1].log10(), r[0] / 1000.])
            .collect(),
        0,
        "MSIS collisions",
        false,
    );
    let mut arrow = Line::new(
        vec![
            [-146., 98.],
            [-159., 98.],
            [-156., 99.],
            [-159., 98.],
            [-156., 97.],
        ],
        "#0072b2",
    );
    arrow.width_pt = 0.8;
    f.panels[p].push(Layer::Line(arrow));
    f.panels[p].label_at("Below scale", [-132., 98.]);
    let p = ps[1];
    let mut ax = Axes::new([1e2, 1e6], [90., 150.], r"Collision rate (s$^{-1}$)", "");
    ax.set_xscale(AxisScale::Log10)?;
    ax.xticks = vec![1e2, 1e3, 1e4, 1e5, 1e6];
    ax.xticklabels = Some(vec![
        r"$10^2$".into(),
        r"$10^3$".into(),
        r"$10^4$".into(),
        r"$10^5$".into(),
        r"$10^6$".into(),
    ]);
    ax.yticks = vec![90., 105., 120., 135., 150.];
    f.panels[p].axes = Some(ax);
    let mut marker = Line::new(vec![[1e2, event], [1e6, event]], "#999999");
    marker.dash = Some("3 3".into());
    marker.width_pt = 0.6;
    f.panels[p].push(Layer::Line(marker));
    let z = read(&h, "collision_profile/height_m")?;
    let mut keys = vec![];
    for (name, label, i, dash) in [
        ("nu_total_s1", r"$\nu_{en}+\nu_{ei}$", 0, false),
        ("nu_en_s1", r"$\nu_{en}$", 1, true),
        ("nu_ei_s1", r"$\nu_{ei}$", 2, true),
    ] {
        let nu = read(&h, &format!("collision_profile/{name}"))?;
        keys.push(line(
            &mut f,
            p,
            nu.iter().zip(&z).map(|(&v, &z)| [v, z / 1000.]).collect(),
            i,
            label,
            dash,
        ));
    }
    f.panels[p].legend = Some(Legend::new(keys));
    f.save_for_article(
        "figures/altitude_sweep.pdf",
        "figures/model/altitude_sweep.pdf",
    )?;
    f.save("figures/altitude_sweep.png")?;
    Ok(())
}
