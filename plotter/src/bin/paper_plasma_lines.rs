// Figure 3. Native spectra and complete display-filter provenance: data/paper_plasma_lines.h5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn read(f: &File, k: &str) -> Result<Vec<f64>> {
    Ok(f.dataset(k)?.read_raw::<f64>()?)
}
fn edges(v: &[f64]) -> Vec<f64> {
    let mut e = vec![v[0] - (v[1] - v[0]) / 2.];
    e.extend(v.windows(2).map(|a| (a[0] + a[1]) / 2.));
    e.push(v[v.len() - 1] + (v[v.len() - 1] - v[v.len() - 2]) / 2.);
    e
}
fn arrow(f: &mut Figure, p: usize, start: [f64; 2], end: [f64; 2]) {
    let r = f.panels[p].rect;
    let sx = r.width / 4.25;
    let sy = r.height / 40.;
    let dx = (end[0] - start[0]) * sx;
    let dy = (end[1] - start[1]) * sy;
    let l = dx.hypot(dy);
    let ux = dx / l;
    let uy = dy / l;
    let tip = [end[0] - ux / sx, end[1] - uy / sy];
    let a = [
        tip[0] - (1.5 * ux + 0.6 * uy) / sx,
        tip[1] - (1.5 * uy - 0.6 * ux) / sy,
    ];
    let b = [
        tip[0] - (1.5 * ux - 0.6 * uy) / sx,
        tip[1] - (1.5 * uy + 0.6 * ux) / sy,
    ];
    let mut line = Line::new(vec![start, tip, a, tip, b], "#ffffff");
    line.width_pt = 0.65;
    f.panels[p].push(Layer::Line(line));
}
fn main() -> Result<()> {
    let h = File::open("data/paper_plasma_lines.h5")?;
    let x = read(&h, "frequency_mhz")?;
    let y = read(&h, "height_km")?;
    let mut fig = ArticleLayout::new(177., 10.)?.figure(1., 124.)?;
    let panels = fig.grid(1, 2)?;
    fig.height_mm = 124.;
    let sc = fig.scale(Scale::new(
        [-15., 5.],
        Normalization::Linear,
        ColorMap::ember(),
    )?);
    for (j, name) in ["enhanced", "ordinary"].iter().enumerate() {
        let p = panels[j];
        fig.panels[p].rect = Rect::new(if j == 0 { 17. } else { 103. }, 10., 70., 84.);
        let mut ax = Axes::new(
            [2., 6.25],
            [100., 140.],
            "Frequency (MHz)",
            if j == 0 {
                "Altitude above radar (km)"
            } else {
                ""
            },
        );
        ax.xticks = vec![2., 3., 4., 5., 6.];
        ax.yticks = vec![100., 110., 120., 130., 140.];
        fig.panels[p].axes = Some(ax);
        let values = read(&h, &format!("{name}/filtered_excess_snr"))?
            .iter()
            .map(|v| 10. * v.max(10_f64.powf(-1.5)).log10())
            .collect();
        fig.panels[p].push(Layer::Pcolormesh(Field::new(
            edges(&x),
            edges(&y),
            values,
            sc,
        )?));
        fig.labels.push(Label::new(
            if j == 0 {
                "(a) 17:03:50 UTC"
            } else {
                "(b) 14:52:31 UTC"
            },
            if j == 0 { 17. } else { 103. },
            3.,
        ));
    }
    let p = panels[0];
    for (prefix, color) in [("ion", "#cbd3df"), ("model", "#57e0dc")] {
        let xx = read(&h, &format!("{prefix}_plasma_line_mhz"))?;
        let yy = read(&h, &format!("{prefix}_height_km"))?;
        // Subdivide only plotting segments to create gaps at the observed narrow peaks.
        let mut pieces: Vec<Vec<[f64; 2]>> = vec![];
        let mut line = vec![];
        for i in 0..xx.len() - 1 {
            let n = ((xx[i + 1] - xx[i]).abs() / 0.005)
                .max((yy[i + 1] - yy[i]).abs() / 0.01)
                .ceil()
                .max(1.) as usize;
            for k in 0..n {
                let t = k as f64 / n as f64;
                let x = xx[i] + t * (xx[i + 1] - xx[i]);
                let z = yy[i] + t * (yy[i + 1] - yy[i]);
                let hide = z < 100.
                    || z > 140.
                    || (prefix == "model" && z > 120.)
                    || ((117.2..=118.7).contains(&z)
                        && ((4.40..=4.62).contains(&x) || (5.20..=5.46).contains(&x)));
                if hide {
                    if line.len() > 1 {
                        pieces.push(std::mem::take(&mut line));
                    } else {
                        line.clear();
                    }
                } else {
                    line.push([x, z]);
                }
            }
        }
        if line.len() > 1 {
            pieces.push(line);
        }
        for points in pieces {
            // Split white overlay at 120 km so ordinary plasma-line pixels remain visible.
            for upper in [false, true] {
                let pts: Vec<_> = points
                    .iter()
                    .copied()
                    .filter(|a| if upper { a[1] >= 120. } else { a[1] <= 120. })
                    .collect();
                if pts.len() > 1 {
                    let mut l = Line::new(pts, if upper { "#ffffff" } else { color });
                    l.width_pt = 0.6;
                    l.opacity = if upper { 0.4 } else { 1. };
                    fig.panels[p].push(Layer::Line(l));
                }
            }
        }
    }
    let mut legend = Legend::new(vec![
        LegendEntry::line("Ion-line fit", "#cbd3df"),
        LegendEntry::line("Smooth model", "#57e0dc"),
    ]);
    legend.position = LegendPosition::LowerRight;
    legend.avoid_data = true;
    fig.panels[p].legend = Some(legend);
    for (label, start, end) in [
        ("4.51 MHz", [4.40, 111.5], [4.5166, 117.75]),
        ("5.34 MHz", [5.70, 113.0], [5.3345, 117.75]),
    ] {
        arrow(&mut fig, p, start, end);
        fig.panels[p]
            .label_at(label, [start[0], start[1] - 1.])
            .color = "#ffffff".into();
    }
    fig.panels[p]
        .label_at("Doppler sidelobes", [5.13, 131.])
        .color = "#ffffff".into();
    for (start, end) in [
        ([4.9, 129.5], [5.32, 122.]),
        ([5.6, 129.5], [5.835, 117.75]),
    ] {
        arrow(&mut fig, p, start, end);
    }
    fig.panels[p].legend.as_mut().unwrap().avoid_layers = (1..fig.panels[p].layers.len()).collect();
    let mut cb = Colorbar::new(
        Rect::new(38., 108., 117., 3.),
        sc,
        "Filtered excess / noise (dB)",
        vec![-15., -10., -5., 0., 5.],
    );
    cb.horizontal = true;
    fig.colorbars.push(cb);
    fig.auto_colorbar_layout = false;
    fig.save_for_article(
        "figures/observed_plasma_lines.pdf",
        "figures/model/observed_plasma_lines.pdf",
    )?;
    fig.save("figures/observed_plasma_lines.png")?;
    Ok(())
}
