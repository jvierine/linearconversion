// Native coupled samples prepared by coupled_sim/scripts/prepare_conversion_animation.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn rd(h: &File, k: &str) -> Result<Vec<f64>> {
    Ok(h.dataset(k)?.read_raw::<f64>()?)
}
fn att(h: &File, k: &str) -> Result<f64> {
    Ok(h.attr(k)?.read_scalar::<f64>()?)
}
fn line(f: &mut Figure, p: usize, x: &[f64], y: &[f64], series: usize) -> LegendEntry {
    let l = Line::series(x.iter().zip(y).map(|(&x, &y)| [x, y]).collect(), series);
    let e = l.legend("");
    f.panels[p].push(Layer::Line(l));
    e
}
fn edges(v: &[f64]) -> Vec<f64> {
    let mut e = vec![v[0] - (v[1] - v[0]) / 2.];
    e.extend(v.windows(2).map(|a| (a[0] + a[1]) / 2.));
    e.push(v[v.len() - 1] + (v[v.len() - 1] - v[v.len() - 2]) / 2.);
    e
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let h = File::open(&args[1])?;
    let out = &args[2];
    std::fs::create_dir_all(out)?;
    let title = args
        .get(3)
        .map(|s| s.as_str())
        .unwrap_or("Topside conversion");
    let z = rd(&h, "z_relative_m")?;
    let x = rd(&h, "x_m")?;
    let t = rd(&h, "time_s")?;
    let n = z.len();
    let d = rd(&h, "density_real_ppm")?;
    let di = rd(&h, "density_imag_ppm")?;
    let e = rd(&h, "Ez_real_ratio")?;
    let de = rd(&h, "density_envelope_ppm")?;
    let ee = rd(&h, "Ez_envelope_ratio")?;
    let profz = rd(&h, "profile_height_km")?;
    let profn = rd(&h, "profile_ne_1e11")?;
    let kx = h.dataset("kx_rad_m")?.read_scalar::<f64>()?;
    let dmax = att(&h, "delta_ne_max_ppm")?.ceil() * 1.12;
    let emax = att(&h, "Ez_max_ratio")?.ceil() * 1.12;
    let angle = att(&h, "angle_deg")?;
    let res = att(&h, "local_resonance_m")?;
    let bragg = att(&h, "bragg_offset_m")?;
    let b = att(&h, "bottom_resonance_m")? / 1000.;
    let top = att(&h, "top_resonance_m")? / 1000.;
    let up = res > 118000.;
    let yr = [z[0], *z.last().unwrap()];
    let frames = if args.iter().any(|a| a == "--preview") {
        1
    } else {
        96
    };
    for frame in 0..frames {
        let ti = frame * t.len() / 96;
        let mut f = Figure::new(IntendedUse::Custom {
            width_mm: 320.,
            font_pt: 11.,
        });
        f.height_mm = 172.;
        f.raster_dpi = 140.;
        let ps = f.grid(1, 4)?;
        f.height_mm = 172.;
        f.tight_export = false;
        for (i, &p) in ps.iter().enumerate() {
            let (xx, w) = [(20., 44.), (88., 52.), (161., 52.), (237., 71.)][i];
            f.panels[p].rect = Rect::new(xx, 32., w, 102.);
        }
        f.labels.push(Label::new(title, 160., 7.));
        f.labels.push(Label::new(
            format!(r"O input: ${angle:.2}^\circ$; $5.340$ MHz; peak $E_{{\rm inc}}=1$ mV/m"),
            160.,
            16.,
        ));
        f.labels.push(Label::new(
            format!(
                r"$t={:.3}$ $\mu$s --- one sustained HF period, slowed down",
                t[ti] * 1e6
            ),
            160.,
            24.,
        ));
        let p = ps[0];
        let mut ax = Axes::new(
            [0.8, 4.2],
            [117.5, 118.8],
            r"$n_e$ ($10^{11}$ m$^{-3}$)",
            "Altitude (km)",
        );
        ax.xticks = vec![1., 2., 3., 4.];
        ax.yticks = vec![117.5, 118., 118.5];
        f.panels[p].axes = Some(ax);
        line(&mut f, p, &profn, &profz, 0);
        for level in [b, top] {
            line(&mut f, p, &[0.8, 4.2], &[level, level], 4);
        }
        let mut label = Label::new(
            if up {
                "Topside zoom"
            } else {
                "Bottomside zoom"
            },
            43.,
            if up { 45. } else { 114. },
        );
        label.leader = Some(f.panels[p].data_to_figure([3.5, res / 1000.]));
        f.labels.push(label);
        for (j, xrange, xlabel) in [
            (1, [-emax, emax], r"$E_z/E_{\rm inc}$"),
            (2, [-dmax, dmax], r"$\delta n_e/n_e$ (ppm)"),
        ] {
            let p = ps[j];
            let mut ax = Axes::new(
                xrange,
                yr,
                xlabel,
                if j == 1 {
                    "Height from local resonance (m)"
                } else {
                    ""
                },
            );
            ax.xticks = vec![xrange[0] / 1.12, 0., xrange[1] / 1.12];
            ax.xticklabels = Some(vec![
                format!("{:.1}", xrange[0] / 1.12),
                "0".into(),
                format!("{:.1}", xrange[1] / 1.12),
            ]);
            ax.yticks = if up {
                vec![0., 5., 10., 15.]
            } else {
                vec![-15., -10., -5., 0.]
            };
            if j == 2 {
                ax.show_y_axis = false;
            }
            f.panels[p].axes = Some(ax);
            let env = if j == 1 { &ee } else { &de };
            let vals = if j == 1 { &e } else { &d };
            let mut amplitude_key = line(&mut f, p, env, &z, 1);
            amplitude_key.label = "Amp.".into();
            line(
                &mut f,
                p,
                &env.iter().map(|a| -a).collect::<Vec<_>>(),
                &z,
                1,
            );
            let mut field_key = line(&mut f, p, &vals[ti * n..(ti + 1) * n], &z, 0);
            field_key.label = "Field".into();
            line(&mut f, p, &xrange, &[0., 0.], 4);
            line(&mut f, p, &xrange, &[bragg, bragg], 2);
            if j == 1 {
                let mut legend = Legend::new(vec![field_key, amplitude_key]);
                legend.position = if up {
                    LegendPosition::UpperRight
                } else {
                    LegendPosition::LowerRight
                };
                // The far-outward right corner is clear for all phases.
                legend.avoid_data = false;
                f.panels[p].legend = Some(legend);
            }
        }
        let p = ps[3];
        let mut ax = Axes::new([-60., 60.], yr, "Horizontal distance (m)", "");
        ax.show_y_axis = false;
        ax.xticks = vec![-50., 0., 50.];
        f.panels[p].axes = Some(ax);
        let scale = f.scale(Scale::new(
            [-dmax / 1.12, dmax / 1.12],
            Normalization::Asinh {
                width: dmax / 1.12 / 15.,
            },
            ColorMap::diverging(),
        )?);
        // Vertical field is finely sampled before forming the raster. Height is up.
        let mut field = Vec::with_capacity(n * x.len());
        for j in 0..n {
            for &xx in &x {
                let phase = kx * xx;
                field.push(d[ti * n + j] * phase.cos() - di[ti * n + j] * phase.sin());
            }
        }
        f.panels[p].push(Layer::Pcolormesh(Field::new(
            edges(&x),
            edges(&z),
            field,
            scale,
        )?));
        line(&mut f, p, &[-60., 60.], &[0., 0.], 4);
        line(&mut f, p, &[-60., 60.], &[bragg, bragg], 2);
        let mut cb = Colorbar::new(
            Rect::new(237., 152., 71., 2.5),
            scale,
            r"$\delta n_e/n_e$ (ppm)",
            vec![-dmax / 1.12, 0., dmax / 1.12],
        );
        cb.horizontal = true;
        f.colorbars.push(cb);
        f.auto_colorbar_layout = false;
        f.labels.push(Label::new(
            if up {
                r"Phase $\uparrow$"
            } else {
                r"Phase $\downarrow$"
            },
            173.,
            if up { 38. } else { 126. },
        ));
        f.labels
            .push(Label::new("Green dotted: 430 MHz Bragg match", 153., 165.));
        if frame == 0 {
            f.save(format!("{out}/preview.pdf"))?;
        }
        f.save(format!("{out}/frame_{frame:03}.png"))?;
    }
    Ok(())
}
