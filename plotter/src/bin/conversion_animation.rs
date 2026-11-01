// Native coupled fields and period-averaged fluxes from prepare_conversion_animation.py.
use anyhow::Result;
use hdf5::{File, Group};
use rustplotlib::*;
fn rd(g: &Group, k: &str) -> Result<Vec<f64>> {
    Ok(g.dataset(k)?.read_raw::<f64>()?)
}
fn att(g: &Group, k: &str) -> Result<f64> {
    Ok(g.attr(k)?.read_scalar::<f64>()?)
}
fn line(f: &mut Figure, p: usize, x: &[f64], y: &[f64], series: usize) -> LegendEntry {
    let l = Line::series(x.iter().zip(y).map(|(&x, &y)| [x, y]).collect(), series);
    let e = l.legend("");
    f.panels[p].push(Layer::Line(l));
    e
}
fn colored(f: &mut Figure, p: usize, pts: Vec<[f64; 2]>, color: &str, width: f64) {
    let mut l = Line::new(pts, color);
    l.width_pt = width;
    f.panels[p].push(Layer::Line(l));
}
fn label(f: &mut Figure, text: impl Into<String>, x: f64, y: f64, color: &str) {
    let mut l = Label::new(text, x, y);
    l.color = color.into();
    f.labels.push(l);
}
fn edges(v: &[f64]) -> Vec<f64> {
    let mut e = vec![v[0] - (v[1] - v[0]) / 2.];
    e.extend(v.windows(2).map(|a| (a[0] + a[1]) / 2.));
    e.push(v[v.len() - 1] + (v[v.len() - 1] - v[v.len() - 2]) / 2.);
    e
}
struct Zoom {
    z: Vec<f64>,
    d: Vec<f64>,
    di: Vec<f64>,
    e: Vec<f64>,
    env: Vec<f64>,
    flux: Vec<f64>,
    res: f64,
    bragg: f64,
}
fn zoom(g: &Group) -> Result<Zoom> {
    Ok(Zoom {
        z: rd(g, "z_relative_m")?,
        d: rd(g, "density_real_ppm")?,
        di: rd(g, "density_imag_ppm")?,
        e: rd(g, "Ez_real_ratio")?,
        env: rd(g, "Ez_envelope_ratio")?,
        flux: rd(g, "plasma_wave_flux_percent")?,
        res: att(g, "local_resonance_m")?,
        bragg: att(g, "bragg_offset_m")?,
    })
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let h = File::open(&args[1])?;
    let root = h.group("/")?;
    let out = &args[2];
    std::fs::create_dir_all(out)?;
    let title = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("HF conversion through the whole layer");
    let topside = zoom(&h.group("topside")?)?;
    let bottomside = zoom(&h.group("bottomside")?)?;
    let x = rd(&root, "x_m")?;
    let t = rd(&root, "time_s")?;
    let profz = rd(&root, "profile_height_km")?;
    let profn = rd(&root, "profile_ne_1e11")?;
    let em = rd(&root, "overview_net_EM_flux_percent")?;
    let kx = h.dataset("kx_rad_m")?.read_scalar::<f64>()?;
    let angle = att(&root, "angle_deg")?;
    let reflected = 100. * att(&root, "reflected_EM_fraction")?;
    let transmitted = 100. * att(&root, "transmitted_EM_fraction")?;
    let frames = if args.iter().any(|a| a == "--preview") {
        1
    } else {
        96
    };
    for frame in 0..frames {
        let ti = frame * t.len() / 96;
        let mut f = Figure::new(IntendedUse::Custom {
            width_mm: 400.,
            font_pt: 11.,
        });
        let ps = f.grid(2, 4)?;
        f.height_mm = 286.;
        f.raster_dpi = 135.;
        f.tight_export = false;
        f.panels[ps[0]].rect = Rect::new(20., 57., 45., 181.);
        f.panels[ps[1]].rect = Rect::new(88., 57., 43., 181.);
        for (row, offset) in [(0, 2), (1, 5)] {
            for (j, xx, w) in [(0, 164., 51.), (1, 237., 77.), (2, 350., 41.)] {
                f.panels[ps[offset + j]].rect = Rect::new(xx, 57. + 120. * row as f64, w, 61.);
            }
        }
        label(&mut f, title, 200., 7., "#171717");
        label(
            &mut f,
            format!(
                r"O-mode input at ${angle:.2}^\circ$; $5.340$ MHz; peak $E_{{\rm inc}}=1$ mV/m"
            ),
            200.,
            16.,
            "#171717",
        );
        label(
            &mut f,
            format!(r"$t={:.3}$ $\mu$s", t[ti] * 1e6),
            367.,
            16.,
            "#171717",
        );
        label(&mut f, "Sustained wave, slowed down", 367., 7., "#555555");
        label(
            &mut f,
            r"Incident HF $\uparrow$: 100\%",
            66.,
            27.,
            "#171717",
        );
        label(
            &mut f,
            format!(r"Reflected HF $\downarrow$: {reflected:.1}\%"),
            200.,
            27.,
            "#171717",
        );
        label(
            &mut f,
            format!(r"Transmitted HF $\uparrow$: {transmitted:.1}\%"),
            333.,
            27.,
            "#171717",
        );
        label(
            &mut f,
            "Whole layer; HF enters from below",
            76.,
            39.,
            "#171717",
        );
        label(&mut f, "Electron density", 42., 48., "#171717");
        label(&mut f, "Net EM Poynting flux", 109., 48., "#171717");
        let mut ax = Axes::new(
            [0.8, 4.2],
            [117.5, 118.8],
            r"$n_e$ ($10^{11}$ m$^{-3}$)",
            "Altitude (km)",
        );
        ax.xticks = vec![1., 2., 3., 4.];
        ax.yticks = vec![117.5, 118., 118.5];
        f.panels[ps[0]].axes = Some(ax);
        line(&mut f, ps[0], &profn, &profz, 4);
        let mut ax = Axes::new(
            [-5., 105.],
            [117.5, 118.8],
            r"$\langle S_z\rangle/S_{\rm inc}$ (\%)",
            "",
        );
        ax.show_y_axis = false;
        ax.xticks = vec![0., 50., 100.];
        f.panels[ps[1]].axes = Some(ax);
        line(&mut f, ps[1], &em, &profz, 0);
        colored(
            &mut f,
            ps[1],
            vec![[0., 117.5], [0., 118.8]],
            "#bbbbbb",
            0.6,
        );
        for (row, (v, key, color)) in [(&topside, "A", "#0072B2"), (&bottomside, "B", "#D55E00")]
            .iter()
            .enumerate()
        {
            let offset = if row == 0 { 2 } else { 5 };
            let y = 57. + 120. * row as f64;
            let yr = [v.z[0], *v.z.last().unwrap()];
            let n = v.z.len();
            let side = if row == 0 { "TOPSIDE" } else { "BOTTOMSIDE" };
            let absolute = v.res / 1000.;
            let low = (v.res + yr[0]) / 1000.;
            let high = (v.res + yr[1]) / 1000.;
            // Matched colored boxes identify the actual 19 m zoom in BOTH overview plots.
            for (p, xr) in [(ps[0], [0.8, 4.2]), (ps[1], [-5., 105.])] {
                colored(
                    &mut f,
                    p,
                    vec![
                        [xr[0], low],
                        [xr[1], low],
                        [xr[1], high],
                        [xr[0], high],
                        [xr[0], low],
                    ],
                    color,
                    1.1,
                );
            }
            let point = f.panels[ps[0]].data_to_figure([4.2, absolute]);
            let mut tag = Label::new(*key, 73., point[1]);
            tag.color = color.to_string();
            tag.leader = Some(point);
            f.labels.push(tag);
            label(
                &mut f,
                format!(r"{key}. {side} --- resonance at {absolute:.3} km"),
                277.,
                y - 18.,
                color,
            );
            label(&mut f, "Vertical electric field", 189.5, y - 9., "#171717");
            label(
                &mut f,
                "Electron-density perturbation",
                275.5,
                y - 9.,
                "#171717",
            );
            label(
                &mut f,
                "Plasma pressure-energy flux",
                370.5,
                y - 9.,
                "#171717",
            );
            // Fixed scales across movies: weak responses are not magnified into apparent
            // strong conversion. Top and bottom have their own fixed physical ranges.
            let emax = if row == 0 { 80. } else { 140. };
            let dmax = if row == 0 { 50. } else { 85. };
            let fx = if row == 0 { [-0.15, 4.] } else { [-13., 0.4] };
            let p = ps[offset];
            let mut ax = Axes::new(
                [-emax, emax],
                yr,
                r"$E_z/E_{\rm inc}$",
                if row == 0 {
                    "Height from A (m)"
                } else {
                    "Height from B (m)"
                },
            );
            ax.xticks = vec![-emax, 0., emax];
            ax.yticks = if row == 0 {
                vec![0., 5., 10., 15.]
            } else {
                vec![-15., -10., -5., 0.]
            };
            f.panels[p].axes = Some(ax);
            let mut amp = line(&mut f, p, &v.env, &v.z, 1);
            amp.label = "Amp.".into();
            line(
                &mut f,
                p,
                &v.env.iter().map(|a| -a).collect::<Vec<_>>(),
                &v.z,
                1,
            );
            let mut wave = line(&mut f, p, &v.e[ti * n..(ti + 1) * n], &v.z, 0);
            wave.label = "Field".into();
            let mut legend = Legend::new(vec![wave, amp]);
            legend.position = if row == 0 {
                LegendPosition::UpperRight
            } else {
                LegendPosition::LowerRight
            };
            legend.avoid_data = false;
            f.panels[p].legend = Some(legend);
            let p = ps[offset + 1];
            let mut ax = Axes::new([-60., 60.], yr, "Horizontal distance (m)", "");
            ax.show_y_axis = false;
            ax.xticks = vec![-50., 0., 50.];
            f.panels[p].axes = Some(ax);
            let sc = f.scale(Scale::new(
                [-dmax, dmax],
                Normalization::Asinh { width: dmax / 15. },
                ColorMap::diverging(),
            )?);
            let mut values = Vec::with_capacity(n * x.len());
            for j in 0..n {
                for &xx in &x {
                    values.push(
                        v.d[ti * n + j] * (kx * xx).cos() - v.di[ti * n + j] * (kx * xx).sin(),
                    );
                }
            }
            f.panels[p].push(Layer::Pcolormesh(Field::new(
                edges(&x),
                edges(&v.z),
                values,
                sc,
            )?));
            let mut cb = Colorbar::new(
                Rect::new(237., y + 77., 77., 2.5),
                sc,
                r"$\delta n_e/n_e$ (ppm)",
                vec![-dmax, 0., dmax],
            );
            cb.horizontal = true;
            f.colorbars.push(cb);
            let p = ps[offset + 2];
            let mut ax = Axes::new(fx, yr, r"$F_{p,z}/S_{\rm inc}$ (\%)", "");
            ax.show_y_axis = false;
            ax.xticks = if row == 0 {
                vec![0., 2., 4.]
            } else {
                vec![-12., -6., 0.]
            };
            f.panels[p].axes = Some(ax);
            line(&mut f, p, &v.flux, &v.z, row);
            colored(&mut f, p, vec![[0., yr[0]], [0., yr[1]]], "#bbbbbb", 0.6);
            // Arrow describes the sign of energy flux, not an assumed ray direction.
            let hint = if row == 0 {
                r"Positive: upward $\uparrow$"
            } else {
                r"Negative: downward $\downarrow$"
            };
            label(
                &mut f,
                hint,
                370.,
                if row == 0 { y + 7. } else { y + 53. },
                color,
            );
            for (p, xr) in [
                (ps[offset], [-emax, emax]),
                (ps[offset + 1], [-60., 60.]),
                (ps[offset + 2], fx),
            ] {
                line(&mut f, p, &xr, &[0., 0.], 4);
                if v.bragg.is_finite() {
                    line(&mut f, p, &xr, &[v.bragg, v.bragg], 2);
                }
            }
        }
        f.auto_colorbar_layout = false;
        label(&mut f, "Fluxes are period averages.", 78., 267., "#171717");
        label(
            &mut f,
            r"Positive $=$ upward; negative $=$ downward.",
            78.,
            276.,
            "#171717",
        );
        label(
            &mut f,
            "Dotted green: 430 MHz Bragg scale",
            174.,
            276.,
            "#171717",
        );
        if frame == 0 {
            f.save(format!("{out}/preview.pdf"))?;
        }
        f.save(format!("{out}/frame_{frame:03}.png"))?;
    }
    Ok(())
}
