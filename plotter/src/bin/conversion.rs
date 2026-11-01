// Native Rustplotlib defaults; numerical provenance is retained here and in HDF5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
const COLORS: [&str; 3] = ["#245774", "#8b3940", "#4c7566"];
fn read(f: &File, key: &str) -> Result<Vec<f64>> { Ok(f.dataset(key)?.read_raw::<f64>()?) }
fn plot(name: &str, x: &[f64], ys: Vec<(Vec<f64>, &str)>,
        xlim: [f64;2], ylim: [f64;2], xlabel: &str, ylabel: &str) -> Result<()> {
    let mut f=Figure::new(IntendedUse::ArticleWide);
    let idx=f.grid(1,1)?[0];
    f.panels[idx].axes=Some(Axes::new(xlim,ylim,xlabel,ylabel));
    let mut entries=vec![];
    for (i,(y,label)) in ys.into_iter().enumerate() {
        let points=x.iter().zip(y).filter(|(a,b)|a.is_finite()&&b.is_finite())
            .map(|(&a,b)|[a,b]).collect();
        f.panels[idx].push(Layer::Line(Line::new(points,COLORS[i])));
        entries.push(LegendEntry::line(label,COLORS[i]));
    }
    if entries.len()>1 {
        let mut legend=Legend::new(entries);legend.position=LegendPosition::AboveCenter;
        f.panels[idx].rect.y+=12.;f.height_mm+=12.;
        f.panels[idx].legend=Some(legend);
    }
    for ext in ["png","svg"] {f.save(format!("figures/{name}.{ext}"))?;}
    Ok(())
}
fn main()->Result<()> {
    let f=File::open("data/vertical_radar_conversion.h5")?;
    let x=read(&f,"incidence/vacuum_incidence_deg")?;
    plot("linear_conversion_efficiency",&x,vec![
        (read(&f,"incidence/conversion")?,"Optimized incident polarization")],
        [-31.,31.],[0.,1.],"HF vacuum-equivalent incidence angle (degrees)",
        "Converted outgoing energy / incident energy")?;
    let x=read(&f,"plot/z_relative_match_m")?;
    plot("linear_conversion_wave",&x,vec![
        (read(&f,"plot/density_real_percent")?,r"$\mathrm{Re}(\delta n_e/n_e)$"),
        (read(&f,"plot/density_envelope_percent")?,"Amplitude")],
        [-2.,2.],[-1.2,1.2],"Distance along layer normal from Bragg match (m)",
        r"Fractional electron density (\%)")?;
    let n=x.len();
    plot("linear_conversion_wavenumber",&x,vec![
        (read(&f,"plot/phase_kz_m1")?,"Density-wave phase gradient"),
        (vec![18.024267188784464;n],"Arecibo Bragg wave number")],
        [-2.,2.],[0.,50.],"Distance along layer normal from Bragg match (m)",
        r"Vertical wave number magnitude (m$^{-1}$)")?;
    Ok(())
}
