// Diagnostics: nonlinear_sim/scripts/diagnostics.py; HDF5 arrays retain provenance.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
const BLUE:&str="#245774";const RED:&str="#8b3940";const GREEN:&str="#4c7566";
fn read(h:&File,n:&str)->Result<Vec<f64>>{Ok(h.dataset(n)?.read_raw()?)}
fn curve(f:&mut Figure,p:usize,x:&[f64],y:&[f64],color:&str,dash:bool){
 let mut l=Line::new(x.iter().zip(y).map(|(&x,&y)|[x,y]).collect(),color);
 if dash {l.dash=Some("4 3".into());l.protect_from_legend=false;} f.panels[p].push(Layer::Line(l));
}
fn main()->Result<()>{
 let path=if std::path::Path::new("nonlinear_sim/reference/nonlinear_validation.h5").exists(){"nonlinear_sim/reference/nonlinear_validation.h5"}else{"linearconversion/nonlinear_sim/reference/nonlinear_validation.h5"};
 let h=File::open(path)?;std::fs::create_dir_all("figures")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let panels=f.grid(1,2)?;f.width_mm=220.;f.height_mm=135.;
 for (i,&p) in panels.iter().enumerate(){f.panels[p].rect=Rect::new(25.+108.*i as f64,8.,77.,100.);}
 let p=panels[0];f.panels[p].axes=Some(Axes::new([-130.,130.],[-10.,5.],r"$\delta n_e/n_e$ (ppm)","Height relative to resonance (m)"));
 let z=read(&h,"weak_refined_lifted/z")?;let y=read(&h,"weak_refined_lifted/real")?.iter().map(|v|v*1e6).collect::<Vec<_>>();
 curve(&mut f,p,&y,&z,BLUE,false);
 let zc=read(&h,"weak_lifted/z")?;let yc=read(&h,"weak_lifted/real")?.iter().map(|v|v*1e6).collect::<Vec<_>>();
 curve(&mut f,p,&yc,&zc,RED,true);
 f.panels[p].legend=Some(Legend::new(vec![LegendEntry::line("2 cm grid",BLUE),LegendEntry::line("4 cm grid",RED)]));
 let p=panels[1];f.panels[p].axes=Some(Axes::new([0.,1.],[-10.,5.],"Vertical wavelength (m)","Height relative to resonance (m)"));
 curve(&mut f,p,&read(&h,"weak_refined_lifted/wavelength_m")?,&z,BLUE,false);
 let bragg=299792458./(2.*430e6+5.340028553893307e6);
 curve(&mut f,p,&[bragg,bragg],&[-10.,5.],GREEN,true);
 f.panels[p].legend=Some(Legend::new(vec![LegendEntry::line("Solved phase",BLUE),LegendEntry::line("430 MHz Bragg",GREEN)]));
 f.save("figures/nonlinear_wave.png")?;f.save("figures/nonlinear_wave.pdf")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let p=f.grid(1,1)?[0];f.width_mm=200.;f.height_mm=125.;f.panels[p].rect=Rect::new(30.,8.,155.,90.);
 let mut a=Axes::new([0.,2.],[1e-12,1e-3],"Time (HF cycles)",r"Density at $2k_x$: $|\delta n_{e,2}|/n_e$");a.set_yscale(AxisScale::Log10)?;f.panels[p].axes=Some(a);
 for (name,col) in [("weak_lifted",BLUE),("medium_full",GREEN),("strong_full",RED)]{
  let t=read(&h,&format!("{name}/cycles"))?;let y=read(&h,&format!("{name}/second"))?.iter().map(|&v|if v>1e-12{v}else{f64::NAN}).collect::<Vec<_>>();curve(&mut f,p,&t,&y,col,false);
 }
 f.panels[p].legend=Some(Legend::new(vec![LegendEntry::line(r"Initial wave: $0.01\%$",BLUE),LegendEntry::line(r"Initial wave: $0.1\%$",GREEN),LegendEntry::line(r"Initial wave: $1\%$",RED)]));
 f.save("figures/nonlinear_harmonics.png")?;f.save("figures/nonlinear_harmonics.pdf")?;Ok(())
}
