// Scientific arrays: validate_fluid_damping.py; Rustplotlib default styling.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn main()->Result<()> {
 let h=File::open("data/fluid_damping_prior.h5")?;
 let read=|name:&str|->Result<Vec<f64>>{Ok(h.dataset(name)?.read_raw::<f64>()?)};
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,2)?;
 f.width_mm=220.;f.height_mm=158.;
 for i in 0..2{f.panels[ps[i]].rect=Rect::new(25.+i as f64*108.,30.,78.,100.);}
 f.panels[ps[0]].axes=Some(Axes::new([0.,6500.],[-40.,40.],r"Momentum loss rate (s$^{-1}$)","Height relative to resonance (m)"));
 f.panels[ps[1]].axes=Some(Axes::new([-180.,0.],[-25.,0.],"Collisional energy change (dB)","Height relative to resonance (m)"));
 let colors=["#245774","#8b3940","#4c7566"];
 let z=read("height_relative_m")?;
 for (i,name) in ["nu_ei_s1","nu_en_s1","nu_total_s1"].iter().enumerate(){
  let x=read(name)?;f.panels[ps[0]].push(Layer::Line(Line::new(x.iter().zip(&z).map(|(&x,&z)|[x,z]).collect(),colors[i])));
  let base=format!("gradient_{i}");let zz=read(&format!("{base}/height_relative_m"))?;let xx=read(&format!("{base}/energy_change_db"))?;
  f.panels[ps[1]].push(Layer::Line(Line::new(xx.iter().zip(&zz).map(|(&x,&z)|[x,z]).collect(),colors[i])));
  let g=h.group(&base)?;let bm=g.attr("bragg_z")?.read_scalar::<f64>()?;let bl=g.attr("bragg_loss_db")?.read_scalar::<f64>()?;
  let mut tick=Line::new(vec![[bl-5.,bm],[bl+5.,bm]],colors[i]);tick.width_pt=2.;f.panels[ps[1]].push(Layer::Line(tick));
 }
 for (i,labels) in [["Electron–ion","Electron–neutral","Total drag"],["Measured gradient","Half gradient","Quarter gradient"]].iter().enumerate(){
  let mut legend=Legend::new(labels.iter().zip(colors).map(|(&l,c)|LegendEntry::line(l,c)).collect());legend.position=LegendPosition::AboveCenter;legend.columns=1;f.panels[ps[i]].legend=Some(legend);
 }
 for ext in ["png","svg","pdf"] {f.save(format!("figures/fluid_collisional_damping.{ext}"))?;}
 Ok(())
}
