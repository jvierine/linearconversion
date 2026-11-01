// Finite-screening audit from audit_debye_density.py; no measured density changed.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn main()->Result<()>{
 let h=File::open("data/debye_density_audit.h5")?;
 let x:Vec<_>=rd(&h,"ne_m3")?.iter().map(|v|v.log10()).collect();
 let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=190.;f.height_mm=105.;let ps=f.grid(1,2)?;f.height_mm=105.;
 let colors=["#245774","#8B3940","#527f6d","#8795a9"];
 for (j,p) in ps.iter().enumerate(){
  f.panels[*p].rect=Rect::new(22.+90.*j as f64,8.,64.,70.);
  let mut ax=Axes::new([7.,11.],[0.,2.05],r"Electron density (m$^{-3}$)",if j==0{"Integrated power / collective limit"}else{"Short-lag response / collective limit"});
  ax.xticks=vec![7.,8.,9.,10.,11.];ax.xticklabels=Some((7..12).map(|i|format!("$10^{{{i}}}$")).collect());ax.yticks=vec![0.,0.5,1.,1.5,2.];f.panels[*p].axes=Some(ax);
 }
 for (j,key) in ["ideal_ion_relative_power","ideal_all_frequency_relative_power"].iter().enumerate(){
  let y=rd(&h,key)?;let pts=x.iter().zip(y).map(|(&x,y)|[x,y]).collect();f.panels[ps[0]].push(Layer::Line(Line::new(pts,colors[j])));
 }
 f.panels[ps[0]].legend=Some(Legend::new(vec![LegendEntry::line("Ion component",colors[0]),LegendEntry::line("All frequencies",colors[1])]));
 let curves=rd(&h,"csp_relative_power")?;let mut entries=vec![];
 for (j,y) in curves.chunks(x.len()).enumerate(){
  let pts=x.iter().zip(y).map(|(&x,&y)|[x,y]).collect();f.panels[ps[1]].push(Layer::Line(Line::new(pts,colors[j])));entries.push(LegendEntry::line(format!("{} km",60+j*10),colors[j]));
 }
 f.panels[ps[1]].legend=Some(Legend::new(entries));
 for ext in ["png","pdf"] {f.save(format!("figures/debye_density.{ext}"))?;}
 Ok(())
}
