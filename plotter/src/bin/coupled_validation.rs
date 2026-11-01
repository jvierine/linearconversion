// Scientific inputs: coupled_sim/reference/physics_validation.h5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn rgb(s:&str)->[f64;3]{[1,3,5].map(|i|u8::from_str_radix(&s[i..i+2],16).unwrap() as f64/255.)}
fn main()->Result<()> {
 let h=File::open("coupled_sim/reference/physics_validation.h5")?;
 let colors=["#245774","#8b3940"];let shapes=[MarkerShape::Circle,MarkerShape::Diamond];
 let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=230.;f.height_mm=115.;
 for panel in 0..2 {
  let mut a=Axes::new(if panel==0{[28.,150.]}else{[0.43,1.15]},if panel==0{[0.0002,0.08]}else{[0.00002,0.0003]},if panel==0{"Cells per wavelength"}else{r"Relative time step $\Delta t/\Delta t_0$"},if panel==0{"Maximum waveform error"}else{"Relative energy-budget error"});
  a.set_xscale(AxisScale::Log10)?;a.set_yscale(AxisScale::Log10)?;
  a.xticks=if panel==0{vec![32.,64.,128.]}else{vec![0.5,1.]};
  a.xticklabels=Some(a.xticks.iter().map(|v|format!("{v}")).collect());
  let mut p=Panel::new(Rect::new(25.+115.*panel as f64,7.,79.,84.),a);
  let keys=if panel==0{["EM_dispersion","Langmuir_dispersion"]}else{["lossless_energy","collision_energy"]};
  let labels=if panel==0{["EM wave","Langmuir wave"]}else{["No collisions","With collisions"]};
  let mut points=Vec::new();
  for j in 0..2 {
   let g=h.group(&format!("native/{}",keys[j]))?;
   let (x,y)=if panel==0 {(vec![32.,64.,128.],g.dataset("max_waveform_errors")?.read_raw::<f64>()?)}else{(vec![0.5,1.],vec![g.attr("fine_relative_budget_error")?.read_scalar::<f64>()?,g.attr("coarse_relative_budget_error")?.read_scalar::<f64>()?])};
   let pts:Vec<_>=x.iter().zip(y).map(|(&x,y)|[x,y]).collect();
   let mut line=Line::new(pts.clone(),colors[j]);line.width_pt=1.;p.push(Layer::Line(line));points.push(pts);
  }
  p.legend=Some(p.marker_groups(&[MarkerGroup{label:labels[0],points:&points[0],style:MarkerStyle{shape:shapes[0],size_pt:7.,color:rgb(colors[0])}},MarkerGroup{label:labels[1],points:&points[1],style:MarkerStyle{shape:shapes[1],size_pt:7.,color:rgb(colors[1])}}])?);
  f.panels.push(p);
 }
 for ext in ["pdf","png"] {f.save(format!("figures/coupled_validation.{ext}"))?;}
 Ok(())
}
