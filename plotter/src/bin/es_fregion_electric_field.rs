// Measured-background comparison; arrays from es_fregion_electric_field.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn main()->Result<()>{
 let h=File::open("reference/coupled/es_fregion_electric_field.h5")?;
 let mut f=ArticleLayout::new(177.,10.)?.figure(0.62,112.)?;
 let p=f.grid(1,1)?[0];f.width_mm=177.*0.62;f.height_mm=112.;
 f.panels[p].rect=Rect::new(23.,5.,81.,88.);
 let mut ax=Axes::new([0.03,200.],[-2.5,1.5],r"$|E_z|/|E_{\rm inc}|$","Height relative to reflection (km)");
 ax.set_xscale(AxisScale::Log10)?;ax.yticks=vec![-2.,-1.,0.,1.];f.panels[p].axes=Some(ax);
 let mut l=Line::new(vec![[0.03,0.],[200.,0.]],"#a8afb5");l.width_pt=0.6;l.dash=Some("3 2".into());f.panels[p].push(Layer::Line(l));
 for (name,label) in [("Es_fine",r"Sporadic E ($L_n=1.0$ km)"),("F_fine",r"F region ($L_n=82$ km)")]{
  let g=h.group(name)?;let turn=g.attr("O_turn_relative_m")?.read_scalar::<f64>()?;
  let z=g.dataset("height_relative_m")?.read_raw::<f64>()?;let x=g.dataset("Ez_over_Einc")?.read_raw::<f64>()?;
  let points=x.iter().zip(z).filter(|(_,z)| *z-turn > -2600. && *z-turn < 1600.).map(|(&x,z)|[x,(z-turn)/1000.]).collect();
  f.panels[p].plot(points,label);
 }
 f.panels[p].legend.as_mut().unwrap().position=LegendPosition::UpperRight;
 for ext in ["png","pdf"]{f.save(format!("figures/es_fregion_electric_field.{ext}"))?;}
 Ok(())
}
