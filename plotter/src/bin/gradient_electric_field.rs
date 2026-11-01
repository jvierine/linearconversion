// Source: coupled_sim/scripts/gradient_electric_field.py; controlled warm-fluid gradients.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn main()->Result<()>{
 let h=File::open("reference/coupled/gradient_electric_field.h5")?;
 let mut f=ArticleLayout::new(177.,10.)?.figure(0.58,115.)?;
 let p=f.grid(1,1)?[0]; f.width_mm=177.*0.58; f.height_mm=115.;
 f.panels[p].rect=Rect::new(23.,5.,74.,92.);
 let mut ax=Axes::new([0.05,250.],[-250.,150.],r"$|E_z|/|E_{\rm inc}|$","Height relative to resonance (m)");
 ax.set_xscale(AxisScale::Log10)?;ax.yticks=vec![-250.,-200.,-150.,-100.,-50.,0.,50.,100.,150.];
 f.panels[p].axes=Some(ax);
 let mut zero=Line::new(vec![[0.05,0.],[250.,0.]],"#a8afb5");zero.width_pt=0.6;zero.dash=Some("3 2".into());f.panels[p].push(Layer::Line(zero));
 let mut entries=Vec::new();
 for (name,label,col) in [("L0p5",r"$L_n=0.5$ km","#245774"),("L1",r"$L_n=1$ km","#8b3940"),("L2",r"$L_n=2$ km","#4c7566"),("L5",r"$L_n=5$ km","#a26b32")]{
  let g=h.group(name)?;let z=g.dataset("height_relative_m")?.read_raw::<f64>()?;let x=g.dataset("Ez_over_Einc")?.read_raw::<f64>()?;
  let mut l=Line::new(x.iter().zip(&z).map(|(&x,&z)|[x,z]).collect(),col);l.width_pt=1.;f.panels[p].push(Layer::Line(l));
  let turn=g.attr("O_turn_relative_m")?.read_scalar::<f64>()?;
  let i=z.iter().enumerate().min_by(|(_,a),(_,b)|((*a-turn).abs()).total_cmp(&((*b-turn).abs()))).unwrap().0;
  f.panels[p].errorbars(&[[x[i],turn]],Some(&[[x[i]/1.16,x[i]*1.16]]),None,col,1.8,0.,1.)?;
  entries.push(LegendEntry::line(label,col));
 }
 let mut leg=Legend::new(entries);leg.position=LegendPosition::LowerRight;leg.avoid_data=false;f.panels[p].legend=Some(leg);
 for ext in ["pdf","png"]{f.save(format!("figures/gradient_electric_field.{ext}"))?;}
 Ok(())
}
