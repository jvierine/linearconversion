// Arrays and audit: coupled_sim/scripts/resonant_access_audit.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],col:&str,dash:bool){let mut l=Line::new(x.iter().zip(y).map(|(&x,&y)|[x,y]).collect(),col);l.width_pt=0.9;if dash{l.dash=Some("3 2".into());}f.panels[p].push(Layer::Line(l));}
fn main()->Result<()>{
 let path=std::env::args().nth(1).unwrap_or("linearconversion/coupled_sim/reference/resonant_access_audit.h5".into());let h=File::open(path)?;
 let center=h.attr("resonance_height_m")?.read_scalar::<f64>()?;let critical=h.attr("critical_height_m")?.read_scalar::<f64>()?-center;
 let ox=rd(&h,"ray_3/horizontal_m")?;let oz=rd(&h,"ray_3/height_m")?;let j=oz.iter().enumerate().max_by(|(_,a),(_,b)|a.total_cmp(b)).unwrap().0;let offset=ox[j];
 let mut f=ArticleLayout::new(177.,10.)?.figure(1.,105.)?;let ps=f.grid(1,2)?;f.height_mm=105.;
 f.panels[ps[0]].rect=Rect::new(22.,7.,70.,84.);f.panels[ps[1]].rect=Rect::new(112.,7.,60.,84.);
 let mut ax=Axes::new([-750.,800.],[-250.,60.],"Horizontal displacement (m)","Height relative to resonance (m)");ax.xticks=vec![-500.,0.,500.];ax.yticks=vec![-200.,-100.,0.];f.panels[ps[0]].axes=Some(ax);
 for (root,col) in [("ray_3","#245774"),("ray_2","#4c7566")]{let x=rd(&h,&format!("{root}/horizontal_m"))?;let z=rd(&h,&format!("{root}/height_m"))?;let xx:Vec<_>=x.iter().map(|v|v-offset).collect();let zz:Vec<_>=z.iter().map(|v|v-center).collect();line(&mut f,ps[0],&xx,&zz,col,false);}
 for (y,col) in [(0.,"#8b3940"),(critical,"#8895a7")]{line(&mut f,ps[0],&[-750.,800.],&[y,y],col,true);}
 let mut leg=Legend::new(vec![LegendEntry::line("O ray","#245774"),LegendEntry::line("X ray","#4c7566")]);leg.position=LegendPosition::LowerRight;leg.avoid_data=false;f.panels[ps[0]].legend=Some(leg);
 let mut ax=Axes::new([0.01,200.],[-250.,60.],r"$|E_z|/|E_{\rm inc}|$","");ax.set_xscale(AxisScale::Log10)?;ax.yticks=vec![-200.,-100.,0.];f.panels[ps[1]].axes=Some(ax);
 line(&mut f,ps[1],&rd(&h,"Ez_over_incident_peak")?,&rd(&h,"Ez_height_relative_m")?,"#245774",false);
 for (y,col) in [(0.,"#8b3940"),(critical,"#8895a7")]{line(&mut f,ps[1],&[0.01,200.],&[y,y],col,true);}
 let mut leg=Legend::new(vec![LegendEntry::line("Resonance","#8b3940"),LegendEntry::line(r"$f_{pe}=f_0$","#8895a7")]);leg.position=LegendPosition::LowerRight;leg.avoid_data=false;f.panels[ps[1]].legend=Some(leg);
 f.labels.push(Label::new("(a)",22.,3.));f.labels.push(Label::new("(b)",112.,3.));
 for ext in ["png","pdf"]{f.save(format!("figures/resonant_access.{ext}"))?;}
 Ok(())
}
