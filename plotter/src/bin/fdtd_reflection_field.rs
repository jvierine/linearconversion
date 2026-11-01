// Source: coupled_sim/scripts/fdtd_reflection_field.py; data stored in HDF5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],col:&str,dash:bool){let mut l=Line::new(x.iter().zip(y).map(|(&x,&y)|[x,y]).collect(),col);l.width_pt=1.1;if dash{l.dash=Some("3 2".into());}f.panels[p].push(Layer::Line(l));}
fn main()->Result<()>{
 let h=File::open("reference/coupled/fdtd_reflection_field.h5")?;let ray=File::open("coupled_sim/reference/resonant_access_audit.h5")?;
 let refl=ray.attr("critical_height_m")?.read_scalar::<f64>()?-h.attr("resonance_height_m")?.read_scalar::<f64>()?;
 let z=rd(&h,"height_relative_m")?;let total=rd(&h,"total_magnitude_over_incident")?;let ez=rd(&h,"vertical_magnitude_over_incident")?;
 let mut f=ArticleLayout::new(177.,11.)?.figure(1.,108.)?;let ps=f.grid(1,2)?;f.height_mm=108.;
 for (j,&p) in ps.iter().enumerate(){f.panels[p].rect=Rect::new(if j==0{23.}else{113.},8.,59.,81.);f.labels.push(Label::new(if j==0{"(a)"}else{"(b)"},if j==0{23.}else{113.},3.));
 let mut ax=Axes::new(if j==0{[0.05,250.]}else{[0.,200.]},if j==0{[-150.,60.]}else{[-1.,1.]},r"Field magnitude / $|E_{\rm inc}|$",if j==0{"Height relative to resonance (m)"}else{""});if j==0{ax.set_xscale(AxisScale::Log10)?;ax.yticks=vec![-150.,-100.,-50.,0.,50.];}else{ax.xticks=vec![0.,100.,200.];ax.yticks=vec![-1.,-0.5,0.,0.5,1.];}f.panels[p].axes=Some(ax);
 line(&mut f,p,&total,&z,"#245774",false);line(&mut f,p,&ez,&z,"#8b3940",true);
 let xx=if j==0{[0.05,250.]}else{[0.,200.]};line(&mut f,p,&xx,&[0.,0.],"#a8afb5",true);
 if j==0{line(&mut f,p,&xx,&[refl,refl],"#4c7566",true);}
 let mut leg=Legend::new(if j==0{vec![LegendEntry::line("Total","#245774"),LegendEntry::line(r"$E_z$","#8b3940")]}else{vec![LegendEntry::line("Total","#245774"),LegendEntry::line(r"$E_z$","#8b3940")]});leg.position=LegendPosition::LowerRight;leg.avoid_data=false;f.panels[p].legend=Some(leg);}
 f.save("figures/fdtd_reflection_field.png")?;Ok(())
}
