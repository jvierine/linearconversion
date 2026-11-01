// Scientific arrays: coupled_sim/scripts/fdtd_ray_benchmark.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],col:&str,dash:bool){let mut l=Line::new(x.iter().zip(y).filter(|(a,b)|a.is_finite()&&b.is_finite()).map(|(&x,&y)|[x,y]).collect(),col);l.width_pt=1.;if dash{l.dash=Some("3 2".into());}f.panels[p].push(Layer::Line(l));}
fn main()->Result<()>{
 let path=std::env::args().nth(1).unwrap_or("reference/coupled/fdtd_ray_benchmark.h5".into());let h=File::open(path)?;
 let g=h.group("scalar_fine")?;let zt=g.attr("turn_height_m")?.read_scalar::<f64>()?;
 let z:Vec<_>=rd(&h,"scalar_fine/z_m")?.iter().map(|x|x-zt).collect();
 let mut f=ArticleLayout::new(177.,11.)?.figure(1.,82.)?;let ps=f.grid(1,2)?;f.height_mm=82.;
 for (j,&p) in ps.iter().enumerate(){f.panels[p].rect=Rect::new(if j==0{23.}else{113.},7.,59.,59.);f.labels.push(Label::new(if j==0{"(a)"}else{"(b)"},if j==0{23.}else{113.},3.));
 let mut ax=Axes::new([0.,if j==0{7.}else{12.}],if j==0{[-450.,100.]}else{[-90.,90.]},r"$|E_y|/|E_{\rm inc}|$",if j==0{"Height from turning point (m)"}else{""});ax.yticks=if j==0{vec![-400.,-200.,0.]}else{vec![-80.,-40.,0.,40.,80.]};f.panels[p].axes=Some(ax);
 line(&mut f,p,&rd(&h,"scalar_fine/E_over_incident")?,&z,"#245774",false);
 line(&mut f,p,&rd(&h,"scalar_fine/WKB_over_incident")?,&z,"#8b3940",true);
 if j==1{let x=rd(&h,"scalar_fine/Airy_over_incident")?;line(&mut f,p,&x,&z,"#4c7566",true);}
 line(&mut f,p,&[0.,if j==0{7.}else{12.}],&[0.,0.],"#a8afb5",true);
 let mut leg=Legend::new(if j==0{vec![LegendEntry::line("FDTD","#245774"),LegendEntry::line("WKB","#8b3940")]}else{vec![LegendEntry::line("Airy","#4c7566")]});leg.position=LegendPosition::UpperRight;leg.avoid_data=false;f.panels[p].legend=Some(leg);}
 for ext in ["pdf","png"]{f.save(format!("figures/fdtd_ray_benchmark.{ext}"))?;}
 Ok(())
}
