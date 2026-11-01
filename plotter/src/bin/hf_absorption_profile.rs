// Physical heating arrays: coupled_sim/scripts/hf_absorption_profile.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],col:&str,dash:bool){let mut l=Line::new(x.iter().zip(y).map(|(&x,&y)|[x,y]).collect(),col);l.width_pt=1.1;if dash{l.dash=Some("3 2".into());}f.panels[p].push(Layer::Line(l));}
fn main()->Result<()>{
 let h=File::open("reference/coupled/hf_absorption_profile.h5")?;
 let mut f=ArticleLayout::new(177.,11.)?.figure(1.,107.)?;let ps=f.grid(1,2)?;f.height_mm=107.;
 for (j,&p) in ps.iter().enumerate(){f.panels[p].rect=Rect::new(if j==0{23.}else{113.},8.,59.,80.);f.labels.push(Label::new(if j==0{"(a)"}else{"(b)"},if j==0{23.}else{113.},3.));}
 let p=ps[0];let mut ax=Axes::new([0.,100.],[-250.,60.],"Cumulative absorption (\\%)","Height relative to resonance (m)");ax.xticks=vec![0.,25.,50.,75.,100.];ax.yticks=vec![-200.,-100.,0.,50.];f.panels[p].axes=Some(ax);
 let mut entries=vec![];
 for (key,label,col,dash) in [("event/O",r"O, $14^\circ$","#245774",false),("event/X",r"X, $14^\circ$","#4c7566",false),("normal/O",r"O, $0^\circ$","#8b3940",true),("oblique25/O",r"O, $25^\circ$","#9b6d32",true)]{
  let x:Vec<_>=rd(&h,&format!("{key}/cumulative_absorbed_fraction"))?.iter().map(|v|100.*v).collect();line(&mut f,p,&x,&rd(&h,&format!("{key}/z_relative_m"))?,col,dash);entries.push(LegendEntry::line(label,col));
 }
 line(&mut f,p,&[0.,100.],&[0.,0.],"#a8afb5",true);
 let mut leg=Legend::new(entries);leg.position=LegendPosition::LowerRight;leg.avoid_data=false;f.panels[p].legend=Some(leg);
 let p=ps[1];let mut ax=Axes::new([0.,190.],[-1.5,1.5],r"Absorption per metre (\% m$^{-1}$)","");ax.xticks=vec![0.,50.,100.,150.];ax.yticks=vec![-1.5,-1.,-0.5,0.,0.5,1.,1.5];f.panels[p].axes=Some(ax);let mut entries=vec![];
 for (key,label,col,dash) in [("half_nu/O",r"$\nu/2$","#8b3940",true),("event/O",r"$\nu$","#245774",false),("double_nu/O",r"$2\nu$","#4c7566",true)]{
  let x:Vec<_>=rd(&h,&format!("{key}/absorption_per_m"))?.iter().map(|v|100.*v).collect();line(&mut f,p,&x,&rd(&h,&format!("{key}/z_relative_m"))?,col,dash);entries.push(LegendEntry::line(label,col));
 }
 line(&mut f,p,&[0.,190.],&[0.,0.],"#a8afb5",true);let mut leg=Legend::new(entries);leg.position=LegendPosition::LowerRight;leg.avoid_data=false;f.panels[p].legend=Some(leg);
 f.save("figures/hf_absorption_profile.png")?;Ok(())
}
