// Arrays and provenance: coupled_sim/scripts/radar_sensitivity_sweeps.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
const BLUE:&str="#245774";const RED:&str="#8b3940";const GREEN:&str="#4c7566";
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn db(x:f64)->f64{10.*x.max(1e-40).log10()}
fn edges(v:&[f64])->Vec<f64>{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|p|(p[0]+p[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e}
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],col:&str,dash:bool){let mut l=Line::new(x.iter().zip(y).map(|(&x,&y)|[x,y]).collect(),col);l.width_pt=1.;if dash{l.dash=Some("3 2".into());}f.panels[p].push(Layer::Line(l));}
fn main()->Result<()>{
 let path=std::env::args().nth(1).unwrap_or("linearconversion/data/radar_sensitivity.h5".into());let h=File::open(path)?;
 let obs=db(h.attr("reference_RCS_m2")?.read_scalar::<f64>()?);
 let mut f=ArticleLayout::new(177.,10.)?.figure(1.,156.)?;let ps=f.grid(2,2)?;f.height_mm=156.;
 for (i,&p) in ps.iter().enumerate(){f.panels[p].rect=Rect::new(if i%2==0{22.}else{112.},if i<2{8.}else{92.},60.,44.);f.labels.push(Label::new(format!("({})",['a','b','c','d'][i]),if i%2==0{22.}else{112.},if i<2{3.}else{87.}));}
 let p=ps[0];let mut ax=Axes::new([-25.,25.],[-125.,-75.],r"Incidence angle ($^\circ$)",r"RCS (dB m$^2$)");ax.xticks=vec![-20.,0.,20.];ax.yticks=vec![-120.,-100.,-80.];f.panels[p].axes=Some(ax);
 line(&mut f,p,&rd(&h,"angle/degrees")?,&rd(&h,"angle/RCS_m2")?.into_iter().map(db).collect::<Vec<_>>(),BLUE,false);line(&mut f,p,&[-25.,25.],&[obs,obs],"#737b83",true);
 f.panels[p].legend=Some(Legend::new(vec![LegendEntry::line("Model",BLUE),LegendEntry::line("Observed","#737b83")]));
 let p=ps[1];let mut ax=Axes::new([0.,90.],[-180.,180.],r"$\tan^{-1}|E_y/E_x|$ ($^\circ$)",r"$\arg(E_y/E_x)$ ($^\circ$)");ax.xticks=vec![0.,45.,90.];ax.yticks=vec![-180.,0.,180.];f.panels[p].axes=Some(ax);
 let sc=f.scale(Scale::new([-125.,-75.],Normalization::Linear,ColorMap::ember())?);
 f.panels[p].push(Layer::Pcolormesh(Field::new(edges(&rd(&h,"polarization/amplitude_angle_deg")?),edges(&rd(&h,"polarization/relative_phase_deg")?),rd(&h,"polarization/RCS_m2")?.into_iter().map(db).collect(),sc)?));
 let opt=rd(&h,"polarization/optimum")?;line(&mut f,p,&[opt[0]-3.,opt[0]+3.],&[opt[1],opt[1]],"#ffffff",false);line(&mut f,p,&[opt[0],opt[0]],&[opt[1]-15.,opt[1]+15.],"#ffffff",false);
 let mut cb=Colorbar::new(Rect::new(112.,70.,60.,2.5),sc,r"RCS (dB m$^2$)",vec![-120.,-100.,-80.]);cb.horizontal=true;f.colorbars.push(cb);f.auto_colorbar_layout=false;
 let p=ps[2];let mut ax=Axes::new([1e-3,1e4],[-120.,-40.],"HF EIRP (W)",r"RCS (dB m$^2$)");ax.set_xscale(AxisScale::Log10)?;ax.xticks=vec![1e-3,1.,1e3];ax.xticklabels=Some(vec![r"$10^{-3}$".into(),"1".into(),r"$10^3$".into()]);ax.yticks=vec![-120.,-100.,-80.,-60.,-40.];f.panels[p].axes=Some(ax);
 let x=rd(&h,"power/EIRP_W")?;for (a,col) in [(0,RED),(14,BLUE),(25,GREEN)]{line(&mut f,p,&x,&rd(&h,&format!("power/RCS_{a}_m2"))?.into_iter().map(db).collect::<Vec<_>>(),col,false);}line(&mut f,p,&[1e-3,1e4],&[obs,obs],"#737b83",true);
 f.panels[p].legend=Some(Legend::new(vec![LegendEntry::line(r"$0^\circ$",RED),LegendEntry::line(r"$14^\circ$",BLUE),LegendEntry::line(r"$25^\circ$",GREEN)]));
 let p=ps[3];let mut ax=Axes::new([0.4,5.1],[-200.,-60.],r"Density scale length $L_n$ (km)",r"RCS (dB m$^2$)");ax.xticks=vec![1.,2.,3.,4.,5.];ax.yticks=vec![-200.,-160.,-120.,-80.];f.panels[p].axes=Some(ax);
 let tab=rd(&h,"gradient/table")?;let x:Vec<_>=tab.chunks(4).map(|r|r[0]).collect();let y:Vec<_>=tab.chunks(4).map(|r|db(r[1])).collect();line(&mut f,p,&x,&y,BLUE,false);line(&mut f,p,&[0.4,5.1],&[obs,obs],"#737b83",true);
 let ln=h.attr("reference_Ln_m")?.read_scalar::<f64>()?/1000.;line(&mut f,p,&[ln,ln],&[-200.,-60.],"#a8afb5",true);
 f.panels[p].legend=Some(Legend::new(vec![LegendEntry::line("Gradient sweep",BLUE),LegendEntry::line(r"Measured $L_n$","#a8afb5")]));
 for (i,&p) in ps.iter().enumerate(){if let Some(l)=f.panels[p].legend.as_mut(){l.position=if i==3{LegendPosition::UpperRight}else{LegendPosition::UpperLeft};l.avoid_data=false;}}
 for ext in ["pdf","png"]{f.save(format!("figures/radar_sensitivity.{ext}"))?;}
 Ok(())
}
