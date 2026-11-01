// Muldrew ray treatment and numerical provenance are in HDF5/source comments.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
const BLUE:&str="#245774";const RED:&str="#8b3940";const GRAY:&str="#8795a9";
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn save(f:&Figure,name:&str)->Result<()>{for ext in ["png","svg","pdf"]{f.save(format!("figures/{name}.{ext}"))?;}Ok(())}
fn legend(f:&mut Figure,p:usize,e:Vec<LegendEntry>,columns:usize){let mut l=Legend::new(e);l.position=LegendPosition::AboveCenter;l.columns=columns;f.panels[p].legend=Some(l);}
fn main()->Result<()>{
 let h=File::open("data/muldrew_parameter_space.h5")?;
 let threshold=h.attr("detection_threshold_fraction")?.read_scalar::<f64>()?;
 let mut fig=Figure::new(IntendedUse::ArticleWide);let p=fig.grid(1,1)?[0];
 fig.panels[p].rect=Rect::new(23.,29.,140.,35.);fig.height_mm=86.;
 fig.panels[p].axes=Some(Axes::new([0.,100.],[-20.,5.],"Horizontal energy displacement (m)","Height from Bragg match (m)"));
 for (freq,color) in [("4.51",BLUE),("5.33",RED)]{
  let g=h.group(&format!("rays/{freq}"))?;let zb=g.attr("bragg_height_m")?.read_scalar::<f64>()?;
  let x=rd(&h,&format!("rays/{freq}/horizontal_displacement_m"))?;
  let z=rd(&h,&format!("rays/{freq}/height_m"))?;
  fig.panels[p].push(Layer::Line(Line::new(x.iter().zip(z.iter()).map(|(&x,&z)|[x,z-zb]).collect(),color)));
 }
 let xb=h.group("rays/5.33")?.attr("horizontal_displacement_m_to_bragg")?.read_scalar::<f64>()?;
 let mut phase=Line::new(vec![[xb,0.],[xb-0.0148,-7.],[xb-0.7,-5.9],[xb-0.0148,-7.],[xb+0.7,-5.9]],GRAY);phase.width_pt=1.;fig.panels[p].push(Layer::Line(phase));
 let mut b=Line::new(vec![[0.,0.],[100.,0.]],GRAY);b.dash=Some("2,3".into());b.width_pt=0.5;fig.panels[p].push(Layer::Line(b));
 legend(&mut fig,p,vec![LegendEntry::line("4.51 MHz energy",BLUE),LegendEntry::line("5.33 MHz energy",RED),LegendEntry::line("5.33 MHz phase direction",GRAY)],2);
 save(&fig,"muldrew_energy_ray")?;
 let mut fig=Figure::new(IntendedUse::ArticleWide);let p=fig.grid(1,1)?[0];fig.panels[p].rect=Rect::new(23.,23.,140.,78.);fig.height_mm=124.;
 fig.panels[p].axes=Some(Axes::new([-34.,34.],[-80.,10.],"HF incidence (degrees; negative northward)","Received power relative to reference (dB)"));
 for (freq,color) in [(5.33,BLUE),(4.51,RED)]{
  let mut pts=vec![];for name in h.group("incidence")?.member_names()?{
   let g=h.group(&format!("incidence/{name}"))?;if (g.attr("hf_mhz")?.read_scalar::<f64>()?-freq).abs()>0.001{continue;}
   pts.push([g.attr("angle_deg")?.read_scalar::<f64>()?,10.*g.attr("optimal_relative_power")?.read_scalar::<f64>()?.log10()]);
  }pts.sort_by(|a,b|a[0].total_cmp(&b[0]));fig.panels[p].push(Layer::Line(Line::new(pts,color)));
 }
 let mut t=Line::new(vec![[-34.,10.*threshold.log10()],[34.,10.*threshold.log10()]],GRAY);t.dash=Some("4,3".into());fig.panels[p].push(Layer::Line(t));
 legend(&mut fig,p,vec![LegendEntry::line("5.33 MHz",BLUE),LegendEntry::line("4.51 MHz",RED),LegendEntry::line("5.33 MHz detection limit",GRAY)],3);save(&fig,"muldrew_incidence")?;
 let mut fig=Figure::new(IntendedUse::ArticleWide);let p=fig.grid(1,1)?[0];fig.panels[p].rect=Rect::new(23.,29.,140.,78.);fig.height_mm=130.;
 fig.panels[p].axes=Some(Axes::new([7.6,8.55],[-70.,40.],r"$\log_{10}(|d n_e/dz|\,/\,\mathrm{m}^{-4})$","Received power / detection limit (dB)"));
 let mut entries=vec![];for (nu,color) in [(100.,BLUE),(1000.,RED),(3000.,"#4c7566"),(10000.,GRAY)]{
  let mut pts=vec![];for name in h.group("gradient_collision")?.member_names()?{
   let g=h.group(&format!("gradient_collision/{name}"))?;if (g.attr("collision_s")?.read_scalar::<f64>()?-nu).abs()>0.1{continue;}
   pts.push([g.attr("density_gradient_m4")?.read_scalar::<f64>()?.log10(),10.*(g.attr("optimal_relative_power")?.read_scalar::<f64>()?/threshold).log10()]);
  }pts.sort_by(|a,b|a[0].total_cmp(&b[0]));fig.panels[p].push(Layer::Line(Line::new(pts,color)));entries.push(LegendEntry::line(format!(r"$\nu_e={nu:.0}\ \mathrm{{s}}^{{-1}}$"),color));
 }
 let mut t=Line::new(vec![[7.6,0.],[8.55,0.]],"#333333");t.dash=Some("4,3".into());t.width_pt=0.6;fig.panels[p].push(Layer::Line(t));legend(&mut fig,p,entries,2);save(&fig,"muldrew_gradient_collision")?;

 let angles=rd(&h,"allowed_map/angle_deg")?;let grads=rd(&h,"allowed_map/density_gradient_m4")?;
 let ys:Vec<_>=grads.iter().map(|v|v.log10()).collect();
 let edges=|v:&Vec<f64>|{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|w|(w[0]+w[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e};
 let xe=edges(&angles);let ye=edges(&ys);
 let mut fig=Figure::new(IntendedUse::ArticleWide);let p=fig.grid(1,1)?[0];fig.panels[p].rect=Rect::new(23.,10.,135.,65.);fig.height_mm=113.;fig.width_mm=190.;
 fig.panels[p].axes=Some(Axes::new([xe[0],xe[xe.len()-1]],[ye[0],ye[ye.len()-1]],"HF incidence (degrees; negative northward)",r"$\log_{10}(|d n_e/dz|\,/\,\mathrm{m}^{-4})$"));
 let sc=fig.scale(Scale::new([-40.,40.],Normalization::Linear,ColorMap::diverging())?);
 fig.panels[p].push(Layer::Pcolormesh(Field::new(xe,ye,rd(&h,"allowed_map/margin_db")?,sc)?));
 let mut cb=Colorbar::new(Rect::new(28.,90.,125.,3.),sc,"Received power / detection limit (dB)",vec![-40.,-20.,0.,20.,40.]);cb.horizontal=true;fig.colorbars.push(cb);fig.layout_colorbars()?;save(&fig,"muldrew_allowed_region")?;
 let dh=File::open("data/landau_bragg_study.h5")?;
 for freq in ["4.51","5.33"]{
  let mut fig=Figure::new(IntendedUse::ArticleWide);let p=fig.grid(1,1)?[0];fig.panels[p].rect=Rect::new(23.,24.,140.,78.);fig.height_mm=125.;
  fig.panels[p].axes=Some(Axes::new([-100.,0.],[-60.,0.],"Power remaining from damping (dB)","Height from plasma resonance (m)"));
  let pre=format!("1525453430/{freq}");let z=rd(&dh,&format!("{pre}/height_relative_resonance_m"))?;
  let mut entries=vec![];
  for (mode,label,color,dash) in [("combined","Combined",RED,None),("collisional","Collisional",BLUE,Some("4,3")),("collisionless","Collisionless","#4c7566",Some("1,3"))]{
   let power=rd(&dh,&format!("{pre}/damping/{mode}/power_survival"))?;
   let mut l=Line::new(power.iter().zip(z.iter()).map(|(&a,&z)|[10.*a.max(1e-100).log10(),z]).collect(),color);l.dash=dash.map(|v|v.into());fig.panels[p].push(Layer::Line(l));entries.push(LegendEntry::line(label,color));
  }
  let dz=dh.group(&pre)?.attr("resonance_to_bragg_m")?.read_scalar::<f64>()?;
  let mut l=Line::new(vec![[-100.,-dz],[0.,-dz]],GRAY);l.dash=Some("3,3".into());l.width_pt=0.5;fig.panels[p].push(Layer::Line(l));legend(&mut fig,p,entries,3);save(&fig,&format!("muldrew_damping_{}",freq.replace('.',"_")))?;
 }
 Ok(())
}
