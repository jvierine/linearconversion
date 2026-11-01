// Data: calculate_airy_fullwave.py. Native Rustplotlib figure/typography defaults.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
const BLUE:&str="#245774"; const RED:&str="#8b3940"; const GREEN:&str="#4c7566"; const GRAY:&str="#8795a9";
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn line(f:&mut Figure,p:usize,x:&[f64],z:&[f64],color:&str){
 f.panels[p].push(Layer::Line(Line::new(x.iter().zip(z).map(|(&x,&z)|[x,z]).collect(),color)));
}
fn marker(f:&mut Figure,p:usize,x:[f64;2],z:f64){let mut l=Line::new(vec![[x[0],z],[x[1],z]],GRAY);l.width_pt=0.6;l.dash=Some("3,3".into());f.panels[p].push(Layer::Line(l));}
fn save(f:&Figure,n:&str)->Result<()>{for ext in ["png","svg","pdf"]{f.save(format!("figures/{n}.{ext}"))?;}Ok(())}
fn main()->Result<()>{
 let h=File::open("data/airy_hf.h5")?;let z=rd(&h,"z_from_cutoff_m")?;let a=rd(&h,"amplitude_over_Eref")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let p=f.grid(1,1)?[0];
 f.panels[p].rect=Rect::new(27.,23.,132.,112.);f.height_mm=155.;
 f.panels[p].axes=Some(Axes::new([0.,4.5],[-1000.,500.],r"Electric-field magnitude $|E|/E_{\rm ref}$","Height relative to reflection (m)"));
 f.panels[p].axes.as_mut().unwrap().yticks=vec![-1000.,-750.,-500.,-250.,0.,250.,500.];
 marker(&mut f,p,[0.,4.5],0.);line(&mut f,p,&a,&z,BLUE);
 let mut leg=Legend::new(vec![LegendEntry::line("Field magnitude",BLUE),LegendEntry::line("Reflection height",GRAY)]);leg.position=LegendPosition::AboveCenter;leg.columns=2;f.panels[p].legend=Some(leg);
 save(&f,"airy_hf_field")?;
 if std::env::args().any(|arg|arg=="--airy-only"){return Ok(());}
 if !std::path::Path::new("data/airy_measured_fullwave_plot.h5").exists(){return Ok(());}
 let h=File::open("data/airy_measured_fullwave_plot.h5")?;
 let z=rd(&h,"overview_height_km")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,3)?;f.width_mm=245.;f.height_mm=155.;
 for (i,(name,label)) in [("ne_1e11",r"$n_e$ ($10^{11}$ m$^{-3}$)"),("Ehorizontal",r"$|E_{\perp z}|/E_{\rm ref}$"),("Ez",r"$|E_z|/E_{\rm ref}$")].iter().enumerate(){
  let x=rd(&h,name)?;let mx=x.iter().copied().fold(0.,f64::max)*1.08;
  let p=ps[i];f.panels[p].rect=Rect::new(25.+i as f64*80.,17.,55.,110.);
  f.panels[p].axes=Some(Axes::new([0.,mx],[117.6,118.6],*label,if i==0{"Altitude above radar (km)"}else{""}));
  marker(&mut f,p,[0.,mx],h.attr("resonance_height_km")?.read_scalar::<f64>()?);
  line(&mut f,p,&x,&z,BLUE);
 }
 save(&f,"incoming_hf_fullwave")?;
 let z=rd(&h,"fine_height_from_resonance_m")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,2)?;f.width_mm=220.;f.height_mm=158.;
 for (i,(name,label)) in [("fine_Ez_real",r"$E_z/E_{\rm ref}$"),("fine_density_real",r"$\delta n_e/n_e$ (\%)")].iter().enumerate(){
  let x=rd(&h,name)?;let env=rd(&h,if i==0{"fine_Ez_abs"}else{"fine_density_abs"})?;
  let mx=env.iter().copied().fold(0.,f64::max)*1.08;let p=ps[i];
  f.panels[p].rect=Rect::new(25.+i as f64*108.,25.,78.,105.);
  f.panels[p].axes=Some(Axes::new([-mx,mx],[-8.,2.],*label,if i==0{"Height relative to resonance (m)"}else{""}));
  marker(&mut f,p,[-mx,mx],0.);
  let zb=h.attr("bragg_from_resonance_m")?.read_scalar::<f64>()?;
  let mut bragg_line=Line::new(vec![[-mx,zb],[mx,zb]],GREEN);bragg_line.width_pt=0.6;bragg_line.dash=Some("5,3".into());f.panels[p].push(Layer::Line(bragg_line));
  line(&mut f,p,&x,&z,BLUE);line(&mut f,p,&env,&z,RED);
  let mut leg=Legend::new(vec![LegendEntry::line("Instantaneous",BLUE),LegendEntry::line("Amplitude",RED),LegendEntry::line("430 MHz match",GREEN)]);leg.position=LegendPosition::AboveCenter;leg.columns=2;f.panels[p].legend=Some(leg);
 }
 save(&f,"incoming_hf_density_wave")?;
 let z=rd(&h,"response_height_from_resonance_m")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,2)?;f.width_mm=220.;f.height_mm=155.;
 for (i,(names,labels)) in [(["response_Ez_intensity","response_density_intensity"],["Vertical E intensity","Density-wave intensity"]),(["poynting_flux","pressure_flux"],["EM energy flux","Pressure energy flux"])].iter().enumerate(){
  let p=ps[i];f.panels[p].rect=Rect::new(25.+i as f64*108.,27.,78.,100.);
  let x0=rd(&h,names[0])?;let x1=rd(&h,names[1])?;
  let lo=if i==0{0.}else{x0.iter().chain(&x1).copied().fold(0.,f64::min)*1.1};
  let hi=if i==0{1.05}else{x0.iter().chain(&x1).copied().fold(0.,f64::max)*1.1};
  f.panels[p].axes=Some(Axes::new([lo,hi],[-500.,250.],if i==0{"Intensity / peak intensity"}else{"Vertical flux / incident flux"},if i==0{"Height relative to resonance (m)"}else{""}));
  f.panels[p].axes.as_mut().unwrap().yticks=vec![-500.,-400.,-300.,-200.,-100.,0.,100.,200.];
  marker(&mut f,p,[lo,hi],0.);line(&mut f,p,&x0,&z,BLUE);line(&mut f,p,&x1,&z,GREEN);
  let mut leg=Legend::new(vec![LegendEntry::line(labels[0],BLUE),LegendEntry::line(labels[1],GREEN)]);leg.position=LegendPosition::AboveCenter;leg.columns=1;f.panels[p].legend=Some(leg);
 }
 save(&f,"incoming_hf_vertical_extent")?;
 let z=rd(&h,"comparison_height_m")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,2)?;f.width_mm=220.;f.height_mm=160.;
 for (i,yr) in [[-500.,250.],[-10.,25.]].iter().enumerate(){
  let p=ps[i];f.panels[p].rect=Rect::new(25.+i as f64*108.,30.,78.,105.);
  let mut ax=Axes::new([0.01,300.],*yr,r"Field magnitude / $E_{\rm ref}$",if i==0{"Height relative to plasma resonance (m)"}else{""});ax.set_xscale(AxisScale::Log10)?;
  ax.yticks=if i==0{vec![-500.,-250.,0.,250.]}else{vec![-10.,0.,10.,20.]};f.panels[p].axes=Some(ax);
  marker(&mut f,p,[0.01,300.],0.);
  for (name,color) in [("comparison_airy",GRAY),("comparison_total",BLUE),("comparison_vertical",RED)]{line(&mut f,p,&rd(&h,name)?,&z,color);}
  let mut leg=Legend::new(vec![LegendEntry::line("Airy transverse",GRAY),LegendEntry::line("Full-wave total",BLUE),LegendEntry::line("Full-wave vertical",RED)]);leg.position=LegendPosition::AboveCenter;leg.columns=1;f.panels[p].legend=Some(leg);
 }
 save(&f,"hf_field_magnitude_comparison")?;
 Ok(())
}
