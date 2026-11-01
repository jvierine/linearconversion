// Data: prepare_fdtd_fluid_figures.py and native Rust fluid solver.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
const BLUE:&str="#245774";const RED:&str="#8b3940";const GREEN:&str="#4c7566";
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn line(f:&mut Figure,p:usize,x:&[f64],z:&[f64],color:&str){
 let xy=x.iter().zip(z).filter(|(x,z)|x.is_finite()&&z.is_finite()).map(|(&x,&z)|[x,z]).collect();
 f.panels[p].push(Layer::Line(Line::new(xy,color)));
}
fn save(f:&Figure,n:&str)->Result<()>{for ext in ["png","pdf"]{f.save(format!("figures/{n}.{ext}"))?;}Ok(())}
fn edges(v:&[f64])->Vec<f64>{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|w|(w[0]+w[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e}
fn profile(h:&File,phase:f64)->Result<Figure>{
 let z=rd(h,"z")?;let zf=rd(h,"zf")?;let es=rd(h,"source_amplitude_mVm")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,3)?;f.width_mm=270.;f.height_mm=143.;
 for i in 0..3{f.panels[ps[i]].rect=Rect::new(25.+89.*i as f64,9.,65.,106.);}
 let yr=[-12.,4.];let m=-h.attr("bragg_distance_m")?.read_scalar::<f64>()?;
 let xs=[[0.,3.],[-0.005,0.004],[0.,2.]];
 for i in 0..3{
  f.panels[ps[i]].axes=Some(Axes::new(xs[i],yr,[r"Imposed $|\widetilde E|$ (mV/m)",r"$\delta n_e/n_0$ (\%)","Wavelength (m)"][i],"Height relative to resonance (m)"));
  let mut l=Line::new(vec![[xs[i][0],m],[xs[i][1],m]],"#a4acb2");l.width_pt=0.6;f.panels[ps[i]].push(Layer::Line(l));
 }
 for (i,col) in [BLUE,GREEN,RED].iter().enumerate(){line(&mut f,ps[0],&es[i*zf.len()..(i+1)*zf.len()],&zf,col);}
 let re=rd(h,"density_real_percent")?;let im=rd(h,"density_imag_percent")?;
 let signed:Vec<_>=re.iter().zip(im).map(|(a,b)|a*phase.cos()+b*phase.sin()).collect();
 line(&mut f,ps[1],&signed,&z,BLUE);line(&mut f,ps[1],&rd(h,"density_amplitude_percent")?,&z,RED);
 for (key,col) in [("wavelength_theory_m",BLUE),("wavelength_simulation_m",RED)]{
  let xx=rd(h,key)?;let pairs:Vec<_>=xx.iter().zip(&z).filter(|(x,z)|**z< -0.4&&**z>-12.&&**x<2.).map(|(&x,&z)|[x,z]).collect();
  f.panels[ps[2]].push(Layer::Line(Line::new(pairs,col)));
 }
 let lb=h.attr("bragg_wavelength_m")?.read_scalar::<f64>()?;line(&mut f,ps[2],&[lb,lb],&yr,GREEN);
 for (p,entries) in [(ps[0],vec![LegendEntry::line(r"$E_x$",BLUE),LegendEntry::line(r"$E_y$",GREEN),LegendEntry::line(r"$E_z$",RED)]),
  (ps[1],vec![LegendEntry::line("Wave",BLUE),LegendEntry::line("Amplitude",RED)]),
  (ps[2],vec![LegendEntry::line("Local dispersion",BLUE),LegendEntry::line("Solved phase",RED),LegendEntry::line("430 MHz Bragg",GREEN)])]{
  let mut l=Legend::new(entries);l.columns=1;f.panels[p].legend=Some(l);
 }
 Ok(f)
}
fn main()->Result<()>{
 let h=File::open("output/fluid/fdtd_fluid_plot.h5")?;
 save(&profile(&h,0.)?,"fdtd_driven_langmuir")?;
 let z=rd(&h,"z")?;let t=rd(&h,"time_ms")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,2)?;f.width_mm=240.;f.height_mm=154.;
 f.panels[ps[0]].rect=Rect::new(24.,10.,100.,102.);f.panels[ps[1]].rect=Rect::new(155.,10.,72.,102.);
 f.panels[ps[0]].axes=Some(Axes::new([0.,4.],[-12.,4.],"Time (ms)","Height relative to resonance (m)"));
 let sc=f.scale(Scale::new([0.000001,0.005],Normalization::Log,ColorMap::ember())?);
 f.panels[ps[0]].push(Layer::Pcolormesh(Field::new(edges(&t),edges(&z),rd(&h,"history_percent")?,sc)?));
 let mut cb=Colorbar::new(Rect::new(24.,132.,100.,2.5),sc,r"Density amplitude (\%)",vec![0.000001,0.00001,0.0001,0.001]);cb.horizontal=true;f.colorbars.push(cb);
 let mut ax=Axes::new([0.,4.],[1e-8,1.2],"Time (ms)","Energy / peak energy");ax.set_yscale(AxisScale::Log10)?;f.panels[ps[1]].axes=Some(ax);
 let en=rd(&h,"energy_relative")?;line(&mut f,ps[1],&t,&en,BLUE);
 for p in &ps{let yr=if *p==ps[0]{[-12.,4.]}else{[1e-8,1.2]};let mut l=Line::new(vec![[1.5,yr[0]],[1.5,yr[1]]],"#a4acb2");l.width_pt=0.7;f.panels[*p].push(Layer::Line(l));}
 f.layout_colorbars()?;save(&f,"fdtd_driven_evolution")?;
 if std::env::args().any(|a|a=="--animation"){
  std::fs::create_dir_all("output/fluid/frames")?;
  for j in 0..48 {profile(&h,2.*std::f64::consts::PI*j as f64/48.)?.save(format!("output/fluid/frames/phase_{j:03}.png"))?;}
 }
 Ok(())
}
