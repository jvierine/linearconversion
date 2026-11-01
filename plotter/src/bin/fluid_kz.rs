// Scientific arrays: extend_fluid_kz.py. Fixed-frequency warm fluid, not kinetic.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
const BLUE:&str="#245774"; const RED:&str="#8b3940"; const GREEN:&str="#4c7566";
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(&format!("plot/{k}"))?.read_raw::<f64>()?)}
fn attr(h:&File,k:&str)->Result<f64>{Ok(h.group("summary")?.attr(k)?.read_scalar::<f64>()?)}
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],col:&str,width:f64){
 let a=f.panels[p].axes.as_ref().unwrap();
 let points=x.iter().zip(y).enumerate().filter(|(i,(x,y))|i%8==0&&x.is_finite()&&y.is_finite()&&**x>=a.xlim[0]&&**x<=a.xlim[1]&&**y>=a.ylim[0]&&**y<=a.ylim[1]).map(|(_,(&x,&y))|[x,y]).collect();
 let mut l=Line::new(points,col);l.width_pt=width;f.panels[p].push(Layer::Line(l));
}
fn segment(f:&mut Figure,p:usize,points:Vec<[f64;2]>,col:&str){let mut l=Line::new(points,col);l.width_pt=0.7;f.panels[p].push(Layer::Line(l));}
fn legend(f:&mut Figure,p:usize,entries:Vec<LegendEntry>){let mut l=Legend::new(entries);l.columns=1;f.panels[p].legend=Some(l);}
fn label(f:&mut Figure,p:usize,text:&str,xy:[f64;2]){let q=f.panels[p].data_to_figure(xy);f.labels.push(Label::new(text,q[0],q[1]));}
fn mark(f:&mut Figure,p:usize,x:f64,y:f64)->Result<()>{
 let mut pan=f.panels[p].clone();pan.markers(f,&[[x,y]],&MarkerStyle{shape:MarkerShape::Circle,size_pt:5.,color:[0.298,0.459,0.4]})?;f.panels[p]=pan;Ok(())
}
fn save(f:&Figure,n:&str)->Result<()>{for ext in ["png","pdf"]{f.save(format!("figures/{n}.{ext}"))?;}Ok(())}
fn main()->Result<()>{
 let h=File::open("output/fluid/fdtd_extended_kz.h5")?;
 let z=rd(&h,"z")?;let zo=rd(&h,"z_out")?;let k=rd(&h,"k_local")?;let kp=rd(&h,"k_phase")?;
 let lambda=rd(&h,"lambda_local")?;let lp:Vec<_>=kp.iter().map(|k|2.*std::f64::consts::PI/k).collect();
 let bragg=attr(&h,"bragg_height_relative_m")?;let kb=attr(&h,"bragg_k_rad_m")?;let lb=attr(&h,"bragg_wavelength_m")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,3)?;f.width_mm=280.;f.height_mm=140.;
 let xs=[[-35.,35.],[0.,1.],[0.,36.]];
 for i in 0..3{f.panels[ps[i]].rect=Rect::new(24.+92.*i as f64,8.,67.,105.);f.panels[ps[i]].axes=Some(Axes::new(xs[i],[-12.,2.],[r"$\delta n_e/n_0$ (ppm)","Wavelength (m)",r"Downward $|k_z|$ (rad/m)"][i],"Height relative to resonance (m)"));segment(&mut f,ps[i],vec![[xs[i][0],bragg],[xs[i][1],bragg]],GREEN);}
 let re:Vec<_>=rd(&h,"real")?.iter().map(|v|v*1e6).collect();let amp:Vec<_>=rd(&h,"amplitude")?.iter().map(|v|v*1e6).collect();
 line(&mut f,ps[0],&re,&z,BLUE,0.8);line(&mut f,ps[0],&amp,&z,RED,1.2);
 legend(&mut f,ps[0],vec![LegendEntry::line("Wave",BLUE),LegendEntry::line("Envelope",RED)]);
 for (i,a,b,bval) in [(1,&lambda,&lp,lb),(2,&k,&kp,kb)]{
  let zs:Vec<_>=z.iter().map(|&v|if v< -0.6 {v}else{f64::NAN}).collect();
  line(&mut f,ps[i],a,&zs,BLUE,1.6);line(&mut f,ps[i],b,&zs,RED,0.65);
  mark(&mut f,ps[i],bval,bragg)?;
  legend(&mut f,ps[i],vec![LegendEntry::line("Local dispersion",BLUE),LegendEntry::line("Solved phase",RED),LegendEntry::circle("430 MHz Bragg",GREEN)]);
 }
 label(&mut f,ps[1],r"$0.3486$ m",[0.56,-4.7]);label(&mut f,ps[2],r"$18.02$ rad/m",[26.,-4.7]);
 save(&f,"langmuir_bragg_match")?;

 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,3)?;f.width_mm=280.;f.height_mm=140.;
 let xs=[[0.,110.],[1e-12,1e-4],[-120.,5.]];
 for i in 0..3{f.panels[ps[i]].rect=Rect::new(24.+92.*i as f64,8.,67.,105.);let mut a=Axes::new(xs[i],[-100.,2.],[r"Downward $|k_z|$ (rad/m)",r"Density amplitude $|\delta n_e/n_0|$","Collisional attenuation (dB)"][i],"Height relative to resonance (m)");if i==1{a.set_xscale(AxisScale::Log10)?;}f.panels[ps[i]].axes=Some(a);segment(&mut f,ps[i],vec![[xs[i][0],bragg],[xs[i][1],bragg]],GREEN);}
 line(&mut f,ps[0],&k,&z,BLUE,1.6);line(&mut f,ps[0],&rd(&h,"k_out")?,&zo,RED,0.65);mark(&mut f,ps[0],kb,bragg)?;
 legend(&mut f,ps[0],vec![LegendEntry::line("Local dispersion",BLUE),LegendEntry::line("Outgoing phase",RED),LegendEntry::circle("430 MHz Bragg",GREEN)]);
 line(&mut f,ps[1],&rd(&h,"amplitude")?,&z,BLUE,0.6);line(&mut f,ps[1],&rd(&h,"amplitude_out")?,&zo,RED,1.3);
 legend(&mut f,ps[1],vec![LegendEntry::line("Driven",BLUE),LegendEntry::line("Outgoing",RED)]);
 line(&mut f,ps[2],&rd(&h,"attenuation_db")?,&rd(&h,"attenuation_z")?,BLUE,1.3);
 legend(&mut f,ps[2],vec![LegendEntry::line("Collisions (WKB)",BLUE)]);
 for db in [20,40]{let zz=attr(&h,&format!("collision_minus{db}db_height_relative_m"))?;mark(&mut f,ps[2],-(db as f64),zz)?;}
 save(&f,"langmuir_kz_continuation")?;

 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,2)?;f.width_mm=220.;f.height_mm=140.;
 let height=rd(&h,"height_km")?;let lim=attr(&h,"formal_fluid_kmax_rad_m")?;
 for i in 0..2{f.panels[ps[i]].rect=Rect::new(24.+108.*i as f64,8.,75.,105.);f.panels[ps[i]].axes=Some(Axes::new(if i==0{[0.,310.]}else{[0.,1.]},[100.,118.],[r"Downward $|k_z|$ (rad/m)",r"$|k_z|\lambda_D$"][i],"Altitude above radar (km)"));}
 line(&mut f,ps[0],&rd(&h,"k_wide")?,&height,BLUE,1.3);segment(&mut f,ps[0],vec![[lim,100.],[lim,118.]],RED);
 legend(&mut f,ps[0],vec![LegendEntry::line("Fluid dispersion",BLUE),LegendEntry::line("Formal upper bound",RED)]);
 line(&mut f,ps[1],&rd(&h,"k_debye")?,&height,BLUE,1.3);
 label(&mut f,ps[1],"Fluid extrapolation",[0.30,115.]);label(&mut f,ps[1],"No Landau damping",[0.30,114.]);
 save(&f,"langmuir_kz_limit")?;
 Ok(())
}
