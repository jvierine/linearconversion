// Scientific arrays from hf_fdtd/examples/summarize_coupled.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
const BLUE:&str="#245774";const RED:&str="#8b3940";const GREEN:&str="#4c7566";
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(&format!("plot/{k}"))?.read_raw::<f64>()?)}
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],col:&str,width:f64){
 let a=f.panels[p].axes.as_ref().unwrap();let pts=x.iter().zip(y).filter(|(x,y)|x.is_finite()&&y.is_finite()&&**x>=a.xlim[0]&&**x<=a.xlim[1]&&**y>=a.ylim[0]&&**y<=a.ylim[1]).map(|(&x,&y)|[x,y]).collect();let mut l=Line::new(pts,col);l.width_pt=width;f.panels[p].push(Layer::Line(l));
}
fn legend(f:&mut Figure,p:usize,e:Vec<LegendEntry>){let mut l=Legend::new(e);l.columns=1;f.panels[p].legend=Some(l);}
fn save(f:&Figure,name:&str)->Result<()>{for ext in ["png","pdf"]{f.save(format!("figures/{name}.{ext}"))?;}Ok(())}
fn main()->Result<()>{
 let h=File::open("output/coupled/summary.h5")?;let z=rd(&h,"height_m")?;let bragg=h.attr("bragg_height_relative_m")?.read_scalar::<f64>()?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,3)?;f.width_mm=285.;f.height_mm=154.;
 let xs=[[0.001,30.],[-12.,12.],[-0.16,1.05]];
 for i in 0..3{f.panels[ps[i]].rect=Rect::new(24.+94.*i as f64,8.,69.,118.);let mut a=Axes::new(xs[i],[-15.,5.],[r"$|E_z|$ (mV/m)",r"$\delta n_e/n_0$ (ppm)","Vertical flux / incident flux"][i],"Height relative to resonance (m)");if i==0{a.set_xscale(AxisScale::Log10)?;}f.panels[ps[i]].axes=Some(a);line(&mut f,ps[i],&xs[i],&[bragg,bragg],"#a4acb2",0.6);}
 line(&mut f,ps[0],&rd(&h,"cold_Ez_mV_m")?,&z,"#8895a7",1.);line(&mut f,ps[0],&rd(&h,"Ez_mV_m")?,&z,BLUE,1.2);
 legend(&mut f,ps[0],vec![LegendEntry::line("Cold","#8895a7"),LegendEntry::line("Coupled",BLUE)]);
 line(&mut f,ps[1],&rd(&h,"density_real_ppm")?,&z,BLUE,0.65);line(&mut f,ps[1],&rd(&h,"density_amplitude_ppm")?,&z,RED,1.1);
 legend(&mut f,ps[1],vec![LegendEntry::line("Wave",BLUE),LegendEntry::line("Envelope",RED)]);
 line(&mut f,ps[2],&rd(&h,"EM_flux")?,&z,BLUE,1.2);line(&mut f,ps[2],&rd(&h,"pressure_flux")?,&z,GREEN,1.2);
 legend(&mut f,ps[2],vec![LegendEntry::line("EM",BLUE),LegendEntry::line("Plasma",GREEN)]);
 save(&f,"coupled_hf_langmuir")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,2)?;f.width_mm=235.;f.height_mm=140.;
 for i in 0..2{f.panels[ps[i]].rect=Rect::new(24.+116.*i as f64,8.,81.,105.);}
 f.panels[ps[0]].axes=Some(Axes::new([0.,1.],[-12.,2.],"Wavelength (m)","Height relative to resonance (m)"));
 let zp:Vec<_>=z.iter().map(|&v|if v< -0.8&&v> -12.{v}else{f64::NAN}).collect();
 line(&mut f,ps[0],&rd(&h,"wavelength_phase_m")?,&zp,BLUE,1.1);
 let lb=h.attr("bragg_wavelength_m")?.read_scalar::<f64>()?;line(&mut f,ps[0],&[lb,lb],&[-12.,2.],GREEN,0.7);line(&mut f,ps[0],&[0.,1.],&[bragg,bragg],"#a4acb2",0.6);
 legend(&mut f,ps[0],vec![LegendEntry::line("Wave",BLUE),LegendEntry::line("Bragg",GREEN)]);
 f.panels[ps[1]].axes=Some(Axes::new([5.27,5.41],[0.,100.],"Frequency magnitude (MHz)","Plasma-line SNR"));
 line(&mut f,ps[1],&rd(&h,"spectrum_frequency_mhz")?,&rd(&h,"spectrum_snr")?,RED,1.1);
 line(&mut f,ps[1],&rd(&h,"frequency_mhz")?,&rd(&h,"observed_snr")?,BLUE,0.8);
 legend(&mut f,ps[1],vec![LegendEntry::line("Data",BLUE),LegendEntry::line("Scaled model",RED)]);
 save(&f,"coupled_bragg_response")?;
 Ok(())
}
