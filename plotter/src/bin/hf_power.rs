// Data from estimate_hf_power.py. Beam filling and phase coherence are distinct.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
const BLUE:&str="#245774";const RED:&str="#8b3940";const GREEN:&str="#4c7566";
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],col:&str,width:f64){
 let a=f.panels[p].axes.as_ref().unwrap();let points=x.iter().zip(y).filter(|(x,y)|x.is_finite()&&y.is_finite()&&**x>=a.xlim[0]&&**x<=a.xlim[1]&&**y>=a.ylim[0]&&**y<=a.ylim[1]).map(|(&x,&y)|[x,y]).collect();
 let mut l=Line::new(points,col);l.width_pt=width;f.panels[p].push(Layer::Line(l));
}
fn legend(f:&mut Figure,p:usize,entries:Vec<LegendEntry>){let mut l=Legend::new(entries);l.columns=1;f.panels[p].legend=Some(l);}
fn main()->Result<()>{
 let h=File::open("output/fluid/hf_power_from_measured_pl.h5")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,3)?;f.width_mm=285.;f.height_mm=147.;
 for i in 0..3 {f.panels[ps[i]].rect=Rect::new(25.+94.*i as f64,9.,68.,107.);}
 f.panels[ps[0]].axes=Some(Axes::new([5.27,5.41],[0.,105.],"Frequency magnitude (MHz)","Plasma-line SNR"));
 let nu:Vec<_>=rd(&h,"line_frequency_hz")?.iter().map(|v|v/1e6).collect();let y=rd(&h,"line_measured_snr")?;
 let fit=rd(&h,"line_fit_fixed")?;let xx:Vec<_>=(0..1001).map(|i|5.27+0.14*i as f64/1000.).collect();
 let yy:Vec<_>=xx.iter().map(|&nu|{let a=std::f64::consts::PI*(nu*1e6-fit[1])*44e-6;fit[0]*(if a.abs()<1e-10{1.}else{a.sin()/a}).powi(2)+fit[2]}).collect();
 line(&mut f,ps[0],&xx,&yy,RED,1.2);
 let mut pan=f.panels[ps[0]].clone();let pts:Vec<_>=nu.iter().zip(&y).map(|(&x,&y)|[x,y]).collect();pan.markers(&mut f,&pts,&MarkerStyle{shape:MarkerShape::Circle,size_pt:3.5,color:[0.141,0.341,0.455]})?;f.panels[ps[0]]=pan;
 legend(&mut f,ps[0],vec![LegendEntry::circle("Data",BLUE),LegendEntry::line("Pulse fit",RED)]);
 f.panels[ps[1]].axes=Some(Axes::new([0.,6.],[100.,140.],r"$\int\mathrm{SNR}\,df$ (MHz)","Altitude above radar (km)"));
 let z:Vec<_>=rd(&h,"ion_height_m")?.iter().map(|v|v/1000.).collect();let arr=rd(&h,"ion_integrated_snr_Hz")?;
 for (v,col,width) in [(arr[z.len()..2*z.len()].to_vec(),BLUE,1.5),(rd(&h,"ion_theory_integrated_snr_Hz")?,"#8895a7",1.),(rd(&h,"ion_scaled_theory_Hz")?,RED,0.8)]{line(&mut f,ps[1],&v.iter().map(|v|v/1e6).collect::<Vec<_>>(),&z,col,width);}
 legend(&mut f,ps[1],vec![LegendEntry::line("Data",BLUE),LegendEntry::line("Nominal", "#8895a7"),LegendEntry::line("Scaled",RED)]);
 let mut a=Axes::new([1.,1000.],[1.,1e4],r"Transverse phase length $\ell_\perp$ (m)","Required HF EIRP (W)");a.set_xscale(AxisScale::Log10)?;a.set_yscale(AxisScale::Log10)?;f.panels[ps[2]].axes=Some(a);
 line(&mut f,ps[2],&rd(&h,"coherence_length_m")?,&rd(&h,"eirp125km_vs_coherence_W")?,BLUE,1.2);
 let v=h.attr("eirp_at125km_W")?.read_scalar::<f64>()?;line(&mut f,ps[2],&[1.,1000.],&[v,v],GREEN,0.7);
 legend(&mut f,ps[2],vec![LegendEntry::line("125 km path",BLUE),LegendEntry::line("Coherent limit",GREEN)]);
 for ext in ["png","pdf"]{f.save(format!("figures/hf_power_from_measured_pl.{ext}"))?;}
 Ok(())
}
