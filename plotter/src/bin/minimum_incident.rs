// Scientific arrays: coupled_sim/scripts/plot_minimum_incident_data.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(&format!("plot/{k}"))?.read_raw::<f64>()?)}
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],color:&str,dash:bool){let mut l=Line::new(x.iter().zip(y).map(|(&x,&y)|[x,y]).collect(),color);l.width_pt=0.9;if dash{l.dash=Some("3 2".into());}f.panels[p].push(Layer::Line(l));}
fn save(f:&Figure,name:&str)->Result<()>{for ext in ["png","pdf"]{f.save(format!("figures/{name}.{ext}"))?;}Ok(())}
fn main()->Result<()>{
 let path=std::env::args().nth(1).unwrap_or("data/minimum_incident_plots.h5".into());let h=File::open(path)?;
 let z=rd(&h,"height_m")?;let bm=h.attr("bragg_height_relative_m")?.read_scalar::<f64>()?;
 let mut f=ArticleLayout::new(177.,10.)?.figure(1.,105.)?;let ps=f.grid(1,2)?;f.height_mm=105.;
 f.panels[ps[0]].rect=Rect::new(20.,8.,44.,62.);f.panels[ps[1]].rect=Rect::new(82.,8.,89.,62.);
 let mut ax=Axes::new([0.001,20.],[-15.,3.],r"$|E_z|$ (mV/m)","Height relative to resonance (m)");ax.set_xscale(AxisScale::Log10)?;ax.yticks=vec![-15.,-10.,-5.,0.];f.panels[ps[0]].axes=Some(ax);
 line(&mut f,ps[0],&rd(&h,"Ez_mV_m")?,&z,"#245774",false);
 line(&mut f,ps[0],&[0.001,20.],&[bm,bm],"#6c7378",true);
 let x=rd(&h,"time_edges_us")?;let y=rd(&h,"z_edges_m")?;
 let mut ax=Axes::new([x[0],*x.last().unwrap()],[-15.,3.],r"Time ($\mu$s)","");ax.show_y_axis=false;ax.xticks=vec![0.,0.2,0.4];f.panels[ps[1]].axes=Some(ax);
 let sc=f.scale(Scale::new([-2.5,2.5],Normalization::Asinh{width:0.06},ColorMap::diverging())?);
 f.panels[ps[1]].push(Layer::Pcolormesh(Field::new(x.clone(),y,rd(&h,"density_timeheight_1e6_m3")?,sc)?));
 line(&mut f,ps[1],&[x[0],*x.last().unwrap()],&[bm,bm],"#555555",true);
 let mut bar=Colorbar::new(Rect::new(82.,87.,89.,2.5),sc,r"$\delta n_e$ ($10^6$ m$^{-3}$)",vec![-2.5,-0.5,0.,0.5,2.5]);bar.horizontal=true;f.colorbars.push(bar);
 f.labels.push(Label::new("(a)",20.,3.));f.labels.push(Label::new("(b)",82.,3.));
 save(&f,"minimum_incident_wave")?;
 let mut f=ArticleLayout::new(177.,10.)?.figure(1.,105.)?;let ps=f.grid(1,2)?;f.height_mm=105.;
 for i in 0..2{f.panels[ps[i]].rect=Rect::new(20.+85.*i as f64,6.,62.,79.);}
 f.panels[ps[0]].axes=Some(Axes::new([0.,1.],[-12.,2.],"Wavelength (m)","Height relative to resonance (m)"));
 let zz:Vec<_>=z.iter().map(|&v|if v< -0.8 && v> -12.{v}else{f64::NAN}).collect();let wave=rd(&h,"wavelength_m")?;
 line(&mut f,ps[0],&wave,&zz,"#245774",false);let lb=h.attr("bragg_wavelength_m")?.read_scalar::<f64>()?;
 line(&mut f,ps[0],&[lb,lb],&[-12.,2.],"#4c7566",true);line(&mut f,ps[0],&[0.,1.],&[bm,bm],"#8895a7",true);
 f.panels[ps[0]].legend=Some(Legend::new(vec![LegendEntry::line("Wave","#245774"),LegendEntry::line("Bragg","#4c7566")]));
 f.panels[ps[1]].axes=Some(Axes::new([5.27,5.41],[0.,100.],"Frequency magnitude (MHz)","Plasma-line SNR"));
 line(&mut f,ps[1],&rd(&h,"model_frequency_mhz")?,&rd(&h,"model_snr")?,"#8b3940",false);
 line(&mut f,ps[1],&rd(&h,"observed_frequency_mhz")?,&rd(&h,"observed_snr")?,"#245774",false);
 f.panels[ps[1]].legend=Some(Legend::new(vec![LegendEntry::line("Data","#245774"),LegendEntry::line("Model","#8b3940")]));
 save(&f,"minimum_incident_bragg")?;
 // Angle scan retained as supporting diagnostics.
 let angle=h.dataset("scan/angle_deg")?.read_raw::<f64>()?;let flux=h.dataset("scan/incident_vertical_flux_W_m2")?.read_raw::<f64>()?;
 let mut pts:Vec<_>=angle.iter().zip(flux).map(|(&x,y)|[x,y]).collect();pts.sort_by(|a,b|a[0].total_cmp(&b[0]));
 let mut f=ArticleLayout::new(177.,10.)?.figure(0.7,85.)?;let p=f.grid(1,1)?[0];f.height_mm=85.;f.panels[p].rect=Rect::new(23.,5.,95.,61.);
 let mut ax=Axes::new([-25.,25.],[1e-12,1e-7],r"Incidence angle ($^\circ$)",r"Required upward flux (W m$^{-2}$)");ax.set_yscale(AxisScale::Log10)?;f.panels[p].axes=Some(ax);f.panels[p].push(Layer::Line(Line::new(pts,"#245774")));save(&f,"minimum_incident_scan")?;
 Ok(())
}
