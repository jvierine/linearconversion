// Arrays: coupled_sim/scripts/coupled_losses.py; HF loss: far_field_budget.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn read(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(&format!("plot/{k}"))?.read_raw::<f64>()?)}
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],color:&str){
 let a=f.panels[p].axes.as_ref().unwrap();let pts=x.iter().zip(y).filter(|(x,y)|x.is_finite()&&y.is_finite()&&**x>=a.xlim[0]&&**x<=a.xlim[1]&&**y>=a.ylim[0]&&**y<=a.ylim[1]).map(|(&x,&y)|[x,y]).collect();let mut l=Line::new(pts,color);l.width_pt=1.15;f.panels[p].push(Layer::Line(l));
}
fn main()->Result<()>{
 let h=File::open("coupled_sim/output/coupled/losses.h5")?;let z=read(&h,"height_m")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,3)?;f.width_mm=285.;f.height_mm=155.;
 let xr=[[0.,1.05],[0.,0.28],[-0.16,1.05]];let yr=[[-600.,100.],[-30.,30.],[-100.,30.]];
 let labels=["Cumulative heating / incident flux",r"Heating / incident flux (m$^{-1}$)","Vertical flux / incident flux"];
 for i in 0..3{f.panels[ps[i]].rect=Rect::new(24.+94.*i as f64,8.,69.,118.);f.panels[ps[i]].axes=Some(Axes::new(xr[i],yr[i],labels[i],"Height relative to resonance (m)"));line(&mut f,ps[i],&xr[i],&[0.,0.],"#b4bac1");}
 line(&mut f,ps[0],&read(&h,"cumulative_loss")?,&z,"#245774");
 line(&mut f,ps[1],&read(&h,"loss_ei_per_m")?,&z,"#245774");line(&mut f,ps[1],&read(&h,"loss_en_per_m")?,&z,"#8b3940");
 f.panels[ps[1]].legend=Some(Legend::new(vec![LegendEntry::line("Electron--ion","#245774"),LegendEntry::line("Electron--neutral","#8b3940")]));
 line(&mut f,ps[2],&read(&h,"EM_flux")?,&z,"#245774");line(&mut f,ps[2],&read(&h,"pressure_flux")?,&z,"#4c7566");
 f.panels[ps[2]].legend=Some(Legend::new(vec![LegendEntry::line("EM","#245774"),LegendEntry::line("Plasma","#4c7566")]));
 for ext in ["pdf","png"]{f.save(format!("figures/coupled_energy_losses.{ext}"))?;}Ok(())
}
