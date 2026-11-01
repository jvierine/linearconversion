// Scientific arrays: coupled_sim/scripts/summarize_fregion_comparison.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(&format!("plot/{k}"))?.read_raw::<f64>()?)}
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],color:&str,w:f64){
 let a=f.panels[p].axes.as_ref().unwrap();let pts=x.iter().zip(y).filter(|(x,y)|x.is_finite()&&y.is_finite()&&**x>=a.xlim[0]&&**x<=a.xlim[1]&&**y>=a.ylim[0]&&**y<=a.ylim[1]).map(|(&x,&y)|[x,y]).collect();let mut l=Line::new(pts,color);l.width_pt=w;f.panels[p].push(Layer::Line(l));
}
fn main()->Result<()>{
 let h=File::open("coupled_sim/output/coupled/gradient_study/observed_f_summary.h5")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,2)?;f.width_mm=250.;f.height_mm=145.;
 for i in 0..2{f.panels[ps[i]].rect=Rect::new(27.+124.*i as f64,9.,90.,103.);let mut ax=Axes::new(if i==0{[0.4,120.]}else{[-26.,26.]},[-125.,30.],if i==0{r"Density scale length $L_n$ (km)"}else{r"Vacuum-equivalent incidence ($^\circ$)"},"Relative Bragg response (dB)");if i==0{ax.set_xscale(AxisScale::Log10)?;}f.panels[ps[i]].axes=Some(ax);let x=f.panels[ps[i]].axes.as_ref().unwrap().xlim;line(&mut f,ps[i],&x,&[0.,0.],"#b4bac1",0.6);}
 for (key,col) in [("E","#245774"),("F","#8b3940")]{line(&mut f,ps[0],&rd(&h,&format!("{key}_Ln_km"))?,&rd(&h,&format!("{key}_response_dB"))?,col,1.3);}
 let p=rd(&h,"measured_point")?;line(&mut f,ps[0],&[p[0]/1.06,p[0]*1.06],&[0.,0.],"#4c7566",2.);line(&mut f,ps[0],&[p[0],p[0]],&[-3.,3.],"#4c7566",2.);
 f.panels[ps[0]].legend=Some(Legend::new(vec![LegendEntry::line("E control, 300 K","#245774"),LegendEntry::line("F background","#8b3940"),LegendEntry::line("Es background","#4c7566")]));
 for (key,col) in [("measured","#4c7566"),("E5","#245774"),("Fobserved","#8b3940")]{line(&mut f,ps[1],&rd(&h,&format!("{key}_angles"))?,&rd(&h,&format!("{key}_angular_dB"))?,col,1.3);}
 f.panels[ps[1]].legend=Some(Legend::new(vec![LegendEntry::line("Es background","#4c7566"),LegendEntry::line("E control, 5 km","#245774"),LegendEntry::line("F background","#8b3940")]));
 for ext in ["pdf","png"]{f.save(format!("figures/gradient_comparison.{ext}"))?;}
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,2)?;f.width_mm=250.;f.height_mm=155.;
 for i in 0..2{f.panels[ps[i]].rect=Rect::new(27.+124.*i as f64,8.,89.,118.);f.panels[ps[i]].axes=Some(Axes::new(if i==0{[2.,5.]}else{[0.,500.]},[200.,270.],if i==0{r"$n_e$ ($10^{11}$ m$^{-3}$)"}else{r"Density scale length $L_n$ (km)"},"Altitude above radar (km)"));}
 let z=rd(&h,"F_altitude_km")?;
 line(&mut f,ps[0],&rd(&h,"F_native_ne")?,&z,"#b4bac1",0.6);
 for (s,col) in [(5,"#8b3940"),(10,"#245774"),(15,"#4c7566")]{
  line(&mut f,ps[0],&rd(&h,&format!("F_ne_{s}km"))?,&z,col,1.2);
  line(&mut f,ps[1],&rd(&h,&format!("F_Ln_{s}km"))?,&z,col,1.2);
 }
 let zr=h.attr("F_center_km")?.read_scalar::<f64>()?;
 for i in 0..2{let lim=f.panels[ps[i]].axes.as_ref().unwrap().xlim;line(&mut f,ps[i],&lim,&[zr,zr],"#b4bac1",0.6);}
 f.panels[ps[0]].legend=Some(Legend::new(vec![LegendEntry::line("Native density","#b4bac1"),LegendEntry::line(r"$\sigma_z=5$ km","#8b3940"),LegendEntry::line(r"$\sigma_z=10$ km","#245774"),LegendEntry::line(r"$\sigma_z=15$ km","#4c7566")]));
 for ext in ["pdf","png"]{f.save(format!("figures/F_region_background.{ext}"))?;}Ok(())
}
