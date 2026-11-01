// Independent Boussinesq model products; not measured ISR quantities.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn read(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn edges(v:&[f64])->Vec<f64>{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|w|(w[0]+w[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e}
fn main()->Result<()> {
 let h=File::open("output/gravity_wave_study/simulation_figures.h5")?;
 let t=read(&h,"baseline/time_min")?;let z=read(&h,"baseline/height_km")?;let x=read(&h,"baseline/x_km")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let ids=f.grid(3,1)?;f.width_mm=190.;f.height_mm=222.;
 for (row,&p) in ids.iter().enumerate(){f.panels[p].rect=Rect::new(18.,12.+row as f64*68.,137.,45.);}
 for row in 0..2 {
  let p=ids[row];let xx=if row==0{x.clone()}else{t.clone()};
  let (ylim,lim,label,key)=if row==0{([110.,126.],[-0.05,0.05],r"Vorticity (s$^{-1}$)","baseline/snapshot_80_min/vorticity_s_1")}
  else{([100.,136.],[-2.,2.],r"$w$ (m s$^{-1}$)","baseline/w_advected_timeheight_m_s")};
  let js:Vec<_>=(0..z.len()).filter(|&j|z[j]>=ylim[0]&&z[j]<=ylim[1]).collect();let zz:Vec<_>=js.iter().map(|&j|z[j]).collect();
  let a=read(&h,key)?;let mut v=vec![];for &j in &js{for i in 0..xx.len(){v.push(if row==0{a[j*x.len()+i]}else{a[i*z.len()+j]});}}
  let mut ax=Axes::new([0.,if row==0{40.}else{90.}],ylim,if row==0{"Horizontal distance (km)"}else{""},"Altitude (km)");
  ax.xticks=if row==0{vec![0.,10.,20.,30.,40.]}else{vec![0.,30.,60.,90.]};ax.yticks=if row==0{vec![110.,118.,126.]}else{vec![100.,110.,118.,130.]};f.panels[p].axes=Some(ax);
  let sc=f.scale(Scale::new(lim,Normalization::Linear,ColorMap::diverging())?);f.panels[p].push(Layer::Pcolormesh(Field::new(edges(&xx),edges(&zz),v,sc)?));
  let r=f.panels[p].rect;f.colorbars.push(Colorbar::new(Rect::new(r.x+r.width+4.,r.y,2.5,r.height),sc,label,vec![lim[0],0.,lim[1]]));
 }
 let up=read(&h,"baseline/upward_flux_m3_s3")?;let down=read(&h,"baseline/downward_flux_m3_s3")?;let control=read(&h,"stable/upward_flux_m3_s3")?;let tc=read(&h,"stable/time_min")?;
 let max=up.iter().chain(&down).fold(0_f64,|a,&b|a.max(b.abs()))*1.1;
 let p=ids[2];let mut ax=Axes::new([0.,90.],[-max,max],"Elapsed time (min)",r"$F_z/\rho_0$ (m$^3$ s$^{-3}$)");ax.xticks=vec![0.,30.,60.,90.];f.panels[p].axes=Some(ax);
 for (xx,yy,col) in [(&t,&up,"#245774"),(&t,&down,"#8b3940"),(&tc,&control,"#4c7566")] {
  f.panels[p].push(Layer::Line(Line::new(xx.iter().zip(yy).map(|(&a,&b)|[a,b]).collect(),col)));
 }
 let mut legend=Legend::new(vec![LegendEntry::line("Above shear","#245774"),LegendEntry::line("Below shear","#8b3940"),LegendEntry::line("Stable control, above","#4c7566")]);legend.position=LegendPosition::AboveCenter;legend.columns=3;f.panels[p].legend=Some(legend);
 f.layout_colorbars()?;for ext in ["png","pdf"]{f.save(format!("output/gravity_wave_study/simulation_radiation.{ext}"))?;}
 if h.group("fine").is_err(){return Ok(());}
 // Independent parameter changes are compared with the same flux definition.
 let mut f=Figure::new(IntendedUse::ArticleWide);let ids=f.grid(2,1)?;f.height_mm=145.;
 for (row,&p) in ids.iter().enumerate(){f.panels[p].rect=Rect::new(24.,20.+row as f64*62.,145.,43.);let mut ax=Axes::new([0.,90.],if row==0{[-7.,3.]}else{[-max,max]},if row==1{"Elapsed time (min)"}else{""},if row==0{r"$\log_{10}[E'/(\mathrm{m}^2\,\mathrm{s}^{-2})]$"}else{r"$F_z/\rho_0$ (m$^3$ s$^{-3}$)"});ax.xticks=vec![0.,30.,60.,90.];f.panels[p].axes=Some(ax);}
 let cases=[("baseline","Baseline","#245774"),("fine","Finer grid","#8b3940"),("tall","Taller domain","#997124"),("stable","Stable control","#4c7566")];
 for (key,_,col) in cases {let tt=read(&h,&format!("{key}/time_min"))?;for row in 0..2{let yy=read(&h,&format!("{key}/{}",if row==0{"source_energy_m2_s2"}else{"upward_flux_m3_s3"}))?;f.panels[ids[row]].push(Layer::Line(Line::new(tt.iter().zip(yy).map(|(&a,b)|[a,if row==0{b.max(1e-12).log10()}else{b}]).collect(),col)));}}
 let mut l=Legend::new(cases.iter().map(|(_,label,col)|LegendEntry::line(*label,*col)).collect());l.position=LegendPosition::AboveCenter;l.columns=4;f.panels[ids[0]].legend=Some(l);
 for ext in ["png","pdf"]{f.save(format!("output/gravity_wave_study/simulation_checks.{ext}"))?;}
 Ok(())
}
