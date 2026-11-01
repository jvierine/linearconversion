// Separate views: measured candidates and simulated gravity-wave velocity.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
#[path = "../time_axes.rs"] mod time_axes;
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn edges(v:&[f64])->Vec<f64>{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|w|(w[0]+w[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e}
fn main()->Result<()> {
 for sim in [false,true]{
  let h=File::open(if sim{"output/gravity_wave_study/simulation_figures.h5"}else{"output/gravity_wave_study/measurement.h5"})?;
  let mut t=rd(&h,if sim{"baseline/time_min"}else{"time_unix_s"})?;if !sim{for v in &mut t{*v=(*v-1525392000.)/3600.;}}
  let z=rd(&h,if sim{"baseline/height_km"}else{"height_km"})?;
  let a=rd(&h,if sim{"baseline/w_advected_timeheight_m_s"}else{"wave_candidate_bandpass_percent"})?;
  let xlim=if sim{[30.,90.]}else{[15.2,19.05]};let ylim=if sim{[122.,138.]}else{[124.,145.]};let limits=if sim{[-1.,1.]}else{[-2.,2.]};
  let ii:Vec<_>=(0..t.len()).filter(|&i|t[i]>=xlim[0]&&t[i]<=xlim[1]).collect();let jj:Vec<_>=(0..z.len()).filter(|&j|z[j]>=ylim[0]&&z[j]<=ylim[1]).collect();
  let xx:Vec<_>=ii.iter().map(|&i|t[i]).collect();let zz:Vec<_>=jj.iter().map(|&j|z[j]).collect();let mut v=vec![];for &j in &jj{for &i in &ii{v.push(a[i*z.len()+j]);}}
  let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=190.;f.height_mm=124.;let p=f.grid(1,1)?[0];f.panels[p].rect=Rect::new(18.,if sim{12.}else{25.},137.,75.);
  let mut ax=Axes::new([xx[0],xx[xx.len()-1]],ylim,if sim{"Elapsed time (min)"}else{""},"Altitude (km)");
  if sim{ax.xticks=vec![30.,40.,50.,60.,70.,80.,90.];ax.yticks=vec![122.,126.,130.,134.,138.];}else{time_axes::time_axes(&mut ax,true,true)?;ax.yticks=vec![125.,130.,135.,140.,145.];}
  f.panels[p].axes=Some(ax);let sc=f.scale(Scale::new(limits,Normalization::Linear,ColorMap::diverging())?);
  f.panels[p].push(Layer::Pcolormesh(Field::new(edges(&xx),edges(&zz),v,sc)?));let r=f.panels[p].rect;
  f.colorbars.push(Colorbar::new(Rect::new(r.x+r.width+4.,r.y,2.5,r.height),sc,if sim{r"$w$ (m s$^{-1}$)"}else{r"$\delta n_e/n_{e,\mathrm{bg}}$ (\%)"},vec![limits[0],0.,limits[1]]));f.layout_colorbars()?;
  let name=if sim{"simulated_upward_waves"}else{"measured_wave_candidates"};for ext in ["png","pdf"]{f.save(format!("output/gravity_wave_study/{name}.{ext}"))?;}
 }
 Ok(())
}
