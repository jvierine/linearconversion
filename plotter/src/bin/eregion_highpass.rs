// Temporal Butterworth filtering is performed on density HDF5 arrays.
// No filtering of raster image pixels and no spatial interpolation.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
#[path = "../time_axes.rs"] mod time_axes;
fn main()->Result<()> {
 let h=File::open("data/eregion_fit/density_highpass.h5")?;
 let t=h.dataset("time_unix_s")?.read_raw::<f64>()?;
 let z=h.dataset("height_km")?.read_raw::<f64>()?;
 let x:Vec<_>=t.iter().map(|v|(v-1525392000.)/3600.).collect();
 let mut xe=vec![x[0]-(x[1]-x[0])/2.];xe.extend(x.windows(2).map(|w|(w[0]+w[1])/2.));xe.push(x[x.len()-1]+(x[x.len()-1]-x[x.len()-2])/2.);
 for (name,key,ylim,limits,mult,label) in [
  ("arecibo_density_highpass","delta_ne_m3",[50.,140.],[-3.,3.],1e-10,r"$\delta n_e$ ($10^{10}$ m$^{-3}$)"),
  ("arecibo_density_highpass_zoom","fractional_percent",[95.,140.],[-25.,25.],1.,r"$\delta n_e / n_{e,\mathrm{bg}}$ (\%)")
 ] {
  let a=h.dataset(&format!("cutoff_30_min/{key}"))?.read_raw::<f64>()?;
  let inds:Vec<_>=(0..z.len()).filter(|&j|z[j]>=ylim[0]-0.15&&z[j]<=ylim[1]+0.15).collect();
  let mut ze=vec![z[inds[0]]-0.15];ze.extend(inds.iter().map(|&j|z[j]+0.15));
  let mut v=vec![];for &j in &inds{for i in 0..t.len(){v.push(a[i*z.len()+j]*mult);}}
  let mut f=Figure::new(IntendedUse::ArticleWide);let p=f.grid(1,1)?[0];
  f.panels[p].rect.y+=15.;f.panels[p].rect.width-=25.;f.panels[p].rect.height=85.;f.height_mm=f.panels[p].rect.y+104.;
  let mut ax=Axes::new([xe[0],xe[xe.len()-1]],ylim,"","Altitude (km)");time_axes::time_axes(&mut ax,true,true)?;
  ax.yticks=if ylim[0]<90.{(5..=14).map(|i|i as f64*10.).collect()}else{(19..=28).map(|i|i as f64*5.).collect()};
  f.panels[p].axes=Some(ax);
  let sc=f.scale(Scale::new(limits,Normalization::Linear,ColorMap::diverging())?);
  f.panels[p].push(Layer::Pcolormesh(Field::new(xe.clone(),ze,v,sc)?));
  let r=f.panels[p].rect;
  let ticks=if mult==1.{vec![-25.,-12.5,0.,12.5,25.]}else{vec![-3.,-1.5,0.,1.5,3.]};
  f.colorbars.push(Colorbar::new(Rect::new(r.x+r.width+4.,r.y,2.5,r.height),sc,label,ticks));f.layout_colorbars()?;
  for ext in ["png","svg"]{f.save(format!("figures/{name}.{ext}"))?;}
 }
 Ok(())
}
