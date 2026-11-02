// Source spectra, catalogue selection and processing metadata are in the HDF5.
use anyhow::Result;
use hdf5::{File};
use rustplotlib::*;
fn rd(g:&hdf5::Group,k:&str)->Result<Vec<f64>> {Ok(g.dataset(k)?.read_raw::<f64>()?)}
fn edges(v:&[f64])->Vec<f64>{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|a|(a[0]+a[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e}
fn main()->Result<()> {
 let h=File::open("data/enhanced_events_animation.h5")?;let mut names=h.group("frames")?.member_names()?;names.sort();
 for name in names.into_iter().take(1) {
  let g=h.group(&format!("frames/{name}"))?;let x=rd(&g,"frequency_mhz")?;let y=rd(&g,"height_km")?;
  let mut f=ArticleLayout::new(177.,10.)?.figure(1.,147.)?;let p=f.grid(1,1)?[0];f.height_mm=147.;f.panels[p].rect=Rect::new(21.,15.,149.,100.);
  let mut ax=Axes::new([2.,6.5],[100.,140.],"Upshifted frequency (MHz)","Altitude above radar (km)");ax.xticks=vec![2.,3.,4.,5.,6.];ax.yticks=vec![100.,110.,120.,130.,140.];f.panels[p].axes=Some(ax);
  let sc=f.scale(Scale::new([-15.,20.],Normalization::Linear,ColorMap::ember())?);
  f.panels[p].push(Layer::Pcolormesh(Field::new(edges(&x),edges(&y),vec![-15.;x.len()*y.len()],sc)?));
  let mut cb=Colorbar::new(Rect::new(40.,131.,112.,3.),sc,"High-pass spectral excess (dB / noise)",vec![-15.,-5.,5.,15.,20.]);cb.horizontal=true;f.colorbars.push(cb);f.auto_colorbar_layout=false;
  f.save("media/enhanced_events/movie_template.png")?;println!("frame {name}");
 }
 Ok(())
}
