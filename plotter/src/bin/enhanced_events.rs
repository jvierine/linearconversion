// Source spectra, catalogue selection and processing metadata are in the HDF5.
use anyhow::Result;
use hdf5::{File,types::VarLenUnicode};
use rustplotlib::*;
fn rd(g:&hdf5::Group,k:&str)->Result<Vec<f64>> {Ok(g.dataset(k)?.read_raw::<f64>()?)}
fn edges(v:&[f64])->Vec<f64>{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|a|(a[0]+a[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e}
fn main()->Result<()> {
 let h=File::open("data/enhanced_events_animation.h5")?;let mut names=h.group("frames")?.member_names()?;names.sort();
 for name in names {
  let g=h.group(&format!("frames/{name}"))?;let x=rd(&g,"frequency_mhz")?;let y=rd(&g,"height_km")?;
  let mut f=ArticleLayout::new(177.,10.)?.figure(1.,147.)?;let p=f.grid(1,1)?[0];f.height_mm=147.;f.panels[p].rect=Rect::new(21.,15.,149.,100.);
  let mut ax=Axes::new([2.,6.5],[100.,140.],"Upshifted frequency (MHz)","Altitude above radar (km)");ax.xticks=vec![2.,3.,4.,5.,6.];ax.yticks=vec![100.,110.,120.,130.,140.];f.panels[p].axes=Some(ax);
  let sc=f.scale(Scale::new([-15.,20.],Normalization::Linear,ColorMap::ember())?);
  f.panels[p].push(Layer::Pcolormesh(Field::new(edges(&x),edges(&y),rd(&g,"snr_db")?,sc)?));
  let label=g.attr("label")?.read_scalar::<VarLenUnicode>()?.as_str().to_owned();f.labels.push(Label::new(label,95.,6.));
  for row in rd(&g,"events")?.chunks(3) {
   let band=row[0] as usize;let freq=row[1].abs();let z=row[2];let start=[freq+0.32,z-4.];let tip=[freq+0.035,z-0.45];
   f.panels[p].push(Layer::Line(Line::new(vec![start,tip,[tip[0]+0.11,tip[1]-0.45],tip,[tip[0]+0.015,tip[1]-0.9]],"#ffffff")));
   let a=f.panels[p].label_at(if band==0{"4.51 MHz"}else{"5.34 MHz"},[start[0],start[1]-0.8]);a.color="#ffffff".into();
  }
  let mut cb=Colorbar::new(Rect::new(40.,131.,112.,3.),sc,"Spectral SNR (dB)",vec![-15.,-5.,5.,15.,20.]);cb.horizontal=true;f.colorbars.push(cb);f.auto_colorbar_layout=false;
  f.save(format!("media/enhanced_events/frame_{name}.png"))?;println!("frame {name}");
 }
 Ok(())
}
