// Empirical PSF estimation and inversion provenance are stored in HDF5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn read(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn edges(v:&[f64])->Vec<f64>{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|w|(w[0]+w[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e}
fn main()->Result<()>{
 let only=std::env::args().nth(1);
 let h=File::open("data/plasma_line_deconvolution.h5")?;
 for epoch in ["1525453430","1525453540"]{
  let x=read(&h,&format!("{epoch}/frequency_mhz"))?;let y=read(&h,&format!("{epoch}/height_km"))?;
  let full=only.as_deref()==Some("full");
  for (key,label) in [("observed_excess","Measured excess"),("restored","Empirical deconvolution"),("residual","Re-convolution residual")]{
   if full && key=="residual"{continue;}
   if !full && only.as_ref().is_some_and(|s| s!=key){continue;}
   let mut fig=Figure::new(IntendedUse::ArticleWide);let p=fig.grid(1,1)?[0];
   fig.panels[p].rect.height=90.;fig.height_mm=145.;
   fig.panels[p].axes=Some(Axes::new(if full{[2.,8.]}else{[4.2,6.6]},if full{[100.,130.]}else{[110.75,124.75]},"Frequency magnitude (MHz)","Altitude above radar (km)"));
   let sc=fig.scale(Scale::new([0.,if full{20.}else{40.}],Normalization::Linear,ColorMap::ember())?);
   let a=read(&h,&format!("{epoch}/{key}"))?;
   let val=a.iter().map(|v|10.*(1.+100.*v.max(0.)).log10()).collect();
   // Signed residual gets its own diverging linear scale; never hide over-subtraction.
   let (sc,val)=if key=="residual"{(fig.scale(Scale::new([-0.3,0.3],Normalization::Linear,ColorMap::diverging())?),a)}else{(sc,val)};
   fig.panels[p].push(Layer::Pcolormesh(Field::new(edges(&x),edges(&y),val,sc)?));
   let r=fig.panels[p].rect;
   let (ct,ticks)=if key=="residual"{("Signed excess power (relative)",vec![-0.3,0.,0.3])}else{("Excess power (dB, relative)",if full{vec![0.,5.,10.,15.,20.]}else{vec![0.,10.,20.,30.,40.]})};
   let mut cb=Colorbar::new(Rect::new(r.x+5.,r.y+r.height+14.,r.width-10.,3.),sc,ct,ticks);cb.horizontal=true;fig.colorbars.push(cb);fig.layout_colorbars()?;
   // Standalone comparison panels named in the page, not source-code annotations.
   let _=label;
   let suffix=if full{"_full"}else{""};
   for ext in ["png","svg"]{fig.save(format!("figures/plasma_deconvolution_{epoch}_{key}{suffix}.{ext}"))?;}
  }
 }
 if only.is_some(){return Ok(());}
 let x=read(&h,"psf/delta_frequency_mhz")?;let y=read(&h,"psf/delta_height_km")?;let a=read(&h,"psf/kernel")?;let max=a.iter().copied().fold(0.,f64::max);
 let mut fig=Figure::new(IntendedUse::ArticleWide);let p=fig.grid(1,1)?[0];fig.panels[p].rect.height=90.;fig.height_mm=145.;
 fig.panels[p].axes=Some(Axes::new([-1.25,1.25],[-6.6,6.6],"Frequency offset (MHz)","Height offset (km)"));
 let sc=fig.scale(Scale::new([-40.,0.],Normalization::Linear,ColorMap::ember())?);
 fig.panels[p].push(Layer::Pcolormesh(Field::new(edges(&x),edges(&y),a.iter().map(|v|10.*(v/max).max(1e-4).log10()).collect(),sc)?));
 let r=fig.panels[p].rect;let mut cb=Colorbar::new(Rect::new(r.x+5.,r.y+r.height+14.,r.width-10.,3.),sc,"Empirical response (dB from peak)",vec![-40.,-30.,-20.,-10.,0.]);cb.horizontal=true;fig.colorbars.push(cb);fig.layout_colorbars()?;
 for ext in ["png"]{fig.save(format!("figures/plasma_line_empirical_psf.{ext}"))?;}
 Ok(())
}
