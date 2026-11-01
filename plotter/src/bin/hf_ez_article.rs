// Input reconstruction: hf_fdtd/examples/field_cut_2d.py; signed Ez and its absolute value.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn main()->Result<()>{
 let h=File::open("output/hf_fdtd/field_cut_2d.h5")?;
 for name in ["overview","resonance"] {
  let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,2)?;f.height_mm=125.;
  let x=rd(&h,&format!("{name}/horizontal_edges_m"))?;
  let y=rd(&h,&format!("{name}/{}",if name=="overview"{"height_edges_km"}else{"height_relative_edges_m"}))?;
  let ez=rd(&h,&format!("{name}/instantaneous_Ez_over_incident"))?;
  for i in 0..2 {
   let p=ps[i];let left=23.+i as f64*83.;f.panels[p].rect=Rect::new(left,6.,66.,84.);
   let ylabel=if i>0{""}else if name=="overview"{"Altitude above radar (km)"}else{"Height relative to resonance (m)"};
   f.panels[p].axes=Some(Axes::new([x[0],*x.last().unwrap()],[y[0],*y.last().unwrap()],"Horizontal distance (m)",ylabel));
   let (scale,ticks,label,values)=if i==0 {
    (Scale::new([-200.,200.],Normalization::Asinh{width:0.5},ColorMap::diverging())?,vec![-200.,-10.,0.,10.,200.],r"$E_z/|E_{\rm inc}|$",ez.clone())
   }else{
    (Scale::new([0.01,200.],Normalization::Log,ColorMap::ember())?,vec![0.01,0.1,1.,10.,200.],r"$|E_z|/|E_{\rm inc}|$",ez.iter().map(|v|v.abs().max(1e-14)).collect())
   };
   let sc=f.scale(scale);f.panels[p].push(Layer::Pcolormesh(Field::new(x.clone(),y.clone(),values,sc)?));
   let mut cb=Colorbar::new(Rect::new(left,108.,66.,2.5),sc,label,ticks);cb.horizontal=true;f.colorbars.push(cb);
  }
  f.layout_colorbars()?;
  for ext in ["png","pdf"]{f.save(format!("figures/hf_ez_article_{name}.{ext}"))?;}
 }
 Ok(())
}
