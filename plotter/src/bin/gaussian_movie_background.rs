// Static Rustplotlib axes/colorbars for transient movie frames. Animated pixels
// use the exact exported transfer-function lookup tables in the same directory.
use anyhow::Result;
use rustplotlib::*;
use std::{fs, path::Path};
fn main()->Result<()> {
 let a:Vec<String>=std::env::args().collect(); let out=Path::new(&a[1]); fs::create_dir_all(out)?;
 let case=&a[2];let maxima=[a[3].parse::<f64>()?,a[4].parse::<f64>()?];
 let mut f=Figure::new(IntendedUse::Custom{width_mm:400.,font_pt:14.});
 f.height_mm=225.;f.raster_dpi=121.92;f.tight_export=false;f.auto_colorbar_layout=false;
 let ps=f.grid(1,2)?;
 f.height_mm=225.;
 for j in 0..2 {
  let p=ps[j];let x=if j==0 {25.} else {224.};
  f.panels[p].rect=Rect::new(x,12.,170.,194.);
  let overview=(case=="overview"||case=="flux")&&j==0;
  let yr=if overview {[115.1,120.8]} else if case=="flux" {[-160.,40.]} else {[-20.,5.]};
  let mut ax=Axes::new([-100.,100.],yr,"Horizontal distance (m)",if overview {"Altitude (km)"} else {"Height from carrier resonance (m)"});
  ax.xticks=vec![-100.,-50.,0.,50.,100.];
  ax.yticks=if overview {vec![116.,117.,118.,119.,120.]} else if case=="flux" {vec![-150.,-100.,-50.,0.]} else {vec![-20.,-15.,-10.,-5.,0.,5.]};
  f.panels[p].axes=Some(ax);
  let unsigned=false;
  let norm=Normalization::Asinh{width:maxima[j]/if overview {300.} else {30.}};
  let sc=Scale::new(if unsigned {[0.,maxima[j]]} else {[-maxima[j],maxima[j]]},norm,if unsigned {ColorMap::ember()} else {ColorMap::diverging()})?;
  let table:Vec<u8>=(0..65536).flat_map(|i| sc.colors.sample(i as f64/65535.)).collect();
  fs::write(out.join(format!("{case}_palette_{j}.rgb")),table)?;
  let si=f.scale(sc);
  let units=match (case.as_str(),j) {
   ("overview",0)=>r"$E_z$ (mV/m)",("overview",_)=>r"$\delta n_e$ ($10^6$ m$^{-3}$)",
   ("fields",0)=>r"$E_z$ (mV/m)",("fields",_)=>r"$\delta n_e$ ($10^6$ m$^{-3}$)",
   ("flux",0)=>r"$\langle S_z\rangle$ (pW/m$^2$)",("flux",_)=>r"$\langle\delta p_e u_z\rangle$ (fW/m$^2$)",
   (_,0)=>r"$J_x$ ($\mu$A/m$^2$)",(_,_)=>r"$J_z$ ($\mu$A/m$^2$)"};
  let ticks=if unsigned {vec![0.,maxima[j]]} else {vec![-maxima[j],0.,maxima[j]]};
  let mut cb=Colorbar::new(Rect::new(x+107.,19.,57.,2.5),si,units,ticks);cb.horizontal=true;f.colorbars.push(cb);
 }
 f.save(out.join(format!("{case}_background.png")))?;
 Ok(())
}
