// Default Rustplotlib publication style. Data provenance is in input HDF5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn read(f:&File,k:&str)->Result<Vec<f64>>{Ok(f.dataset(k)?.read_raw::<f64>()?)}
fn edges(v:&[f64])->Vec<f64>{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|w|(w[0]+w[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e}
// Presentation-only gaps uncover measured line enhancements. Neither the
// equilibrium supplied to the solver nor any spectral pixel is changed.
fn overlay_segments(x:&[f64],y:&[f64],epoch:&str)->Vec<Vec<[f64;2]>>{
 let heights=if epoch=="1525453430" {[117.2,118.7]}else{[113.2,115.2]};
 let hidden=|f:f64,z:f64| z>=heights[0]&&z<=heights[1]
     && ((f>=4.40&&f<=4.62)||(f>=5.20&&f<=5.46));
 let mut segments=vec![];let mut line=vec![];
 // Subdivide straight plotting segments so even sparsely sampled ion-line
 // curves break at the gap, without bridging across it or changing their shape.
 for i in 0..x.len()-1{
  let n=((x[i+1]-x[i]).abs()/0.005).max((y[i+1]-y[i]).abs()/0.01).ceil().max(1.) as usize;
  for j in 0..n{
   let t=j as f64/n as f64;let f=x[i]+t*(x[i+1]-x[i]);let z=y[i]+t*(y[i+1]-y[i]);
   if z<100.||z>130.||hidden(f,z){
    if line.len()>1{segments.push(std::mem::take(&mut line));}else{line.clear();}
   }else{line.push([f,z]);}
  }
 }
 let f=x[x.len()-1];let z=y[y.len()-1];
 if z>=100.&&z<=130.&&!hidden(f,z){line.push([f,z]);}
 if line.len()>1{segments.push(line);}
 segments
}
fn main()->Result<()>{
 let epoch=std::env::args().nth(1).unwrap_or("1525460596".into());
 let zoom=std::env::args().nth(2).as_deref()==Some("zoom");
 let h=File::open(format!("data/conversion_overlay_{epoch}.h5"))?;
 let x=read(&h,"frequency_mhz")?;let y=read(&h,"height_km")?;
 let mut fig=Figure::new(IntendedUse::ArticleWide);let p=fig.grid(1,1)?[0];
 fig.panels[p].rect.height=115.;fig.height_mm+=65.;
 fig.panels[p].axes=Some(Axes::new(if zoom {[4.2,6.2]}else{[2.,8.]},if zoom {[116.5,119.5]}else{[100.,130.]},"Frequency magnitude (MHz)","Altitude above radar (km)"));
 let ceiling=if epoch=="1525453430" {40.} else {15.};
 let sc=fig.scale(Scale::new([0.,ceiling],Normalization::Linear,ColorMap::ember())?);
 let values=read(&h,"displayed_power")?.iter().map(|v|10.*(1.+100.*v.max(0.)).log10()).collect();
 fig.panels[p].push(Layer::Pcolormesh(Field::new(edges(&x),edges(&y),values,sc)?));
 for (prefix,color) in [("ion","#cbd3df"),("model","#57e0dc")]{
  let xx=read(&h,&format!("{prefix}_fpe_mhz"))?;let yy=read(&h,&format!("{prefix}_height_km"))?;
  for points in overlay_segments(&xx,&yy,&epoch){fig.panels[p].push(Layer::Line(Line::new(points,color)));}
 }
 // Inside the quiet lower-right map region; retain the data and existing curve gaps.
 let mut legend=Legend::new(vec![
  LegendEntry::line("Ion-line plasma frequency", "#cbd3df"),
  LegendEntry::line("Smooth density model", "#57e0dc")]);
 legend.position=LegendPosition::LowerRight;legend.avoid_data=false;
 fig.panels[p].legend=Some(legend);
 if epoch=="1525453430" && !zoom {
  // Arrowheads use physical page distances so axis aspect does not distort them.
  for (label,start,end) in [
   ("4.51 MHz",[4.52,113.0],[4.5166015625,117.37]),
   ("5.34 MHz",[6.55,114.5],[5.33447265625,117.52])]{
   let r=fig.panels[p].rect;let sx=r.width/6.;let sy=r.height/30.;
   let dx=(end[0]-start[0])*sx;let dy=(end[1]-start[1])*sy;
   let length=dx.hypot(dy);let ux=dx/length;let uy=dy/length;
   let a=[end[0]-(2.*ux+0.8*uy)/sx,end[1]-(2.*uy-0.8*ux)/sy];
   let b=[end[0]-(2.*ux-0.8*uy)/sx,end[1]-(2.*uy+0.8*ux)/sy];
   fig.panels[p].push(Layer::Line(Line::new(vec![start,end,a,end,b],"#ffffff")));
   let text=fig.panels[p].label_at(label,[start[0],start[1]-0.85]);
   text.color="#ffffff".into();
  }
 }
 let r=fig.panels[p].rect;
 let mut cb=Colorbar::new(Rect::new(r.x+6.,r.y+r.height+14.,r.width-12.,3.),sc,"Plasma-line excess power (dB, relative)",vec![0.,ceiling/4.,ceiling/2.,3.*ceiling/4.,ceiling]);
 cb.horizontal=true;fig.colorbars.push(cb);fig.layout_colorbars()?;
 fig.height_mm=fig.height_mm.max(fig.colorbars[0].rect.y+22.);
 let suffix=if zoom {"_zoom"}else{""};
 for ext in ["png","svg","pdf"]{fig.save(format!("figures/conversion_overlay_{epoch}{suffix}.{ext}"))?;}
 Ok(())
}
