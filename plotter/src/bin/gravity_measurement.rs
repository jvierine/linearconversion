// Scientific source provenance is retained in measurement.h5 and source comments.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
#[path = "../time_axes.rs"] mod time_axes;
fn read(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn edges(v:&[f64])->Vec<f64>{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|w|(w[0]+w[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e}
fn main()->Result<()> {
 let h=File::open("output/gravity_wave_study/measurement.h5")?;
 let t=read(&h,"time_unix_s")?;let x:Vec<_>=t.iter().map(|v|(v-1525392000.)/3600.).collect();
 let z=read(&h,"height_km")?;let pars=read(&h,"layer_parameters")?;
 let js:Vec<_>=(0..z.len()).filter(|&j|z[j]>=110.&&z[j]<=145.).collect();let zz:Vec<_>=js.iter().map(|&j|z[j]).collect();
 for sensitivity in [false,true] {
  let mut f=Figure::new(IntendedUse::ArticleWide);let ids=f.grid(3,1)?;
  for (row,&p) in ids.iter().enumerate(){f.panels[p].rect=Rect::new(18.,25.+row as f64*57.,137.,45.);}
  f.height_mm=25.+3.*57.+12.;f.width_mm=190.;
  for (row,&p) in ids.iter().enumerate(){
   let mut ax=Axes::new([x[0],x[x.len()-1]],[110.,145.],"","Altitude (km)");
   time_axes::time_axes(&mut ax,row==2,row==0)?;ax.yticks=vec![110.,120.,130.,140.];f.panels[p].axes=Some(ax);
   let key=if sensitivity{format!("cutoff_{}_min/display_layer_removed_percent",[20,30,60][row])}
     else{["ne_m3","cutoff_30_min/display_fractional_percent","cutoff_30_min/display_layer_removed_percent"][row].into()};
   let a=read(&h,&key)?;let mut v=vec![];
   for &j in &js {for i in 0..t.len(){v.push(if !sensitivity&&row==0{a[i*z.len()+j].max(1.).log10()}else{a[i*z.len()+j]});}}
   let (lim,label,ticks,palette)=if !sensitivity&&row==0{([10.9,11.8],r"$\log_{10}(n_e\,/\,\mathrm{m}^{-3})$".to_string(),vec![11.,11.4,11.8],ColorMap::ember())}
   else{([-6.,6.],if sensitivity{format!("{} min cutoff (\\%)",[20,30,60][row])}else{r"$\delta n_e / n_{e,\mathrm{bg}}$ (\%)".into()},vec![-6.,-3.,0.,3.,6.],ColorMap::diverging())};
   let sc=f.scale(Scale::new(lim,Normalization::Linear,palette)?);
   f.panels[p].push(Layer::Pcolormesh(Field::new(edges(&x),edges(&zz),v,sc)?));
   if row==2 || sensitivity {
    let points:Vec<_>=(25..t.len()).map(|i|[x[i],pars[i*5+3]]).collect();
    let mut line=Line::new(points,"#333333");line.width_pt=0.6;f.panels[p].push(Layer::Line(line));
   }
   let r=f.panels[p].rect;f.colorbars.push(Colorbar::new(Rect::new(r.x+r.width+4.,r.y,2.5,r.height),sc,label,ticks));
  }
  f.layout_colorbars()?;
  let name=if sensitivity{"measurement_filter_sensitivity"}else{"measurement_gravity_waves"};
  for ext in ["png","pdf"]{f.save(format!("output/gravity_wave_study/{name}.{ext}"))?;}
 }
 Ok(())
}
