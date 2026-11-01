// Native Rustplotlib defaults. Full time coverage; no smoothing across missing fits.
// Calculation provenance and quality flags are in plasma_parameters.h5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
#[path = "../time_axes.rs"] mod time_axes;
fn read(f:&File,p:&str)->Result<Vec<f64>> {Ok(f.dataset(p)?.read_raw::<f64>()?)}
fn edges(x:&[f64])->Vec<f64>{let mut e=vec![x[0]-(x[1]-x[0])/2.];e.extend(x.windows(2).map(|w|(w[0]+w[1])/2.));e.push(x[x.len()-1]+(x[x.len()-1]-x[x.len()-2])/2.);e}
fn main()->Result<()> {
 let h=File::open("data/eregion_fit/plasma_parameters.h5")?;
 let raw=File::open("data/eregion_fit/compact_inputs.h5")?;
 let z=read(&h,"height_km")?;let t=read(&h,"time_unix_s")?;
 let xe:Vec<_>=read(&h,"time_edges_unix_s")?.iter().map(|v|(v-1525392000.)/3600.).collect();
 let pt=read(&raw,"plasma/time_unix_s")?;
 let xlim=[(pt[0]-1525392000.)/3600.,(pt[pt.len()-1]+10.-1525392000.)/3600.];
 let mut f=Figure::new(IntendedUse::ArticleWide);let ids=f.grid(4,1)?;
 let top=f.panels[0].rect.y+15.;let panel_height=34.;let spacing=11.;
 for (row,&p) in ids.iter().enumerate(){
  f.panels[p].rect.y=top+row as f64*(panel_height+spacing);
  f.panels[p].rect.height=panel_height;f.panels[p].rect.width-=25.;
 }
 f.height_mm=top+4.*(panel_height+spacing)+8.;
 let specs=[("ne_m3",[8.,12.],r"$\log_{10}(n_e\,/\,\mathrm{m}^{-3})$",1.),
            ("te_K",[100.,800.],r"$T_e$ (K)",1.),
            ("ti_K",[100.,700.],r"$T_i$ (K)",1.),
            ("vi_m_s",[-100.,100.],r"$v_i$ (m/s)",1.)];
 for (row,(&p,(key,lim,label,mult))) in ids.iter().zip(specs).enumerate(){
  let mut a=Axes::new(xlim,[100.,140.],if row==3{"Time (UTC, 4 May 2018)"}else{""},"Altitude (km)");
  time_axes::time_axes(&mut a,row==3,row==0)?;
  a.yticks=vec![100.,110.,120.,130.,140.];f.panels[p].axes=Some(a);
  let data=read(&h,key)?;let mut v=vec![f64::NAN;data.len()];
  for i in 0..t.len(){for j in 0..z.len(){let k=i*z.len()+j;v[j*t.len()+i]=if row==0{data[k].max(1e8).log10()}else{data[k]*mult};}}
  let palette=if row==3 {ColorMap::diverging()}else{ColorMap::ember()};
  let sc=f.scale(Scale::new(lim,Normalization::Linear,palette)?);
  f.panels[p].push(Layer::Pcolormesh(Field::new(xe.clone(),edges(&z),v,sc)?));
  let r=f.panels[p].rect;
  let ticks=match row{0=>vec![8.,9.,10.,11.,12.],1=>vec![200.,400.,600.,800.],2=>vec![100.,300.,500.,700.],_=>vec![-100.,-50.,0.,50.,100.]};
  f.colorbars.push(Colorbar::new(Rect::new(r.x+r.width+4.,r.y,2.5,r.height),sc,label,ticks));
 }
 f.layout_colorbars()?;
 for ext in ["png","svg"]{f.save(format!("figures/arecibo_plasma_parameters_full_experiment.{ext}"))?;}
 Ok(())
}
