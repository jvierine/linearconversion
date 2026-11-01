// Scientific diagnostics from the exact native-cycle inputs. Native Rustplotlib style.
use anyhow::Result;
use hdf5::{File,Group};
use rustplotlib::*;
fn read(g:&Group,k:&str)->Result<Vec<f64>> {Ok(g.dataset(k)?.read_raw::<f64>()?)}
fn main()->Result<()> {
 let h=File::open("data/eregion_fit/full_altitude_fit_audit.h5")?;
 let root=h.group("acf")?;
 for time in [0,65,130,200] {
  let mut names=root.member_names()?;names.sort();
  let selected:Vec<_>=names.iter().filter(|n|n.starts_with(&format!("t{time:03}_"))).collect();
  for (page,chunk) in selected.chunks(6).enumerate(){
   let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=195.;let ids=f.grid(3,2)?;f.height_mm=252.;
   let ts=root.group(chunk[0])?.attr("time_unix_s")?.read_scalar::<f64>()?;
   let minutes=((ts-1525392000.)/60.).round() as i64;
   f.labels.push(Label::new(format!("4 May 2018, {:02}:{:02} UTC ({:02}:{:02} AST)",minutes/60,minutes%60,minutes/60-4,minutes%60),97.5,7.));
   for (ix,name) in chunk.iter().enumerate(){
    let g=root.group(name)?;let p=ids[ix];let row=ix/2;let col=ix%2;
    f.panels[p].rect=Rect::new(23.+col as f64*91.,27.+row as f64*73.,68.,51.);
    let lag=read(&g,"lag_us")?;let obs_r=read(&g,"observed_real")?;let obs_i=read(&g,"observed_imag")?;
    let mod_r=read(&g,"model_real")?;let mod_i=read(&g,"model_imag")?;let sigma=read(&g,"sigma_component")?;
    let norm=obs_r[0].abs().max(sigma[0]);
    let extent=obs_r.iter().chain(&obs_i).chain(&mod_r).chain(&mod_i).fold(1.0_f64,|a,&b|a.max(b.abs()/norm))*1.15;
    let lo=lag.iter().copied().fold(f64::INFINITY,f64::min).min(0.);
    let hi=lag.iter().copied().fold(f64::NEG_INFINITY,f64::max).max(0.);
    f.panels[p].axes=Some(Axes::new([lo,hi],[-extent,extent],r"Lag ($\mu$s)",if col==0{"Normalized ACF"}else{""}));
    for (obs,model,color,rgb,shape) in [(&obs_r,&mod_r,"#245774",[0.141,0.341,0.455],MarkerShape::Circle),(&obs_i,&mod_i,"#8b3940",[0.545,0.224,0.251],MarkerShape::Square)]{
     let points:Vec<_>=lag.iter().zip(obs).map(|(&x,&y)|[x,y/norm]).collect();
     let bounds:Vec<_>=obs.iter().zip(&sigma).map(|(&v,&s)|[(v-s)/norm,(v+s)/norm]).collect();
     f.panels[p].errorbars(&points,None,Some(&bounds),color,0.35,1.,0.4)?;
     let mut panel=f.panels[p].clone();panel.markers(&mut f,&points,&MarkerStyle{shape,size_pt:2.5,color:rgb})?;f.panels[p]=panel;
     let mut xy:Vec<_>=lag.iter().zip(model).map(|(&x,&y)|[x,y/norm]).collect();xy.sort_by(|a,b|a[0].total_cmp(&b[0]));
     f.panels[p].push(Layer::Line(Line::new(xy,color)));
    }
    let z=g.attr("height_km")?.read_scalar::<f64>()?;
    f.panels[p].label_at(format!("{z:.1} km"),[lo+0.13*(hi-lo),extent*0.83]);
    if ix<2 {let mut l=Legend::new(vec![LegendEntry::marker("Real",MarkerShape::Circle,"#245774"),LegendEntry::marker("Imaginary",MarkerShape::Square,"#8b3940")]);l.position=LegendPosition::AboveCenter;l.columns=2;f.panels[p].legend=Some(l);}
   }
   f.save(format!("figures/arecibo_acf_audit_t{time:03}_{page}.png"))?;
   f.save(format!("figures/arecibo_acf_audit_t{time:03}_{page}.pdf"))?;
  }
 }
 Ok(())
}
