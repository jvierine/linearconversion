// Native Rustplotlib rendering; measurement and model provenance is stored in HDF5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
#[path = "../time_axes.rs"] mod time_axes;
fn read(f:&File,p:&str)->Result<Vec<f64>> {Ok(f.dataset(p)?.read_raw::<f64>()?)}
fn edges(x:&[f64])->Vec<f64> {let mut e=vec![x[0]-(x[1]-x[0])/2.];e.extend(x.windows(2).map(|w|(w[0]+w[1])/2.));e.push(x[x.len()-1]+(x[x.len()-1]-x[x.len()-2])/2.);e}
fn heat(name:&str,x:&[f64],y:&[f64],values:Vec<f64>,lim:[f64;2],yl:&str,cl:&str)->Result<()> {
 let mut f=Figure::new(IntendedUse::ArticleWide);let p=f.grid(1,1)?[0];
 f.panels[p].rect.y+=15.;f.height_mm+=15.;
 let mut ax=Axes::new([x[0],x[x.len()-1]],[y[0],y[y.len()-1]],"",yl);
 time_axes::time_axes(&mut ax,true,true)?;
 f.panels[p].axes=Some(ax);
 let sc=f.scale(Scale::new(lim,Normalization::Linear,ColorMap::ember())?);
 f.panels[p].push(Layer::Pcolormesh(Field::new(edges(x),edges(y),values,sc)?));
 let r=f.panels[p].rect;let mut cb=Colorbar::new(Rect::new(r.x+6.,r.y+r.height+12.,r.width-12.,2.5),sc,cl,(0..6).map(|i|lim[0]+(lim[1]-lim[0])*i as f64/5.).collect());
 cb.horizontal=true;f.colorbars.push(cb);f.layout_colorbars()?;
 f.height_mm=f.height_mm.max(f.colorbars[0].rect.y+f.colorbars[0].rect.height+18.);
 for ext in ["png","svg"] {f.save(format!("figures/{name}.{ext}"))?;}
 Ok(())
}
fn main()->Result<()> {
 let f=File::open("data/eregion_fit/compact_inputs.h5")?;
 let t=read(&f,"plasma/time_unix_s")?;let x:Vec<_>=t.iter().map(|v|(v-1525392000.)/3600.).collect();
 let y=read(&f,"plasma/frequency_MHz")?;
 for (side,name) in [("negative","arecibo_enhancement_time_frequency"),("positive","arecibo_opposite_sideband_time_frequency")] {
  let q=read(&f,&format!("plasma/{side}_excess"))?;let mut z=vec![0.;q.len()];
  for i in 0..x.len(){for j in 0..y.len(){z[j*x.len()+i]=10.*(1.+q[i*y.len()+j]).max(1.).log10();}}
  heat(name,&x,&y,z,[0.,25.],"Plasma-line frequency magnitude (MHz)","Peak power / background (dB)")?;
  if side=="negative" {
   let ix:Vec<_>=(0..x.len()).filter(|&i|x[i]>=16.75&&x[i]<=17.4).collect();
   let iy:Vec<_>=(0..y.len()).filter(|&j|y[j]>=4.2&&y[j]<=6.1).collect();
   let xx:Vec<_>=ix.iter().map(|&i|x[i]).collect();let yy:Vec<_>=iy.iter().map(|&j|y[j]).collect();
   let mut zz=vec![];for &j in &iy{for &i in &ix{zz.push(10.*(1.+q[i*y.len()+j]).max(1.).log10());}}
   heat("arecibo_enhancement_detail",&xx,&yy,zz,[0.,20.],"Plasma-line frequency magnitude (MHz)","Peak power / background (dB)")?;
  }
 }
 if std::path::Path::new("data/eregion_fit/plasma_parameters.h5").exists(){fit_plots()?;}
 Ok(())
}
fn fit_plots()->Result<()> {
 let h=File::open("data/eregion_fit/plasma_parameters.h5")?;
 let z=read(&h,"profile/height_km")?;
 let mut fig=Figure::new(IntendedUse::ArticleWide);let ids=fig.grid(1,3)?;
 fig.height_mm+=18.;for p in &mut fig.panels {p.rect.y+=18.;}
 let axes=[([0.,5.],r"$n_e$ ($10^{11}$ m$^{-3}$)"),([0.,800.],"Temperature (K)"),([-80.,80.],"Apparent velocity (m/s)")];
 for (n,&idx) in ids.iter().enumerate(){fig.panels[idx].axes=Some(Axes::new(axes[n].0,[100.,140.],axes[n].1,if n==0{"Altitude (km)"}else{""}));}
 let colors=["#245774","#8b3940","#4c7566"];
 for (idx,key,color) in [(ids[0],"ne_1e11_m3",colors[0]),(ids[1],"ti_K",colors[0]),(ids[1],"te_K",colors[1]),(ids[1],"Tn_K",colors[2]),(ids[2],"vi_m_s",colors[0])]{
  let v=read(&h,&format!("profile/{key}"))?;fig.panels[idx].push(Layer::Line(Line::new(v.iter().zip(&z).map(|(&a,&b)|[a,b]).collect(),color)));
 }
 let nx=read(&h,"profile/plasma_line_density_1e11_m3")?;let nz=read(&h,"profile/plasma_line_height_km")?;
 let points:Vec<_>=nx.iter().zip(nz).map(|(&a,b)|[a,b]).collect();
 let mut panel=fig.panels[ids[0]].clone();panel.markers(&mut fig,&points,&MarkerStyle{shape:MarkerShape::Circle,size_pt:4.,color:[0.545,0.224,0.251]})?;fig.panels[ids[0]]=panel;
 let mut l=Legend::new(vec![LegendEntry::line("Ion-line fit",colors[0]),LegendEntry::marker("Plasma line",MarkerShape::Circle,colors[1])]);l.position=LegendPosition::AboveCenter;fig.panels[ids[0]].legend=Some(l);
 let mut l=Legend::new(vec![LegendEntry::line(r"$T_i$",colors[0]),LegendEntry::line(r"$T_e$",colors[1]),LegendEntry::line(r"$T_n$ (MSIS)",colors[2])]);l.position=LegendPosition::AboveCenter;fig.panels[ids[1]].legend=Some(l);
 for ext in ["png","svg"]{fig.save(format!("figures/arecibo_fitted_profiles.{ext}"))?;}
 let tt=read(&h,"time_unix_s")?;let x:Vec<_>=tt.iter().map(|v|(v-1525392000.)/3600.).collect();let nr=z.len();
 for (key,name,lim,label,scale) in [("ne_m3","arecibo_fitted_density",[0.,5.],r"$n_e$ ($10^{11}$ m$^{-3}$)",1e-11),("ti_K","arecibo_fitted_Ti",[100.,700.],r"$T_i$ (K; NO$^+$ model)",1.),("te_K","arecibo_fitted_Te",[100.,800.],r"$T_e$ (K; NO$^+$ model)",1.)]{
  let data=read(&h,key)?;let mut field=vec![0.;data.len()];for i in 0..x.len(){for j in 0..nr{field[j*x.len()+i]=data[i*nr+j]*scale;}}
  heat(name,&x,&z,field,lim,"Altitude (km)",label)?;
 }
 let mut fig=Figure::new(IntendedUse::ArticleWide);let idx=fig.grid(1,1)?[0];fig.panels[idx].rect.y+=15.;fig.height_mm+=15.;
 fig.panels[idx].axes=Some(Axes::new([0.,440.],[-0.4,1.2],"Lag (microseconds)","Normalized ion-line ACF"));
 let lag=read(&h,"profile/acf_lag_us")?;let err=read(&h,"profile/acf_sigma")?;
 for (suffix,color,rgb,shape) in [("real",colors[0],[0.141,0.341,0.455],MarkerShape::Circle),("imag",colors[1],[0.545,0.224,0.251],MarkerShape::Square)]{
  let obs=read(&h,&format!("profile/acf_measured_{suffix}"))?;let model=read(&h,&format!("profile/acf_model_{suffix}"))?;
  let points:Vec<_>=lag.iter().zip(&obs).map(|(&a,&b)|[a,b]).collect();let bounds:Vec<_>=obs.iter().zip(&err).map(|(&a,&b)|[a-b,a+b]).collect();
  fig.panels[idx].errorbars(&points,None,Some(&bounds),color,0.5,2.,0.6)?;
  let mut panel=fig.panels[idx].clone();panel.markers(&mut fig,&points,&MarkerStyle{shape,size_pt:3.,color:rgb})?;fig.panels[idx]=panel;
  fig.panels[idx].push(Layer::Line(Line::new(lag.iter().zip(model).map(|(&a,b)|[a,b]).collect(),color)));
 }
 let mut l=Legend::new(vec![LegendEntry::marker("Measured real",MarkerShape::Circle,colors[0]),LegendEntry::marker("Measured imaginary",MarkerShape::Square,colors[1]),LegendEntry::line("Fitted real",colors[0]),LegendEntry::line("Fitted imaginary",colors[1])]);l.position=LegendPosition::AboveCenter;l.columns=2;fig.panels[idx].legend=Some(l);
 for ext in ["png","svg"]{fig.save(format!("figures/arecibo_acf_fit.{ext}"))?;}
 Ok(())
}
