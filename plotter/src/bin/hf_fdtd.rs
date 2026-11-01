// Scientific arrays and metadata prepared by hf_fdtd/examples/prepare_arecibo_figure.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
const BLUE:&str="#245774";const RED:&str="#8b3940";const GREEN:&str="#4c7566";
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn line(f:&mut Figure,p:usize,x:&[f64],z:&[f64],color:&str){f.panels[p].push(Layer::Line(Line::new(x.iter().zip(z).map(|(&x,&z)|[x,z]).collect(),color)));}
fn save(f:&Figure,n:&str)->Result<()>{for ext in ["png","svg","pdf"]{f.save(format!("figures/{n}.{ext}"))?;}Ok(())}
fn edges(v:&[f64])->Vec<f64>{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|w|(w[0]+w[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e}
fn main()->Result<()>{
 let h=File::open("output/hf_fdtd/field_plot.h5")?;let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,3)?;f.width_mm=270.;f.height_mm=142.;
 for i in 0..3{f.panels[ps[i]].rect=Rect::new(25.+89.*i as f64,10.,62.,104.);}
 let z=rd(&h,"wide/z_km")?;
 f.panels[ps[0]].axes=Some(Axes::new([0.,4.5],[117.4,118.6],r"$n_e$ ($10^{11}$ m$^{-3}$)","Altitude above radar (km)"));
 line(&mut f,ps[0],&rd(&h,"wide/density")?,&z,BLUE);
 let mut ax=Axes::new([0.01,300.],[117.4,118.6],r"$|E|/|E_{\rm inc}|$","Altitude above radar (km)");ax.set_xscale(AxisScale::Log10)?;f.panels[ps[1]].axes=Some(ax);
 for (comp,col) in [("x",BLUE),("y",GREEN),("z",RED)]{line(&mut f,ps[1],&rd(&h,&format!("wide/total_{comp}"))?,&z,col);}
 let zf=rd(&h,"fine/z_relative_m")?;
 f.panels[ps[2]].axes=Some(Axes::new([0.,300.],[-3.,3.],r"$|E_z|/|E_{\rm inc}|$","Height relative to resonance (m)"));
 line(&mut f,ps[2],&rd(&h,"fine/total_z")?,&zf,RED);line(&mut f,ps[2],&rd(&h,"fine/transverse_z")?,&zf,BLUE);
 for (p,entries) in [(ps[1],vec![LegendEntry::line("Horizontal x",BLUE),LegendEntry::line("Horizontal y",GREEN),LegendEntry::line("Vertical",RED)]),(ps[2],vec![LegendEntry::line("Total cold field",RED),LegendEntry::line("Transverse drive",BLUE)])]{let mut l=Legend::new(entries);l.columns=1;f.panels[p].legend=Some(l);}
 save(&f,"hf_fdtd_field")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let p=f.grid(1,1)?[0];f.panels[p].rect=Rect::new(28.,12.,135.,100.);f.height_mm=160.;
 let t=rd(&h,"time_ns")?;let z=rd(&h,"map_height_km")?;let vals=rd(&h,"Ex_time_over_incident")?;
 f.panels[p].axes=Some(Axes::new([t[0],*t.last().unwrap()],[117.5,118.5],"Time within recorded steady wave (ns)","Altitude above radar (km)"));
 let lim=(2.*vals.iter().fold(0f64,|a,&v|a.max(v.abs()))).ceil()/2.;let sc=f.scale(Scale::new([-lim,lim],Normalization::Linear,ColorMap::diverging())?);
 f.panels[p].push(Layer::Pcolormesh(Field::new(edges(&t),edges(&z),vals,sc)?));
 let mut cb=Colorbar::new(Rect::new(32.,132.,127.,3.),sc,r"$E_x/|E_{\rm inc}|$",vec![-lim,0.,lim]);cb.horizontal=true;f.colorbars.push(cb);f.layout_colorbars()?;
 save(&f,"hf_fdtd_recorded_field")?;
 Ok(())
}
