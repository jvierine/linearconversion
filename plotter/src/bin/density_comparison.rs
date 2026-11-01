// Source arrays and user-selected density palette: prepare_density_comparison.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
#[path = "../time_axes.rs"] mod time_axes;
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn edges(v:&[f64])->Vec<f64>{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|w|(w[0]+w[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e}
fn curve(f:&mut Figure,p:usize,x:&[f64],y:&[f64],color:&str,width:f64){let mut l=Line::new(x.iter().zip(y).map(|(&x,&y)|[x,y]).collect(),color);l.width_pt=width;f.panels[p].push(Layer::Line(l));}
fn save(f:&Figure,n:&str)->Result<()>{for ext in ["png","pdf"]{f.save(format!("figures/{n}.{ext}"))?;}Ok(())}
fn main()->Result<()>{
 let h=File::open("output/coupled/density_comparison.h5")?;
 let t=rd(&h,"map/time_edges_utc_hour")?;let z=rd(&h,"map/height_km")?;let values=rd(&h,"map/ne_m3")?.iter().map(|x|x.max(1e8).log10()).collect();
 let palette=rd(&h,"palette/density_rgb")?.chunks_exact(3).map(|v|[v[0],v[1],v[2]]).collect();
 let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=200.;f.height_mm=145.;let p=f.grid(1,1)?[0];f.height_mm=145.;f.panels[p].rect=Rect::new(22.,24.,147.,99.);
 let mut a=Axes::new([t[0],t[t.len()-1]],[50.,140.],"","Altitude above radar (km)");time_axes::time_axes(&mut a,true,true)?;a.yticks=(5..=14).map(|i|i as f64*10.).collect();f.panels[p].axes=Some(a);
 let sc=f.scale(Scale::new([8.,12.],Normalization::Linear,ColorMap::listed(palette)?)?);
 f.panels[p].push(Layer::Pcolormesh(Field::new(t,edges(&z),values,sc)?));
 let time=h.attr("selected_utc_hour")?.read_scalar::<f64>()?;let mut l=Line::new(vec![[time,50.],[time,140.]],"#ffffff");l.dash=Some("4 3".into());l.width_pt=0.8;f.panels[p].push(Layer::Line(l));
 let r=f.panels[p].rect;f.colorbars.push(Colorbar::new(Rect::new(r.x+r.width+4.,r.y,3.,r.height),sc,r"$\log_{10}(n_e\,/\,\mathrm{m}^{-3})$",(16..=24).map(|i|i as f64/2.).collect()));f.layout_colorbars()?;
 save(&f,"coupled_measured_density")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=235.;f.height_mm=146.;let ps=f.grid(1,2)?;f.height_mm=146.;
 let zm=rd(&h,"profile/csp_height_km")?;let nm:Vec<_>=rd(&h,"profile/csp_ne_m3")?.iter().map(|v|v/1e11).collect();
 let ns:Vec<_>=rd(&h,"smooth/ne_m3")?.iter().map(|v|v/1e11).collect();let zs=rd(&h,"smooth/height_km")?;
 let nc:Vec<_>=rd(&h,"simulation/ne_m3")?.iter().map(|v|v/1e11).collect();let zc=rd(&h,"simulation/height_km")?;
 let nr:Vec<_>=rd(&h,"smooth_csp/ne_m3")?.iter().map(|v|v/1e11).collect();let zr=rd(&h,"smooth_csp/height_km")?;
 for i in 0..2{let p=ps[i];let yr=if i==0{[100.,130.]}else{[117.,119.5]};f.panels[p].rect=Rect::new(24.+116.*i as f64,8.,82.,112.);
 f.panels[p].axes=Some(Axes::new([0.,5.2],yr,r"$n_e$ ($10^{11}$ m$^{-3}$)","Altitude above radar (km)"));
 curve(&mut f,p,&nm,&zm,"#8795a9",0.8);
 if i==0{curve(&mut f,p,&ns,&zs,"#245774",1.3);}else{curve(&mut f,p,&nc,&zc,"#245774",1.3);}
 curve(&mut f,p,&nr,&zr,"#527f6d",1.3);
 f.panels[p].legend=Some(Legend::new(vec![LegendEntry::line("Short-pulse data","#8795a9"),LegendEntry::line("Reference fit","#245774"),LegendEntry::line("Short-pulse fit","#527f6d")]));
 }
 save(&f,"coupled_density_profile_comparison")?;Ok(())
}
