// Archived CSP density reconstruction, not a new kinetic fit.
// Source: evidence/seid_event/avg_acf_surf.py and ne_result.h5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
#[path = "../time_axes.rs"] mod time_axes;
fn main()->Result<()> {
 let h=File::open("evidence/seid_event/ne_result.h5")?;
 let t=h.dataset("unix_seconds")?.read_raw::<f64>()?;
 let z=h.dataset("range_km")?.read_raw::<f64>()?;
 let ne=h.dataset("Ne_m3")?.read_raw::<f64>()?;
 let x:Vec<f64>=t.iter().map(|v|(v-1525392000.)/3600.).collect();
 let mut xe=vec![x[0]-(x[1]-x[0])/2.];xe.extend(x.windows(2).map(|w|(w[0]+w[1])/2.));xe.push(x[x.len()-1]+(x[x.len()-1]-x[x.len()-2])/2.);
 let mut ze=vec![z[0]-0.15];ze.extend(z.iter().map(|v|v+0.15));
 let mut vals=vec![0.;ne.len()];
 for i in 0..t.len(){for j in 0..z.len(){
   // Negative/noise-floor estimates are under-range, never mirrored with abs().
   vals[j*t.len()+i]=ne[i*z.len()+j].max(1e8).log10();
 }}
 let mut f=Figure::new(IntendedUse::ArticleWide);let p=f.grid(1,1)?[0];
 f.panels[p].rect.y+=15.;f.panels[p].rect.width-=25.;f.panels[p].rect.height=85.;
 f.height_mm=f.panels[p].rect.y+104.;
 let mut ax=Axes::new([14.+35./60.+46./3600.,19.+5./60.+6./3600.],[50.,140.],"","Altitude (km)");
 time_axes::time_axes(&mut ax,true,true)?;ax.yticks=(5..=14).map(|i|i as f64*10.).collect();f.panels[p].axes=Some(ax);
 let sc=f.scale(Scale::new([8.,12.],Normalization::Linear,ColorMap::ember())?);
 f.panels[p].push(Layer::Pcolormesh(Field::new(xe,ze,vals,sc)?));
 let r=f.panels[p].rect;
 f.colorbars.push(Colorbar::new(Rect::new(r.x+r.width+4.,r.y,2.5,r.height),sc,r"$\log_{10}(n_e\,/\,\mathrm{m}^{-3})$",vec![8.,9.,10.,11.,12.]));
 f.layout_colorbars()?;
 for ext in ["png","svg"]{f.save(format!("figures/arecibo_archived_density_utc_ast.{ext}"))?;}
 Ok(())
}
