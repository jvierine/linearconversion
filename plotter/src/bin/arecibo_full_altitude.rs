// Native Rustplotlib defaults. Provenance and fit diagnostics live in HDF5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
#[path = "../time_axes.rs"] mod time_axes;
fn read(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn edges(v:&[f64])->Vec<f64>{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|a|(a[0]+a[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e}
fn native_image(field:&Field,scale:&Scale)->image::RgbaImage {
 let nx=field.x_edges.len()-1;let ny=field.y_edges.len()-1;let width=nx*8;
 let mut out=image::RgbaImage::new(width as u32,ny as u32);
 for x in 0..width {let xx=field.x_edges[0]+(x as f64+0.5)/width as f64*(field.x_edges[nx]-field.x_edges[0]);let i=field.x_edges.partition_point(|&q|q<=xx).saturating_sub(1).min(nx-1);
  for j in 0..ny {out.put_pixel(x as u32,(ny-1-j) as u32,image::Rgba(scale.map(field.values[j*nx+i]).expect("finite fitted cell")));}
 }
 out
}
fn main()->Result<()> {
 let h=File::open("data/eregion_fit/full_altitude_fits.h5")?;
 let t=read(&h,"time_unix_s")?;let z=read(&h,"height_km")?;let a=read(&h,"results")?;
 let xe:Vec<_>=read(&h,"time_edges_unix_s")?.iter().map(|v|(v-1525392000.)/3600.).collect();
 for (name,yl,yt,limits) in [
  ("full_altitude",[35.,1500.],vec![100.,300.,600.,900.,1200.,1500.],[[8.,12.],[0.,5000.],[0.,4000.],[-300.,300.]]),
  ("de_region",[50.,180.],vec![50.,75.,100.,125.,150.,175.],[[8.,12.],[100.,1500.],[100.,1000.],[-100.,100.]]),
  ("f_region",[150.,700.],vec![200.,300.,400.,500.,600.,700.],[[9.,12.],[500.,4000.],[300.,3000.],[-300.,300.]])]{
  let jj:Vec<_>=(0..z.len()).filter(|&j|z[j]>=yl[0]&&z[j]<=yl[1]).collect();let zz:Vec<_>=jj.iter().map(|&j|z[j]).collect();
  let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=195.;let ids=f.grid(4,1)?;f.height_mm=304.;
  for (row,&p) in ids.iter().enumerate(){
   f.panels[p].rect=Rect::new(22.,26.+row as f64*65.,138.,53.);
   let ze=edges(&zz);let mut ax=Axes::new([xe[0],xe[xe.len()-1]],[ze[0],ze[ze.len()-1]],"","Altitude (km)");ax.yticks=yt.clone();time_axes::time_axes(&mut ax,row==3,row==0)?;f.panels[p].axes=Some(ax);
   let mut v=vec![];for &j in &jj{for i in 0..t.len(){let x=a[(i*z.len()+j)*16+row];assert!(x.is_finite(),"unfitted cell in final plot");v.push(if row==0{x.max(1e8).log10()}else{x});}}
   let sc=f.scale(Scale::new(limits[row],Normalization::Linear,if row==3{ColorMap::diverging()}else{ColorMap::ember()})?);
   let field=Field::new(xe.clone(),edges(&zz),v,sc)?;let img=native_image(&field,&f.scales[sc]);f.panels[p].push(Layer::Image(img));let r=f.panels[p].rect;
   let label=[r"$\log_{10}(n_e\,/\,\mathrm{m}^{-3})$",r"$T_e$ (K)",r"$T_i$ (K)",r"Apparent $v_i$ (m s$^{-1}$)"][row];
   let ticks=if row==0{(8..=12).map(|x|x as f64).filter(|&x|x>=limits[row][0]&&x<=limits[row][1]).collect()}else if row==3{vec![limits[row][0],0.,limits[row][1]]}else{vec![limits[row][0],(limits[row][0]+limits[row][1])/2.,limits[row][1]]};
   f.colorbars.push(Colorbar::new(Rect::new(r.x+r.width+4.,r.y,2.5,r.height),sc,label,ticks));
  }
  f.layout_colorbars()?;
  f.save(format!("figures/arecibo_parameters_{name}.png"))?;
  // PDF retains the full field for zooming and scientific reuse.
  f.save(format!("figures/arecibo_parameters_{name}.pdf"))?;
 }
 let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=195.;let ids=f.grid(3,1)?;f.height_mm=240.;
 for (row,&p) in ids.iter().enumerate(){
  f.panels[p].rect=Rect::new(22.,26.+row as f64*65.,138.,53.);let mut ax=Axes::new([xe[0],xe[xe.len()-1]],[35.,1500.],"","Altitude (km)");ax.yticks=vec![100.,300.,600.,900.,1200.,1500.];time_axes::time_axes(&mut ax,row==2,row==0)?;f.panels[p].axes=Some(ax);
  let mut v=vec![];for j in 0..z.len(){for i in 0..t.len(){let k=(i*z.len()+j)*16;let value=if row==0{(a[k+4]/a[k].max(1.)).min(3.)}else{a[k+10+row]};v.push(value);}}
  let limit=if row==0{[0.,3.]}else{[0.,1.]};let sc=f.scale(Scale::new(limit,Normalization::Linear,ColorMap::ember())?);f.panels[p].push(Layer::Pcolormesh(Field::new(xe.clone(),edges(&z),v,sc)?));let r=f.panels[p].rect;
  let label=[r"$\sigma_{n_e}/n_e$",r"$T_e$ prior variance reduction",r"$T_i$ prior variance reduction"][row];
  f.colorbars.push(Colorbar::new(Rect::new(r.x+r.width+4.,r.y,2.5,r.height),sc,label,vec![0.,limit[1]/2.,limit[1]]));
 }
 f.layout_colorbars()?;f.save("figures/arecibo_parameters_information.png")?;
 Ok(())
}
