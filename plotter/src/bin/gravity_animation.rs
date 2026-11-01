// Animation frames are rendered by native Rustplotlib from simulated HDF5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn edges(v:&[f64])->Vec<f64>{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|w|(w[0]+w[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e}
fn main()->Result<()> {
 let h=File::open("output/gravity_wave_study/kh_baseline.h5")?;
 let t=h.dataset("time_s")?.read_raw::<f64>()?;let z=h.dataset("height_km")?.read_raw::<f64>()?;let x=h.dataset("x_km")?.read_raw::<f64>()?;
 let w=h.dataset("w_m_s")?.read_raw::<f32>()?;let vort=h.dataset("vorticity_s_1")?.read_raw::<f32>()?;
 std::fs::create_dir_all("output/gravity_wave_study/frames")?;
 for (frame,i) in (0..t.len()).step_by(3).enumerate(){
  let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=190.;f.height_mm=154.;f.raster_dpi=140.;let ids=f.grid(2,1)?;
  for (row,&p) in ids.iter().enumerate(){
   f.panels[p].rect=Rect::new(18.,12.+row as f64*67.,137.,47.);
   let (ylim,lim,data,label)=if row==0{([110.,126.],[-0.05,0.05],&vort,r"Vorticity (s$^{-1}$)")}else{([98.,138.],[-2.,2.],&w,r"$w$ (m s$^{-1}$)")};
   let js:Vec<_>=(0..z.len()).filter(|&j|z[j]>=ylim[0]&&z[j]<=ylim[1]).collect();let zz:Vec<_>=js.iter().map(|&j|z[j]).collect();
   let mut v=vec![];for &j in &js{for k in 0..x.len(){v.push(data[(i*z.len()+j)*x.len()+k] as f64);}}
   let mut ax=Axes::new([0.,40.],ylim,if row==0{"Horizontal distance (km)".to_string()}else{format!("Horizontal distance (km); elapsed time {:.1} min",t[i]/60.)},"Altitude (km)");ax.xticks=vec![0.,10.,20.,30.,40.];ax.yticks=if row==0{vec![110.,118.,126.]}else{vec![100.,110.,118.,130.]};f.panels[p].axes=Some(ax);
   let sc=f.scale(Scale::new(lim,Normalization::Linear,ColorMap::diverging())?);f.panels[p].push(Layer::Pcolormesh(Field::new(edges(&x),edges(&zz),v,sc)?));
   let r=f.panels[p].rect;f.colorbars.push(Colorbar::new(Rect::new(r.x+r.width+4.,r.y,2.5,r.height),sc,label,vec![lim[0],0.,lim[1]]));
  }
  f.layout_colorbars()?;f.save(format!("output/gravity_wave_study/frames/frame_{frame:04}.png"))?;
 }
 Ok(())
}
