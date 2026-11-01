// Native-spectrum RCS catalogue and diagnostics; analysis provenance in HDF5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
#[path="../time_axes.rs"] mod time_axes;
const B:&str="#245774";const R:&str="#8B3940";
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn markers(f:&mut Figure,p:usize,points:&[[f64;2]],color:[f64;3],shape:MarkerShape)->Result<()>{let mut panel=f.panels[p].clone();panel.markers(f,points,&MarkerStyle{shape,size_pt:3.,color})?;f.panels[p]=panel;Ok(())}
fn main()->Result<()>{
 let h=File::open("data/plasma_line_rcs.h5")?;
 let t:Vec<_>=rd(&h,"time_unix_s")?.iter().map(|v|(v-1525392000.)/3600.).collect();
 let rows=rd(&h,"catalog/rows")?;
 // Copernicus manuscript is 177 mm wide with 10 pt body type.
 let mut f=ArticleLayout::new(177.,10.)?.figure(1.,92.)?;
 let p=f.grid(1,1)?[0];f.height_mm=92.;
 f.panels[p].rect=Rect::new(22.,16.,131.,58.);
 let mut ax=Axes::new([t[0],*t.last().unwrap()],[-115.,-60.],"",r"Equivalent RCS (dB m$^2$)");
 ax.yticks=vec![-110.,-100.,-90.,-80.,-70.,-60.];
 time_axes::time_axes(&mut ax,true,true)?;
 // A common display conversion uses the median event RCS-to-SNR offset.
 // Column 7 is fitted line power/noise power in a 22.727 kHz bandwidth.
 // Preserve the individual calibrated RCS and SNR values in the HDF5 catalogue.
 let mut offsets:Vec<f64>=rows.chunks(14)
  .filter(|r|r[0]<2. && r[7]>0. && r[7].is_finite() && r[9].is_finite())
  .map(|r|10.*r[7].log10()-r[9]).collect();
 offsets.sort_by(f64::total_cmp);
 anyhow::ensure!(!offsets.is_empty(),"No valid RCS/SNR calibration pairs");
 let n=offsets.len();
 let offset=if n%2==0 {(offsets[n/2-1]+offsets[n/2])/2.}else{offsets[n/2]};
 ax.secondary_y=Some(SecondaryAxis::new("SNR (dB)",vec![-10.,0.,10.,20.,30.,40.],|db|db-offset)?);
 f.panels[p].axes=Some(ax);
 for j in 0..2 {
  let rgb=if j==0{[0.141,0.341,0.455]}else{[0.545,0.224,0.251]};
  let shape=if j==0{MarkerShape::Circle}else{MarkerShape::Diamond};
  let mut selected:Vec<_>=rows.chunks(14).filter(|r|r[0] as usize==j).collect();
  selected.sort_by(|a,b|a[2].total_cmp(&b[2]));
  let rcs:Vec<_>=selected.iter().map(|r|[(r[2]-1525392000.)/3600.,r[9]]).collect();
  markers(&mut f,p,&rcs,rgb,shape)?;
 }
 let mut legend=Legend::new(vec![
  LegendEntry::marker("4.51 MHz",MarkerShape::Circle,B),
  LegendEntry::marker("5.34 MHz",MarkerShape::Diamond,R)]);
 legend.columns=1;legend.position=LegendPosition::UpperRight;
 f.panels[p].legend=Some(legend);
 for ext in ["png","pdf"]{f.save(format!("figures/plasma_line_rcs_time.{ext}"))?;}
 if std::env::args().any(|s|s=="--time-only"){return Ok(());}
 // Native spectral-core fits for each detected branch's strongest return.
 let sums=rd(&h,"catalog/peak_summary")?;let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=190.;f.height_mm=220.;let ps=f.grid(3,1)?;
 for (panel,s) in sums.chunks(10).enumerate(){
  let j=s[0] as usize;let i=s[1] as usize;let p=ps[panel];let x=rd(&h,&format!("band{j}/frequency_mhz"))?;let n=x.len();let all=rd(&h,&format!("band{j}/observed"))?;let y=&all[i*n..(i+1)*n];let vals=rd(&h,&format!("band{j}/values"))?;let v=&vals[i*14..(i+1)*14];let xmin=x.iter().copied().fold(f64::INFINITY,f64::min);let xmax=x.iter().copied().fold(f64::NEG_INFINITY,f64::max);
  f.panels[p].rect=Rect::new(27.,12.+panel as f64*67.,155.,47.);let ymax=y.iter().copied().chain(rd(&h,&format!("band{j}/peak_dense_fit"))?).fold(0.,f64::max)*1.18;
  f.panels[p].axes=Some(Axes::new([xmin,xmax],[-0.05*ymax,ymax],"Signed frequency offset (MHz)","Excess / noise power"));
  let xx=rd(&h,&format!("band{j}/peak_dense_frequency_mhz"))?;let yy=rd(&h,&format!("band{j}/peak_dense_fit"))?;let pts=xx.iter().zip(yy).map(|(&x,y)|[x,y]).collect();let mut line=Line::new(pts,R);line.width_pt=1.;f.panels[p].push(Layer::Line(line));markers(&mut f,p,&x.iter().zip(y).map(|(&x,&y)|[x,y]).collect::<Vec<_>>(),[0.141,0.341,0.455],MarkerShape::Circle)?;
  let secs=(s[2]-1525392000.) as i64;f.panels[p].label_at(format!("{} UTC, {:0.2} km",format!("{:02}:{:02}:{:02}",secs/3600,secs%3600/60,secs%60),s[4]),[xmin+0.20*(xmax-xmin),0.90*ymax]);
  f.panels[p].legend=Some(Legend::new(vec![LegendEntry::circle("Native spectrum",B),LegendEntry::line("Broadened pulse response",R)]));
 }
 for ext in ["png","pdf"]{f.save(format!("figures/plasma_line_rcs_fits.{ext}"))?;}
 Ok(())
}
