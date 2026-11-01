// Native-spectrum RCS catalogue and diagnostics; analysis provenance in HDF5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
#[path="../time_axes.rs"] mod time_axes;
const B:&str="#245774";const R:&str="#8B3940";const G:&str="#8895a7";
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn markers(f:&mut Figure,p:usize,points:&[[f64;2]],color:[f64;3],shape:MarkerShape)->Result<()>{let mut panel=f.panels[p].clone();panel.markers(f,points,&MarkerStyle{shape,size_pt:3.,color})?;f.panels[p]=panel;Ok(())}
fn main()->Result<()>{
 let h=File::open("data/plasma_line_rcs.h5")?;let cal=rd(&h,"ion_check/per_record_multiplier")?;let t:Vec<_>=rd(&h,"time_unix_s")?.iter().map(|v|(v-1525392000.)/3600.).collect();let rows=rd(&h,"catalog/rows")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=190.;f.height_mm=175.;let ps=f.grid(2,1)?;
 for j in 0..2 {
  let p=ps[j];f.panels[p].rect=Rect::new(27.,26.+j as f64*70.,155.,48.);
  let mut ax=Axes::new([t[0],*t.last().unwrap()],[-120.,-60.],"",r"Equivalent RCS (dB m$^2$)");ax.yticks=vec![-120.,-100.,-80.,-60.];time_axes::time_axes(&mut ax,j==1,j==0)?;f.panels[p].axes=Some(ax);
  let sens=rd(&h,&format!("band{j}/rcs_sensitivity_m2"))?;let pts:Vec<_>=t.iter().zip(sens).enumerate().filter(|(i,_)|cal[*i]>0.05&&cal[*i]<1.).map(|(_,(&x,y))|[x,10.*y.log10()]).collect();let mut line=Line::new(pts,G);line.width_pt=0.5;f.panels[p].push(Layer::Line(line));
  let mut pts=vec![];let mut opp=vec![];
  for r in rows.chunks(14){if r[0] as usize==j{pts.push([(r[2]-1525392000.)/3600.,r[9]])}else if r[0] as usize==j+2{opp.push([(r[2]-1525392000.)/3600.,r[9]])}}
  markers(&mut f,p,&pts,if j==0{[0.141,0.341,0.455]}else{[0.322,0.498,0.427]},MarkerShape::Circle)?;markers(&mut f,p,&opp,[0.545,0.224,0.251],MarkerShape::Square)?;
  f.panels[p].label_at(if j==0{"4.51 MHz"}else{"5.34 MHz"},[14.82,-66.]);
  let mut l=Legend::new(vec![LegendEntry::circle("Enhanced branch",if j==0{B}else{"#527f6d"}),LegendEntry::marker("Opposite candidate",MarkerShape::Square,R),LegendEntry::line("Sensitivity",G)]);l.columns=1;f.panels[p].legend=Some(l);
 }
 for ext in ["png","pdf"]{f.save(format!("figures/plasma_line_rcs_time.{ext}"))?;}
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
