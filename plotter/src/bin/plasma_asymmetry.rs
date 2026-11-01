// Scientific arrays: check_plasma_line_asymmetry.py; native spectral powers.
// Display labels follow the manuscript's adopted RF sideband convention.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],col:&str){let mut l=Line::new(x.iter().zip(y).map(|(&a,&b)|[a,b]).collect(),col);l.width_pt=1.2;f.panels[p].push(Layer::Line(l));}
fn main()->Result<()>{
 let h=File::open("output/fluid/plasma_line_asymmetry.h5")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let p=f.grid(1,1)?[0];
 f.width_mm=180.;f.height_mm=76.;
 f.panels[p].rect=Rect::new(23.,5.,150.,53.);
 f.panels[p].axes=Some(Axes::new([5.27,5.41],[-2.,80.],"Frequency magnitude (MHz)","Excess power / noise power"));
 for (name,color) in [("enhanced","#245774"),("opposite","#8b3940")]{
  let g=h.group(&format!("1525453430/{name}"))?;
  let x=g.dataset("frequency_magnitude_mhz")?.read_raw::<f64>()?;
  let y=g.dataset("signed_excess_snr")?.read_raw::<f64>()?;
  line(&mut f,p,&x,&y,color);
 }
 let mut l=Legend::new(vec![LegendEntry::line("Upshifted","#245774"),LegendEntry::line("Downshifted","#8b3940")]);
 l.columns=1;f.panels[p].legend=Some(l);
 for ext in ["png","pdf"]{f.save(format!("figures/plasma_line_asymmetry.{ext}"))?;}
 Ok(())
}
