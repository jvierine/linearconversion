// Scientific arrays: check_plasma_line_asymmetry.py; signed native spectra.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],col:&str){let mut l=Line::new(x.iter().zip(y).map(|(&a,&b)|[a,b]).collect(),col);l.width_pt=1.2;f.panels[p].push(Layer::Line(l));}
fn main()->Result<()>{
 let h=File::open("output/fluid/plasma_line_asymmetry.h5")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let ps=f.grid(1,2)?;f.width_mm=235.;f.height_mm=130.;
 for i in 0..2 {f.panels[ps[i]].rect=Rect::new(25.+116.*i as f64,8.,81.,95.);f.panels[ps[i]].axes=Some(Axes::new([5.27,5.41],if i==0{[-2.,80.]}else{[-0.06,0.06]},"Frequency magnitude (MHz)","Excess power / noise power"));line(&mut f,ps[i],&[5.27,5.41],&[0.,0.],"#a4acb2");}
 for (name,color,label) in [("enhanced","#245774","Enhanced branch"),("opposite","#8b3940","Opposite branch")]{
  let g=h.group(&format!("1525453430/{name}"))?;let x=g.dataset("frequency_magnitude_mhz")?.read_raw::<f64>()?;let y=g.dataset("signed_excess_snr")?.read_raw::<f64>()?;
  line(&mut f,ps[0],&x,&y,color);if name=="opposite"{line(&mut f,ps[1],&x,&y,color);f.panels[ps[1]].legend=Some(Legend::new(vec![LegendEntry::line(label,color)]));}
 }
 let mut l=Legend::new(vec![LegendEntry::line("Enhanced","#245774"),LegendEntry::line("Opposite","#8b3940")]);l.columns=1;f.panels[ps[0]].legend=Some(l);
 for ext in ["png","pdf"]{f.save(format!("figures/plasma_line_asymmetry.{ext}"))?;}Ok(())
}
