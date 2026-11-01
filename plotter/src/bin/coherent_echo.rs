// Data and equation provenance: coupled_sim/scripts/coherent_echo.py, HDF5 metadata.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
const BLUE:&str="#245774";const RED:&str="#8b3940";const GREEN:&str="#4c7566";
fn rd(h:&File,n:&str)->Result<Vec<f64>>{Ok(h.dataset(n)?.read_raw()?)}
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],col:&str){let mut v=Vec::new();let a=f.panels[p].axes.as_ref().unwrap();for(&x,&y)in x.iter().zip(y){if x>=a.xlim[0]&&x<=a.xlim[1]{v.push([x,y]);}}f.panels[p].push(Layer::Line(Line::new(v,col)));}
fn main()->Result<()>{
 let root=if std::path::Path::new("reference/coupled/coherent_echo.h5").exists(){"reference"}else{"linearconversion/reference"};let h=File::open(format!("{root}/coupled/coherent_echo.h5"))?;
 let mut f=ArticleLayout::new(177.,10.)?.figure(1.,107.)?;let ps=f.grid(1,2)?;f.height_mm=107.;
 for(i,&p)in ps.iter().enumerate(){f.panels[p].rect=Rect::new(20.+87.*i as f64,5.,66.,76.);}
 f.panels[ps[0]].axes=Some(Axes::new([0.,0.75],[-1.7,1.7],r"Time ($\mu$s)",r"Voltage ($\mu$V, $50\,\Omega$)"));
 let t=rd(&h,"plot/time_us")?;line(&mut f,ps[0],&t,&rd(&h,"plot/I_uV")?,BLUE);line(&mut f,ps[0],&t,&rd(&h,"plot/Q_uV")?,RED);
 f.panels[ps[0]].legend=Some(Legend::new(vec![LegendEntry::line("I",BLUE),LegendEntry::line("Q",RED)]));
 f.panels[ps[1]].axes=Some(Axes::new([5.27,5.41],[0.,105.],"Frequency magnitude (MHz)","Plasma-line SNR"));
 line(&mut f,ps[1],&rd(&h,"plot/dense_frequency_mhz")?,&rd(&h,"plot/model_snr")?,RED);
 line(&mut f,ps[1],&rd(&h,"plot/observed_frequency_mhz")?,&rd(&h,"observed_snr")?,BLUE);
 f.panels[ps[1]].legend=Some(Legend::new(vec![LegendEntry::line("Data",BLUE),LegendEntry::line("Model",RED)]));
 std::fs::create_dir_all("figures")?;f.save("figures/coherent_radar_echo.png")?;f.save("figures/coherent_radar_echo.pdf")?;let _=GREEN;Ok(())
}
