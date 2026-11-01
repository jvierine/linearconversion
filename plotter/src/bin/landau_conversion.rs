// Kinetic closure and smooth-density provenance are stored in the input HDF5.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
const COLORS:[&str;2]=["#245774","#8b3940"];
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn main()->Result<()>{
 let h=File::open("data/landau_bragg_study.h5")?;
 for epoch in ["1525453430","1525460596"]{
  for kind in ["wavelength","attenuation"]{
   let mut f=Figure::new(IntendedUse::ArticleWide);let p=f.grid(1,1)?[0];
   f.panels[p].rect.height=90.;f.panels[p].rect.y+=12.;f.height_mm=136.;
   let short=kind=="wavelength";
   f.panels[p].axes=Some(Axes::new(if short{[0.,1.3]}else{[0.,1.05]},if short{[-20.,0.]}else{[-90.,0.]},if short{"Plasma-wave wavelength (m)"}else{"Amplitude remaining from collisionless damping"},"Height relative to plasma resonance (m)"));
   let mut entries=vec![];
   for (i,freq) in ["4.51","5.33"].iter().enumerate(){
    let pre=format!("{epoch}/{freq}");let z=rd(&h,&format!("{pre}/height_relative_resonance_m"))?;
    let xx=if short{rd(&h,&format!("{pre}/kz_real_m1"))?.iter().map(|v|-2.*std::f64::consts::PI/v).collect()}else{rd(&h,&format!("{pre}/amplitude_survival"))?};
    let pts=xx.iter().zip(z.iter()).filter(|(x,z)|x.is_finite()&&**z>=if short{-20.}else{-90.}).map(|(&x,&z)|[x,z]).collect();
    f.panels[p].push(Layer::Line(Line::new(pts,COLORS[i])));entries.push(LegendEntry::line(format!("{freq} MHz"),COLORS[i]));
    let dz=h.group(&pre)?.attr("resonance_to_bragg_m")?.read_scalar::<f64>()?;
    let mut line=Line::new(vec![[0.,-dz],[if short{0.3485958814}else{1.},-dz]],COLORS[i]);line.dash=Some("3,3".into());line.width_pt=0.6;f.panels[p].push(Layer::Line(line));
   }
   if short{let mut line=Line::new(vec![[0.3485958814,-20.],[0.3485958814,0.]],"#8795a9");line.dash=Some("3,3".into());f.panels[p].push(Layer::Line(line));entries.push(LegendEntry::line("430 MHz Bragg wavelength","#8795a9"));}
   let mut legend=Legend::new(entries);legend.position=LegendPosition::AboveCenter;legend.columns=if short{3}else{2};f.panels[p].legend=Some(legend);
   for ext in ["png","svg"]{f.save(format!("figures/landau_{kind}_{epoch}.{ext}"))?;}
  }
 }
 let mut fig=Figure::new(IntendedUse::ArticleWide);let p=fig.grid(1,1)?[0];
 fig.panels[p].rect.height=90.;fig.panels[p].rect.y+=12.;fig.height_mm=136.;
 fig.panels[p].axes=Some(Axes::new([-0.85,0.85],[-5.,5.],r"Fractional electron density (\%)","Height relative to 430 MHz Bragg match (m)"));
 let zz=rd(&h,"1525453430/5.33/micro_height_relative_bragg_m")?;
 for (i,key) in ["micro_density_real_percent","micro_density_amplitude_percent"].iter().enumerate(){
  let aa=rd(&h,&format!("1525453430/5.33/{key}"))?;
  fig.panels[p].push(Layer::Line(Line::new(aa.iter().zip(zz.iter()).map(|(&a,&z)|[a,z]).collect(),COLORS[i])));
 }
 let mut legend=Legend::new(vec![LegendEntry::line(r"$\mathrm{Re}(\delta n_e/n_e)$",COLORS[0]),LegendEntry::line("Amplitude",COLORS[1])]);legend.position=LegendPosition::AboveCenter;legend.columns=2;fig.panels[p].legend=Some(legend);
 for ext in ["png","svg"]{fig.save(format!("figures/landau_density_wave_1525453430.{ext}"))?;}
 Ok(())
}
