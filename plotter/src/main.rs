// Numerical products are in HDF5. Rustplotlib defaults own typography and axes.
// Series colors follow the native rustplotlib gallery; unsigned fields use
// its dark-low/bright-high ember palette, consistent with the user's rule.
use anyhow::Result;
use rustplotlib::*;
use hdf5::File;
const COLORS:[&str;4]=["#245774","#8b3940","#4c7566","#ac792e"];
fn read(f:&File,p:&str)->Result<Vec<f64>>{Ok(f.dataset(p)?.read_raw::<f64>()?)}
fn scaled(v:Vec<f64>,a:f64)->Vec<f64>{v.into_iter().map(|x|x*a).collect()}
fn save(f:&Figure,name:&str)->Result<()>{for ext in ["png","svg"]{f.save(format!("figures/{name}.{ext}"))?;}println!("Rendered {name}");Ok(())}
fn lineplot(name:&str,x:&[f64],ys:Vec<(Vec<f64>,String)>,xlim:[f64;2],ylim:[f64;2],xl:&str,yl:&str,log:bool)->Result<()> {
 let mut f=Figure::new(IntendedUse::ArticleWide); let ids=f.grid(1,1)?;let idx=ids[0];
 let mut ax=Axes::new(xlim,ylim,xl,yl);if log{ax.set_yscale(AxisScale::Log10)?;}
 f.panels[idx].axes=Some(ax);let mut entries=vec![];
 for (i,(y,label)) in ys.into_iter().enumerate(){
  f.panels[idx].push(Layer::Line(Line::new(x.iter().zip(y).map(|(&a,b)|[a,b]).collect(),COLORS[i])));
  if !label.is_empty(){entries.push(LegendEntry::line(label,COLORS[i]));}
 }
 if !entries.is_empty(){f.panels[idx].legend=Some(Legend::new(entries));}
 save(&f,name)
}
fn edges(x:&[f64])->Vec<f64>{let mut e=vec![x[0]-(x[1]-x[0])/2.];e.extend(x.windows(2).map(|w|(w[0]+w[1])/2.));e.push(x[x.len()-1]+(x[x.len()-1]-x[x.len()-2])/2.);e}
fn heatmap(name:&str,x:&[f64],y:&[f64],vals:Vec<f64>,lim:[f64;2],ylabel:&str,clabel:&str,cticks:Vec<f64>,contours:Vec<(Vec<f64>,&str)>)->Result<()> {
 let mut f=Figure::new(IntendedUse::ArticleWide);let ids=f.grid(1,1)?;let idx=ids[0];
 // Reserve physical room for attached colorbar using the native layout engine.
 let r=f.panels[idx].rect;
 f.panels[idx].axes=Some(Axes::new([x[0],x[x.len()-1]],[y[0],y[y.len()-1]],r"Angle between $\mathbf{k}$ and $\mathbf{B}$ (degrees)",ylabel));
 let sc=f.scale(Scale::new(lim,Normalization::Log,ColorMap::ember())?);
 f.panels[idx].push(Layer::Pcolormesh(Field::new(edges(x),edges(y),vals,sc)?));
 for (v,color) in contours{
  let co=Contour::new(x.to_vec(),y.to_vec(),v,vec![25.,26.],sc,false)?;
  for path in co.paths(25.) { f.panels[idx].push(Layer::Line(Line::new(path,color))); }
 }
 let mut bar=Colorbar::new(Rect::new(r.x+6.,r.y+r.height+12.,r.width-12.,2.5),sc,clabel,cticks);
 bar.horizontal=true; f.colorbars.push(bar);
 f.layout_colorbars()?;
 f.height_mm=f.height_mm.max(f.colorbars[0].rect.y+f.colorbars[0].rect.height+18.);
 save(&f,name)
}
fn main()->Result<()> {
 let f=File::open("data/spectra.h5")?;
 let x=scaled(read(&f,"thermal_ion/f_hz")?,1e-3);
 lineplot("thermal_ion",&x,(1..=3).map(|n|Ok((scaled(read(&f,&format!("thermal_ion/Te_{}",1000*n))?,1e3),format!("$T_e/T_i={n}$")))).collect::<Result<Vec<_>>>()?,[-12.,12.],[0.,0.06],"Frequency offset (kHz)",r"$S_{ee}(f)$ (kHz$^{-1}$)",false)?;
 let x=read(&f,"thermal_plasma/f_hz")?;let mid=x[x.len()/2];let xx:Vec<_>=x.iter().map(|v|(v-mid)/1e3).collect();
 let y=scaled(read(&f,"thermal_plasma/S_per_hz")?,1e3);let ymax=y.iter().cloned().fold(0.,f64::max)*1.08;
 lineplot("thermal_plasma",&xx,vec![(y,"".into())],[-5.,5.],[0.,ymax],&format!("Frequency relative to {:.4} MHz (kHz)",mid/1e6),r"$S_{ee}(f)$ (kHz$^{-1}$)",false)?;
 let x=read(&f,"photoelectron/E_eV")?;
 lineplot("energy_distribution",&x,vec![(read(&f,"photoelectron/p_pedestal")?,"Smooth tail".into()),(read(&f,"photoelectron/p_bump")?,"20 eV peak".into())],[1.,60.],[1e-5,2.],"Electron energy (eV)",r"Energy probability density (eV$^{-1}$)",true)?;
 let x=read(&f,"photoelectron/fpe_MHz")?;
 lineplot("photoelectron_enhancement",&x,vec![(read(&f,"photoelectron/pedestal_enhancement")?,"Smooth tail".into()),(read(&f,"photoelectron/bump_enhancement")?,"20 eV peak".into())],[2.5,10.],[1.,2000.],"Electron plasma frequency (MHz)","Integrated line power / thermal",true)?;
 let x=scaled(read(&f,"photoelectron_line/offset_hz")?,0.001);
 lineplot("photoelectron_line",&x,vec![(scaled(read(&f,"photoelectron_line/thermal")?,1000.),"Thermal".into()),(scaled(read(&f,"photoelectron_line/tail")?,1000.),"20 eV tail".into())],[-3.,3.],[1e-6,100.],"Frequency relative to line center (kHz)",r"$S_{ee}(f)$ (kHz$^{-1}$)",true)?;
 let x=scaled(read(&f,"magnetic_ion/f_hz")?,0.001);
 lineplot("magnetic_ion",&x,[0,60,89,90].into_iter().map(|a|Ok((scaled(read(&f,&format!("magnetic_ion/theta_{a}"))?,1000.),format!("$\\theta={a}^\\circ$")))).collect::<Result<Vec<_>>>()?,[-10.,10.],[1e-4,5.],"Frequency offset (kHz)",r"$S_{ee}(f)$ (kHz$^{-1}$)",true)?;
 heatmap("magnetized_tail",&read(&f,"magnetized_tail/theta_deg")?,&read(&f,"magnetized_tail/fpe_MHz")?,read(&f,"magnetized_tail/enhancement")?,[1.,1e4],"Electron plasma frequency (MHz)","Line power / thermal",vec![1.,10.,100.,1000.,1e4],vec![])?;
 // Density profile is plotted against altitude; no synthetic absolute event power.
 let h=read(&f,"hf_schematic/height_km")?;let fp=scaled(read(&f,"hf_schematic/fpe_hz")?,1e-6);let fl=scaled(read(&f,"hf_schematic/radar_mode_hz")?,1e-6);
 let mut ff=Figure::new(IntendedUse::ArticleWide);let idx=ff.grid(1,1)?[0];ff.panels[idx].rect.y+=16.;ff.height_mm+=16.;ff.panels[idx].axes=Some(Axes::new([2.5,7.5],[115.5,120.5],"Frequency (MHz)","Altitude (km)"));
 let mut ent=vec![];
 for (i,(v,label)) in [(fp,r"$f_{pe}(h)$"),(fl,"Radar-scale wave frequency")].into_iter().enumerate(){ff.panels[idx].push(Layer::Line(Line::new(v.into_iter().zip(h.iter()).map(|(a,&b)|[a,b]).collect(),COLORS[i])));ent.push(LegendEntry::line(label,COLORS[i]));}
 for (i,pump) in [4.51,5.33].iter().enumerate(){let mut l=Line::new(vec![[*pump,115.5],[*pump,120.5]],COLORS[i+2]);l.dash=Some("4 3".into());ff.panels[idx].push(Layer::Line(l));let mut entry=LegendEntry::line(format!("{pump:.2} MHz pump"),COLORS[i+2]);if let LegendSymbol::Line{dash,..}=&mut entry.symbol{*dash=Some("4 3".into());}ent.push(entry);}
 let mut lg=Legend::new(ent);lg.position=LegendPosition::AboveCenter;lg.columns=2;ff.panels[idx].legend=Some(lg);save(&ff,"hf_matching")?;
 let f=File::open("data/longley_profile.h5")?;let a=read(&f,"theta_deg")?;let h=read(&f,"height_km")?;let n=a.len();let fr=read(&f,"resonance_hz")?;let kk=read(&f,"k_m1")?;let b=read(&f,"B_T")?;
 let en=|order:f64|->Vec<f64>{fr.iter().enumerate().map(|(i,&v)|{let kp=kk[i]*a[i%n].to_radians().cos();9.1093837139e-31/2./1.602176634e-19*((2.*std::f64::consts::PI*v-order*1.602176634e-19*b[i/n]/9.1093837139e-31)/kp).powi(2)}).collect()};
 heatmap("longley_profile",&a,&h,read(&f,"area")?,[0.01,10.],"Altitude (km)","Integrated line power",vec![0.01,0.1,1.,10.],vec![(en(0.),"#41c8dc"),(en(1.),"#65d77c")])?;
 let dr=File::open("data/driven_response.h5")?;
 let x=scaled(read(&dr,"fpe_hz")?,1e-6);
 lineplot("driven_response",&x,[0,45,90].into_iter().map(|a|Ok((read(&dr,&format!("theta_{a}"))?,format!("$\\theta={a}^\\circ$")))).collect::<Result<Vec<_>>>()?,[5.,5.34],[0.01,10000.],"Electron plasma frequency (MHz)",r"$|\delta n_e/n_e|\,/\,|E_{\rm ext}|$ ((V/m)$^{-1}$)",true)?;
 Ok(())
}
