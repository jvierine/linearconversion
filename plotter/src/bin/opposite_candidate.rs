// Diagnostic input: prepare_opposite_candidate.py. All spectrum powers native.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
#[path="../time_axes.rs"] mod time_axes;
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn edges(v:&[f64])->Vec<f64>{let mut e=vec![v[0]-(v[1]-v[0])/2.];e.extend(v.windows(2).map(|w|(w[0]+w[1])/2.));e.push(v[v.len()-1]+(v[v.len()-1]-v[v.len()-2])/2.);e}
fn rgb(c:&str)->[f64;3]{let s=c.trim_start_matches('#');[0,2,4].map(|i|u8::from_str_radix(&s[i..i+2],16).unwrap() as f64/255.)}
fn marks(f:&mut Figure,p:usize,pts:&[[f64;2]],c:&str,size:f64,shape:MarkerShape)->Result<()>{let mut pan=f.panels[p].clone();pan.markers(f,pts,&MarkerStyle{shape,size_pt:size,color:rgb(c)})?;f.panels[p]=pan;Ok(())}
fn line(f:&mut Figure,p:usize,x:&[f64],y:&[f64],c:&str)->Result<()>{let mut l=Line::new(x.iter().zip(y).map(|(&x,&y)|[x,y]).collect(),c);l.width_pt=1.;f.panels[p].push(Layer::Line(l));Ok(())}
fn save(f:&Figure,name:&str)->Result<()>{for ext in ["png","pdf"]{f.save(format!("figures/{name}.{ext}"))?;}Ok(())}
// Symbols have a fixed six-point diameter, independent of axis aspect.
fn plain_symbols(panel:&mut Panel,points:&[[f64;2]],cross:bool){
 let a=panel.axes.as_ref().unwrap();let r=3.*25.4/72.27;
 let dx=r*(a.xlim[1]-a.xlim[0])/panel.rect.width;
 let dy=r*(a.ylim[1]-a.ylim[0])/panel.rect.height;
 let mut path=Vec::new();
 for &[x,y] in points {
  if cross {path.extend([[x-dx,y-dy],[x+dx,y+dy],[f64::NAN;2],[x-dx,y+dy],[x+dx,y-dy],[f64::NAN;2]]);}
  else {for i in 0..=48 {let t=i as f64*std::f64::consts::TAU/48.;path.push([x+dx*t.cos(),y+dy*t.sin()]);}path.push([f64::NAN;2]);}
 }
 let mut line=Line::new(path,"#000000");line.width_pt=0.8;panel.push(Layer::Line(line));
}
fn main()->Result<()>{
 let h=File::open("data/opposite_candidate.h5")?;let dh=File::open("data/density_reference_debye.h5")?;
 let colors=["#245774","#527f6d","#8B3940","#765a90","#b86f25"];
 let palette:Vec<[f64;3]>=rd(&dh,"palette/rgb")?.chunks_exact(3).map(|v|[v[0],v[1],v[2]]).collect();
 let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=235.;let ps=f.grid(1,2)?;f.height_mm=122.;
 let p=ps[0];f.panels[p].rect=Rect::new(23.,10.,65.,85.);let ax=Axes::new([-5.45,-5.23],[115.,121.],"Signed frequency offset (MHz)","Altitude above radar (km)");f.panels[p].axes=Some(ax);
 let sc=f.scale(Scale::new([0.,5.],Normalization::Linear,ColorMap::listed(palette.clone())?)?);
 f.panels[p].push(Layer::Pcolormesh(Field::new(edges(&rd(&h,"map/frequency_mhz")?),edges(&rd(&h,"map/height_km")?),rd(&h,"map/power_over_noise_db")?,sc)?));
 f.colorbars.push(Colorbar::new(Rect::new(91.,10.,3.,85.),sc,"Power / noise (dB)",vec![0.,1.,2.,3.,4.,5.]));f.layout_colorbars()?;
 let p=ps[1];f.panels[p].rect=Rect::new(132.,10.,93.,85.);f.panels[p].axes=Some(Axes::new([5.25,5.43],[-0.2,2.9],"Frequency magnitude (MHz)","Excess / noise power"));
 let x=rd(&h,"core/frequency_magnitude_mhz")?;
 for (key,c) in [("earlier",colors[3]),("later",colors[1]),("candidate",colors[2])]{let y=rd(&h,&format!("core/{key}"))?;line(&mut f,p,&x,&y,c)?;}
 let xp=rd(&h,"core/positive_frequency_mhz")?;line(&mut f,p,&xp,&rd(&h,"core/positive")?,colors[0])?;
 marks(&mut f,p,&x.iter().zip(rd(&h,"core/candidate")?).map(|(&x,y)|[x,y]).collect::<Vec<_>>(),colors[2],2.,MarkerShape::Circle)?;
 f.panels[p].legend=Some(Legend::new(vec![LegendEntry::line("Opposite, 17:14:23 UTC",colors[2]),LegendEntry::line("Main, same time",colors[0]),LegendEntry::line("Opposite, 10 s earlier",colors[3]),LegendEntry::line("Opposite, 10 s later",colors[1])]));
 save(&f,"opposite_candidate")?;
 // Main-branch return locations only; candidates remain in the separate audit.
 let t=rd(&dh,"map/time_edges_utc_hour")?;let z=rd(&dh,"map/height_km")?;let rows=rd(&h,"enhancements")?;let mut points:Vec<Vec<[f64;2]>>=vec![vec![];2];
 for r in rows.chunks(14){let j=r[0] as usize;if j<2{points[j].push([(r[2]-1525392000.)/3600.,r[4]]);}}
 // Match the reference's 496:420 axis aspect, rather than stretching time.
 let map_height=147.*420./496.;
 let mut f=Figure::new(IntendedUse::ArticleWide);f.width_mm=200.;let p=f.grid(1,1)?[0];f.height_mm=171.;
 f.panels[p].rect=Rect::new(22.,24.,147.,map_height);let mut ax=Axes::new([t[0],*t.last().unwrap()],[50.,140.],"","Altitude above radar (km)");time_axes::time_axes(&mut ax,true,true)?;ax.yticks=(5..=14).map(|i|i as f64*10.).collect();f.panels[p].axes=Some(ax);
 let density_palette=rd(&dh,"palette/density_rgb")?.chunks_exact(3).map(|v|[v[0],v[1],v[2]]).collect();
 // Peak measured density is 4.94e11 m^-3; 11.7 dex keeps it at the palette maximum.
 let sc=f.scale(Scale::new([8.,11.7],Normalization::Linear,ColorMap::listed(density_palette)?)?);
 f.panels[p].push(Layer::Pcolormesh(Field::new(t,edges(&z),rd(&dh,"map/ne_m3")?.iter().map(|x|x.max(1e8).log10()).collect(),sc)?));
 // Plain black symbols, without halos; coordinates remain measured values.
 for (j,pts) in points.iter().enumerate(){plain_symbols(&mut f.panels[p],pts,j==0);}
 // Compact in-axis legend using the identical symbol paths as the data.
 let left=f.panels[p].axes.as_ref().unwrap().xlim[0]+0.08;
 let white=f.scale(Scale::new([0.,1.],Normalization::Linear,ColorMap::new(vec![[1.;3],[1.;3]])?)?);
 f.panels[p].push(Layer::PolygonMesh(PolygonMesh::new(vec![vec![[left,51.5],[left+0.9,51.5],[left+0.9,61.5],[left,61.5]]],vec![0.5],white)?));
 for (j,z) in [58.8,54.4].iter().enumerate(){
  plain_symbols(&mut f.panels[p],&[[left+0.12,*z]],j==0);
  f.panels[p].label_at(if j==0{"4.51 MHz"}else{"5.34 MHz"},[left+0.55,*z]);
 }
 f.colorbars.push(Colorbar::new(Rect::new(173.,24.,3.,map_height),sc,r"$\log_{10}(n_e\,/\,\mathrm{m}^{-3})$",vec![8.,8.5,9.,9.5,10.,10.5,11.,11.5,11.7]));f.layout_colorbars()?;
 save(&f,"density_enhancement_times")?;
 // Antialiased screen preview, not filtering or resampling of scientific data.
 let raster=image::open("figures/density_enhancement_times.png")?;
 let height=(800. * raster.height() as f64/raster.width() as f64).round() as u32;
 raster.resize_exact(800,height,image::imageops::FilterType::Lanczos3).save("figures/density_enhancement_times_preview.png")?;
 Ok(())
}
