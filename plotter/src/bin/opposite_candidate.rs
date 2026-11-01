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
 let sc=f.scale(Scale::new([8.,12.],Normalization::Linear,ColorMap::listed(density_palette)?)?);
 f.panels[p].push(Layer::Pcolormesh(Field::new(t,edges(&z),rd(&dh,"map/ne_m3")?.iter().map(|x|x.max(1e8).log10()).collect(),sc)?));
 let marker_colors=["#0072B2","#D55E00"];
 let shapes=[MarkerShape::Circle,MarkerShape::Diamond];
 // Open symbols and a shared halo pass preserve overlapping frequency classes.
 let groups=[MarkerGroup{label:"4.51 MHz",points:&points[0],style:MarkerStyle{shape:shapes[0],size_pt:8.,color:rgb(marker_colors[0])}},MarkerGroup{label:"5.34 MHz",points:&points[1],style:MarkerStyle{shape:shapes[1],size_pt:8.,color:rgb(marker_colors[1])}}];
 let mut legend=f.panels[p].marker_groups(&groups)?;legend.position=LegendPosition::LowerLeft;legend.avoid_data=false;f.panels[p].legend=Some(legend);
 f.colorbars.push(Colorbar::new(Rect::new(173.,24.,3.,map_height),sc,r"$\log_{10}(n_e\,/\,\mathrm{m}^{-3})$",(16..=24).map(|i|i as f64/2.).collect()));f.layout_colorbars()?;
 save(&f,"density_enhancement_times")?;
 // Antialiased screen preview, not filtering or resampling of scientific data.
 let raster=image::open("figures/density_enhancement_times.png")?;
 let height=(800. * raster.height() as f64/raster.width() as f64).round() as u32;
 raster.resize_exact(800,height,image::imageops::FilterType::Lanczos3).save("figures/density_enhancement_times_preview.png")?;
 Ok(())
}
