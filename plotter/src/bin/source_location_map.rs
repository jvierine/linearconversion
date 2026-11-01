// Geographic arrays and coupling weights: source_location_map.py and
// prepare_source_map_plot.py. Scientific provenance remains in the HDF5 file.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
fn read(h:&File,name:&str)->Result<Vec<f64>>{Ok(h.dataset(name)?.read_raw::<f64>()?)}
fn points(h:&File,name:&str)->Result<Vec<[f64;2]>>{Ok(read(h,name)?.chunks(2).map(|a|[a[0],a[1]]).collect())}
fn line(f:&mut Figure,p:usize,pts:Vec<[f64;2]>,color:&str,width:f64){let mut l=Line::new(pts,color);l.width_pt=width;f.panels[p].push(Layer::Line(l));}
fn loc(x:f64,y:f64)->[f64;2]{[20.+151.*(x+67.4)/1.9,5.+100.5*(19.-y)/1.2]}
fn label(f:&mut Figure,text:&str,xy:[f64;2],offset:[f64;2]){let a=loc(xy[0],xy[1]);let mut l=Label::new(text,a[0]+offset[0],a[1]+offset[1]);l.background=Some("#ffffff".into());l.padding_mm=0.3;f.labels.push(l);}
fn main()->Result<()>{
 let h=File::open("reference/coupled/source_location_map.h5")?;
 let memo=std::env::args().any(|a|a=="--memo");
 let width=if memo{179.324}else{177.};let font=if memo{11.}else{10.};
 let mut f=ArticleLayout::new(width,font)?.figure(1.,145.)?;let p=f.grid(1,1)?[0];f.width_mm=width;f.height_mm=145.;
 f.panels[p].rect=Rect::new(20.,5.,151.,100.5);
 let mut ax=Axes::new([-67.4,-65.5],[17.8,19.],"","");
 ax.xticks=vec![-67.25,-66.75,-66.25,-65.75];ax.xticklabels=Some(vec![r"$67.25^\circ$ W".into(),r"$66.75^\circ$ W".into(),r"$66.25^\circ$ W".into(),r"$65.75^\circ$ W".into()]);
 ax.yticks=vec![18.,18.25,18.5,18.75,19.];ax.yticklabels=Some(ax.yticks.iter().map(|y|format!(r"${y}^\circ$ N")).collect());f.panels[p].axes=Some(ax);
 let land=f.scale(Scale::new([0.,1.],Normalization::Linear,ColorMap::new(vec![[0.92,0.92,0.88],[0.92,0.92,0.88]])?)?);
 let mut polys=vec![];for name in h.group("map/land")?.member_names()?{polys.push(points(&h,&format!("map/land/{name}"))?);}
 let n=polys.len();f.panels[p].push(Layer::PolygonMesh(PolygonMesh::new(polys,vec![0.5;n],land)?));
 let sc=f.scale(Scale::new([-30.,0.],Normalization::Linear,ColorMap::ember())?);
 f.panels[p].push(Layer::Pcolormesh(Field::new(read(&h,"map/longitude_edges_deg")?,read(&h,"map/latitude_edges_deg")?,read(&h,"map/relative_response_db")?,sc)?));
 for name in h.group("map/coast")?.member_names()?{let pts=points(&h,&format!("map/coast/{name}"))?;line(&mut f,p,pts.clone(),"#ffffff",1.1);line(&mut f,p,pts,"#52585e",0.45);}
 let towns=points(&h,"map/towns/lonlat")?;
 let names=h.dataset("map/towns/names")?.read_raw::<hdf5::types::VarLenUnicode>()?;
 for (xy,name) in towns.iter().zip(names){
  line(&mut f,p,vec![[xy[0]-0.005,xy[1]],[xy[0]+0.005,xy[1]]],"#171717",1.5);
  let name=name.as_str();let txt=if name=="Mayagüez"{r#"Mayag\"uez"#}else{name};
  let offset=match name{"Ponce"=>[0.,4.],"Arecibo"=>[0.,-4.],"Adjuntas"=>[-12.,0.],_=>[0.,-4.]};label(&mut f,txt,*xy,offset);
 }
 let radar=[h.attr("radar_longitude_deg")?.read_scalar::<f64>()?,h.attr("radar_latitude_deg")?.read_scalar::<f64>()?];
 line(&mut f,p,vec![[radar[0]-0.015,radar[1]],[radar[0]+0.015,radar[1]]],"#0072B2",1.8);
 line(&mut f,p,vec![[radar[0],radar[1]-0.014],[radar[0],radar[1]+0.014]],"#0072B2",1.8);
 label(&mut f,"Radar",radar,[-8.,3.5]);
 let best=read(&h,"best")?;let xy=[best[3],best[2]];
 let circle=(0..=64).map(|i|{let t=i as f64*std::f64::consts::TAU/64.;[xy[0]+0.014*t.cos(),xy[1]+0.013*t.sin()]}).collect();line(&mut f,p,circle,"#171717",1.2);
 label(&mut f,"Strongest coupling",xy,[22.,0.]);
 line(&mut f,p,vec![radar,xy],"#0072B2",0.8);
 label(&mut f,"Atlantic Ocean",[-65.9,18.8],[0.,0.]);label(&mut f,"Caribbean Sea",[-65.9,17.93],[0.,0.]);
 let mut cb=Colorbar::new(Rect::new(20.,123.,151.,3.),sc,"Relative modeled response (dB)",vec![-30.,-20.,-10.,0.]);cb.horizontal=true;f.colorbars.push(cb);f.auto_colorbar_layout=false;
 let suffix=if memo{"_memo"}else{""};
 for ext in ["png","pdf"]{f.save(format!("figures/source_location_map{suffix}.{ext}"))?;}
 Ok(())
}
