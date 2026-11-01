// Source: trace_incoming_hf.py; cold EM Hamiltonian rays, measured smooth n_e.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;
const BLUE:&str="#245774";const RED:&str="#8b3940";const GRAY:&str="#8795a9";const GREEN:&str="#4c7566";
fn rd(h:&File,k:&str)->Result<Vec<f64>>{Ok(h.dataset(k)?.read_raw::<f64>()?)}
fn arrow(f:&mut Figure,p:usize,start:[f64;2],v:[f64;2],color:&str){
 let end=[start[0]+v[0],start[1]+v[1]];let l=v[0].hypot(v[1]);let a=[v[0]/l,v[1]/l];let s=l*0.16;
 let line=Line::new(vec![start,end,[end[0]-s*a[0]+s*0.5*a[1],end[1]-s*a[1]-s*0.5*a[0]],end,[end[0]-s*a[0]-s*0.5*a[1],end[1]-s*a[1]+s*0.5*a[0]]],color);
 f.panels[p].push(Layer::Line(line));
}
fn save(f:&Figure,n:&str)->Result<()>{for ext in ["png","svg","pdf"]{f.save(format!("figures/{n}.{ext}"))?;}Ok(())}
fn main()->Result<()>{
 let h=File::open("data/incoming_hf_rays.h5")?;
 let x=rd(&h,"-20.0/1/horizontal_m")?;let z=rd(&h,"-20.0/1/height_m")?;
 let e=rd(&h,"-20.0/1/electric_inplane_axis")?;let k=rd(&h,"-20.0/1/phase_unit_vector")?;
 let j=z.iter().enumerate().max_by(|(_,a),(_,b)|a.total_cmp(b)).unwrap().0;
 let xt=-x[j];let zb=117937.32516295004;let zr=117941.5654266378;
 let mut f=Figure::new(IntendedUse::ArticleWide);let p=f.grid(1,1)?[0];
 f.panels[p].rect=Rect::new(24.,24.,126.,126.);f.width_mm=178.;f.height_mm=174.;
 f.panels[p].axes=Some(Axes::new([0.,20.],[100.,120.],"Horizontal distance northward (km)","Height along layer normal (km)"));
 for (mode,color) in [("0",GRAY),("1",BLUE)]{
  let xx=rd(&h,&format!("-20.0/{mode}/horizontal_m"))?;let zz=rd(&h,&format!("-20.0/{mode}/height_m"))?;
  f.panels[p].push(Layer::Line(Line::new(xx.iter().zip(zz.iter()).map(|(&x,&z)|[-x/1000.,z/1000.]).collect(),color)));
 }
 for target in [105000.,110000.,115000.]{
  let i=(0..j).min_by(|&a,&b|(z[a]-target).abs().total_cmp(&(z[b]-target).abs())).unwrap();
  let a=[-x[i]/1000.,z[i]/1000.];
  arrow(&mut f,p,a,[-k[2*i]*0.9,k[2*i+1]*0.9],GREEN);
  arrow(&mut f,p,[a[0]+e[2*i]*0.45,a[1]-e[2*i+1]*0.45],[-e[2*i]*0.9,e[2*i+1]*0.9],RED);
 }
 let mut leg=Legend::new(vec![LegendEntry::line("O-like energy ray",BLUE),LegendEntry::line("X-like energy ray",GRAY),LegendEntry::line("HF phase direction",GREEN),LegendEntry::line("HF electric field",RED)]);leg.position=LegendPosition::AboveCenter;leg.columns=2;f.panels[p].legend=Some(leg);
 save(&f,"incoming_hf_ray")?;
 let mut f=Figure::new(IntendedUse::ArticleWide);let p=f.grid(1,1)?[0];
 f.panels[p].rect=Rect::new(24.,29.,140.,70.);f.width_mm=185.;f.height_mm=124.;
 f.panels[p].axes=Some(Axes::new([-100.,100.],[-60.,40.],"Northward displacement from HF turn (m)","Height from plasma-wave Bragg match (m)"));
 f.panels[p].push(Layer::Line(Line::new(x.iter().zip(z.iter()).filter(|(x,_)|(-**x-xt).abs()<110.).map(|(&x,&z)|[-x-xt,z-zb]).collect(),BLUE)));
 let a=[0.,z[j]-zb];arrow(&mut f,p,a,[-k[2*j]*28.,k[2*j+1]*28.],GREEN);
 arrow(&mut f,p,[a[0]+e[2*j]*14.,a[1]-e[2*j+1]*14.],[-e[2*j]*28.,e[2*j+1]*28.],RED);
 for y in [0.,zr-zb]{let mut l=Line::new(vec![[-100.,y],[100.,y]],GRAY);l.width_pt=0.5;l.dash=Some(if y==0.{"4,3"}else{"1,3"}.into());f.panels[p].push(Layer::Line(l));}
 let mut leg=Legend::new(vec![LegendEntry::line("HF energy ray",BLUE),LegendEntry::line("HF phase direction",GREEN),LegendEntry::line("HF electric field",RED)]);leg.position=LegendPosition::AboveCenter;leg.columns=3;f.panels[p].legend=Some(leg);
 save(&f,"incoming_hf_turn")?;
 Ok(())
}
