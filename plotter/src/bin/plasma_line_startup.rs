// Arrays and provenance: data/plasma_line_startup.h5; coupled_sim/scripts/plasma_line_startup.py.
use anyhow::Result;
use hdf5::File;
use rustplotlib::*;

fn read(h:&File,name:&str)->Result<Vec<f64>> { Ok(h.dataset(name)?.read_raw()?) }

fn main()->Result<()> {
    let h=File::open("data/plasma_line_startup.h5")?;
    let t:Vec<_>=read(&h,"ramp_10us/time_s")?.into_iter().map(|x|x*1000.).collect();
    let raw=read(&h,"ramp_10us/power_normalized")?;
    let recorded=read(&h,"ramp_10us/power_44us_normalized")?;
    let mut f=ArticleLayout::new(179.324,11.)?.figure(1.,85.)?;
    let ps=f.grid(1,2)?;
    f.height_mm=85.;
    let gray=f.scale(Scale::new([0.,1.],Normalization::Linear,
        ColorMap::new(vec![[0.94,0.94,0.94],[0.94,0.94,0.94]])?)?);
    for (i,&p) in ps.iter().enumerate() {
        f.panels[p].rect=Rect::new(if i==0 {22.} else {108.},7.,66.,57.);
        let limits=if i==0 {[-0.03,1.6]} else {[1.10,1.33]};
        let mut ax=Axes::new(limits,[0.,1.4],"",if i==0 {"Plasma-line power / steady power"}else{""});
        ax.xticks=if i==0 {vec![0.,0.4,0.8,1.2,1.6]}else{vec![1.1,1.2,1.3]};
        ax.yticks=vec![0.,0.5,1.0];
        f.panels[p].axes=Some(ax);
        f.panels[p].push(Layer::Pcolormesh(Field::new(limits.to_vec(),vec![0.95,1.05],vec![0.5],gray)?));
        for (values,index) in [(&raw,0),(&recorded,2)] {
            let points=t.iter().zip(values).filter(|(x,_)|**x>=limits[0]&&**x<=limits[1])
                .map(|(&x,&y)|[x,y]).collect();
            let line=Line::series(points,index);
            f.panels[p].push(Layer::Line(line));
        }
        f.labels.push(Label::new(if i==0 {"(a)"}else{"(b)"},if i==0 {22.}else{108.},2.));
    }
    let key1=Line::series(vec![[0.,0.],[1.,1.]],0).legend("Instantaneous");
    let key2=Line::series(vec![[0.,0.],[1.,1.]],2).legend(r"44 $\mu$s record");
    let mut legend=Legend::new(vec![key1,key2]);legend.position=LegendPosition::UpperLeft;
    f.panels[ps[0]].legend=Some(legend);
    f.labels.push(Label::new("Time after HF switch-on midpoint (ms)",98.,77.));
    f.auto_colorbar_layout=false;
    f.save_for_article("figures/plasma_line_startup.pdf","memos/memo_62/plasma_line_startup.pdf")?;
    f.save("figures/plasma_line_startup.png")?;
    Ok(())
}
