// All input coordinates are decimal UTC hours on 2018-05-04.
// AST is UTC-4 year-round; both axes use the same physical tick positions.
use anyhow::Result;
use rustplotlib::{Axes, SecondaryAxis};
pub fn clock_label(hours: f64) -> String {
    let minutes=(hours*60.).round() as i64;
    format!("{:02}:{:02}",(minutes.div_euclid(60)).rem_euclid(24),minutes.rem_euclid(60))
}
pub fn time_axes(a: &mut Axes, primary: bool, secondary: bool) -> Result<()> {
    let step=if a.xlim[1]-a.xlim[0]<1. {1./6.}else{1.};
    let ticks:Vec<f64>=((a.xlim[0]/step).ceil() as i64..=(a.xlim[1]/step).floor() as i64).map(|i|i as f64*step).collect();
    if primary {
        a.xlabel="Time (UTC, 4 May 2018)".into();
        a.xticks=ticks.clone();
        a.xticklabels=Some(ticks.iter().map(|v|clock_label(*v)).collect());
    }else{a.xlabel.clear();a.xticks.clear();a.xticklabels=None;}
    if secondary {
        let mut s=SecondaryAxis::new("Time (AST, UTC$-4$)",ticks.iter().map(|v|v-4.).collect(),|v|v+4.)?;
        s.tick_labels=Some(ticks.iter().map(|v|clock_label(v-4.)).collect());
        a.secondary_x=Some(s);
    }
    Ok(())
}
