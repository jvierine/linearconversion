use nonlinear_conversion::{Config, scalar::ScalarSolver};
use std::fs;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg: Config = serde_json::from_slice(&fs::read(std::env::args().nth(1).ok_or("config")?)?)?;
    let mut s = ScalarSolver::new(cfg);
    let mut records = Vec::new();
    let mut times = Vec::new();
    for step in 0..=s.cfg.steps {
        if step % s.cfg.stride == 0 || step == s.cfg.steps {
            assert!(
                s.state
                    .iter()
                    .all(|q| q[0] > 0. && q.iter().all(|v| v.is_finite()))
            );
            times.push(step as f64 * s.cfg.dt / s.cfg.omega_si);
            for q in &s.state {
                records.extend_from_slice(q);
            }
            eprintln!("total-density step {step}/{}", s.cfg.steps);
        }
        if step < s.cfg.steps {
            s.step(step as f64 * s.cfg.dt);
        }
    }
    let h = hdf5::File::create(&s.cfg.output)?;
    let nz = s.cfg.z.len();
    h.new_dataset::<f64>()
        .shape([times.len(), s.cfg.nx, nz, 10])
        .create("state")?
        .write_raw(&records)?;
    h.new_dataset::<f64>()
        .shape(times.len())
        .create("time_s")?
        .write_raw(&times)?;
    let n: Vec<_> = records
        .chunks_exact(10)
        .map(|q| q[0] * s.cfg.density_si)
        .collect();
    h.new_dataset::<f64>()
        .shape([times.len(), s.cfg.nx, nz])
        .create("electron_density_m3")?
        .write_raw(&n)?;
    for (name, v) in [
        ("density_si", s.cfg.density_si),
        ("electric_si", s.cfg.electric_si),
        ("omega_si", s.cfg.omega_si),
        ("beta", s.cfg.beta),
        ("dx_normalized", s.cfg.dx),
        ("dt_normalized", s.cfg.dt),
    ] {
        h.new_attr::<f64>().create(name)?.write_scalar(&v)?;
    }
    Ok(())
}
