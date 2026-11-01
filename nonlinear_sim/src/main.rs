use nonlinear_conversion::{Config, Solver};
use std::{fs, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg: Config = serde_json::from_slice(&fs::read(
        std::env::args().nth(1).ok_or("config path required")?,
    )?)?;
    let path = cfg.output.clone();
    let steps = cfg.steps;
    let stride = cfg.stride;
    let nz = cfg.z.len();
    let nx = cfg.nx;
    let dt = cfg.dt;
    let mut s = Solver::new(cfg);
    let clock = Instant::now();
    let mut frames: Vec<f64> = Vec::new();
    let mut times = Vec::new();
    let mut audit = Vec::new();
    for step in 0..=steps {
        if step % stride == 0 || step == steps {
            let (g, b, n) = s.audit_at(step as f64 * dt);
            audit.extend([g, b, n]);
            times.push(step as f64 * dt / s.cfg.omega_si);
            // All x positions retained: DC/harmonics and pressure are inspectable.
            for v in &s.state {
                frames.extend(v);
            }
            eprintln!(
                "step {step}/{steps} t={:.3} cycles max dn/n={n:.4e}, elapsed={:.1}s",
                step as f64 * dt / (2. * std::f64::consts::PI),
                clock.elapsed().as_secs_f64()
            );
        }
        if step < steps {
            s.step(step as f64 * dt);
        }
    }
    let f = hdf5::File::create(path)?;
    f.new_dataset::<f64>()
        .shape([times.len(), nx, nz, 16])
        .create("state")?
        .write_raw(&frames)?;
    f.new_dataset::<f64>()
        .shape(times.len())
        .create("time_s")?
        .write_raw(&times)?;
    f.new_dataset::<f64>()
        .shape([times.len(), 3])
        .create("audit_gauss_divB_max_dn")?
        .write_raw(&audit)?;
    for (name, v) in [
        ("z_normalized", &s.cfg.z),
        ("background_density_normalized", &s.cfg.n0),
        ("equilibrium_Ez_normalized", &s.cfg.eq_ez),
        ("collision_normalized", &s.cfg.nu),
        ("buffer_normalized", &s.cfg.sponge),
    ] {
        f.new_dataset::<f64>()
            .shape(v.len())
            .create(name)?
            .write_raw(v)?;
    }
    for (name, v) in [
        ("omega_si", s.cfg.omega_si),
        ("length_si", s.cfg.length_si),
        ("density_si", s.cfg.density_si),
        ("electric_si", s.cfg.electric_si),
        ("altitude_origin_m", s.cfg.altitude_origin_m),
        ("beta", s.cfg.beta),
        ("dt_normalized", s.cfg.dt),
        ("dx_normalized", s.cfg.dx),
    ] {
        f.new_attr::<f64>().create(name)?.write_scalar(&v)?;
    }
    f.new_attr::<u8>()
        .create("linearized")?
        .write_scalar(&(s.cfg.linear as u8))?;
    Ok(())
}
