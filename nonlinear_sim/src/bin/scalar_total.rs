use nonlinear_conversion::{Config, scalar::ScalarSolver};
use std::fs;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input: serde_json::Value =
        serde_json::from_slice(&fs::read(std::env::args().nth(1).ok_or("config")?)?)?;
    let ramp_time = input
        .get("ramp_time")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.);
    let cfg: Config = serde_json::from_value(input)?;
    let mut s = ScalarSolver::new(cfg);
    assert!(ramp_time >= 0. && ramp_time.is_finite());
    if ramp_time > 0. {
        assert!(
            s.cfg.initial.iter().flatten().all(|v| *v == [0., 0.]),
            "Source ramp requires equilibrium initialization"
        );
    }
    s.ramp_time = ramp_time;
    let mut records = Vec::new();
    let mut times = Vec::new();
    let mut gauss = Vec::new();
    let mut envelopes = Vec::new();
    for step in 0..=s.cfg.steps {
        if step % s.cfg.stride == 0 || step == s.cfg.steps {
            assert!(
                s.state
                    .iter()
                    .all(|q| q[0] > 0. && q.iter().all(|v| v.is_finite()))
            );
            times.push(step as f64 * s.cfg.dt / s.cfg.omega_si);
            gauss.push(s.gauss().iter().map(|r| r.abs()).fold(0., f64::max));
            envelopes.push(s.source_envelope(step as f64 * s.cfg.dt));
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
    h.new_dataset::<f64>()
        .shape(gauss.len())
        .create("gauss_max_normalized")?
        .write_raw(&gauss)?;
    h.new_dataset::<f64>()
        .shape(envelopes.len())
        .create("source_envelope")?
        .write_raw(&envelopes)?;
    h.new_dataset::<f64>()
        .shape([s.cfg.nx, nz])
        .create("final_source_charge_normalized")?
        .write_raw(&s.source_charge)?;
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
        ("ramp_time_s", s.ramp_time / s.cfg.omega_si),
    ] {
        h.new_attr::<f64>().create(name)?.write_scalar(&v)?;
    }
    Ok(())
}
