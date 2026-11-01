//! Coupling driver: the existing FDTD and electron-fluid Rust crates exchange
//! E and current at each timestep. This executable contains no field/fluid RHS.
//! E and pressure variable q at integer times; H and normalized current at
//! half times. q=sqrt(m cs^2/eps0) dn/sqrt(n0); all ten variables share the
//! same quadratic energy normalization. No independent Poisson field/source.
use arecibo_coupled::advance_coupled;
use arecibo_fluid::coupling::{FieldCoupledFluid, YeeFluidCoupler};
use arecibo_fluid::maxwell_coupling::CouplingGeometry;
use meteor_fdtd::stratified::phase;
use serde::{Deserialize, Serialize};
use std::{fs, io::Write};
type Z = [f64; 2];
#[derive(Deserialize)]
struct Input {
    n: usize,
    dz_cells: Vec<f64>,
    dt: f64,
    omega: f64,
    kx: f64,
    gyro: [f64; 3],
    wp: Vec<Vec<f64>>,
    nu: Vec<Vec<f64>>,
    sponge: Vec<Vec<f64>>,
    sqrt_ne: Vec<Vec<f64>>,
    beta: f64,
    initial: Vec<Vec<Z>>,
    source: Vec<Vec<Z>>,
    steps: usize,
    record_stride: usize,
    output: String,
}
#[derive(Serialize)]
struct Report {
    steps: usize,
    dt_s: f64,
    frames: usize,
    relative_max_e_error: f64,
    relative_max_q_error: f64,
    finite: bool,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg: Input = serde_json::from_slice(&fs::read(std::env::args().nth(1).ok_or("config")?)?)?;
    let n = cfg.n;
    let dt = cfg.dt;
    let a = cfg.omega * dt;
    let k = cfg.kx;
    assert!(
        n > 4 && dt > 0. && dt / cfg.dz_cells.iter().copied().fold(f64::INFINITY, f64::min) < 0.5
    );
    let mut e = cfg.initial[0..3].to_vec();
    let mut h = cfg.initial[3..6].to_vec();
    let mut j = cfg.initial[6..9].to_vec();
    let q = cfg.initial[9].clone();
    for f in h.iter_mut().chain(j.iter_mut()) {
        for v in f {
            *v = phase(*v, a / 2.);
        }
    }
    let geometry = CouplingGeometry {
        widths: &cfg.dz_cells,
        kx: k,
        gyro: cfg.gyro,
        plasma_frequency: &cfg.wp,
        collision_rate: &cfg.nu,
        sqrt_density: &cfg.sqrt_ne,
        thermal_speed: cfg.beta,
        density_buffer: &cfg.sponge[0],
    };
    let mut fluid = YeeFluidCoupler::new(geometry, dt, j, q)?;
    let mut file = fs::File::create(&cfg.output)?;
    let mut frames = 0;
    for step in 0..cfg.steps {
        // One field/current exchange at the common Maxwell timestep. The FDTD
        // electron-current update is REPLACED by the warm-fluid current, not added.
        advance_coupled(
            &mut e,
            &mut h,
            &mut fluid,
            &cfg.wp,
            &cfg.dz_cells,
            k,
            dt,
            &cfg.sponge,
            &cfg.source,
            -a * (step as f64 + 0.5),
        )?;
        if step % cfg.record_stride == 0 || step + 1 == cfg.steps {
            file.write_all(&((step + 1) as f64 * dt / 299792458.).to_le_bytes())?;
            for f in e
                .iter()
                .map(Vec::as_slice)
                .chain(std::iter::once(fluid.density()))
            {
                for v in f {
                    for val in v {
                        file.write_all(&val.to_le_bytes())?;
                    }
                }
            }
            frames += 1;
        }
    }
    let mut errors = [0f64; 2];
    let mut scales = [0f64; 2];
    let mut finite = true;
    for (v, f) in e
        .iter()
        .map(Vec::as_slice)
        .chain(std::iter::once(fluid.density()))
        .enumerate()
    {
        let kind = usize::from(v == 3);
        let idx = if v == 3 { 9 } else { v };
        for i in 0..n {
            let reference = phase(cfg.initial[idx][i], -a * cfg.steps as f64);
            for r in 0..2 {
                finite &= f[i][r].is_finite();
                errors[kind] = errors[kind].max((f[i][r] - reference[r]).abs());
                scales[kind] = scales[kind].max(reference[r].abs());
            }
        }
    }
    let report = Report {
        steps: cfg.steps,
        dt_s: dt / 299792458.,
        frames,
        relative_max_e_error: errors[0] / scales[0].max(1e-100),
        relative_max_q_error: errors[1] / scales[1].max(1e-100),
        finite,
    };
    println!("{}", serde_json::to_string(&report)?);
    if !finite || report.relative_max_e_error > 1e-5 || report.relative_max_q_error > 1e-5 {
        return Err("Coupled FDTD consistency failed".into());
    }
    Ok(())
}
