use nonlinear_conversion::{Config, scalar::ScalarSolver};
use std::f64::consts::TAU;
fn cfg() -> Config {
    let nx = 32;
    let nz = 7;
    Config {
        nx,
        z: (0..nz).map(|i| i as f64).collect(),
        n0: vec![1.; nz],
        eq_ez: vec![0.; nz],
        nu: vec![0.; nz],
        sponge: vec![0.; nz],
        beta: 0.1,
        gyro: [0.; 3],
        dx: TAU / nx as f64,
        dt: 0.005,
        steps: 400,
        stride: 10,
        initial: vec![vec![[0.; 2]; nz]; 16],
        source: vec![vec![[0.; 2]; nz]; 16],
        linear: false,
        output: String::new(),
        omega_si: 1.,
        length_si: 1.,
        density_si: 1.,
        electric_si: 1.,
        altitude_origin_m: 0.,
        initial_pressure_lift: false,
    }
}
fn wave(a: f64, linear: bool) -> (ScalarSolver, f64) {
    let mut c = cfg();
    c.linear = linear;
    let kd = (8. * c.dx.sin() - (2. * c.dx).sin()) / (6. * c.dx);
    let w = (1. + 3. * c.beta * c.beta * kd * kd).sqrt();
    for z in 0..c.z.len() {
        c.initial[0][z] = [a, 0.];
        c.initial[1][z] = [w * a / (c.beta * kd), 0.];
        c.initial[10][z] = [0., a / (c.beta * kd)];
    }
    (ScalarSolver::new(c), w)
}
#[test]
fn total_density_equilibrium_stays_stationary() {
    let mut c = cfg();
    c.n0 = (0..7)
        .map(|z| 1. + 0.3 * (-((z as f64 - 3.) / 1.5).powi(2)).exp())
        .collect();
    for z in 1..6 {
        c.eq_ez[z] = -3. * c.beta * (c.n0[z + 1] - c.n0[z - 1]) / (2. * c.n0[z]);
    }
    c.nu.fill(0.01);
    c.gyro = [0.2, 0., -0.4];
    let mut s = ScalarSolver::new(c);
    let before = s.state.clone();
    for i in 0..100 {
        s.step(i as f64 * s.cfg.dt);
    }
    assert_eq!(s.state, before);
}
#[test]
fn scalar_density_recovers_bohm_gross_and_conserves_charge() {
    let (mut s, w) = wave(1e-4, true);
    let mass: f64 = s.state.iter().map(|q| q[0]).sum();
    for i in 0..400 {
        s.step(i as f64 * s.cfg.dt);
    }
    let t = 400. * s.cfg.dt;
    let nz = s.cfg.z.len();
    let mut err = 0f64;
    for x in 0..s.cfg.nx {
        let expected = 1. + 1e-4 * (TAU * x as f64 / s.cfg.nx as f64 - w * t).cos();
        err = err.max((s.state[x * nz + 3][0] - expected).abs());
    }
    assert!(err < 1e-11, "density error {err}");
    assert!(s.gauss().iter().all(|r| r.abs() < 3e-14));
    assert!((s.state.iter().map(|q| q[0]).sum::<f64>() - mass).abs() < 1e-10);
}
#[test]
fn full_density_equations_approach_linear_limit() {
    let mut errors = Vec::new();
    for a in [1e-4, 2e-4] {
        let (mut full, _) = wave(a, false);
        let (mut lin, _) = wave(a, true);
        for i in 0..200 {
            let t = i as f64 * full.cfg.dt;
            full.step(t);
            lin.step(t);
        }
        let err = full
            .state
            .iter()
            .zip(&lin.state)
            .map(|(a, b)| (a[0] - b[0]).powi(2))
            .sum::<f64>()
            .sqrt();
        errors.push(err);
        assert!(full.gauss().iter().all(|r| r.abs() < 3e-14));
    }
    assert!((errors[1] / errors[0] - 4.).abs() < 0.03, "{errors:?}");
}
