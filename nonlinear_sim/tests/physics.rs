use nonlinear_conversion::{Config, Solver};
use std::f64::consts::PI;
fn config(nx: usize) -> Config {
    let nz = 7;
    Config {
        nx,
        z: (0..nz).map(|z| z as f64).collect(),
        n0: vec![1.; nz],
        eq_ez: vec![0.; nz],
        nu: vec![0.; nz],
        sponge: vec![0.; nz],
        beta: 0.1,
        gyro: [0.; 3],
        dx: 2. * PI / nx as f64,
        dt: 0.01,
        steps: 100,
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
fn standing(nx: usize, a: f64, linear: bool) -> Solver {
    let mut c = config(nx);
    c.linear = linear;
    let kd = (8. * c.dx.sin() - (2. * c.dx).sin()) / (6. * c.dx);
    for z in 0..c.z.len() {
        c.initial[0][z] = [a, 0.];
        c.initial[4][z] = [3. * a, 0.];
        c.initial[7][z] = [a, 0.];
        c.initial[9][z] = [a, 0.];
        c.initial[10][z] = [0., a / (c.beta * kd)];
    }
    Solver::new(c)
}
#[test]
fn nonuniform_equilibrium_is_stationary() {
    let mut c = config(8);
    c.n0 = (0..7)
        .map(|z| 1. + 0.2 * (-((z as f64 - 3.) / 1.5).powi(2)).exp())
        .collect();
    for z in 1..6 {
        c.eq_ez[z] = -c.beta * (c.n0[z + 1] - c.n0[z - 1]) / (2. * c.n0[z]);
    }
    c.gyro = [0.2, 0., -0.4];
    c.nu.fill(0.01);
    let mut s = Solver::new(c);
    for i in 0..300 {
        s.step(i as f64 * 0.01);
    }
    let err = s.state.iter().flatten().fold(0f64, |a, &b| a.max(b.abs()));
    assert!(err < 2e-15, "{err}");
}
#[test]
fn uniform_langmuir_dispersion_and_gauss() {
    let mut s = standing(32, 1e-6, true);
    let dt = s.cfg.dt;
    let kd = (8. * s.cfg.dx.sin() - (2. * s.cfg.dx).sin()) / (6. * s.cfg.dx);
    let omega = (1. + 3. * s.cfg.beta.powi(2) * kd * kd).sqrt();
    for i in 0..1300 {
        s.step(i as f64 * dt);
    }
    let expected = 1e-6 * (omega * 1300. * dt).cos();
    let err = (s.state[3][0] - expected).abs() / 1e-6;
    assert!(err < 1e-8, "{err}");
    let (g, b, _) = s.audit();
    assert!(g < 1e-17, "{g}");
    assert!(b < 1e-18);
}
#[test]
fn nonlinear_limit_converges_to_linear_model() {
    let mut errors = Vec::new();
    for a in [1e-3, 1e-4] {
        let mut full = standing(24, a, false);
        let mut lin = standing(24, a, true);
        for i in 0..150 {
            full.step(i as f64 * 0.01);
            lin.step(i as f64 * 0.01);
        }
        let err = full
            .state
            .iter()
            .zip(&lin.state)
            .map(|(a, b)| (a[0] - b[0]).abs())
            .fold(0., f64::max)
            / a;
        errors.push(err);
    }
    assert!(errors[1] < errors[0] / 8., "{errors:?}");
    assert!(errors[1] > 0.);
}
#[test]
fn zero_heat_flux_energy_is_conserved() {
    let energy = |s: &Solver| {
        s.state
            .iter()
            .map(|q| 0.5 * (q[4] + q[7] + q[9] + q[10..16].iter().map(|v| v * v).sum::<f64>()))
            .sum::<f64>()
    };
    let mut errors = Vec::new();
    for nx in [16, 32] {
        let mut s = standing(nx, 0.001, false);
        let initial = energy(&s);
        for i in 0..200 {
            s.step(i as f64 * 0.01);
        }
        errors.push((energy(&s) - initial).abs() / initial.abs());
    }
    assert!(errors[1] < 1e-8, "{errors:?}");
}
#[test]
fn bgk_homogeneous_current_damping_matches_analytic() {
    let mut c = config(8);
    c.nu.fill(0.2);
    let mut s = Solver::new(c);
    // Homogeneous electron current and its self-consistent electric field.
    for q in &mut s.state {
        q[1] = 0.01;
        q[4] = 0.0001;
    }
    for i in 0..100 {
        s.step(i as f64 * 0.01);
    }
    let w = (1f64 - 0.1 * 0.1).sqrt();
    let expected = 0.01 * (-0.1f64).exp() * (w.cos() - 0.1 / w * w.sin());
    assert!((s.state[3][1] - expected).abs() < 1e-11);
    assert!(s.audit().0 < 1e-15);
}
#[test]
fn nonlinear_initialization_retains_positive_central_pressure() {
    let mut c = config(8);
    c.initial_pressure_lift = true;
    for z in 0..c.z.len() {
        c.initial[1][z] = [2., 0.];
    }
    let s = Solver::new(c);
    for cell in &s.state {
        let mut q = std::array::from_fn(|a| cell[a]);
        q[0] += 1.;
        for a in [4, 7, 9] {
            q[a] += 1.;
        }
        assert!(arecibo_fluid::ten_moment::admissible(&q));
        let p = arecibo_fluid::ten_moment::pressure(&q);
        assert!((p[0][0] - 1.).abs() < 1e-14);
    }
}
#[test]
fn rk4_time_refinement_has_fourth_order_error() {
    let mut errors = Vec::new();
    for dt in [0.2, 0.1, 0.05] {
        let mut s = standing(16, 1e-6, true);
        s.cfg.dt = dt;
        let kd = (8. * s.cfg.dx.sin() - (2. * s.cfg.dx).sin()) / (6. * s.cfg.dx);
        let w = (1. + 3. * s.cfg.beta.powi(2) * kd * kd).sqrt();
        let steps = (2. / dt).round() as usize;
        for i in 0..steps {
            s.step(i as f64 * dt);
        }
        errors.push((s.state[3][0] / 1e-6 - (2. * w).cos()).abs());
    }
    assert!(
        errors[0] / errors[1] > 12. && errors[1] / errors[2] > 12.,
        "{errors:?}"
    );
}
#[test]
fn electromagnetic_wave_has_correct_dispersion() {
    let mut c = config(32);
    c.beta = 0.;
    c.linear = true;
    let a = 1e-6;
    // Traveling cold-plasma transverse wave, exp(i k x - i omega t).
    let kd = (8. * c.dx.sin() - (2. * c.dx).sin()) / (6. * c.dx);
    let w = (1. + kd * kd).sqrt();
    for z in 0..c.z.len() {
        c.initial[11][z] = [a, 0.];
        c.initial[15][z] = [kd / w * a, 0.];
        c.initial[2][z] = [0., -a / w];
    }
    let mut s = Solver::new(c);
    for i in 0..1000 {
        s.step(i as f64 * 0.01);
    }
    assert!((s.state[3][11] - a * (w * 10.).cos()).abs() / a < 1e-8);
}
