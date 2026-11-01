//! Independent physical constraints on the exact timestep used by main.rs.
//! Scaled homogeneous tests are not estimates of event plasma parameters.
use arecibo_coupled::advance_coupled;
use arecibo_fluid::{
    coupling::{FieldCoupledFluid, YeeFluidCoupler},
    maxwell_coupling::CouplingGeometry,
};
use std::f64::consts::TAU;
type Z = [f64; 2];
fn add(a: Z, b: Z) -> Z {
    [a[0] + b[0], a[1] + b[1]]
}
fn sub(a: Z, b: Z) -> Z {
    [a[0] - b[0], a[1] - b[1]]
}
fn mul(a: Z, b: f64) -> Z {
    [a[0] * b, a[1] * b]
}
fn ik(a: Z, k: f64) -> Z {
    [-k * a[1], k * a[0]]
}
fn norm(a: Z) -> f64 {
    a[0] * a[0] + a[1] * a[1]
}
fn mag(a: Z) -> f64 {
    norm(a).sqrt()
}
fn zeros(n: usize) -> Vec<Vec<Z>> {
    vec![vec![[0.; 2]; n]; 3]
}
fn report(name: &str, values: serde_json::Value) {
    println!(
        "VALIDATION {}",
        serde_json::json!({"name":name,"metrics":values})
    );
}
struct Grid {
    d: Vec<f64>,
    rho: Vec<Vec<f64>>,
    wp: Vec<Vec<f64>>,
    nu: Vec<Vec<f64>>,
    sponge: Vec<Vec<f64>>,
    kx: f64,
    beta: f64,
    b: [f64; 3],
}
impl Grid {
    fn uniform(n: usize, nu: f64) -> Self {
        Self {
            d: vec![TAU / n as f64; n],
            rho: vec![vec![1.; n]; 3],
            wp: vec![vec![0.8; n]; 3],
            nu: vec![vec![nu; n]; 3],
            sponge: vec![vec![0.; n]; 6],
            kx: 0.,
            beta: 0.35,
            b: [0.; 3],
        }
    }
    fn varied(n: usize, nu: f64) -> Self {
        let mut g = Self::uniform(n, nu);
        g.kx = 0.43;
        g.b = [0.2, 0.1, 0.3];
        g.beta = 0.25;
        for i in 0..n {
            let a = TAU * i as f64 / n as f64;
            g.d[i] *= 1. + 0.2 * a.sin();
            for q in 0..3 {
                let x = a + if q == 2 { TAU / (2. * n as f64) } else { 0. };
                g.rho[q][i] = 1. + 0.23 * x.cos();
                g.wp[q][i] = 0.8 * g.rho[q][i];
            }
        }
        g
    }
    fn fluid<'a>(&'a self, dt: f64, j: Vec<Vec<Z>>, q: Vec<Z>) -> YeeFluidCoupler<'a> {
        YeeFluidCoupler::new(
            CouplingGeometry {
                widths: &self.d,
                kx: self.kx,
                gyro: self.b,
                plasma_frequency: &self.wp,
                collision_rate: &self.nu,
                sqrt_density: &self.rho,
                thermal_speed: self.beta,
                density_buffer: &self.sponge[0],
            },
            dt,
            j,
            q,
        )
        .unwrap()
    }
    fn step(&self, dt: f64, e: &mut [Vec<Z>], h: &mut [Vec<Z>], f: &mut YeeFluidCoupler) {
        advance_coupled(
            e,
            h,
            f,
            &self.wp,
            &self.d,
            self.kx,
            dt,
            &self.sponge,
            &zeros(self.d.len()),
            0.,
        )
        .unwrap();
    }
    fn dual(&self, i: usize) -> f64 {
        (self.d[i] + self.d[(i + self.d.len() - 1) % self.d.len()]) / 2.
    }
    fn div(&self, e: &[Vec<Z>], i: usize) -> Z {
        add(
            ik(e[0][i], self.kx),
            mul(
                sub(e[2][i], e[2][(i + self.d.len() - 1) % self.d.len()]),
                1. / self.dual(i),
            ),
        )
    }
    fn initial(&self) -> (Vec<Vec<Z>>, Vec<Vec<Z>>, Vec<Vec<Z>>, Vec<Z>) {
        let n = self.d.len();
        let mut e = zeros(n);
        let mut j = zeros(n);
        for q in 0..3 {
            for i in 0..n {
                let a = TAU * i as f64 / n as f64 + q as f64 * 0.37;
                e[q][i] = [a.cos(), 0.3 * (2. * a).sin()];
                j[q][i] = [0.2 * a.sin(), 0.1 * (3. * a).cos()];
            }
        }
        let q = (0..n)
            .map(|i| mul(self.div(&e, i), -self.beta / self.wp[0][i]))
            .collect();
        (e, zeros(n), j, q)
    }
}

#[test]
fn nonuniform_magnetized_timesteps_satisfy_all_equations_and_gauss_law() {
    let n = 48;
    let g = Grid::varied(n, 0.07);
    let dt = 0.01;
    let (mut e, mut h, j, q) = g.initial();
    let mut f = g.fluid(dt, j, q);
    let mut err = [0f64; 6];
    for _ in 0..100 {
        let eo = e.clone();
        let ho = h.clone();
        let jo = f.current().to_vec();
        let qo = f.density().to_vec();
        g.step(dt, &mut e, &mut h, &mut f);
        let j = f.current();
        let q = f.density();
        let mid: Vec<Vec<Z>> = (0..3)
            .map(|r| (0..n).map(|i| mul(add(jo[r][i], j[r][i]), 0.5)).collect())
            .collect();
        for i in 0..n {
            let im = (i + n - 1) % n;
            let ip = (i + 1) % n;
            let df = |r: usize| mul(sub(eo[r][ip], eo[r][i]), 1. / g.d[i]);
            let db = |r: usize| mul(sub(h[r][i], h[r][im]), 1. / g.dual(i));
            let ce = [
                mul(df(1), -1.),
                sub(df(0), ik(eo[2][i], g.kx)),
                ik(eo[1][i], g.kx),
            ];
            let ch = [
                mul(db(1), -1.),
                sub(db(0), ik(h[2][i], g.kx)),
                ik(h[1][i], g.kx),
            ];
            let p = [
                ik(qo[i], g.beta * g.kx),
                [0.; 2],
                mul(
                    sub(mul(qo[ip], 1. / g.rho[0][ip]), mul(qo[i], 1. / g.rho[0][i])),
                    g.beta * g.rho[2][i] / g.d[i],
                ),
            ];
            let jz = mul(
                add(mul(mid[2][i], g.d[i]), mul(mid[2][im], g.d[im])),
                1. / (g.d[i] + g.d[im]),
            );
            let jx = mul(add(mid[0][i], mid[0][ip]), 0.5);
            let jy = mul(add(mid[1][i], mid[1][ip]), 0.5);
            let gyro = [
                sub(mul(jz, g.b[1]), mul(mid[1][i], g.b[2])),
                sub(mul(mid[0][i], g.b[2]), mul(jz, g.b[0])),
                sub(mul(jy, g.b[0]), mul(jx, g.b[1])),
            ];
            for r in 0..3 {
                err[0] = err[0].max(mag(sub(
                    mul(sub(e[r][i], eo[r][i]), 1. / dt),
                    sub(ch[r], mul(j[r][i], g.wp[r][i])),
                )));
                err[1] = err[1].max(mag(add(mul(sub(h[r][i], ho[r][i]), 1. / dt), ce[r])));
                let force = sub(
                    add(add(mul(eo[r][i], g.wp[r][i]), p[r]), gyro[r]),
                    mul(mid[r][i], g.nu[r][i]),
                );
                err[2] = err[2].max(mag(sub(mul(sub(j[r][i], jo[r][i]), 1. / dt), force)));
            }
            let divj = add(
                ik(j[0][i], g.kx),
                mul(
                    sub(mul(j[2][i], g.rho[2][i]), mul(j[2][im], g.rho[2][im])),
                    1. / (g.dual(i) * g.rho[0][i]),
                ),
            );
            err[3] = err[3].max(mag(sub(mul(sub(q[i], qo[i]), 1. / dt), mul(divj, g.beta))));
            err[4] = err[4].max(mag(add(g.div(&e, i), mul(q[i], g.wp[0][i] / g.beta))));
            err[5] = err[5].max(mag(add(
                ik(h[0][i], g.kx),
                mul(sub(h[2][ip], h[2][i]), 1. / g.d[i]),
            )));
        }
    }
    report(
        "equation_residuals",
        serde_json::json!({"Ampere":err[0],"Faraday":err[1],"momentum":err[2],"continuity":err[3],"Gauss_electric":err[4],"Gauss_magnetic":err[5],"limit":1e-10}),
    );
    assert!(err.iter().all(|e| *e < 1e-10), "equation residuals {err:?}");
}

fn wave_error(n: usize, em: bool, nu: f64) -> (f64, f64) {
    let g = Grid::uniform(n, nu);
    let dt = 0.3 * g.d[0];
    let kd = 2. * (g.d[0] / 2.).sin() / g.d[0];
    let freq = (0.8f64.powi(2) + if em { 1. } else { g.beta * g.beta }).sqrt();
    let damped = (freq * freq - nu * nu / 4.).sqrt();
    let mut e = zeros(n);
    let mut h = zeros(n);
    let mut j = zeros(n);
    let mut q = vec![[0.; 2]; n];
    for i in 0..n {
        let z = i as f64 * g.d[0];
        let face = z + g.d[0] / 2.;
        if em {
            e[0][i] = [z.cos(), 0.];
            h[1][i] = [-0.5 * dt * kd * face.sin(), 0.];
            j[0][i] = [-0.5 * dt * 0.8 * z.cos(), 0.];
        } else {
            e[2][i] = [face.cos(), 0.];
            q[i] = [g.beta * kd / 0.8 * z.sin(), 0.];
            j[2][i] = [
                -0.5 * dt * (0.8 + g.beta * g.beta * kd * kd / 0.8) * face.cos(),
                0.,
            ];
        }
    }
    let norm = e[if em { 0 } else { 2 }][0][0];
    let mut f = g.fluid(dt, j, q);
    let mut err = 0f64;
    let mut trace = vec![1.];
    let steps = (8. * TAU / (damped * dt)).ceil() as usize;
    for s in 1..=steps {
        g.step(dt, &mut e, &mut h, &mut f);
        let t = s as f64 * dt;
        let v = e[if em { 0 } else { 2 }][0][0] / norm;
        let exact =
            (-nu * t / 2.).exp() * ((damped * t).cos() + nu / (2. * damped) * (damped * t).sin());
        err = err.max((v - exact).abs());
        trace.push(v);
    }
    let mut peaks = Vec::new();
    for i in 1..trace.len() - 1 {
        if trace[i] > 0. && trace[i] > trace[i - 1] && trace[i] > trace[i + 1] {
            let y = trace[i];
            let a = trace[i - 1];
            let b = trace[i + 1];
            let shift = 0.5 * (a - b) / (a - 2. * y + b);
            let peak = y - 0.25 * (a - b) * shift;
            peaks.push(((i as f64 + shift) * dt, peak.ln()));
        }
    }
    let mt = peaks.iter().map(|p| p.0).sum::<f64>() / peaks.len() as f64;
    let my = peaks.iter().map(|p| p.1).sum::<f64>() / peaks.len() as f64;
    let gamma = -peaks.iter().map(|p| (p.0 - mt) * (p.1 - my)).sum::<f64>()
        / peaks.iter().map(|p| (p.0 - mt).powi(2)).sum::<f64>();
    (err, gamma)
}
#[test]
fn electromagnetic_and_langmuir_waves_converge_to_continuum_dispersion() {
    for em in [false, true] {
        let errors: Vec<_> = [32, 64, 128]
            .iter()
            .map(|&n| wave_error(n, em, 0.).0)
            .collect();
        report(
            if em {
                "EM_dispersion"
            } else {
                "Langmuir_dispersion"
            },
            serde_json::json!({"cells":[32,64,128],"max_waveform_errors":errors}),
        );
        assert!(
            errors[0] / errors[1] > 3.5 && errors[1] / errors[2] > 3.5,
            "{errors:?}"
        );
        assert!(errors[2] < 0.005);
    }
}
#[test]
fn coupled_langmuir_amplitude_decays_at_half_the_collision_rate() {
    let nu = 0.08;
    let (err, gamma) = wave_error(128, false, nu);
    report(
        "collisional_decay",
        serde_json::json!({"measured_amplitude_decay":gamma,"expected_nu_over_2":nu/2.,"max_waveform_error":err}),
    );
    assert!((gamma / (nu / 2.) - 1.).abs() < 0.001);
    assert!(err < 0.002);
}
fn energy_error(dt: f64, nu: f64) -> f64 {
    let n = 48;
    let g = Grid::varied(n, nu);
    let (mut e, mut h, j, q) = g.initial();
    let mut f = g.fluid(dt, j, q);
    let mut u0 = 0.;
    let mut heating = 0.;
    let mut old_rate = 0.;
    let mut error = 0f64;
    for step in 0..(20. / dt).round() as usize {
        let eo = e.clone();
        let ho = h.clone();
        let jo = f.current().to_vec();
        let qo = f.density().to_vec();
        g.step(dt, &mut e, &mut h, &mut f);
        let mut u = 0.;
        let mut rate = 0.;
        for i in 0..n {
            u += 0.5 * g.dual(i) * norm(qo[i]);
            for r in 0..3 {
                let we = if r == 2 { g.d[i] } else { g.dual(i) };
                let wh = if r == 2 { g.dual(i) } else { g.d[i] };
                let jm = mul(add(jo[r][i], f.current()[r][i]), 0.5);
                let hm = mul(add(ho[r][i], h[r][i]), 0.5);
                u += 0.5 * (we * (norm(eo[r][i]) + norm(jm)) + wh * norm(hm));
                rate += we * nu * norm(jm);
            }
        }
        if step == 0 {
            u0 = u;
        } else {
            heating += 0.5 * dt * (old_rate + rate);
            error = error.max(((u + heating - u0) / u0).abs());
        }
        old_rate = rate;
    }
    error
}
#[test]
fn energy_is_conserved_or_becomes_collision_heat_with_second_order_error() {
    for nu in [0., 0.07] {
        let coarse = energy_error(0.02, nu);
        let fine = energy_error(0.01, nu);
        report(
            if nu == 0. {
                "lossless_energy"
            } else {
                "collision_energy"
            },
            serde_json::json!({"coarse_relative_budget_error":coarse,"fine_relative_budget_error":fine,"refinement_ratio":coarse/fine}),
        );
        assert!(fine < 0.0001, "energy closure {fine}");
        assert!(coarse / fine > 3.5, "energy convergence {}", coarse / fine);
    }
}
