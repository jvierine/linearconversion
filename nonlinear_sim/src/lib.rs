//! Real x-z, nonlinear ten-moment/Maxwell evolution. Fields and moments use
//! the SAME commuting derivatives and RK stages, preserving Gauss's law.
//! n_ref=eps0*m*omega^2/e^2, v_ref=sqrt(kT_ref/m), length=c/omega,
//! time=1/omega, E_ref=m*v_ref*omega/e, H=c*B/E_ref. beta=v_ref/c.
//! All state entries are deviations from a force-balanced equilibrium.
use arecibo_fluid::ten_moment::{self as tm, PAIRS};
use meteor_fdtd::stratified::maxwell_rhs_xz;
use serde::{Deserialize, Serialize};
pub mod scalar;
pub type Cell = [f64; 16];
#[derive(Clone, Deserialize, Serialize)]
pub struct Config {
    pub nx: usize,
    pub z: Vec<f64>,
    pub n0: Vec<f64>,
    pub eq_ez: Vec<f64>,
    pub nu: Vec<f64>,
    pub sponge: Vec<f64>,
    pub beta: f64,
    pub gyro: [f64; 3],
    pub dx: f64,
    pub dt: f64,
    pub steps: usize,
    pub stride: usize,
    pub initial: Vec<Vec<[f64; 2]>>,
    pub source: Vec<Vec<[f64; 2]>>,
    pub linear: bool,
    pub output: String,
    pub omega_si: f64,
    pub length_si: f64,
    pub density_si: f64,
    pub electric_si: f64,
    pub altitude_origin_m: f64,
    #[serde(default)]
    pub initial_pressure_lift: bool,
}
pub struct Solver {
    pub cfg: Config,
    pub state: Vec<Cell>,
    fx: Vec<Cell>,
    fz: Vec<Cell>,
    k: Vec<Cell>,
    sum: Vec<Cell>,
    temp: Vec<Cell>,
    base: Vec<Cell>,
}
impl Solver {
    pub fn new(cfg: Config) -> Self {
        let nz = cfg.z.len();
        assert!(cfg.nx >= 5 && nz >= 3);
        assert!(cfg.dt > 0. && cfg.dx > 0. && cfg.n0.iter().all(|&v| v > 0.));
        let mut state = vec![[0.; 16]; cfg.nx * nz];
        for x in 0..cfg.nx {
            let phase = 2. * std::f64::consts::PI * x as f64 / cfg.nx as f64;
            for z in 0..nz {
                for a in 0..16 {
                    state[x * nz + z][a] =
                        cfg.initial[a][z][0] * phase.cos() - cfg.initial[a][z][1] * phase.sin();
                }
            }
        }
        // The harmonic pressure perturbation is a central moment. Include the
        // quadratic bulk-flow contribution when initializing nonlinear raw moments.
        if cfg.initial_pressure_lift && !cfg.linear {
            for (i, q) in state.iter_mut().enumerate() {
                let n = cfg.n0[i % nz] + q[0];
                for (k, &(a, b)) in PAIRS.iter().enumerate() {
                    q[4 + k] += q[1 + a] * q[1 + b] / n;
                }
            }
        }
        let blank = vec![[0.; 16]; state.len()];
        Self {
            cfg,
            state,
            fx: blank.clone(),
            fz: blank.clone(),
            k: blank.clone(),
            sum: blank.clone(),
            temp: blank.clone(),
            base: blank,
        }
    }
    /// Fourth-order centered horizontal derivative; periodic x.
    fn dx(&self, v: &[Cell], x: usize, z: usize, a: usize) -> f64 {
        let nx = self.cfg.nx;
        let nz = self.cfg.z.len();
        let at = |j: usize| v[j * nz + z][a];
        (-at((x + 2) % nx) + 8. * at((x + 1) % nx) - 8. * at((x + nx - 1) % nx)
            + at((x + nx - 2) % nx))
            / (12. * self.cfg.dx)
    }
    /// Conservative centered derivative on dual cells of a smoothly graded grid.
    fn dz(&self, v: &[Cell], x: usize, z: usize, a: usize) -> f64 {
        let nz = self.cfg.z.len();
        if z == 0 || z + 1 == nz {
            return 0.;
        }
        (v[x * nz + z + 1][a] - v[x * nz + z - 1][a]) / (self.cfg.z[z + 1] - self.cfg.z[z - 1])
    }
    pub fn rhs(&mut self, state: &[Cell], time: f64) -> Vec<Cell> {
        let nz = self.cfg.z.len();
        let nx = self.cfg.nx;
        // Perturbation fluxes ensure exactly stationary nonuniform equilibrium.
        for x in 0..nx {
            for z in 0..nz {
                let i = x * nz + z;
                let d = &state[i];
                let n0 = self.cfg.n0[z];
                let mut q = std::array::from_fn(|a| d[a]);
                q[0] += n0;
                for a in [4, 7, 9] {
                    q[a] += n0;
                }
                if self.cfg.linear {
                    for (axis, out) in [(0, &mut self.fx[i]), (2, &mut self.fz[i])] {
                        out.fill(0.);
                        out[0] = d[1 + axis];
                        let m = tm::matrix(&std::array::from_fn(|a| d[a]));
                        for a in 0..3 {
                            out[1 + a] = m[a][axis];
                        }
                        for (k, &(a, b)) in PAIRS.iter().enumerate() {
                            out[4 + k] = if a == b { d[1 + axis] } else { 0. }
                                + if b == axis { d[1 + a] } else { 0. }
                                + if a == axis { d[1 + b] } else { 0. };
                        }
                    }
                } else {
                    let fx = tm::flux(&q, 0);
                    let fz = tm::flux(&q, 2);
                    self.fx[i][..10].copy_from_slice(&fx);
                    self.fz[i][..10].copy_from_slice(&fz);
                    self.fx[i][1] -= n0;
                    self.fz[i][3] -= n0;
                }
                // Buffer electric damping is an artificial charge current. Its matching
                // continuity flux is retained; it is excluded from physical loss accounting.
                self.fx[i][0] -= self.cfg.sponge[z] * d[10];
                self.fz[i][0] -= self.cfg.sponge[z] * d[12];
            }
        }
        let mut out = vec![[0.; 16]; state.len()];
        for x in 0..nx {
            let phase = 2. * std::f64::consts::PI * x as f64 / nx as f64 - time;
            let (sn, cs) = phase.sin_cos();
            for z in 0..nz {
                let i = x * nz + z;
                let d = &state[i];
                let n0 = self.cfg.n0[z];
                let sig = self.cfg.sponge[z];
                let mut q: tm::Moments = std::array::from_fn(|a| d[a]);
                q[0] += n0;
                for a in [4, 7, 9] {
                    q[a] += n0;
                }
                let ee = [d[10], d[11], self.cfg.eq_ez[z] + d[12]];
                let g = std::array::from_fn(|a| {
                    self.cfg.gyro[a]
                        + if self.cfg.linear {
                            0.
                        } else {
                            self.cfg.beta * d[13 + a]
                        }
                });
                let mut src = if self.cfg.linear {
                    let mut v = tm::source(
                        &std::array::from_fn(|a| d[a]),
                        [0., 0., -self.cfg.eq_ez[z]],
                        g,
                        self.cfg.nu[z] + sig,
                        1.,
                    );
                    for a in 0..3 {
                        v[1 + a] -= n0 * d[10 + a];
                    }
                    v
                } else {
                    tm::source(&q, ee.map(|v| -v), g, self.cfg.nu[z] + sig, 1.)
                };
                // Subtract equilibrium electric force, exactly paired with removed flux.
                if !self.cfg.linear {
                    src[3] += n0 * self.cfg.eq_ez[z];
                }
                for a in 0..10 {
                    out[i][a] = src[a]
                        - self.cfg.beta * (self.dx(&self.fx, x, z, a) + self.dz(&self.fz, x, z, a));
                }
                let dx_e = std::array::from_fn(|a| self.dx(state, x, z, 10 + a));
                let dz_e = std::array::from_fn(|a| self.dz(state, x, z, 10 + a));
                let dx_h = std::array::from_fn(|a| self.dx(state, x, z, 13 + a));
                let dz_h = std::array::from_fn(|a| self.dz(state, x, z, 13 + a));
                let (er, hr) = maxwell_rhs_xz(dx_e, dz_e, dx_h, dz_h, [-d[1], -d[2], -d[3]]);
                for a in 0..3 {
                    out[i][10 + a] = er[a] - sig * d[10 + a];
                    out[i][13 + a] = hr[a];
                }
                for a in 0..16 {
                    let s = self.cfg.source[a][z];
                    out[i][a] += s[0] * cs - s[1] * sn;
                }
            }
        }
        out
    }
    pub fn step(&mut self, time: f64) {
        // Shared RK4 stages keep the current in continuity and Ampere identical.
        self.base.copy_from_slice(&self.state);
        self.temp.copy_from_slice(&self.state);
        self.sum.fill([0.; 16]);
        for stage in 0..4 {
            let t = time + self.cfg.dt * [0., 0.5, 0.5, 1.][stage];
            let tmp = std::mem::take(&mut self.temp);
            self.k = self.rhs(&tmp, t);
            self.temp = tmp;
            let factor = [0.5, 0.5, 1., 0.][stage];
            let w = [1., 2., 2., 1.][stage];
            for i in 0..self.state.len() {
                for a in 0..16 {
                    self.sum[i][a] += w * self.k[i][a];
                    self.temp[i][a] = self.base[i][a] + self.cfg.dt * factor * self.k[i][a];
                }
            }
        }
        for i in 0..self.state.len() {
            for a in 0..16 {
                self.state[i][a] = self.base[i][a] + self.cfg.dt * self.sum[i][a] / 6.;
            }
        }
    }
    pub fn audit(&self) -> (f64, f64, f64) {
        self.audit_at(0.)
    }
    pub fn audit_at(&self, time: f64) -> (f64, f64, f64) {
        let nz = self.cfg.z.len();
        let mut gauss = 0f64;
        let mut db = 0f64;
        let mut dn = 0f64;
        for x in 0..self.cfg.nx {
            for z in 2..nz - 2 {
                let i = x * nz + z;
                let d = self.state[i];
                let mut q = std::array::from_fn(|a| d[a]);
                q[0] += self.cfg.n0[z];
                for a in [4, 7, 9] {
                    q[a] += self.cfg.n0[z];
                }
                assert!(
                    tm::admissible(&q),
                    "Nonpositive density or pressure at x={x},z={z}"
                );
                // Charge on the impressed source sheet follows its own continuity law.
                let phase = 2. * std::f64::consts::PI * x as f64 / self.cfg.nx as f64 - time;
                let angle = 2. * std::f64::consts::PI / self.cfg.nx as f64;
                let kd = (8. * angle.sin() - (2. * angle).sin()) / (6. * self.cfg.dx);
                let ds: [f64; 2] = std::array::from_fn(|a| {
                    (self.cfg.source[12][z + 1][a] - self.cfg.source[12][z - 1][a])
                        / (self.cfg.z[z + 1] - self.cfg.z[z - 1])
                });
                let sx = self.cfg.source[10][z];
                let external = self.cfg.beta
                    * ((-kd * sx[0] - ds[1]) * phase.cos() - (-kd * sx[1] + ds[0]) * phase.sin());
                gauss = gauss.max(
                    (self.cfg.beta
                        * (self.dx(&self.state, x, z, 10) + self.dz(&self.state, x, z, 12))
                        + d[0]
                        - external)
                        .abs(),
                );
                db =
                    db.max((self.dx(&self.state, x, z, 13) + self.dz(&self.state, x, z, 15)).abs());
                dn = dn.max((d[0] / self.cfg.n0[z]).abs());
            }
        }
        (gauss, db, dn)
    }
}
