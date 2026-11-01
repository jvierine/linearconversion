//! Total-density scalar barotropic electrons coupled to Maxwell.
//! State: [N, N ux, N uy, N uz, Ex, Ey, Ez, Hx, Hy, Hz]. N is TOTAL
//! density, normalized by density_si; E,H are deviations from equilibrium.
//! p_eff=3 n kB T_ref. Equilibrium electric support is subtracted exactly.
//! Same normalization and mesh as Config; no pressure tensor is evolved.
use crate::Config;
use meteor_fdtd::stratified::maxwell_rhs_xz;
pub type Cell = [f64; 10];
pub struct ScalarSolver {
    pub cfg: Config,
    pub state: Vec<Cell>,
}
impl ScalarSolver {
    pub fn new(cfg: Config) -> Self {
        assert!(cfg.nx >= 5 && cfg.z.len() >= 3 && cfg.n0.iter().all(|n| *n > 0.));
        let nz = cfg.z.len();
        let mut state = vec![[0.; 10]; cfg.nx * nz];
        for x in 0..cfg.nx {
            let ph = std::f64::consts::TAU * x as f64 / cfg.nx as f64;
            for z in 0..nz {
                for a in 0..10 {
                    let old = if a < 4 { a } else { a + 6 };
                    state[x * nz + z][a] =
                        cfg.initial[old][z][0] * ph.cos() - cfg.initial[old][z][1] * ph.sin();
                }
                state[x * nz + z][0] += cfg.n0[z];
            }
        }
        Self { cfg, state }
    }
    fn dx(&self, v: &[Cell], x: usize, z: usize, a: usize) -> f64 {
        let nx = self.cfg.nx;
        let nz = self.cfg.z.len();
        let at = |j: usize| v[j * nz + z][a];
        (-at((x + 2) % nx) + 8. * at((x + 1) % nx) - 8. * at((x + nx - 1) % nx)
            + at((x + nx - 2) % nx))
            / (12. * self.cfg.dx)
    }
    fn dz(&self, v: &[Cell], x: usize, z: usize, a: usize) -> f64 {
        let nz = self.cfg.z.len();
        if z == 0 || z + 1 == nz {
            return 0.;
        }
        (v[x * nz + z + 1][a] - v[x * nz + z - 1][a]) / (self.cfg.z[z + 1] - self.cfg.z[z - 1])
    }
    pub fn rhs(&self, state: &[Cell], time: f64) -> Vec<Cell> {
        let nz = self.cfg.z.len();
        let nx = self.cfg.nx;
        let mut fx = vec![[0.; 10]; state.len()];
        let mut fz = fx.clone();
        for x in 0..nx {
            for z in 0..nz {
                let i = x * nz + z;
                let d = state[i];
                let n = d[0];
                assert!(
                    n > 0. && n.is_finite(),
                    "Nonpositive total electron density"
                );
                for (axis, f) in [(0, &mut fx[i]), (2, &mut fz[i])] {
                    f[0] = d[1 + axis] - self.cfg.sponge[z] * d[4 + axis];
                    for a in 0..3 {
                        f[1 + a] = if self.cfg.linear {
                            0.
                        } else {
                            d[1 + a] * d[1 + axis] / n
                        };
                    }
                    // Subtract the equilibrium pressure before differentiation.
                    f[1 + axis] += 3. * (n - self.cfg.n0[z]);
                }
            }
        }
        let mut out = vec![[0.; 10]; state.len()];
        for x in 0..nx {
            let ph = std::f64::consts::TAU * x as f64 / nx as f64 - time;
            for z in 0..nz {
                let i = x * nz + z;
                let d = state[i];
                let n = d[0];
                let dn = n - self.cfg.n0[z];
                for a in 0..4 {
                    out[i][a] = -self.cfg.beta * (self.dx(&fx, x, z, a) + self.dz(&fz, x, z, a));
                }
                let g: [f64; 3] = std::array::from_fn(|a| {
                    self.cfg.gyro[a]
                        + if self.cfg.linear {
                            0.
                        } else {
                            self.cfg.beta * d[7 + a]
                        }
                });
                let cross = [
                    g[1] * d[3] - g[2] * d[2],
                    g[2] * d[1] - g[0] * d[3],
                    g[0] * d[2] - g[1] * d[1],
                ];
                for a in 0..3 {
                    out[i][1 + a] += cross[a]
                        - (self.cfg.nu[z] + self.cfg.sponge[z]) * d[1 + a]
                        - if self.cfg.linear {
                            self.cfg.n0[z] * d[4 + a]
                        } else {
                            n * d[4 + a]
                        };
                }
                out[i][3] -= dn * self.cfg.eq_ez[z];
                let dx_e = std::array::from_fn(|a| self.dx(state, x, z, 4 + a));
                let dz_e = std::array::from_fn(|a| self.dz(state, x, z, 4 + a));
                let dx_h = std::array::from_fn(|a| self.dx(state, x, z, 7 + a));
                let dz_h = std::array::from_fn(|a| self.dz(state, x, z, 7 + a));
                let (er, hr) = maxwell_rhs_xz(dx_e, dz_e, dx_h, dz_h, [-d[1], -d[2], -d[3]]);
                for a in 0..3 {
                    out[i][4 + a] = er[a] - self.cfg.sponge[z] * d[4 + a];
                    out[i][7 + a] = hr[a];
                }
                for a in 0..10 {
                    let old = if a < 4 { a } else { a + 6 };
                    let s = self.cfg.source[old][z];
                    out[i][a] += s[0] * ph.cos() - s[1] * ph.sin();
                }
            }
        }
        out
    }
    pub fn step(&mut self, time: f64) {
        let base = self.state.clone();
        let mut tmp = base.clone();
        let mut sum = vec![[0.; 10]; base.len()];
        for stage in 0..4 {
            let k = self.rhs(&tmp, time + self.cfg.dt * [0., 0.5, 0.5, 1.][stage]);
            for i in 0..base.len() {
                for a in 0..10 {
                    sum[i][a] += [1., 2., 2., 1.][stage] * k[i][a];
                    tmp[i][a] = base[i][a] + self.cfg.dt * [0.5, 0.5, 1., 0.][stage] * k[i][a];
                }
            }
        }
        for i in 0..base.len() {
            for a in 0..10 {
                self.state[i][a] = base[i][a] + self.cfg.dt * sum[i][a] / 6.;
            }
        }
    }
    /// Discrete perturbation Gauss residual, before impressed-source charge.
    pub fn gauss(&self) -> Vec<f64> {
        let nz = self.cfg.z.len();
        let mut r = Vec::new();
        for x in 0..self.cfg.nx {
            for z in 1..nz - 1 {
                r.push(
                    self.cfg.beta * (self.dx(&self.state, x, z, 4) + self.dz(&self.state, x, z, 6))
                        + self.state[x * nz + z][0]
                        - self.cfg.n0[z],
                );
            }
        }
        r
    }
}
