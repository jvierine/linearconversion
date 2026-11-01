//! Production timestep shared by the executable and physics validation tests.
use arecibo_fluid::coupling::{CouplingError, FieldCoupledFluid};
use meteor_fdtd::stratified::{Complex, advance_electric, advance_magnetic};

/// E and density are at integer times; H and current at half times. Advance
/// the electron current exactly once, then give that same current to Ampere.
#[allow(clippy::too_many_arguments)]
pub fn advance_coupled(
    electric: &mut [Vec<Complex>],
    magnetic: &mut [Vec<Complex>],
    fluid: &mut impl FieldCoupledFluid,
    wp: &[Vec<f64>],
    widths: &[f64],
    kx: f64,
    dt: f64,
    sponge: &[Vec<f64>],
    source: &[Vec<Complex>],
    source_phase: f64,
) -> Result<(), CouplingError> {
    advance_magnetic(electric, magnetic, widths, kx, dt, sponge);
    fluid.advance(electric)?;
    advance_electric(
        electric,
        magnetic,
        fluid.current(),
        wp,
        widths,
        kx,
        dt,
        sponge,
        source,
        source_phase,
    );
    Ok(())
}
