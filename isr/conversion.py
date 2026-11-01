"""Linear 1-D warm-electron Maxwell scattering with oblique incidence.

This is a controlled fluid closure, not a full Vlasov solver. Fixed ions,
prescribed equilibrium n0(z), scalar pressure force -3 kB Te grad(n1/n0),
and electron drag nu are assumed. An external equilibrium force supports
the prescribed profile. An optional complex pressure response represents local kinetic dispersion
and Landau loss. Nonlinear coupling and transmitter identification are omitted.

Convention exp(i kx x - i omega t); x-coordinate below is omega*z/c.
u is normalized by e Eunit/(me omega); b=c B1/Eunit;
s=sqrt(beta)*pressure_response*(n1/n0)/(e Eunit/(me omega c)), beta=3 kB Te/(me c^2).
State y=(Ex,Ey,bx,by,uz,s). Positive gyrofrequency vector is e B0/(me omega).
The first-order system follows from continuity, electron momentum, Faraday
and Ampere laws. Boundary eigenmodes impose incident and outgoing waves;
no absorbing layer or artificial longitudinal forcing is used.

Source context: Gondarenko et al. 2003, doi:10.1029/2003JA009985 (cold model);
Gondarenko et al. 2009, doi:10.1029/2009GL038379 (cold/warm comparison).
"""
from dataclasses import dataclass
import numpy as np
from scipy.constants import c, e, m_e, k as kb, epsilon_0
from scipy.linalg import solve_banded


@dataclass(frozen=True)
class Parameters:
    frequency_hz: float = 5.33e6
    temperature_k: float = 1000.
    magnetic_t: float = 40e-6
    magnetic_tilt_deg: float = 42.  # B relative to upward density gradient
    q: float = 0.15  # conserved kx/(omega/c), NOT local propagation angle

    @property
    def omega(self): return 2*np.pi*self.frequency_hz
    @property
    def k0(self): return self.omega/c
    @property
    def beta(self): return 3*kb*self.temperature_k/(m_e*c*c)
    @property
    def gyro(self):
        a = np.deg2rad(self.magnetic_tilt_deg)
        return e*self.magnetic_t/(m_e*self.omega)*np.array([np.sin(a), 0., np.cos(a)])


def components(y, p, drag, par, pressure_response=1.):
    """Reconstruct normalized E, b, u, and density amplitude from state(s)."""
    y = np.asarray(y)
    ex, ey, bx, by, uz, s = np.moveaxis(y, -1, 0)
    a = np.asarray(drag)-1j
    gx, _, gz = par.gyro
    r1 = -ex-1j*par.q*np.sqrt(par.beta)*s
    r2 = -ey-gx*uz
    ux = (a*r1-gz*r2)/(a*a+gz*gz)
    uy = (gz*r1+a*r2)/(a*a+gz*gz)
    ez = -par.q*by+1j*np.asarray(p)*uz
    E = np.stack([ex, ey, ez], axis=-1)
    b = np.stack([bx, by, par.q*ey], axis=-1)
    u = np.stack([ux, uy, uz], axis=-1)
    return E, b, u, s/(np.sqrt(par.beta)*np.asarray(pressure_response))


def matrix(p, logarithmic_gradient, drag, par, pressure_response=1.):
    """Return A for dy/d(k0*z)=A y, scalar or vectorized profiles."""
    if par.temperature_k <= 0:
        raise ValueError('Warm outgoing-wave solver requires positive temperature')
    p, logarithmic_gradient, drag, xi = np.broadcast_arrays(p, logarithmic_gradient, drag, pressure_response)
    basis = np.broadcast_to(np.eye(6, dtype=complex), p.shape+(6, 6))
    E, _, u, _ = components(basis, p[..., None], drag[..., None], par)
    ex, ey, bx, by, uz, s = np.moveaxis(basis, -1, 0)
    ux, uy, _ = np.moveaxis(u, -1, 0)
    ez = E[..., 2]
    gx = par.gyro[0]
    deriv = np.stack([
        1j*by+1j*par.q*ez,
        -1j*bx,
        1j*(par.q**2-1)*ey-p[..., None]*uy,
        1j*ex+p[..., None]*ux,
        1j*s/(np.sqrt(par.beta)*xi[...,None])-1j*par.q*ux-logarithmic_gradient[..., None]*uz,
        (-ez-(drag[..., None]-1j)*uz+gx*uy)/np.sqrt(par.beta),
    ], axis=-1)
    return np.swapaxes(deriv, -1, -2)


def flux(y, p, par):
    """Total z energy flux in units epsilon0*c*Eunit^2/2, including pressure."""
    ex, ey, bx, by, uz, s = np.moveaxis(np.asarray(y), -1, 0)
    return np.real(ex*np.conj(by)-ey*np.conj(bx)
                   +np.sqrt(par.beta)*np.asarray(p)*s*np.conj(uz))


def modes(p, par):
    """Collisionless reservoir modes, normalized to unit flux if propagating.

direction=+1: outgoing at the upper boundary; -1: outgoing at the lower.
Evanescent waves use the decay direction. No O/X naming is inferred.
"""
    if par.magnetic_t == 0:
        # Resolve exactly degenerate transverse eigenvalues into p/s modes;
        # generic eigensolvers can return non-flux-orthogonal combinations.
        values, columns = [], []
        for sign in (1, -1):
            nz = sign*np.sqrt(complex(1-p-par.q**2))
            n = np.array([par.q, 0., nz])
            for E in (np.array([nz, 0., -par.q]), np.array([0., 1., 0.])):
                b, u = np.cross(n, E), -1j*E
                columns.append([E[0], E[1], b[0], b[1], u[2], 0.])
                values.append(1j*nz)
            nz = sign*np.sqrt(complex((1-p)/par.beta-par.q**2))
            n = np.array([par.q, 0., nz])
            u = -1j*n/p
            columns.append([n[0], 0., 0., 0., u[2], np.sqrt(par.beta)*(n@u)])
            values.append(1j*nz)
        eigenvalues, vectors = np.array(values), np.array(columns, complex).T
        vectors /= np.linalg.norm(vectors, axis=0)
    else:
        eigenvalues, vectors = np.linalg.eig(matrix(p, 0., 0., par))
    directions, propagating = [], []
    for j, mu in enumerate(eigenvalues):
        v = vectors[:, j]
        propagates = abs(mu.real) < 1e-7*max(1., abs(mu.imag))
        f = flux(v, p, par)
        if propagates:
            if abs(f) < 1e-15:
                raise ValueError('Degenerate zero-flux mode: move reservoir from cutoff')
            vectors[:, j] /= np.sqrt(abs(f))
            directions.append(int(np.sign(f)))
        else:
            directions.append(-int(np.sign(mu.real)))
        propagating.append(propagates)
    return eigenvalues, vectors, np.array(directions), np.array(propagating)


def scatter(z_m, density_ratio, dlogp_dz, collision_s, par, branch=0, pressure_response=1., cell_order=2, incident_weights=None):
    """Solve linear scattering through a layer with collisionless reservoirs.

branch selects the incoming EM modes ordered by |kz| at the lower boundary.
The third, short-wavelength incident mode is excluded. Boundary nu must be 0.
Implicit midpoint collocation gives a block-banded system, stable through
evanescent regions. Grid convergence is required for conversion/field claims.
"""
    z, p, grad, nu = np.broadcast_arrays(z_m, density_ratio, dlogp_dz, collision_s)
    if z.ndim != 1 or len(z) < 3 or np.any(np.diff(z) <= 0) or np.any(p <= 0):
        raise ValueError('Need increasing 1-D grid and positive density')
    if abs(nu[0])+abs(nu[-1]) > 1e-12 or np.any(nu < 0):
        raise ValueError('Nonnegative drag must vanish at reservoir boundaries')
    xi=np.broadcast_to(np.asarray(pressure_response,dtype=complex),p.shape)
    if np.any(xi.real<=0) or np.any(xi.imag>1e-12) or not np.allclose(xi[[0,-1]],1.):
        raise ValueError("Passive pressure response must be real-positive, Im<=0, and unity at reservoirs")
    x = par.k0*z
    ml, vl, dl, pl = modes(p[0], par)
    mr, vr, dr, pr = modes(p[-1], par)
    il, ir = np.flatnonzero(dl == 1), np.flatnonzero(dr == -1)
    if len(il)+len(ir) != 6 or len(il) != 3:
        raise ValueError('Unexpected radiation-condition mode count')
    em = sorted([j for j in il if pl[j] and abs(ml[j]) < 5],
                key=lambda j: abs(ml[j]))
    if branch not in range(len(em)):
        raise ValueError('Requested incoming electromagnetic branch unavailable')
    incoming = em[branch]
    n = len(x)
    band = np.zeros((17, 6*n), dtype=complex)
    rhs = np.zeros(6*n, dtype=complex)

    def insert(rows, cols, values):
        rows, cols, values = np.broadcast_arrays(rows, cols, values)
        band[8+rows-cols, cols] = values

    left = np.linalg.inv(vl)[il]
    right = np.linalg.inv(vr)[ir]
    insert(np.arange(3)[:, None], np.arange(6)[None, :], left)
    weights=np.zeros(len(em),dtype=complex);weights[branch]=1.
    if incident_weights is not None:
        weights=np.asarray(incident_weights,dtype=complex)
        if weights.shape!=(len(em),) or not np.isclose(np.vdot(weights,weights).real,1.):
            raise ValueError('Incident EM weights must have unit squared norm')
    for j,weight in zip(em,weights):rhs[np.flatnonzero(il==j)[0]]=weight
    insert((6*n-3+np.arange(3))[:, None], (6*n-6+np.arange(6))[None, :], right)
    # Chunk assembly bounds temporary memory for kilometre-scale measured layers.
    # Midpoint profiles are second-order approximations to supplied samples.
    for start in range(0,n-1,20000):
        j=np.arange(start,min(start+20000,n-1))
        A=matrix((p[j+1]+p[j])/2,(grad[j+1]+grad[j])/(2*par.k0),
                 (nu[j+1]+nu[j])/(2*par.omega),par,pressure_response=(xi[j+1]+xi[j])/2)
        half=(x[j+1]-x[j])[:,None,None]*A/2
        rows=(3+6*j)[:,None,None]+np.arange(6)[None,:,None]
        cols=(6*j)[:,None,None]+np.arange(6)[None,None,:]
        if cell_order==4:
            # [2/2] Pade cell propagator. Coefficients sampled at midpoint;
            # fourth-order constant-medium phase, second-order coefficient
            # variation, whose scale is much longer than the wavelength.
            fourth=(half@half)/3
        elif cell_order==2:fourth=0.
        else:raise ValueError('cell_order must be 2 or 4')
        insert(rows,cols,-np.eye(6)-half-fourth)
        insert(rows,cols+6,np.eye(6)-half+fourth)
    y = solve_banded((8, 8), band, rhs, overwrite_ab=True, overwrite_b=True,
                     check_finite=False).reshape(n, 6)
    cl, cr = np.linalg.solve(vl, y[0]), np.linalg.solve(vr, y[-1])
    outgoing_left = np.flatnonzero((dl == -1) & pl)
    # At the low-density reservoir, identify the Langmuir branch by large |kz|.
    lang = max(outgoing_left, key=lambda j: abs(ml[j]))
    if abs(ml[lang]) < 5:
        raise ValueError('Reservoir does not separate EM and Langmuir wavelengths')
    conversion = abs(cl[lang])**2
    reflection = sum(abs(cl[j])**2 for j in outgoing_left if j != lang)
    transmission = sum(abs(cr[j])**2 for j in np.flatnonzero((dr == 1) & pr))
    E, b, u, h = components(y, p, nu/par.omega, par, xi)
    absorption_collision = np.trapezoid(p*nu/par.omega*np.sum(abs(u)**2, axis=1), x)
    absorption_landau = np.trapezoid(p*np.imag(1/xi)*abs(y[:,5])**2, x)
    absorption = absorption_collision+absorption_landau
    return dict(z_m=z, p=p, pressure_response=xi, absorption_collision=float(absorption_collision), absorption_landau=float(absorption_landau), state=y, E=E, b=b, u=u, h=h, flux=flux(y, p, par),
                conversion=float(conversion), reflection=float(reflection),
                outgoing_langmuir_amplitude=cl[lang],
                transmission=float(transmission), absorption=float(absorption),
                energy_error=float(1-conversion-reflection-transmission-absorption),
                incoming_kz_m1=float(ml[incoming].imag*par.k0),
                outgoing_langmuir_kz_m1=float(ml[lang].imag*par.k0),
                incoming_weights=weights,incoming_E=components(vl[:,em]@weights, p[0], 0., par)[0])


def optimal_polarization(z, p, grad, nu, par, target_z_m=None, pressure_response=1.):
    """Favorable coherent polarization for outgoing flux or local density.

If l_j are the unit-flux EM->Langmuir amplitudes, maximizing |sum(l_j a_j)|^2
under sum|a_j|^2=1 gives a_j=conj(l_j)/||l|| and Cmax=sum|l_j|^2.
The phases are specified at the lower reservoir. Real propagation from a
transmitter need not supply this optimum superposition of EM eigenmodes.
If target_z_m is supplied, maximize local |n1/n0| there instead of boundary
Langmuir flux. This matters when collisions attenuate waves before the boundary.
"""
    results=[scatter(z,p,grad,nu,par,branch=0,pressure_response=pressure_response)]
    try: results.append(scatter(z,p,grad,nu,par,branch=1,pressure_response=pressure_response))
    except ValueError as exc:
        if 'unavailable' not in str(exc): raise
    amplitudes=np.array([r['outgoing_langmuir_amplitude'] for r in results])
    # In a long collisional layer, boundary flux can be exponentially attenuated.
    # Optionally optimize the local complex density response at the radar match.
    response=amplitudes if target_z_m is None else np.array([
        np.interp(target_z_m,z,r['h']) for r in results])
    norm=float(np.linalg.norm(response))
    weights=np.conj(response)/norm if norm>0 else np.eye(1,len(results))[0]
    C=float(abs(amplitudes@weights)**2)
    r=results[0].copy()
    for key in ['state','E','b','u','h','incoming_E']:
        r[key]=sum(a*v[key] for a,v in zip(weights,results))
    r['flux']=flux(r['state'],p,par)
    r['conversion']=C
    r['reflection']=float(1-r['flux'][0]-C)
    r['transmission']=float(r['flux'][-1])
    r['absorption_collision']=float(np.trapezoid(p*nu/par.omega*np.sum(abs(r['u'])**2,axis=1),par.k0*z))
    r['absorption_landau']=float(np.trapezoid(p*np.imag(1/np.asarray(pressure_response))*abs(r['state'][:,5])**2,par.k0*z))
    r['absorption']=r['absorption_collision']+r['absorption_landau']
    r['energy_error']=1-C-r['reflection']-r['transmission']-r['absorption']
    r['outgoing_langmuir_amplitude']=amplitudes@weights
    # A superposition has no single incident kz. Preserve its modal content.
    r.pop('incoming_kz_m1')
    r['incoming_mode_kz_m1']=np.array([v['incoming_kz_m1'] for v in results])
    r['incoming_weights']=weights
    return r


def layer(z_m, scale_m, low=0.65, high=1.3, collision_s=0.):
    """Monotonic bottomside test profile; not a fitted full sporadic-E layer."""
    t = np.tanh(np.asarray(z_m)/scale_m)
    p = low+(high-low)*(1+t)/2
    grad = (high-low)*(1-t*t)/(2*scale_m*p)
    # Smooth drag taper makes the boundary modal decomposition unambiguous.
    nu = collision_s*np.sin(np.pi*(z_m-z_m[0])/(z_m[-1]-z_m[0]))**2
    nu[[0, -1]] = 0.
    return p, grad, nu


def matching_ratio(k_vector_m1, par):
    """Exact local collisionless fluid-Maxwell density for a specified k.

Positive eigenvalue P=omega_pe^2/omega^2 of i*M*Dvac. Return all real positive
solutions; caller must select the physical branch. Includes displacement
current and magnetic field, unlike the high-frequency approximate formula.
"""
    n = np.asarray(k_vector_m1)/par.k0
    gx, gy, gz = par.gyro
    cross = np.array([[0, gz, -gy], [-gz, 0, gx], [gy, -gx, 0]])
    M = -1j*np.eye(3)+cross+1j*par.beta*np.outer(n, n)
    D = (1-n@n)*np.eye(3)+np.outer(n, n)
    vals = np.linalg.eigvals(1j*M@D)
    return np.sort(vals.real[(abs(vals.imag) < 1e-7) & (vals.real > 0)])


def physical_density(h, par, incident_flux_w_m2):
    """Complex fractional density for a flux-normalized scattering solution."""
    Eunit = np.sqrt(2*incident_flux_w_m2/(epsilon_0*c))
    return h*e*Eunit/(m_e*par.omega*c)
