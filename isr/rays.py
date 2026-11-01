"""Muldrew-style ray geometry alongside the driven full-wave solution.

Muldrew (1993), Radio Science 28, 379-388, doi:10.1029/92RS00273:
conserve frequency/tangential k, distinguish group from phase direction,
retain signed incidence, and integrate loss along propagation time.
The cold tensor benchmark reproduces his horizontal-group-velocity point.
The event plasma ray uses the longitudinal Maxwellian dielectric only after
the wave is electrostatic. The fluid-Maxwell boundary solution still handles
launch, reflection, tunnelling and interference; rays do not set launch power.
"""
from dataclasses import replace
import numpy as np
from scipy.constants import c
from scipy.optimize import root
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import PchipInterpolator
from .landau import dielectric


def spitze_angle_deg(par):
    """Cold magnetic-meridian O/Z critical vacuum angle (absolute value)."""
    Y = np.linalg.norm(par.gyro)
    return np.rad2deg(np.arcsin(np.sqrt(Y/(1+Y))*abs(np.sin(np.deg2rad(par.magnetic_tilt_deg)))))


def electromagnetic_dispersion(k_vector, density_ratio, par, cold=False):
    """Hermitian local cold/warm-fluid Maxwell dispersion matrix, real k,w."""
    n = np.asarray(k_vector)/par.k0
    gx, gy, gz = par.gyro
    cross = np.array([[0, gz, -gy], [-gz, 0, gx], [gy, -gx, 0]])
    M = -1j*np.eye(3)+cross
    if not cold:
        M = M+1j*par.beta*np.outer(n, n)
    return (1-n@n)*np.eye(3)+np.outer(n, n)+1j*density_ratio*np.linalg.inv(M)


def cold_horizontal_group_point(par, initial_nz=-4.74, initial_ratio=.993):
    """Solve D=0 and dD/dkz=0, not a claim about access or transmission."""
    def equations(v):
        nz, p = v
        k = np.array([par.q*par.k0, 0., nz*par.k0])
        values, vectors = np.linalg.eigh(electromagnetic_dispersion(k, p, par, cold=True))
        j = np.argmin(abs(values)); E = vectors[:, j]
        shift = np.array([0., 0., par.k0*1e-4])
        derivative = (electromagnetic_dispersion(k+shift, p, par, cold=True)
                      - electromagnetic_dispersion(k-shift, p, par, cold=True))/(2e-4)
        return [values[j], np.vdot(E, derivative@E).real]
    sol = root(equations, [initial_nz, initial_ratio], tol=1e-10)
    if np.linalg.norm(sol.fun) > 1e-8:
        raise ArithmeticError('Horizontal group point did not converge')
    nz, p = sol.x
    return dict(nz=float(nz), density_ratio=float(p),
                refractive_index=float(np.hypot(par.q, nz)), residual=float(np.linalg.norm(sol.fun)))


def electromagnetic_ray(z_start, kz_start, density_and_gradient, par,
                        cold=True, max_time_s=.002, max_index=7., max_step_s=1e-6):
    """Hamiltonian trace of an incident local EM mode, including reflection.

    kx and omega are fixed. Density callback returns P and dP/dz in SI metres.
    Rays cannot cross an evanescent interval: tunnelling/branch launch remains
    the full-wave boundary problem. Stop at index 7 for cold tracing, following
    Muldrew's stated limit before hot-plasma dispersion becomes necessary.
    """
    kx=par.q*par.k0
    def local(z,kz):
        p,dp=density_and_gradient(z);kv=np.array([kx,0.,kz])
        D=electromagnetic_dispersion(kv,p,par,cold=cold)
        values,vectors=np.linalg.eigh(D);j=np.argmin(abs(values));E=vectors[:,j]
        if cold:
            n=kv/par.k0
            deriv=[]
            for i in [0,2]:
                ei=np.eye(3)[i]
                d=(np.outer(ei,n)+np.outer(n,ei)-2*n[i]*np.eye(3))/par.k0
                deriv.append(np.vdot(E,d@E).real)
        else:
            step=par.k0*1e-4;deriv=[]
            for i in [0,2]:
                delta=np.eye(3)[i]*step
                d=(electromagnetic_dispersion(kv+delta,p,par)-electromagnetic_dispersion(kv-delta,p,par))/(2*step)
                deriv.append(np.vdot(E,d@E).real)
        dw=par.omega*1e-5
        def freq(w):
            other=replace(par,frequency_hz=w/(2*np.pi),q=kx*c/w)
            return electromagnetic_dispersion(kv,p*(par.omega/w)**2,other,cold=cold)
        Domega=np.vdot(E,((freq(par.omega+dw)-freq(par.omega-dw))/(2*dw))@E).real
        Dp=electromagnetic_dispersion(kv,1.,par,cold=cold)-electromagnetic_dispersion(kv,0.,par,cold=cold)
        vg=-np.asarray(deriv)/Domega
        kdot=np.vdot(E,Dp@E).real*dp/Domega
        return vg,kdot,E,values[j]
    # Project the supplied boundary mode onto this dispersion surface.
    def residual(k):
        p,_=density_and_gradient(z_start)
        vals=np.linalg.eigvalsh(electromagnetic_dispersion([kx,0.,k[0]],p,par,cold=cold))
        return [vals[np.argmin(abs(vals))]]
    projected=root(residual,[kz_start],tol=1e-11)
    if abs(projected.fun[0])>1e-8:raise ArithmeticError('Incident ray projection failed')
    def rhs(t,y):
        vg,kdot,_,_=local(y[1],y[2])
        return [vg[0],vg[1],kdot]
    def lower(t,y):return y[1]-z_start+.1
    lower.terminal=True;lower.direction=-1
    def shortwave(t,y):return max_index-np.hypot(kx,y[2])/par.k0
    shortwave.terminal=True;shortwave.direction=-1
    def turning(t,y):return local(y[1],y[2])[0][1]
    turning.terminal=False;turning.direction=0
    sol=solve_ivp(rhs,[0,max_time_s],[0,z_start,projected.x[0]],rtol=2e-8,
                  atol=[1e-4,1e-4,1e-11],max_step=max_step_s,events=[lower,shortwave,turning],dense_output=True)
    if not sol.success:raise ArithmeticError(sol.message)
    time=np.unique(np.r_[np.linspace(0,sol.t[-1],1201),sol.t,
        *[np.linspace(max(0,t-2e-6),min(sol.t[-1],t+2e-6),401) for t in sol.t_events[2]]])
    state=sol.sol(time).T
    details=[local(z,k) for _,z,k in state]
    vg=np.array([v[0] for v in details]);E=np.array([v[2] for v in details])
    reason='returned to lower boundary' if len(sol.t_events[0]) else 'cold index limit; requires hot/full-wave continuation' if len(sol.t_events[1]) else 'time limit'
    return dict(time_s=time,horizontal_m=state[:,0],height_m=state[:,1],kz_m1=state[:,2],
                kx_m1=np.full_like(time,kx),group_velocity_m_s=vg,electric_polarization=E,
                dispersion_residual=np.array([v[3] for v in details]),turning_time_s=sol.t_events[2],stop_reason=reason)


def kinetic_group_velocity(par, density_ratio, kdown, step=.02):
    """[vg_x,vg_z] from Re epsilon=0, holding physical density fixed.

    Valid for a weakly damped electrostatic mode. With appreciable loss this
    real-dispersion derivative is not an exact transport velocity.
    """
    # A five-point stencil avoids cancellation in 1+zeta*Z(zeta) when the
    # horizontal group speed is small; tiny k increments amplify that error.
    def derivative(fn):
        return (-fn(2*step)+8*fn(step)-8*fn(-step)+fn(-2*step))/(12*step)
    dx = derivative(lambda shift: dielectric(kdown,density_ratio,replace(par,q=par.q+shift/par.k0)).real)
    dz = -derivative(lambda shift: dielectric(kdown+shift,density_ratio,par).real)
    dw = par.omega*1e-5
    def at_frequency(w):
        p = replace(par, frequency_hz=w/(2*np.pi), q=par.q*par.omega/w)
        return dielectric(kdown, density_ratio*(par.omega/w)**2, p).real
    dt = (at_frequency(par.omega+dw)-at_frequency(par.omega-dw))/(2*dw)
    return -np.array([dx, dz])/dt


def electrostatic_ray(z_m, density_ratio, kdown_m1, par, collision_alpha_m1=None):
    """Trace a downward ES segment; arrays ordered from high z to low z.

    Horizontal displacement dx/dz=vg_x/vg_z and dt/dz=1/vg_z.
    Spatial amplitude loss alpha integrates over vertical distance. Equivalently
    gamma=alpha*abs(vg_z) integrates over time; never use |vg| in its place.
    Stop before a vertical turning point or strong spatial damping.
    """
    z, p, kd = np.asarray(z_m), np.asarray(density_ratio), np.asarray(kdown_m1)
    if np.any(np.diff(z) >= 0):
        raise ValueError('Ray heights must descend')
    vg = np.array([kinetic_group_velocity(par, pp, kk.real) for pp, kk in zip(p, kd)])
    valid = (vg[:, 1] < 0) & (kd.imag/np.maximum(kd.real, 1e-30) < .02)
    invalid = np.flatnonzero(~valid)
    count = int(invalid[0]) if len(invalid) else len(z)
    if count < 2:
        raise ValueError('No weak-loss downward electrostatic ray segment')
    z, p, kd, vg = z[:count], p[:count], kd[:count], vg[:count]
    x = cumulative_trapezoid(vg[:, 0]/vg[:, 1], z, initial=0)
    time = cumulative_trapezoid(1/vg[:, 1], z, initial=0)
    # Path length is positive as z decreases.
    arc = cumulative_trapezoid(np.linalg.norm(vg, axis=1)/vg[:, 1], z, initial=0)
    ak = np.maximum(kd.imag, 0)
    ac = np.zeros_like(z) if collision_alpha_m1 is None else np.asarray(collision_alpha_m1)[:count]
    out = dict(height_m=z, horizontal_displacement_m=x, travel_time_s=time,
               path_length_m=arc, density_ratio=p, group_velocity_m_s=vg,
               wavevector_m1=np.column_stack([np.full_like(z, par.q*par.k0), -kd.real]),
               phase_angle_from_down_deg=np.rad2deg(np.arctan2(abs(par.q*par.k0), kd.real)),
               group_angle_from_down_deg=np.rad2deg(np.arctan2(abs(vg[:, 0]), -vg[:, 1])))
    for name, alpha in [('collisionless', ak), ('collisional', ac), ('combined', ak+ac)]:
        tau = -cumulative_trapezoid(alpha, z, initial=0)
        out[name+'_power_survival'] = np.exp(-2*tau)
        out[name+'_temporal_amplitude_loss_s1'] = alpha*abs(vg[:, 1])
    return out


def solution_ray(z, density_ratio, collision_s, par, table, match_z):
    """Ray diagnostic for a full-wave solution, in the same z coordinates."""
    from .conversion import matrix
    from .landau import bragg_match
    pp=table['density_ratio']; kk=table['kdown_m1']; xx=table['pressure_response']
    rise=np.flatnonzero((density_ratio[:-1]<pp[-1])&(density_ratio[1:]>=pp[-1]))
    if not len(rise):return None
    j=rise[0];top=float(np.interp(pp[-1],density_ratio[j:j+2],z[j:j+2]))
    bottom=max(float(z[0]),float(match_z)-20.)
    if top<=bottom:return None
    zz=np.linspace(top-1e-5,bottom,401);p=np.interp(zz,z,density_ratio)
    kd=PchipInterpolator(pp,kk.real)(p)+1j*PchipInterpolator(pp,kk.imag)(p)
    xi=PchipInterpolator(pp,xx.real)(p)
    nu=np.interp(zz,z,collision_s)
    eig=np.linalg.eigvals(matrix(p,np.zeros_like(p),nu/par.omega,par,xi))
    chosen=np.argmin(eig.imag,axis=1)
    alpha=np.maximum((1j*par.k0*eig[np.arange(len(zz)),chosen]).imag,0)
    result=electrostatic_ray(zz,p,kd,par,alpha)
    kr=4*np.pi*430e6/c;pb,_=bragg_match(par,kr)
    kb=np.sqrt(kr**2-(par.q*par.k0)**2)
    vg=kinetic_group_velocity(par,pb,kb)
    result['bragg_group_velocity_m_s']=vg
    result['bragg_phase_angle_deg']=abs(np.rad2deg(np.arctan2(par.q*par.k0,kb)))
    result['bragg_group_angle_deg']=np.rad2deg(np.arctan2(abs(vg[0]),-vg[1]))
    for name in ['horizontal_displacement_m','travel_time_s','path_length_m']:
        result[name+'_to_bragg']=float(np.interp(match_z,result['height_m'][::-1],result[name][::-1]))
    result['spitze_angle_deg']=spitze_angle_deg(par)
    return result
