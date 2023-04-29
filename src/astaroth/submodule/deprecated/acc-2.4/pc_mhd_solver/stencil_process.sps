#include "stencil_definition.sdh"

Vector
value(in VectorField uu)
{
    return (Vector){value(uu.x), value(uu.y), value(uu.z)};
}

#if LUPWD
Scalar
upwd_der6(in VectorField uu, in ScalarField lnrho)
{
    Scalar uux = fabs(value(uu).x);
    Scalar uuy = fabs(value(uu).y);
    Scalar uuz = fabs(value(uu).z);
    return (Scalar){uux * der6x_upwd(lnrho) + uuy * der6y_upwd(lnrho) + uuz * der6z_upwd(lnrho)};
}
#endif

Matrix
gradients(in VectorField uu)
{
    return (Matrix){gradient(uu.x), gradient(uu.y), gradient(uu.z)};
}

Scalar
continuity(in VectorField uu, in ScalarField lnrho)
{
    return -dot(value(uu), gradient(lnrho))
#if LUPWD
           // This is a corrective hyperdiffusion term for upwinding.
           + upwd_der6(uu, lnrho)
#endif
           - divergence(uu);
}

#if LENTROPY
Vector
momentum(in VectorField uu, in ScalarField lnrho, in ScalarField ss, in VectorField aa)
{
    const Matrix S   = stress_tensor(uu);
    const Scalar cs2 = AC_cs2_sound * exp(AC_gamma * value(ss) / AC_cp_sound +
                                          (AC_gamma - 1) * (value(lnrho) - AC_lnrho0));
    const Vector j   = (Scalar(1.) / AC_mu0) *
                     (gradient_of_divergence(aa) - laplace_vec(aa)); // Current density
    const Vector B = curl(aa);
    // TODO: DOES INTHERMAL VERSTION INCLUDE THE MAGNETIC FIELD?
    const Scalar inv_rho = Scalar(1.) / exp(value(lnrho));

    // Regex replace CPU constants with get\(AC_([a-zA-Z_0-9]*)\)
    // \1
    const Vector mom = -mul(gradients(uu), value(uu)) -
                       cs2 * ((Scalar(1.) / AC_cp_sound) * gradient(ss) + gradient(lnrho)) +
                       inv_rho * cross(j, B) +
                       AC_nu_visc *
                           (laplace_vec(uu) + Scalar(1. / 3.) * gradient_of_divergence(uu) +
                            Scalar(2.) * mul(S, gradient(lnrho))) +
                       AC_zeta * gradient_of_divergence(uu);
    return mom;
}
#elif LTEMPERATURE
Vector
momentum(in VectorField uu, in ScalarField lnrho, in ScalarField tt)
{
    Vector mom;

    const Matrix S = stress_tensor(uu);

    const Vector pressure_term = (AC_cp_sound - AC_cv_sound) *
                                 (gradient(tt) + value(tt) * gradient(lnrho));

    mom = -mul(gradients(uu), value(uu)) - pressure_term +
          AC_nu_visc * (laplace_vec(uu) + Scalar(1. / 3.) * gradient_of_divergence(uu) +
                        Scalar(2.) * mul(S, gradient(lnrho))) +
          AC_zeta * gradient_of_divergence(uu);

#if LGRAVITY
    mom = mom - (Vector){0, 0, -10.0};
#endif

    return mom;
}
#else
Vector
momentum(in VectorField uu, in ScalarField lnrho)
{
    Vector mom;

    const Matrix S = stress_tensor(uu);

    // Isothermal: we have constant speed of sound

    mom = -mul(gradients(uu), value(uu)) - AC_cs2_sound * gradient(lnrho) +
          AC_nu_visc * (laplace_vec(uu) + Scalar(1. / 3.) * gradient_of_divergence(uu) +
                        Scalar(2.) * mul(S, gradient(lnrho))) +
          AC_zeta * gradient_of_divergence(uu);

#if LGRAVITY
    mom = mom - (Vector){0, 0, -10.0};
#endif

    return mom;
}
#endif

Vector
induction(in VectorField uu, in VectorField aa)
{
    // Note: We do (-nabla^2 A + nabla(nabla dot A)) instead of (nabla x (nabla
    // x A)) in order to avoid taking the first derivative twice (did the math,
    // yes this actually works. See pg.28 in arXiv:astro-ph/0109497)
    // u cross B - AC_eta * AC_mu0 * (AC_mu0^-1 * [- laplace A + grad div A ])
    const Vector B        = curl(aa);
    const Vector grad_div = gradient_of_divergence(aa);
    const Vector lap      = laplace_vec(aa);

    // Note, AC_mu0 is cancelled out
    const Vector ind = cross(value(uu), B) - AC_eta * (grad_div - lap);

    return ind;
}

#if LENTROPY
Scalar
lnT(in ScalarField ss, in ScalarField lnrho)
{
    const Scalar lnT = AC_lnT0 + AC_gamma * value(ss) / AC_cp_sound +
                       (AC_gamma - Scalar(1.)) * (value(lnrho) - AC_lnrho0);
    return lnT;
}

// Nabla dot (K nabla T) / (rho T)
Scalar
heat_conduction(in ScalarField ss, in ScalarField lnrho)
{
    const Scalar inv_AC_cp_sound = AcReal(1.) / AC_cp_sound;

    const Vector grad_ln_chi = -gradient(lnrho);

    const Scalar first_term = AC_gamma * inv_AC_cp_sound * laplace(ss) +
                              (AC_gamma - AcReal(1.)) * laplace(lnrho);
    const Vector second_term = AC_gamma * inv_AC_cp_sound * gradient(ss) +
                               (AC_gamma - AcReal(1.)) * gradient(lnrho);
    const Vector third_term = AC_gamma * (inv_AC_cp_sound * gradient(ss) + gradient(lnrho)) +
                              grad_ln_chi;

    const Scalar chi = AC_THERMAL_CONDUCTIVITY / (exp(value(lnrho)) * AC_cp_sound);
    return AC_cp_sound * chi * (first_term + dot(second_term, third_term));
}

Scalar
heating(const int i, const int j, const int k)
{
    return 1;
}

Scalar
entropy(in ScalarField ss, in VectorField uu, in ScalarField lnrho, in VectorField aa)
{
    const Matrix S      = stress_tensor(uu);
    const Scalar inv_pT = Scalar(1.) / (exp(value(lnrho)) * exp(lnT(ss, lnrho)));
    const Vector j      = (Scalar(1.) / AC_mu0) *
                     (gradient_of_divergence(aa) - laplace_vec(aa)); // Current density
    const Scalar RHS = H_CONST - C_CONST + AC_eta * (AC_mu0)*dot(j, j) +
                       Scalar(2.) * exp(value(lnrho)) * AC_nu_visc * contract(S) +
                       AC_zeta * exp(value(lnrho)) * divergence(uu) * divergence(uu);

    return -dot(value(uu), gradient(ss)) + inv_pT * RHS + heat_conduction(ss, lnrho);
}
#endif

#if LTEMPERATURE
Scalar
heat_transfer(in VectorField uu, in ScalarField lnrho, in ScalarField tt)
{
    const Matrix S                  = stress_tensor(uu);
    const Scalar heat_diffusivity_k = 0.0008; // 8e-4;
    return -dot(value(uu), gradient(tt)) + heat_diffusivity_k * laplace(tt) +
           heat_diffusivity_k * dot(gradient(lnrho), gradient(tt)) +
           AC_nu_visc * contract(S) * (Scalar(1.) / AC_cv_sound) -
           (AC_gamma - 1) * value(tt) * divergence(uu);
}
#endif

#if LFORCING
Vector
simple_vortex_forcing(Vector a, Vector b, Scalar magnitude)
{
    return magnitude * cross(normalized(b - a), (Vector){0, 0, 1}); // Vortex
}

Vector
simple_outward_flow_forcing(Vector a, Vector b, Scalar magnitude)
{
    return magnitude * (1 / length(b - a)) * normalized(b - a); // Outward flow
}

// The Pencil Code forcing_hel_noshear(), manual Eq. 222, inspired forcing function with adjustable
// helicity
Vector
helical_forcing(Scalar magnitude, Vector k_force, Vector xx, Vector ff_re, Vector ff_im, Scalar phi)
{
    // JP: This looks wrong:
    //    1) Should it be AC_dsx * AC_nx instead of AC_dsx * AC_ny?
    //    2) Should you also use globalGrid.n instead of the local n?
    //    MV: You are rigth. Made a quickfix. I did not see the error  because multigpu is split
    //        in z direction not y direction.
    //    3) Also final point: can we do this with vectors/quaternions instead?
    //       Tringonometric functions are much more expensive and inaccurate/
    //    MV: Good idea. No an immediate priority.
    // Fun related article:
    // https://randomascii.wordpress.com/2014/10/09/intel-underestimates-error-bounds-by-1-3-quintillion/
    xx.x = xx.x * (2.0 * M_PI / (AC_dsx * globalGridN.x));
    xx.y = xx.y * (2.0 * M_PI / (AC_dsy * globalGridN.y));
    xx.z = xx.z * (2.0 * M_PI / (AC_dsz * globalGridN.z));

    Scalar cos_phi     = cos(phi);
    Scalar sin_phi     = sin(phi);
    Scalar cos_k_dot_x = cos(dot(k_force, xx));
    Scalar sin_k_dot_x = sin(dot(k_force, xx));
    // Phase affect only the x-component
    // Scalar real_comp       = cos_k_dot_x;
    // Scalar imag_comp       = sin_k_dot_x;
    Scalar real_comp_phase = cos_k_dot_x * cos_phi - sin_k_dot_x * sin_phi;
    Scalar imag_comp_phase = cos_k_dot_x * sin_phi + sin_k_dot_x * cos_phi;

    Vector force = (Vector){ff_re.x * real_comp_phase - ff_im.x * imag_comp_phase,
                            ff_re.y * real_comp_phase - ff_im.y * imag_comp_phase,
                            ff_re.z * real_comp_phase - ff_im.z * imag_comp_phase};

    return force;
}

// PC-style helical forcing with support for profiles
Vector
forcing(int3 vertexIdx, int3 globalVertexIdx, Scalar dt, ScalarArray profx_ampl, ScalarArray profy_ampl,
        ScalarArray profz_ampl, ScalarArray profx_hel, ScalarArray profy_hel, ScalarArray profz_hel)
{
    Vector pos = (Vector){(globalVertexIdx.x - DCONST(AC_nx_min)) * AC_dsx,
                          (globalVertexIdx.y - DCONST(AC_ny_min)) * AC_dsy,
                          (globalVertexIdx.z - DCONST(AC_nz_min)) * AC_dsz};

    Complex fx = AC_fact * exp(Complex(0, AC_kk.x * AC_k1_ff * pos.z + AC_phase));
    Complex fy = exp(Complex(0, AC_kk.y * AC_k1_ff * pos.y));

    Complex fz;
    if (AC_iforcing_zsym == 0) {
        fz = exp(Complex(0., AC_kk.z * AC_k1_ff * pos.z));
    }
    else if (AC_iforcing_zsym == 1) {
        fz = Complex(cos(AC_kk.z * AC_k1_ff * pos.z), 0);
    }
    else if (AC_iforcing_zsym == -1) {
        fz = Complex(sin(AC_kk.z * AC_k1_ff * pos.z), 0);
    }
    else {
        // Failure
    }

    Complex fxyz = fx * fy * fz;

    // TODO recheck indices
    Scalar force_ampl    = profx_ampl[vertexIdx.x - NGHOST] * profy_ampl[vertexIdx.y] * profz_ampl[vertexIdx.z];
    Scalar prof_hel_ampl = profx_hel[vertexIdx.x - NGHOST] * profy_hel[vertexIdx.y] *
                           profz_hel[vertexIdx.z];

    return force_ampl * AC_fda * (Complex(AC_coef1.x, prof_hel_ampl * AC_coef2.x) * fxyz).x;
}
#endif // LFORCING

// Declare input and output arrays using locations specified in the
// array enum in astaroth.h
in ScalarField lnrho(VTXBUF_LNRHO);
out ScalarField out_lnrho(VTXBUF_LNRHO);

in VectorField uu(VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
out VectorField out_uu(VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);

#if LMAGNETIC
in VectorField aa(VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ);
out VectorField out_aa(VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ);
#endif

#if LENTROPY
in ScalarField ss(VTXBUF_ENTROPY);
out ScalarField out_ss(VTXBUF_ENTROPY);
#endif

#if LTEMPERATURE
in ScalarField tt(VTXBUF_TEMPERATURE);
out ScalarField out_tt(VTXBUF_TEMPERATURE);
#endif

Kernel void
solve()
{
    Scalar dt = AC_dt;
    out_lnrho = rk3(out_lnrho, lnrho, continuity(uu, lnrho), dt);

#if LMAGNETIC
    out_aa = rk3(out_aa, aa, induction(uu, aa), dt);
#endif

#if LENTROPY
    out_uu = rk3(out_uu, uu, momentum(uu, lnrho, ss, aa), dt);
    out_ss = rk3(out_ss, ss, entropy(ss, uu, lnrho, aa), dt);
#elif LTEMPERATURE
    out_uu = rk3(out_uu, uu, momentum(uu, lnrho, tt), dt);
    out_tt = rk3(out_tt, tt, heat_transfer(uu, lnrho, tt), dt);
#else
    out_uu = rk3(out_uu, uu, momentum(uu, lnrho), dt);
#endif

#if LFORCING
    if (step_number == 2) {
        out_uu = out_uu + forcing(vertexIdx, globalVertexIdx, dt, AC_profx_ampl, AC_profy_ampl, AC_profz_ampl,
                                  AC_profx_hel, AC_profy_hel, AC_profz_hel);
    }
#endif
}
