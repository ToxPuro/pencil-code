#define LMAGNETIC (1)
#define LENTROPY (1)


// Declare uniforms (i.e. device constants)
uniform Scalar cs2_sound;
uniform Scalar nu_visc;
uniform Scalar cp_sound;
uniform Scalar mu0;
uniform Scalar eta;
uniform Scalar gamma;
uniform Scalar chi;
uniform Scalar zeta;

uniform int nx_min;
uniform int ny_min;
uniform int nz_min;
uniform int nx;
uniform int ny;
uniform int nz;

uniform Scalar xorig;
uniform Scalar yorig;
uniform Scalar zorig;

//Star position
uniform Scalar star_pos_x;
uniform Scalar star_pos_z;
uniform Scalar GM_star;

//Needed for gravity
uniform Scalar dsx;
uniform Scalar dsy;
uniform Scalar dsz;
uniform Scalar inv_dsx;
uniform Scalar inv_dsy;
uniform Scalar inv_dsz;

Scalar
distance_x(Vector a, Vector b)
{
    return sqrt(dot(a-b, a-b));
}

Vector
value(in VectorField uu)
{
    return (Vector){value(uu.x), value(uu.y), value(uu.z)};
}

Matrix
gradients(in VectorField uu)
{
    return (Matrix){gradient(uu.x), gradient(uu.y), gradient(uu.z)};
}

Scalar
continuity(in VectorField uu, in ScalarField lnrho) {
    return -dot(value(uu), gradient(lnrho)) - divergence(uu);
}

// Gravitation for in negative x-direction.
Vector
grav_force_line(const int3 vertexIdx)
{
    Vector vertex_pos = (Vector){dsx * vertexIdx.x - xorig, dsy * vertexIdx.y - yorig, dsz * vertexIdx.z - zorig};
    Vector star_pos   = (Vector){star_pos_x,                dsy * vertexIdx.y - yorig, dsz * vertexIdx.z - zorig};

    const Scalar RR = vertex_pos.x - star_pos.x;

    const Scalar G_force_abs = GM_star / (RR*RR); // Force per unit mass;

    Vector G_force = (Vector){ - G_force_abs,
                                 AcReal(0.0),
                                 AcReal(0.0)};

    return G_force;
}

#if LENTROPY
Vector
momentum(in VectorField uu, in ScalarField lnrho, in ScalarField ss, in VectorField aa, const int3 vertexIdx) {
  Vector mom;

  const Matrix S = stress_tensor(uu);

  mom = -mul(gradients(uu), value(uu)) -
    cs2_sound * gradient(lnrho) +
    nu_visc *
    (laplace_vec(uu) + Scalar(1. / 3.) * gradient_of_divergence(uu) +
      Scalar(2.) * mul(S, gradient(lnrho))) + zeta * gradient_of_divergence(uu);

  mom = mom - cs2_sound * (Scalar(1.) / cp_sound) * gradient(ss);

  const Vector grad_div = gradient_of_divergence(aa);
  const Vector lap = laplace_vec(aa);
  const Vector j = (Scalar(1.) / mu0) * (grad_div - lap);
  const Vector B = curl(aa);
  mom = mom + (Scalar(1.) / exp(value(lnrho))) * cross(j, B);

  mom = mom + grav_force_line(vertexIdx);

  return mom;
}
#else
Vector
momentum(in VectorField uu, in ScalarField lnrho, const int3 vertexIdx) {
  Vector mom;

  const Matrix S = stress_tensor(uu);

  mom = -mul(gradients(uu), value(uu)) -
    cs2_sound * gradient(lnrho) +
    nu_visc *
    (laplace_vec(uu) + Scalar(1. / 3.) * gradient_of_divergence(uu) +
      Scalar(2.) * mul(S, gradient(lnrho))) + zeta * gradient_of_divergence(uu);

  mom = mom + grav_force_line(vertexIdx);

  return mom;
}
#endif


Vector
induction(in VectorField uu, in VectorField aa) {
  // Note: We do (-nabla^2 A + nabla(nabla dot A)) instead of (nabla x (nabla
  // x A)) in order to avoid taking the first derivative twice (did the math,
  // yes this actually works. See pg.28 in arXiv:astro-ph/0109497)
  // u cross B - ETA * mu0 * (mu0^-1 * [- laplace A + grad div A ])
  const Vector B = curl(aa);
  const Vector grad_div = gradient_of_divergence(aa);
  const Vector lap = laplace_vec(aa);

  // Note, mu0 is cancelled out
  const Vector ind = cross(value(uu), B) - eta * (grad_div - lap);

  return ind;
}


#if LENTROPY
Scalar
lnT( in ScalarField ss, in ScalarField lnrho) {
  const Scalar lnT = LNT0 + value(ss) / cp_sound +
    (gamma - AcReal(1.)) * (value(lnrho) - LNRHO0);
  return lnT;
}

// Nabla dot (K nabla T) / (rho T)
Scalar
heat_conduction( in ScalarField ss, in ScalarField lnrho) {
  const Scalar inv_cp_sound = AcReal(1.) / cp_sound;

  const Vector grad_ln_chi = (Vector) {
    0,
    0,
    0
  }; // TODO not used

  const Scalar first_term = gamma * inv_cp_sound * laplace(ss) +
    (gamma - AcReal(1.)) * laplace(lnrho);
  const Vector second_term = gamma * inv_cp_sound * gradient(ss) +
    (gamma - AcReal(1.)) * gradient(lnrho);
  const Vector third_term = gamma * (inv_cp_sound * gradient(ss) +
    gradient(lnrho)) + grad_ln_chi;

  return cp_sound * chi * (first_term + dot(second_term, third_term));
}

Scalar
heating(const int i, const int j, const int k) {
  return 1;
}

Scalar
entropy(in ScalarField ss, in VectorField uu, in ScalarField lnrho, in VectorField aa) {
    const Matrix S = stress_tensor(uu);

    // nabla x nabla x A / mu0 = nabla(nabla dot A) - nabla^2(A)
    const Vector j = gradient_of_divergence(aa) - laplace_vec(aa);

    const Scalar inv_pT = AcReal(1.) / (exp(value(lnrho)) + exp(lnT(ss, lnrho)));

    return -dot(value(uu), gradient(ss)) +
      inv_pT * (H_CONST - C_CONST +
        eta * mu0 * dot(j, j) +
        AcReal(2.) * exp(value(lnrho)) * nu_visc * contract(S) +
        zeta * exp(value(lnrho)) * divergence(uu) * divergence(uu)
      ) + heat_conduction(ss, lnrho);
}
#endif

// Declare input and output arrays using locations specified in the
// array enum in astaroth.h
in ScalarField lnrho(VTXBUF_LNRHO);
out ScalarField out_lnrho(VTXBUF_LNRHO);

in VectorField uu(VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ);
out VectorField out_uu(VTXBUF_UUX,VTXBUF_UUY,VTXBUF_UUZ);


#if LMAGNETIC
in VectorField aa(VTXBUF_AX,VTXBUF_AY,VTXBUF_AZ);
out VectorField out_aa(VTXBUF_AX,VTXBUF_AY,VTXBUF_AZ);
#endif

#if LENTROPY
in ScalarField ss(VTXBUF_ENTROPY);
out ScalarField out_ss(VTXBUF_ENTROPY);
#endif

Kernel void
solve(Scalar dt) {
    WRITE(out_lnrho, RK3(out_lnrho, lnrho, continuity(uu, lnrho), dt));

    #if LMAGNETIC
        WRITE(out_aa,    RK3(out_aa, aa, induction(uu, aa), dt));
    #endif


    #if LENTROPY
        WRITE(out_uu, RK3(out_uu, uu, momentum(uu, lnrho, ss, aa, vertexIdx), dt));
        WRITE(out_ss,    RK3(out_ss, ss, entropy(ss, uu, lnrho, aa), dt));
    #else
        WRITE(out_uu, RK3(out_uu, uu, momentum(uu, lnrho, vertexIdx), dt));
    #endif
}
