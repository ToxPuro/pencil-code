int AC_step_number
real AC_dt

hostdefine RK_ORDER (3)

rk3(s0, s1, roc) {
#if RK_ORDER == 1
    // Euler
    real alpha= 0, 0.0, 0.0, 0.0
    real beta = 0, 1.0, 0.0, 0.0
#elif RK_ORDER == 2
    real alpha= 0,     0.0, -1.0/2.0, 0.0
    real beta = 0, 1.0/2.0,      1.0, 0.0
#elif RK_ORDER == 3
    real alpha = 0., -5./9., -153. / 128.
    real beta = 1. / 3., 15./ 16., 8. / 15.
#endif
    /*
    // This conditional has abysmal performance on AMD for some reason, better performance on NVIDIA than the workaround below
    if AC_step_number > 0 {
        return s1 + beta[AC_step_number] * ((alpha[AC_step_number] / beta[AC_step_number - 1]) * (s1 - s0) + roc * AC_dt)
    } else {
        return s1 + beta[AC_step_number] * roc * AC_dt
    }
    */
    // Workaround
    return s1 + beta[AC_step_number + 1] * ((alpha[AC_step_number] / beta[AC_step_number]) * (s1 - s0) + roc * AC_dt)
}
