#include "derivs.h"

vecvalue(v) {
    return real3(value(v.x), value(v.y), value(v.z))
}

gradient(s) {
    return real3(derx(s), dery(s), derz(s))
}

gradients(v) {
    return Matrix(gradient(v.x), gradient(v.y), gradient(v.z))
}

divergence(v) {
    return derx(v.x) + dery(v.y) + derz(v.z)
}

curl(v) {
    return real3(dery(v.z) - derz(v.y), derz(v.x) - derx(v.z), derx(v.y) - dery(v.x))
}

laplace(s) {
    return derxx(s) + deryy(s) + derzz(s)
}

veclaplace(v) {
    return real3(laplace(v.x), laplace(v.y), laplace(v.z))
}

stress_tensor(v) {
    Matrix S

    S.data[0][0] = (2.0 / 3.0) * derx(v.x) - (1.0 / 3.0) * (dery(v.y) + derz(v.z))
    S.data[0][1] = (1.0 / 2.0) * (dery(v.x) + derx(v.y))
    S.data[0][2] = (1.0 / 2.0) * (derz(v.x) + derx(v.z))

    S.data[1][0] = S.data[0][1]
    S.data[1][1] = (2.0 / 3.0) * dery(v.y) - (1.0 / 3.0) * (derx(v.x) + derz(v.z))
    S.data[1][2] = (1.0 / 2.0) * (derz(v.y) + dery(v.z))

    S.data[2][0] = S.data[0][2]
    S.data[2][1] = S.data[1][2]
    S.data[2][2] = (2.0 / 3.0) * derz(v.z) - (1.0 / 3.0) * (derx(v.x) + dery(v.y))

    return S
}

gradient_of_divergence(v) {
    return real3(
        derxx(v.x) + derxy(v.y) + derxz(v.z),
        derxy(v.x) + deryy(v.y) + deryz(v.z),
        derxz(v.x) + deryz(v.y) + derzz(v.z)
    )
}

contract(mat) {
    return dot(mat.row(0), mat.row(0)) +
           dot(mat.row(1), mat.row(1)) +
           dot(mat.row(2), mat.row(2))
}

