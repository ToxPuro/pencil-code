#include "simulation_rng.h"
#include <random>

static std::mt19937 rng;

void
seed_rng(uint32_t seed)
{
    rng.seed(seed);
    // Some parts of astaroth still use srand, I think...
    srand(seed);
}

std::mt19937&
get_rng()
{
    return rng;
}

AcReal
random_uniform_real_01()
{
    std::uniform_real_distribution<AcReal> u01(0, 1);
    return u01(get_rng());
}
