#pragma once
#include <astaroth.h>
#include <random>

// Low-level API
void seed_rng(uint32_t seed);
std::mt19937& get_rng();

// High-level API
AcReal random_uniform_real_01();
