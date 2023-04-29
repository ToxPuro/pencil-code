#pragma once
#include "errchk.h"

typedef struct {
    size_t count;
    double* data;
    bool on_device;
} Array;

static Array
arrayCreate(const size_t count, const bool on_device)
{
    Array a = (Array){
        .count     = count,
        .data      = NULL,
        .on_device = on_device,
    };

    const size_t bytes = count * sizeof(a.data[0]);
    if (on_device) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&a.data, bytes));
    }
    else {
        a.data = (double*)malloc(bytes);
        ERRCHK_ALWAYS(a.data);
    }

    return a;
}

static void
arrayDestroy(Array* a)
{
    if (a->on_device)
        cudaFree(a->data);
    else
        free(a->data);
    a->data  = NULL;
    a->count = 0;
}

/**
    Simple rng for doubles in range [0...1].
    Not suitable for generating full-precision f64 randoms.
*/
static double
randd(void)
{
    return (double)rand() / RAND_MAX;
}

static inline void
arrayRandomize(Array* a)
{
    if (!a->on_device) {
        for (size_t i = 0; i < a->count; ++i)
            a->data[i] = randd();
    }
    else {
        Array b = arrayCreate(a->count, false);
        arrayRandomize(&b);
        const size_t bytes = a->count * sizeof(b.data[0]);
        cudaMemcpy(a->data, b.data, bytes, cudaMemcpyHostToDevice);
        arrayDestroy(&b);
    }
}