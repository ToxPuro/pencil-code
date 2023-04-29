// clang-format off
/**
  Code generator for unrolling and reordering memory accesses

  Key structures:
    
      stencils

      int stencils_accessed[kernel][field][stencil]: Set if `stencil` accessed for `field` in `kernel`
      char* stencils[stencil][depth][height][width]: contains the expression to compute the stencil coefficient

      char* stencil_unary_ops[stencil]: contains the function name of the unary operation used to process `stencil`
      char* stencil_binary_ops[stencil]: contains the function name of the binary operation used to process `stencil`

      A stencil is defined (formally) as

        f: R -> R       | Map operator
        g: R^{|s|} -> R | Reduce operator
        p: stencil points
        w: stencil element weights (coefficients)

        f(p_i) = ...
        s_i = w_i f(p_i)
        g(s) = ...

        For example for an ordinary stencil
        f(p_i) = p_i
        g(s) = sum_{i=1}^{|s|} s_i = sum s_i

      Alternatively by recursion
        G(p_0) = w_i f(p_0)
        G(p_i) = g(w_i f(p_i), G(p_{i-1}))

      Could also simplify notation by incorporating w into f

      CS view:
        res = f(p[0])
        for i in 1,len(p):
          res = g(f(p[i]), res)
*/
// clang-format on
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "user_defines.h"

#include "stencil_accesses.h"
#include "stencilgen.h"

#include "implementation.h"

void
raise_error(const char* str)
{
  // Make sure the error does not go unnoticed
  //
  // It is not clear how the CMake building process
  // could be stopped if a part of code generation
  // fails but an infinite loop is an easy and
  // effective way to inform the user something went wrong
  while (1)
    fprintf(stderr, "FATAL ERROR: %s\n", str);
  exit(EXIT_FAILURE);
}

void
gen_stencil_definitions(void)
{
  if (!NUM_FIELDS)
    raise_error("Must declare at least one Field in the DSL code!");

  if (!NUM_STENCILS)
    raise_error("Must declare at least one Stencil in the DSL code!");

  printf(
      "static __device__ /*const*/ AcReal /*__restrict__*/ "
      "stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH]={");
  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("{");
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      printf("{");
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        printf("{");
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          printf("%s,", stencils[stencil][depth][height][width]
                            ? stencils[stencil][depth][height][width]
                            : "0");
        }
        printf("},");
      }
      printf("},");
    }
    printf("},");
  }
  printf("};");
}

void
gen_kernel_prefix(void)
{
  printf("const int3 vertexIdx = (int3){"
         "threadIdx.x + blockIdx.x * blockDim.x + start.x,"
         "threadIdx.y + blockIdx.y * blockDim.y + start.y,"
         "threadIdx.z + blockIdx.z * blockDim.z + start.z,"
         "};");
  printf("const int3 globalVertexIdx = (int3){"
         "d_multigpu_offset.x + vertexIdx.x,"
         "d_multigpu_offset.y + vertexIdx.y,"
         "d_multigpu_offset.z + vertexIdx.z,"
         "};");
  printf("const int3 globalGridN = d_mesh_info.int3_params[AC_global_grid_n];");
  printf("const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);");

  printf("(void)globalVertexIdx;"); // Silence unused warning
  printf("(void)globalGridN;");     // Silence unused warning

  // Read vba.out
#if 0
  // Original (compute when needed)
  // SINGLEPASS_INTEGRATION=ON, 4.97 ms (full step, 128^3)
  // SINGLEPASS_INTEGRATION=OFF, 6.09 ms (full step, 128^3)
  printf("const auto previous __attribute__((unused)) =[&](const Field field)"
         "{ return vba.out[field][idx]; };");
#else
  // Prefetch output fields
  // SINGLEPASS_INTEGRATION=ON, 4.18 ms (full step, 128^3)
  // SINGLEPASS_INTEGRATION=OFF, 4.77 ms (full step, 128^3)
  for (int field = 0; field < NUM_FIELDS; ++field)
    printf("const auto f%d_prev = vba.out[%d][idx];", field, field);

  printf("const auto previous __attribute__((unused)) = [&](const Field field)"
         "{ switch (field) {");
  for (int field = 0; field < NUM_FIELDS; ++field)
    printf("case %d: { return f%d_prev; }", field, field);

  printf("default: return (AcReal)NAN;"
         "}");
  printf("};");
#endif

// Write vba.out
#if 1
  // Original
  printf("const auto write=[&](const Field field, const AcReal value)"
         "{ vba.out[field][idx] = value; };");
#else
  // Buffered, no effect on performance
  // !Remember to emit write insructions in ac.y if this is enabled!
  printf("AcReal out_buffer[NUM_FIELDS];");
  for (int field = 0; field < NUM_FIELDS; ++field)
    printf("out_buffer[%d] = (AcReal)NAN;", field);

  printf("const auto write=[&](const Field field, const AcReal value)"
         "{ out_buffer[field] = value; };");
/*
for (int field = 0; field < NUM_FIELDS; ++field)
printf("vba.out[%d][idx] = out_buffer[%d];", field, field);
*/
#endif
}

void
gen_kernel_prefix_with_boundcheck(void)
{
  gen_kernel_prefix();
  printf("if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || "
         "vertexIdx.z >= end.z) { return; }");
}

void
gen_stencil_accesses(void)
{
  gen_kernel_prefix_with_boundcheck();
  printf(
      "AcReal /*__restrict__*/ processed_stencils[NUM_FIELDS][NUM_STENCILS];");

  for (size_t i = 0; i < NUM_STENCILS; ++i)
    printf("const auto %s=[&](const auto field)"
           "{stencils_accessed[field][stencil_%s]=1;return AcReal(1.0);};",
           stencil_names[i], stencil_names[i]);
}

#if IMPLEMENTATION == ORIGINAL
// Original
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix_with_boundcheck();
  printf(
      "AcReal /*__restrict__*/ processed_stencils[NUM_FIELDS][NUM_STENCILS];");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
      for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
        for (int height = 0; height < STENCIL_HEIGHT; ++height) {
          for (int width = 0; width < STENCIL_WIDTH; ++width) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              printf("processed_stencils[%d][%d] = ", field, stencil);
              if (!stencil_initialized[field][stencil]) {
                printf("%s(stencils[%d][%d][%d][%d]*"
                       "vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                       "vertexIdx.z+(%d))]);",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, field, -STENCIL_ORDER / 2 + width,
                       -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf( //
                    "%s(processed_stencils[%d][%d],%s(stencils[%d][%d][%d][%d]*"
                    "vba.in[%d][IDX(vertexIdx.x+(%d)"
                    ",vertexIdx.y+(%d),vertexIdx.z+(%d))]));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil], stencil, depth, height, width,
                    field, -STENCIL_ORDER / 2 + width,
                    -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);
              }
            }
          }
        }
      }
    }
  }

  for (size_t i = 0; i < NUM_STENCILS; ++i)
    printf("const auto %s __attribute__((unused)) =[&](const auto field)"
           "{return processed_stencils[field][stencil_%s];};",
           stencil_names[i], stencil_names[i]);
}
#elif IMPLEMENTATION == ORIGINAL_WITH_ILP
// Original + improved ILP (field-stencil to inner loop)
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix_with_boundcheck();
  printf(
      "AcReal /*__restrict__*/ processed_stencils[NUM_FIELDS][NUM_STENCILS];");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int field = 0; field < NUM_FIELDS; ++field) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              printf("processed_stencils[%d][%d] = ", field, stencil);
              if (!stencil_initialized[field][stencil]) {
                printf("%s(stencils[%d][%d][%d][%d]*"
                       "vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                       "vertexIdx.z+(%d))]);",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, field, -STENCIL_ORDER / 2 + width,
                       -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf( //
                    "%s(processed_stencils[%d][%d],%s(stencils[%d][%d][%d][%d]*"
                    "vba.in[%d][IDX(vertexIdx.x+(%d)"
                    ",vertexIdx.y+(%d),vertexIdx.z+(%d))]));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil], stencil, depth, height, width,
                    field, -STENCIL_ORDER / 2 + width,
                    -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);
              }
            }
          }
        }
      }
    }
  }

  for (size_t i = 0; i < NUM_STENCILS; ++i)
    printf("const auto %s __attribute__((unused)) =[&](const auto field)"
           "{return processed_stencils[field][stencil_%s];};",
           stencil_names[i], stencil_names[i]);
}
#elif IMPLEMENTATION == EXPL_REG_VARS
// Explicit register variables instead of a processed_stencils array
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix_with_boundcheck();

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int field = 0; field < NUM_FIELDS; ++field) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("%s(stencils[%d][%d][%d][%d]*"
                       "vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                       "vertexIdx.z+(%d))]);",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, field, -STENCIL_ORDER / 2 + width,
                       -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf( //
                    "%s(f%d_s%d,%s(stencils[%d][%d][%d][%d]*"
                    "vba.in[%d][IDX(vertexIdx.x+(%d)"
                    ",vertexIdx.y+(%d),vertexIdx.z+(%d))]));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil], stencil, depth, height, width,
                    field, -STENCIL_ORDER / 2 + width,
                    -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);
              }
            }
          }
        }
      }
    }
  }

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#elif IMPLEMENTATION == FULLY_EXPL_REG_VARS
// Everything prefetched
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix_with_boundcheck();

  // Prefetch stencil elements to local memory
  int cell_initialized[NUM_FIELDS][STENCIL_DEPTH][STENCIL_HEIGHT]
                      [STENCIL_WIDTH] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width] &&
                !cell_initialized[field][depth][height][width]) {
              printf("const auto f%d_%d_%d_%d = ", //
                     field, depth, height, width);
              printf("__ldg(&vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                     "vertexIdx.z+(%d))]);",
                     field, -STENCIL_ORDER / 2 + width,
                     -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

              cell_initialized[field][depth][height][width] = 1;
            }
          }
        }
      }
    }
  }

  // Prefetch stencil coefficients to local memory
  int coeff_initialized[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT]
                       [STENCIL_WIDTH] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

          int stencil_accessed = 0;
          for (int field = 0; field < NUM_FIELDS; ++field)
            stencil_accessed |= stencils_accessed[curr_kernel][field][stencil];
          if (!stencil_accessed)
            continue;

          if (stencils[stencil][depth][height][width] &&
              !coeff_initialized[stencil][depth][height][width]) {
            printf("const auto s%d_%d_%d_%d = ", //
                   stencil, depth, height, width);

            // CT const
            // printf("%s;", stencils[stencil][depth][height][width]);
            printf("stencils[%d][%d][%d][%d];", stencil, depth, height, width);

            coeff_initialized[stencil][depth][height][width] = 1;
          }
        }
      }
    }
  }

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int field = 0; field < NUM_FIELDS; ++field) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("%s(s%d_%d_%d_%d*"
                       "f%d_%d_%d_%d);",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, field, depth, height, width);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf( //
                    "%s(f%d_s%d,%s(s%d_%d_%d_%d*"
                    "f%d_%d_%d_%d));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil], stencil, depth, height, width,
                    field, depth, height, width);
              }
            }
          }
        }
      }
    }
  }

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#elif IMPLEMENTATION == EXPL_REG_VARS_AND_CT_CONST_STENCILS
// Explicit register variables & compile-time constant stencils
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix_with_boundcheck();

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int field = 0; field < NUM_FIELDS; ++field) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("%s((%s)*"
                       "vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                       "vertexIdx.z+(%d))]);",
                       stencil_unary_ops[stencil],
                       stencils[stencil][depth][height][width], field,
                       -STENCIL_ORDER / 2 + width, -STENCIL_ORDER / 2 + height,
                       -STENCIL_ORDER / 2 + depth);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf( //
                    "%s(f%d_s%d,%s((%s)*"
                    "vba.in[%d][IDX(vertexIdx.x+(%d)"
                    ",vertexIdx.y+(%d),vertexIdx.z+(%d))]));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil],
                    stencils[stencil][depth][height][width], field,
                    -STENCIL_ORDER / 2 + width, -STENCIL_ORDER / 2 + height,
                    -STENCIL_ORDER / 2 + depth);
              }
            }
          }
        }
      }
    }
  }

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#elif IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS
// Vectorized loads to local memory. Strict alignment requirements.
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix(); // Note no bounds check
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sz = blockDim.z + STENCIL_DEPTH - 1;");
  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x + "
         "threadIdx.z * blockDim.x * blockDim.y;");

  printf("const int veclen = %s;", veclen_str);
  printf("const int bx = sx / veclen;"); // Vectorized block dimensions
  printf("const int by = sy;");
  printf("const int bz = sz;");
  printf("const int tpb = blockDim.x * blockDim.y * blockDim.z;");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {

    printf("for (int curr = sid; curr < bx * by * bz; curr += tpb) {");
    printf("const int i = curr %% bx;");
    printf("const int j = (curr %% (bx * by)) / bx;");
    printf("const int k = curr / (bx * by);");
    printf("reinterpret_cast<%s%s*>("
           "&smem[j * sx + k * sx * sy])[i] = ",
           realtype, veclen_str);
    // clang-format off
    printf("reinterpret_cast<%s%s*>(&vba.in[%d][IDX(baseIdx.x, baseIdx.y + j, baseIdx.z + k)])[i];", realtype, veclen_str, field);
    // clang-format on
    printf("}");
    printf("__syncthreads();");

    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("%s(stencils[%d][%d][%d][%d]*",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width);
                printf("smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "(threadIdx.z + %d) * sx * sy]);",
                       width, height, depth);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d,%s(stencils[%d][%d][%d][%d]*",
                       stencil_binary_ops[stencil], field, stencil,
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width);
                printf("smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "(threadIdx.z + %d) * sx * sy]));",
                       width, height, depth);
              }
            }
          }
        }
      }
    }
    printf("__syncthreads();");
  }

  printf("if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || "
         "vertexIdx.z >= end.z) { return; }");

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#elif IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_PINGPONG
// Vectorized loads to local memory. Strict alignment requirements.
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix();
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sz = blockDim.z + STENCIL_DEPTH - 1;");
  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x + "
         "threadIdx.z * blockDim.x * blockDim.y;");

  printf("const int veclen = %s;", veclen_str);
  printf("const int bx = sx / veclen;"); // Vectorized block dimensions
  printf("const int by = sy;");
  printf("const int bz = sz;");
  printf("const int tpb = blockDim.x * blockDim.y * blockDim.z;");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};

  printf("for (int curr = sid; curr < bx * by * bz; curr += tpb) {");
  printf("const int i = curr %% bx;");
  printf("const int j = (curr %% (bx * by)) / bx;");
  printf("const int k = curr / (bx * by);");
  printf("reinterpret_cast<%s%s*>(&smem[j * sx + k * sx * sy])[i] = ", realtype,
         veclen_str);
  // clang-format off
  printf("reinterpret_cast<%s%s*>(&vba.in[%d][IDX(baseIdx.x, baseIdx.y + j, baseIdx.z + k)])[i];", realtype, veclen_str, 0);
  // clang-format on
  printf("}");

  for (int field = 0; field < NUM_FIELDS; ++field) {

    printf("__syncthreads();");
    printf("for (int curr = sid; curr < bx * by * bz; curr += tpb) {");
    printf("const int i = curr %% bx;");
    printf("const int j = (curr %% (bx * by)) / bx;");
    printf("const int k = curr / (bx * by);");
    printf("reinterpret_cast<%s%s*>("
           "&smem[j * sx + k * sx * sy + %d * sx * sy * sz])[i] = ",
           realtype, veclen_str, (field + 1) % 2);
    // clang-format off
    printf("reinterpret_cast<%s%s*>(&vba.in[%d][IDX(baseIdx.x, baseIdx.y + j, baseIdx.z + k)])[i];", realtype, veclen_str, field+1);
    // clang-format on
    printf("}");

    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("%s(stencils[%d][%d][%d][%d]*",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width);
                printf("smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "(threadIdx.z + %d) * sx * sy +"
                       "%d * sx * sy *sz]);",
                       width, height, depth, field % 2);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d,%s(stencils[%d][%d][%d][%d]*",
                       stencil_binary_ops[stencil], field, stencil,
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width);
                printf("smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "(threadIdx.z + %d) * sx * sy +"
                       "%d * sx * sy *sz]));",
                       width, height, depth, field % 2);
              }
            }
          }
        }
      }
    }
  }

  printf("if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || "
         "vertexIdx.z >= end.z) { return; }");

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#elif IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_FULL
// Vectorized loads to local memory. Strict alignment requirements.
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix();
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sz = blockDim.z + STENCIL_DEPTH - 1;");
  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x + "
         "threadIdx.z * blockDim.x * blockDim.y;");

  printf("const int veclen = %s;", veclen_str);
  printf("const int bx = sx / veclen;"); // Vectorized block dimensions
  printf("const int by = sy;");
  printf("const int bz = sz;");
  printf("const int tpb = blockDim.x * blockDim.y * blockDim.z;");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};

#if 1
  // Fetch each field in a pipelined fashion (~4.8ms)
  printf("for (int curr = sid; curr < bx * by * bz; curr += tpb) {");
  printf("const int i = curr %% bx;");
  printf("const int j = (curr %% (bx * by)) / bx;");
  printf("const int k = curr / (bx * by);");
  for (int field = 0; field < NUM_FIELDS; ++field) {
    printf("reinterpret_cast<%s%s*>("
           "&smem[j * sx + k * sx * sy + %d * sx * sy * sz])[i] = ",
           realtype, veclen_str, field);
    // clang-format off
    printf("reinterpret_cast<%s%s*>(&vba.in[%d][IDX(baseIdx.x, baseIdx.y + j, baseIdx.z + k)])[i];", realtype, veclen_str, field);
    // clang-format on
  }
  printf("}");
  // No effect on performance if __syncthreads() removed:
  // therefore async pipelines would not provide benefits
  printf("__syncthreads();");
#else
  // Fetch everything at once (~6ms)
  printf(
      "for (int curr = sid; curr < bx * by * bz * NUM_FIELDS; curr += tpb) {");
  printf("const int i = curr %% bx;");
  printf("const int j = (curr %% (bx * by)) / bx;");
  printf("const int k = (curr %% (bx * by * bz)) / (bx * by);");
  printf("const int w = curr / (bx * by * bz);");
  printf("reinterpret_cast<%s%s*>("
         "&smem[j * sx + k * sx * sy + w * sx * sy * sz])[i] = ",
         realtype, veclen_str);
  // clang-format off
    printf("reinterpret_cast<%s%s*>(&vba.in[w][IDX(baseIdx.x, baseIdx.y + j, baseIdx.z + k)])[i];", realtype, veclen_str);
  // clang-format on
  printf("}");
  printf("__syncthreads();");
#endif

  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("%s(stencils[%d][%d][%d][%d]*",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width);
                printf("smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "(threadIdx.z + %d) * sx * sy +"
                       "%d * sx * sy *sz]);",
                       width, height, depth, field);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d,%s(stencils[%d][%d][%d][%d]*",
                       stencil_binary_ops[stencil], field, stencil,
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width);
                printf("smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "(threadIdx.z + %d) * sx * sy +"
                       "%d * sx * sy *sz]));",
                       width, height, depth, field);
              }
            }
          }
        }
      }
    }
  }

  printf("if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || "
         "vertexIdx.z >= end.z) { return; }");

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#elif IMPLEMENTATION ==                                                        \
    SMEM_AND_VECTORIZED_LOADS_AND_PINGPONG_AND_ONDEMAND_STENCIL_COMPUTATION
// Does not actually use smem, just a quick test on what happens if we
// compute the stencil on-demand

void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix_with_boundcheck();

  int stencil_initialized[NUM_STENCILS] = {0};
  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          if (stencils[stencil][depth][height][width]) {
            if (!stencil_initialized[stencil]) {
              printf("auto tmp = ");
              printf("stencils[%d][%d][%d][%d] * ", //
                     stencil, depth, height, width);
              printf("%s(", stencil_unary_ops[stencil]);
              printf("__ldg(&");
              printf("vba.in[field][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                     "vertexIdx.z+(%d))]",
                     -STENCIL_ORDER / 2 + width, -STENCIL_ORDER / 2 + height,
                     -STENCIL_ORDER / 2 + depth);
              printf(")");
              printf(");");
              stencil_initialized[stencil] = 1;
            }
            else {
              printf("tmp =");
              printf("%s(tmp, ", stencil_binary_ops[stencil]);
              printf("stencils[%d][%d][%d][%d] * ", //
                     stencil, depth, height, width);
              printf("%s(", stencil_unary_ops[stencil]);
              printf("__ldg(&");
              printf("vba.in[field][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                     "vertexIdx.z+(%d))]",
                     -STENCIL_ORDER / 2 + width, -STENCIL_ORDER / 2 + height,
                     -STENCIL_ORDER / 2 + depth);
              printf(")");
              printf(")");
              printf(");");
            }
          }
        }
      }
    }
    printf("return tmp;");
    printf("};");
  }
}
#elif IMPLEMENTATION == FULLY_EXPL_REG_VARS_AND_PINGPONG_REGISTERS
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix_with_boundcheck();

  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
      if (stencils_accessed[curr_kernel][field][stencil]) {
        printf("const AcReal* __restrict__ in%d = vba.in[%d];", field, field);
        break;
      }
    }
  }

  // Prefetch stencil coefficients to local memory
  int coeff_initialized[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT]
                       [STENCIL_WIDTH] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

          int stencil_accessed = 0;
          for (int field = 0; field < NUM_FIELDS; ++field)
            stencil_accessed |= stencils_accessed[curr_kernel][field][stencil];
          if (!stencil_accessed)
            continue;

          if (stencils[stencil][depth][height][width] &&
              !coeff_initialized[stencil][depth][height][width]) {
            printf("const auto s%d_%d_%d_%d = ", //
                   stencil, depth, height, width);

            // CT const
            // printf("%s;", stencils[stencil][depth][height][width]);
            printf("stencils[%d][%d][%d][%d];", stencil, depth, height, width);

            coeff_initialized[stencil][depth][height][width] = 1;
          }
        }
      }
    }
  }

  const int prefetch_size = 2;
  // Prefetch stencil elements to local memory
  int cell_initialized[NUM_FIELDS][STENCIL_DEPTH][STENCIL_HEIGHT]
                      [STENCIL_WIDTH] = {0};
  for (int field = 0; field < prefetch_size; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width] &&
                !cell_initialized[field][depth][height][width]) {
              printf("const auto f%d_%d_%d_%d = ", //
                     field, depth, height, width);
              printf("in%d[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                     "vertexIdx.z+(%d))];",
                     field, -STENCIL_ORDER / 2 + width,
                     -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

              cell_initialized[field][depth][height][width] = 1;
            }
          }
        }
      }
    }
  }
  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int field = prefetch_size; field < NUM_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (stencils_accessed[curr_kernel][field][stencil]) {
              if (stencils[stencil][depth][height][width] &&
                  !cell_initialized[field][depth][height][width]) {
                printf("const auto f%d_%d_%d_%d = ", //
                       field, depth, height, width);
                printf("in%d[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                       "vertexIdx.z+(%d))];",
                       field, -STENCIL_ORDER / 2 + width,
                       -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

                cell_initialized[field][depth][height][width] = 1;
              }
            }

            if (stencils_accessed[curr_kernel][field - prefetch_size]
                                 [stencil]) {
              if (stencils[stencil][depth][height][width]) {
                if (!stencil_initialized[field - prefetch_size][stencil]) {
                  printf("auto f%d_s%d = ", field - prefetch_size, stencil);
                  printf("%s(s%d_%d_%d_%d*"
                         "f%d_%d_%d_%d);",
                         stencil_unary_ops[stencil], stencil, depth, height,
                         width, field - prefetch_size, depth, height, width);

                  stencil_initialized[field - prefetch_size][stencil] = 1;
                }
                else {
                  printf("f%d_s%d = ", field - prefetch_size, stencil);
                  printf( //
                      "%s(f%d_s%d,%s(s%d_%d_%d_%d*"
                      "f%d_%d_%d_%d));",
                      stencil_binary_ops[stencil], field - prefetch_size,
                      stencil, stencil_unary_ops[stencil], stencil, depth,
                      height, width, field - prefetch_size, depth, height,
                      width);
                }
              }
            }
          }
        }
      }
    }
  }

  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int field = NUM_FIELDS - prefetch_size; field < NUM_FIELDS;
             ++field) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("%s(s%d_%d_%d_%d*"
                       "f%d_%d_%d_%d);",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, field, depth, height, width);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf( //
                    "%s(f%d_s%d,%s(s%d_%d_%d_%d*"
                    "f%d_%d_%d_%d));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil], stencil, depth, height, width,
                    field, depth, height, width);
              }
            }
          }
        }
      }
    }
  }

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#elif IMPLEMENTATION == SMEM_AND_VECTORIZED_LOADS_FULL_ASYNC
// Vectorized, asynchronous loads to local memory. Strict alignment
// requirements.
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix();
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sz = blockDim.z + STENCIL_DEPTH - 1;");
  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x + "
         "threadIdx.z * blockDim.x * blockDim.y;");

  printf("const int veclen = %s;", veclen_str);
  printf("const int bx = sx / veclen;"); // Vectorized block dimensions
  printf("const int by = sy;");
  printf("const int bz = sz;");
  printf("const int tpb = blockDim.x * blockDim.y * blockDim.z;");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};

#if 1
  // Fetch each field in a pipelined fashion (~4.8ms)
  for (int field = 0; field < NUM_FIELDS; ++field) {
    printf("for (int curr = sid; curr < bx * by * bz; curr += tpb) {");
    printf("const int i = curr %% bx;");
    printf("const int j = (curr %% (bx * by)) / bx;");
    printf("const int k = curr / (bx * by);");

    printf("{const %s%s* smem_ptr = ", realtype, veclen_str);
    printf("&reinterpret_cast<%s%s*>("
           "&smem[j * sx + k * sx * sy + %d * sx * sy * sz])[i];",
           realtype, veclen_str, field);
    printf("const %s%s* in_ptr = ", realtype, veclen_str);
    // clang-format off
    printf("&reinterpret_cast<%s%s*>(&vba.in[%d][IDX(baseIdx.x, baseIdx.y + j, baseIdx.z + k)])[i];", realtype, veclen_str, field);
    // clang-format on
    printf("__pipeline_memcpy_async(smem_ptr, in_ptr, sizeof(%s%s));", realtype,
           veclen_str);
    printf("}");
    printf("}");
    printf("__pipeline_commit();");
  }

  // No effect on performance if __syncthreads() removed:
  // therefore async pipelines would not provide benefits
  // printf("__syncthreads();");
#else
  // Fetch everything at once (~6ms)
  printf(
      "for (int curr = sid; curr < bx * by * bz * NUM_FIELDS; curr += tpb) {");
  printf("const int i = curr %% bx;");
  printf("const int j = (curr %% (bx * by)) / bx;");
  printf("const int k = (curr %% (bx * by * bz)) / (bx * by);");
  printf("const int w = curr / (bx * by * bz);");
  printf("reinterpret_cast<%s%s*>("
         "&smem[j * sx + k * sx * sy + w * sx * sy * sz])[i] = ",
         realtype, veclen_str);
  // clang-format off
    printf("reinterpret_cast<%s%s*>(&vba.in[w][IDX(baseIdx.x, baseIdx.y + j, baseIdx.z + k)])[i];", realtype, veclen_str);
  // clang-format on
  printf("}");
  printf("__syncthreads();");
#endif

  for (int field = 0; field < NUM_FIELDS; ++field) {
    printf("__pipeline_wait_prior(0);");
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("%s(stencils[%d][%d][%d][%d]*",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width);
                printf("smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "(threadIdx.z + %d) * sx * sy +"
                       "%d * sx * sy *sz]);",
                       width, height, depth, field);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d,%s(stencils[%d][%d][%d][%d]*",
                       stencil_binary_ops[stencil], field, stencil,
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width);
                printf("smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "(threadIdx.z + %d) * sx * sy +"
                       "%d * sx * sy *sz]));",
                       width, height, depth, field);
              }
            }
          }
        }
      }
    }
  }

  printf("if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || "
         "vertexIdx.z >= end.z) { return; }");

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#elif IMPLEMENTATION == SMEM_HIGH_OCCUPANCY
// Doesn't work (illegal memory access), long compile times
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix();

  printf(
      "AcReal /*__restrict__*/ processed_stencils[NUM_FIELDS][NUM_STENCILS];");
  for (size_t j = 0; j < NUM_FIELDS; ++j)
    for (size_t i = 0; i < NUM_STENCILS; ++i)
      printf("processed_stencils[%lu][%lu] = NAN;", j, i);

  printf("extern __shared__ AcReal smem[];");

  // TB working set
  printf("const int mx = blockDim.x + (STENCIL_WIDTH-1)/2;");
  printf("const int my = blockDim.y + (STENCIL_HEIGHT-1)/2;");
  printf("const int mz = blockDim.z + (STENCIL_DEPTH-1)/2;");

  // TB computational domain
  printf("const int nx = blockDim.x;");
  printf("const int ny = blockDim.y;");
  printf("const int nz = blockDim.z;");

  // Stencil size
  printf("const int sx = STENCIL_WIDTH;");
  printf("const int sy = STENCIL_HEIGHT;");
  printf("const int sz = STENCIL_DEPTH;");

  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  printf("const int m0 = IDX(baseIdx);");
  printf("const int n0 = IDX(vertexIdx);");
  printf("const int s0 = threadIdx.x + "
         "threadIdx.y * blockDim.x + "
         "threadIdx.z * blockDim.x * blockDim.y;");

  printf("int in = 0;");

  for (int field = 0; field < NUM_FIELDS; ++field) {
    printf("for (int k = 0; k < mz; ++k) {");
    printf("for (int j = 0; j < my; ++j) {");

    printf("const int base = m0 + IDX(0, j, k);");
    printf("if (s0 < mx && s0+base < IDX(end)) smem[s0] = s0 + base;");
    printf("__syncthreads();");

#define RX ((STENCIL_WIDTH - 1) / 2)
#define RY ((STENCIL_HEIGHT - 1) / 2)
#define RZ ((STENCIL_DEPTH - 1) / 2)

    for (int depth = -RZ; depth <= RZ; ++depth) {
      for (int height = -RY; height <= RY; ++height) {
        for (int width = -RX; width <= RX; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            // Skip if the stencil coefficient is zero
            if (!stencils[stencil][depth][height][width])
              continue;

            printf("in = IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                   "vertexIdx.z+(%d)) - base;",
                   width, height, depth);
            printf("if (in >= 0 && in < mx)");
            printf("if (processed_stencils[%d][%d]) {", field, stencil);
            printf("processed_stencils[%d][%d] = ", field, stencil);
            printf("%s(processed_stencils[%d][%d],%s(stencils[%d][%d][%d][%d]*"
                   "smem[in]));",
                   stencil_binary_ops[stencil], field, stencil,
                   stencil_unary_ops[stencil], stencil, depth, height, width);
            printf("}else{");
            printf("processed_stencils[%d][%d] = ", field, stencil);
            printf("%s(stencils[%d][%d][%d][%d]*"
                   "smem[in]);",
                   stencil_unary_ops[stencil], stencil, depth, height, width);
            printf("}");
          }
        }
      }
    }
    printf("}");
    printf("}");
  }

  /*
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x + "
         "threadIdx.z * blockDim.x * blockDim.y;");
  printf("const int bid = IDX(baseIdx);");
  printf("const int bx = blockDim.x;");
  printf("const int bxy = blockDim.x * blockDim.y;");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {

        printf("if (sid < sx)");
        printf("smem[sid] = vba.in[%d][sid + bid + IDX(0, height, width)];",
               field);
        printf("__syncthreads();");

        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          printf("const int x = width + bid + IDX(0, height, width);");
          printf("const int y = ")
        }
      }
    }
  }*/

  /*
  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int field = 0; field < NUM_FIELDS; ++field) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("%s(stencils[%d][%d][%d][%d]*"
                       "vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                       "vertexIdx.z+(%d))]);",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, field, -STENCIL_ORDER / 2 + width,
                       -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf( //
                    "%s(f%d_s%d,%s(stencils[%d][%d][%d][%d]*"
                    "vba.in[%d][IDX(vertexIdx.x+(%d)"
                    ",vertexIdx.y+(%d),vertexIdx.z+(%d))]));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil], stencil, depth, height, width,
                    field, -STENCIL_ORDER / 2 + width,
                    -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);
              }
            }
          }
        }
      }
    }
  }*/

  printf("if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || "
         "vertexIdx.z >= end.z) { return; }");

  for (size_t i = 0; i < NUM_STENCILS; ++i)
    printf("const auto %s __attribute__((unused)) =[&](const auto field)"
           "{return processed_stencils[field][stencil_%s];};",
           stencil_names[i], stencil_names[i]);
  /*
  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
  */
}
#elif IMPLEMENTATION == 1000000 // SMEM_HIGH_OCCUPANCY_CT_CONST_TB
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix();

  const size_t rx = ((STENCIL_WIDTH - 1) / 2);
  const size_t ry = ((STENCIL_HEIGHT - 1) / 2);
  const size_t rz = ((STENCIL_DEPTH - 1) / 2);
  const size_t mx = nx + 2 * rx;
  const size_t my = ny + 2 * ry;
  const size_t mz = nz + 2 * rz;

  printf("extern __shared__ AcReal smem[];");

  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * %lu + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * %lu + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * %lu + start.z - (STENCIL_DEPTH-1)/2};",
         nx, ny, nz);
  printf("const int m0 = IDX(baseIdx);");
  printf("const int s0 = threadIdx.x + "
         "threadIdx.y * %lu + "
         "threadIdx.z * %lu;",
         nx, nx * ny);

  for (int field = 0; field < NUM_FIELDS; ++field)
    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil)
      if (stencils_accessed[curr_kernel][field][stencil])
        printf("AcReal f%d_s%d;", field, stencil);

  bool stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        printf("if (s0 < %lu) smem[s0] = "
               "vba.in[%d][IDX(baseIdx.x, baseIdx.y + %d, baseIdx.z + %d)];",
               mx, field, j, k);
        printf("__syncthreads();");
        for (int i = 0; i < mx; ++i) {
          for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
            for (int height = 0; height < STENCIL_HEIGHT; ++height) {
              for (int width = 0; width < STENCIL_WIDTH; ++width) {
                fprintf(stderr, "%d, %d, %d, %d, %d, %d\n", width, height,
                        depth, i, j, k);
                printf("if (((%d + threadIdx.x) == %d) && "
                       "((%d + threadIdx.y) == %d) && "
                       "((%d + threadIdx.z) == %d)) { ",
                       width, i, height, j, depth, k);

                for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                  if (stencils_accessed[curr_kernel][field][stencil]) {
                    if (stencils[stencil][depth][height][width]) {
                      if (!stencil_initialized[field][stencil]) {
                        printf("f%d_s%d = ", field, stencil);
                        printf("%s(stencils[%d][%d][%d][%d]*"
                               "smem[%d]);",
                               stencil_unary_ops[stencil], stencil, depth,
                               height, width, i);

                        stencil_initialized[field][stencil] = 1;
                      }
                      else {
                        printf("f%d_s%d = ", field, stencil);
                        printf( //
                            "%s(f%d_s%d,%s(stencils[%d][%d][%d][%d]*"
                            "smem[%d]));",
                            stencil_binary_ops[stencil], field, stencil,
                            stencil_unary_ops[stencil], stencil, depth, height,
                            width, i);
                      }
                    }
                  }
                }
                printf("}");
              }
            }
          }
        }
      }
    }
  }

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#elif IMPLEMENTATION == SMEM_HIGH_OCCUPANCY_CT_CONST_TB
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix();

  const size_t rx = ((STENCIL_WIDTH - 1) / 2);
  const size_t ry = ((STENCIL_HEIGHT - 1) / 2);
  const size_t rz = ((STENCIL_DEPTH - 1) / 2);
  const size_t mx = nx + 2 * rx;
  const size_t my = ny + 2 * ry;
  const size_t mz = nz + 2 * rz;

  printf("extern __shared__ AcReal smem[];");

  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * %lu + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * %lu + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * %lu + start.z - (STENCIL_DEPTH-1)/2};",
         nx, ny, nz);
  printf("const int m0 = IDX(baseIdx);");
  printf("const int s0 = threadIdx.x + "
         "threadIdx.y * %lu + "
         "threadIdx.z * %lu;",
         nx, nx * ny);

  for (int field = 0; field < NUM_FIELDS; ++field)
    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil)
      if (stencils_accessed[curr_kernel][field][stencil])
        printf("AcReal f%d_s%d = NAN;", field, stencil);

  bool stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int k = 0; k < (int)mz; ++k) {
      for (int j = 0; j < (int)my; ++j) {
        printf("__syncthreads();");
        printf("if (s0 < %lu) smem[s0] = "
               "vba.in[%d][IDX(baseIdx.x, baseIdx.y + %d, baseIdx.z + %d)];",
               mx, field, j, k);
        printf("__syncthreads();");

        for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
          printf("if (threadIdx.z + %d == %d) {", depth, k);
          for (int height = 0; height < STENCIL_HEIGHT; ++height) {
            printf("if (threadIdx.y + %d == %d) {", height, j);
            for (int width = 0; width < STENCIL_WIDTH; ++width) {
              for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                if (!stencils_accessed[curr_kernel][field][stencil])
                  continue;
                if (!stencils[stencil][depth][height][width])
                  continue;

                printf("if (!f%d_s%d)", field, stencil);
                printf("f%d_s%d = ", field, stencil);
                printf("%s(stencils[%d][%d][%d][%d]*"
                       "smem[threadIdx.x + %d]);",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, width);
                printf("else ");
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d,%s(stencils[%d][%d][%d][%d]*"
                       "smem[threadIdx.x + %d]));",
                       stencil_binary_ops[stencil], field, stencil,
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, width);
              }
            }
            printf("}");
          }
          printf("}");
        }

        /*
        printf("for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {");
        printf("for (int height = 0; height < STENCIL_HEIGHT; ++height) {");
        printf("for (int width = 0; width < STENCIL_WIDTH; ++width) {");
        for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
          if (stencils_accessed[curr_kernel][field][stencil]) {

          }
        }
        printf("}");
        printf("}");
        printf("}");
        printf("}");
        */

        /*
        // No memory faults but takes very long to compile and possibly
        incorrect printf("for (int depth = 0; depth < STENCIL_DEPTH; ++depth)
        {");

        printf("const auto bz = threadIdx.z + depth;");
        printf("if (bz == %d) {", k);
        printf("for (int height = 0; height < STENCIL_HEIGHT; ++height) {");

        printf("const auto by = threadIdx.y + height;");
        printf("if (by == %d) {", j);

        printf("for (int width = 0; width < STENCIL_WIDTH; ++width) {");
        for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
          if (stencils_accessed[curr_kernel][field][stencil]) {
            printf("if (!f%d_s%d)", field, stencil);
            printf("f%d_s%d = ", field, stencil);
            printf("%s(stencils[%d][depth][height][width]*"
                   "smem[threadIdx.x + width]);",
                   stencil_unary_ops[stencil], stencil);
            printf("else ");
            printf("f%d_s%d = ", field, stencil);
            printf("%s(f%d_s%d,%s(stencils[%d][depth][height][width]*"
                   "smem[threadIdx.x + width]));",
                   stencil_binary_ops[stencil], field, stencil,
                   stencil_unary_ops[stencil], stencil);
          }
        }
        printf("}");
        printf("}");
        printf("}");
        printf("}");
        printf("}");
        */
        /*
        for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {

          printf("const auto bz = threadIdx.z + depth;");
          printf("if (bz == %d) {", k);
          for (int height = 0; height < STENCIL_HEIGHT; ++height) {

            printf("const auto by = threadIdx.y + height;");
            printf("if (by == %d) {", j);

            for (int width = 0; width < STENCIL_WIDTH; ++width) {
              for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                if (stencils_accessed[curr_kernel][field][stencil]) {
                  printf("if (!f%d_s%d)", field, stencil);
                  printf("f%d_s%d = ", field, stencil);
                  printf("%s(stencils[%d][%d][%d][%d]*"
                         "smem[threadIdx.x + %d]);",
                         stencil_unary_ops[stencil], stencil, depth, height,
                         width, width);
                  printf("else ");
                  printf("f%d_s%d = ", field, stencil);
                  printf("%s(f%d_s%d,%s(stencils[%d][%d][%d][%d]*"
                         "smem[threadIdx.x + %d]));",
                         stencil_binary_ops[stencil], field, stencil,
                         stencil_unary_ops[stencil], stencil, depth, height,
                         width, width);
                }
              }
            }

            printf("}");
          }
          printf("}");
        }
        */

        /*
        for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
          for (int height = 0; height < STENCIL_HEIGHT; ++height) {
            printf("const auto by = threadIdx.y + height;");
            printf("const auto bz = threadIdx.z + depth;");

            printf("if (by == %d && bz == %d) {", j, k);

            for (int width = 0; width < STENCIL_WIDTH; ++width) {
              printf("const auto bx = threadIdx.x + width;");

              for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                // Processing [field][stencil]
                // bx = threadIdx.x + width
                // by = threadIdx.y + height
                // bz = threadIdx.z + depth
                // bx < mx, then can access smem[bx]
                // by == j
                // bz == k

                // 0 + threadIdx.x + width
                // j + threadIdx.y + height
                // in = smem[threadIdx.x + width]
                // coeff = stencils[stencil][depth][my - (threadIdx.y +
                // height)][width] mask =
              }
            } //
            printf("}");
          }
        }
        */
        /*
        for (int i = 0; i < mx; ++i) {
          // Map i,j,k to stencil space
          printf("{");
          printf("const int sx = %d - threadIdx.x;", i);
          printf("const int sy = %d - threadIdx.y;", j);
          printf("const int sz = %d - threadIdx.z;", k);
          printf("if (sx >= 0 && sy >= 0 && sz >= 0)");
          printf("if (sx < STENCIL_WIDTH)");
          printf("if (sy < STENCIL_HEIGHT)");
          printf("if (sz < STENCIL_DEPTH)");
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
            if (stencils_accessed[curr_kernel][field][stencil]) {
              printf("if (!f%d_s%d)", field, stencil);
              printf("f%d_s%d = ", field, stencil);
              printf("%s(stencils[%d][sz][sy][sx]*"
                     "smem[%d]);",
                     stencil_unary_ops[stencil], stencil, i);
              printf("else ");
              printf("f%d_s%d = ", field, stencil);
              printf( //
                  "%s(f%d_s%d,%s(stencils[%d][sz][sy][sx]*"
                  "smem[%d]));",
                  stencil_binary_ops[stencil], field, stencil,
                  stencil_unary_ops[stencil], stencil, i);
            }
          }
          printf("}");
        }
        */
      }
    }
  }

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#elif IMPLEMENTATION == SMEM_GENERIC_BLOCKED
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix(); // Note no bounds check
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sz = blockDim.z + STENCIL_DEPTH - 1;");
  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x + "
         "threadIdx.z * blockDim.x * blockDim.y;");

  printf("const int veclen = %s;", veclen_str);
  printf("const int bx = sx / veclen;"); // Vectorized block dimensions
  printf("const int by = sy;");
  printf("const int bz = sz;");
  printf("const int tpb = blockDim.x * blockDim.y * blockDim.z;");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {

    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      printf("for (int curr = sid; curr < bx * by; curr += tpb) {");
      printf("const int i = curr %% bx;");
      printf("const int j = curr / bx;");
      printf("const int k = %d;", depth);
      printf("reinterpret_cast<%s%s*>("
             "&smem[j * sx])[i] = ",
             realtype, veclen_str);
      // clang-format off
    printf("reinterpret_cast<%s%s*>(&vba.in[%d][IDX(baseIdx.x, baseIdx.y + j, baseIdx.z + k)])[i];", realtype, veclen_str, field);
      // clang-format on
      printf("}");
      printf("__syncthreads();");
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("%s(stencils[%d][%d][%d][%d]*",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width);
                printf("smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx]);",
                       width, height);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d,%s(stencils[%d][%d][%d][%d]*",
                       stencil_binary_ops[stencil], field, stencil,
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width);
                printf("smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx]));",
                       width, height);
              }
            }
          }
        }
      }
      printf("__syncthreads();");
    }
  }

  printf("if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || "
         "vertexIdx.z >= end.z) { return; }");

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#elif IMPLEMENTATION == SMEM_GENERIC_BLOCKED_1D
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix(); // Note no bounds check
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sz = blockDim.z + STENCIL_DEPTH - 1;");
  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x + "
         "threadIdx.z * blockDim.x * blockDim.y;");

  printf("const int veclen = %s;", veclen_str);
  printf("const int bx = sx / veclen;"); // Vectorized block dimensions
  printf("const int by = sy;");
  printf("const int bz = sz;");
  printf("const int tpb = blockDim.x * blockDim.y * blockDim.z;");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {

    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        printf("for (int curr = sid; curr < bx; curr += tpb) {");
        printf("const int i = curr %% bx;");
        printf("const int j = %d;", height);
        printf("const int k = %d;", depth);
        printf("reinterpret_cast<%s%s*>("
               "&smem[0])[i] = ",
               realtype, veclen_str);
        // clang-format off
    printf("reinterpret_cast<%s%s*>(&vba.in[%d][IDX(baseIdx.x, baseIdx.y + j, baseIdx.z + k)])[i];", realtype, veclen_str, field);
        // clang-format on
        printf("}");
        printf("__syncthreads();");
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("%s(stencils[%d][%d][%d][%d]*",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width);
                printf("smem[(threadIdx.x + %d)]);", width);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d,%s(stencils[%d][%d][%d][%d]*",
                       stencil_binary_ops[stencil], field, stencil,
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width);
                printf("smem[(threadIdx.x + %d)]));", width);
              }
            }
          }
        }
        printf("__syncthreads();");
      }
    }
  }

  printf("if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || "
         "vertexIdx.z >= end.z) { return; }");

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#elif IMPLEMENTATION == FULLY_EXPL_REG_VARS_AND_HALO_THREADS
// Allocates threads for the halo for simplified shuffling, but
// very bad performance (50x worse) and do not know whether shuffling
// could be able to help much
// Note requires MAX_THREADS_PER_BLOCK >= 512 for 6th order
// Not much room for tpb optimization, requires >= (2r+1)^3 tpbs
void
gen_kernel_body(const int curr_kernel)
{

  printf("const int RX = (STENCIL_WIDTH - 1)/2;");
  printf("const int RY = (STENCIL_HEIGHT - 1)/2;");
  printf("const int RZ = (STENCIL_DEPTH - 1)/2;");
  printf("const int3 vertexIdx = (int3){"
         "threadIdx.x + blockIdx.x * (blockDim.x - 2*RX) + start.x - RX,"
         "threadIdx.y + blockIdx.y * (blockDim.y - 2*RY) + start.y - RY,"
         "threadIdx.z + blockIdx.z * (blockDim.z - 2*RZ) + start.z - RZ,"
         "};");

  printf("if (vertexIdx.x >= end.x + RX) return;");
  printf("if (vertexIdx.y >= end.y + RY) return;");
  printf("if (vertexIdx.z >= end.z + RZ) return;");

  /*
  printf("const int3 vertexIdx = (int3){"
         "threadIdx.x + blockIdx.x * blockDim.x + start.x,"
         "threadIdx.y + blockIdx.y * blockDim.y + start.y,"
         "threadIdx.z + blockIdx.z * blockDim.z + start.z,"
         "};");
         */
  printf("const int3 globalVertexIdx = (int3){"
         "d_multigpu_offset.x + vertexIdx.x,"
         "d_multigpu_offset.y + vertexIdx.y,"
         "d_multigpu_offset.z + vertexIdx.z,"
         "};");
  printf("const int3 globalGridN = d_mesh_info.int3_params[AC_global_grid_n];");
  printf("const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);");

  printf("(void)globalVertexIdx;"); // Silence unused warning
  printf("(void)globalGridN;");     // Silence unused warning

  for (int field = 0; field < NUM_FIELDS; ++field)
    printf("const auto f%d_prev = vba.out[%d][idx];", field, field);

  printf("const auto previous __attribute__((unused)) = [&](const Field field)"
         "{ switch (field) {");
  for (int field = 0; field < NUM_FIELDS; ++field)
    printf("case %d: { return f%d_prev; }", field, field);

  printf("default: return (AcReal)NAN;"
         "}");
  printf("};");
  printf("const auto write=[&](const Field field, const AcReal value)"
         "{ vba.out[field][idx] = value; };");

  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
      if (stencils_accessed[curr_kernel][field][stencil]) {
        printf("const AcReal* __restrict__ in%d = vba.in[%d];", field, field);
        break;
      }
    }
  }

#if 1
  for (int field = 0; field < NUM_FIELDS; ++field) {
    printf("const auto f%d = in%d[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
           "vertexIdx.z+(%d))];",
           field, field, 0, 0, 0);
  }
  int cell_initialized[NUM_FIELDS][STENCIL_DEPTH][STENCIL_HEIGHT]
                      [STENCIL_WIDTH] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width] &&
                !cell_initialized[field][depth][height][width]) {
              printf("const auto f%d_%d_%d_%d = f%d;", //
                     field, depth, height, width, field);

              cell_initialized[field][depth][height][width] = 1;
            }
          }
        }
      }
    }
  }
#else // Uncomment to make this work
  // Prefetch stencil elements to local memory
  const int mid = (STENCIL_WIDTH - 1) / 2;
  int cell_initialized[NUM_FIELDS][STENCIL_DEPTH][STENCIL_HEIGHT]
                      [STENCIL_WIDTH] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        const int width = mid;
        for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

          // Skip if the stencil is not used
          if (!stencils_accessed[curr_kernel][field][stencil])
            continue;

          if (stencils[stencil][depth][height][width] &&
              !cell_initialized[field][depth][height][width]) {
            printf("const auto f%d_%d_%d_%d = ", //
                   field, depth, height, width);
            printf("in%d[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                   "vertexIdx.z+(%d))];",
                   field, -STENCIL_ORDER / 2 + width,
                   -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

            cell_initialized[field][depth][height][width] = 1;
          }
        }
      }
    }
  }

  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          if (width == mid)
            continue;
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width] &&
                !cell_initialized[field][depth][height][width]) {
              printf("const auto f%d_%d_%d_%d = ", //
                     field, depth, height, width);
              printf("in%d[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                     "vertexIdx.z+(%d))];",
                     field, -STENCIL_ORDER / 2 + width,
                     -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

              cell_initialized[field][depth][height][width] = 1;
            }
          }
        }
      }
    }
  }
#endif

  // Prefetch stencil coefficients to local memory
  int coeff_initialized[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT]
                       [STENCIL_WIDTH] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

          int stencil_accessed = 0;
          for (int field = 0; field < NUM_FIELDS; ++field)
            stencil_accessed |= stencils_accessed[curr_kernel][field][stencil];
          if (!stencil_accessed)
            continue;

          if (stencils[stencil][depth][height][width] &&
              !coeff_initialized[stencil][depth][height][width]) {
            printf("const auto s%d_%d_%d_%d = ", //
                   stencil, depth, height, width);

            // CT const
            // printf("%s;", stencils[stencil][depth][height][width]);
            printf("stencils[%d][%d][%d][%d];", stencil, depth, height, width);

            coeff_initialized[stencil][depth][height][width] = 1;
          }
        }
      }
    }
  }

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int field = 0; field < NUM_FIELDS; ++field) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("%s(s%d_%d_%d_%d*"
                       "f%d_%d_%d_%d);",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, field, depth, height, width);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf( //
                    "%s(f%d_s%d,%s(s%d_%d_%d_%d*"
                    "f%d_%d_%d_%d));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil], stencil, depth, height, width,
                    field, depth, height, width);
              }
            }
          }
        }
      }
    }
  }

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }

  printf("if (threadIdx.x < RX) return;");
  printf("if (threadIdx.y < RY) return;");
  printf("if (threadIdx.z < RZ) return;");
  printf("if (threadIdx.x >= blockDim.x - RX) return;");
  printf("if (threadIdx.y >= blockDim.y - RY) return;");
  printf("if (threadIdx.z >= blockDim.z - RZ) return;");
}
#if 0
// Does not work and bad performance (1.5x worse)
void
gen_kernel_body(const int curr_kernel)
{
  gen_kernel_prefix_with_boundcheck();

  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
      if (stencils_accessed[curr_kernel][field][stencil]) {
        printf("const AcReal* __restrict__ in%d = vba.in[%d];", field, field);
        break;
      }
    }
  }

  // Prefetch stencil elements to local memory
  int cell_initialized[NUM_FIELDS][STENCIL_DEPTH][STENCIL_HEIGHT]
                      [STENCIL_WIDTH] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {

        const int mid = (STENCIL_WIDTH - 1) / 2;
        // Declare stencil elements, fetch mid
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width] &&
                !cell_initialized[field][depth][height][width]) {

              if (width == mid) {
                printf("const AcReal f%d_%d_%d_%d = ", //
                       field, depth, height, width);
                printf("in%d[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                       "vertexIdx.z+(%d))];",
                       field, -STENCIL_ORDER / 2 + width,
                       -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);
              }
              else {
                printf("AcReal f%d_%d_%d_%d;", //
                       field, depth, height, width);
              }

              cell_initialized[field][depth][height][width] = 1;
            }
          }
        }

        // Shuffle stencil elements

        for (int width = 0; width < mid; ++width) {
          if (cell_initialized[field][depth][height][width]) {
            printf("{");
            printf("const auto tmp = f%d_%d_%d_%d;", //
                   field, depth, height, mid);
            printf("f%d_%d_%d_%d = __shfl_down_sync(0xffffffff, tmp, %d);", //
                   field, depth, height, width, mid - width);
            printf("}");
          }

          if (cell_initialized[field][depth][height]
                              [STENCIL_WIDTH - 1 - width]) {
            printf("{");
            printf("const auto tmp = f%d_%d_%d_%d;", //
                   field, depth, height, mid);
            printf("f%d_%d_%d_%d = __shfl_up_sync(0xffffffff, tmp, %d);", //
                   field, depth, height, STENCIL_WIDTH - 1 - width,
                   mid - width);
            printf("}");
          }
        }

        // Fill in undefined stencil elements
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          if (cell_initialized[field][depth][height][width]) {
            if (width == mid)
              continue;

            printf("if (threadIdx.x + (%d) < (%d) || threadIdx.x + (%d) >= "
                   "blockDim.x + (%d)) {",
                   width, mid, width, mid);
            printf("f%d_%d_%d_%d = ", //
                   field, depth, height, width);
            printf("in%d[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                   "vertexIdx.z+(%d))];",
                   field, -STENCIL_ORDER / 2 + width,
                   -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);
            printf("}");
          }
        }

        /*
  const int mid = (STENCIL_WIDTH - 1) / 2;
  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
  // Skip if the stencil is not used
  if (!stencils_accessed[curr_kernel][field][stencil])
    continue;

  if (stencils[stencil][depth][height][mid] &&
      !cell_initialized[field][depth][height][mid]) {
    printf("const auto f%d_%d_%d_%d = ", //
           field, depth, height, mid);
    printf("in%d[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
           "vertexIdx.z+(%d))];",
           field, -STENCIL_ORDER / 2 + mid, -STENCIL_ORDER / 2 + height,
           -STENCIL_ORDER / 2 + depth);

    cell_initialized[field][depth][height][mid] = 1;
  }
  }


  for (int width = 0; width < STENCIL_WIDTH; ++width) {
  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

    // Skip if the stencil is not used
    if (!stencils_accessed[curr_kernel][field][stencil])
      continue;

    if (stencils[stencil][depth][height][width] &&
        !cell_initialized[field][depth][height][width]) {

      printf("AcReal f%d_%d_%d_%d;", //
             field, depth, height, width);

      printf("if (vertexIdx.x + (%d) >= %d &&"
             "vertexIdx.x + (%d) < blockDim.x + (%d)) {",
             width, STENCIL_ORDER / 2, width, STENCIL_ORDER / 2);
      printf("AcReal tmp = f%d_%d_%d_%d;", field, depth, height, mid);
      // Shuffle
      printf("const int lane = (blockDim.x + threadIdx.x + %d) %% "
             "blockDim.x;",
             width - mid);
      printf("f%d_%d_%d_%d = ", //
             field, depth, height, width);
      printf("__shfl_sync(0xffffffff, tmp, lane);");

      printf("} else { ");
      printf("f%d_%d_%d_%d = ", //
             field, depth, height, width);
      printf("in%d[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
             "vertexIdx.z+(%d))];",
             field, -STENCIL_ORDER / 2 + width,
             -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);
      printf("}");

      cell_initialized[field][depth][height][width] = 1;
    }
  }
  }
  */

        /*
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width] &&
                !cell_initialized[field][depth][height][width]) {
              printf("const auto f%d_%d_%d_%d = ", //
                     field, depth, height, width);
              printf("in%d[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                     "vertexIdx.z+(%d))];",
                     field, -STENCIL_ORDER / 2 + width,
                     -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

              cell_initialized[field][depth][height][width] = 1;
            }
          }
        }
        */
      }
    }
  }
  /*
  // Prefetch stencil elements to local memory
  int cell_initialized[NUM_FIELDS][STENCIL_DEPTH][STENCIL_HEIGHT]
                      [STENCIL_WIDTH] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width] &&
                !cell_initialized[field][depth][height][width]) {
              printf("const auto f%d_%d_%d_%d = ", //
                     field, depth, height, width);
              printf("in%d[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                     "vertexIdx.z+(%d))];",
                     field, -STENCIL_ORDER / 2 + width,
                     -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);

              cell_initialized[field][depth][height][width] = 1;
            }
          }
        }
      }
    }
  }
  */

  // Prefetch stencil coefficients to local memory
  int coeff_initialized[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT]
                       [STENCIL_WIDTH] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

          int stencil_accessed = 0;
          for (int field = 0; field < NUM_FIELDS; ++field)
            stencil_accessed |= stencils_accessed[curr_kernel][field][stencil];
          if (!stencil_accessed)
            continue;

          if (stencils[stencil][depth][height][width] &&
              !coeff_initialized[stencil][depth][height][width]) {
            printf("const auto s%d_%d_%d_%d = ", //
                   stencil, depth, height, width);

            // CT const
            // printf("%s;", stencils[stencil][depth][height][width]);
            printf("stencils[%d][%d][%d][%d];", stencil, depth, height, width);

            coeff_initialized[stencil][depth][height][width] = 1;
          }
        }
      }
    }
  }

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int field = 0; field < NUM_FIELDS; ++field) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("%s(s%d_%d_%d_%d*"
                       "f%d_%d_%d_%d);",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, field, depth, height, width);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf( //
                    "%s(f%d_s%d,%s(s%d_%d_%d_%d*"
                    "f%d_%d_%d_%d));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil], stencil, depth, height, width,
                    field, depth, height, width);
              }
            }
          }
        }
      }
    }
  }

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}
#endif // 0
#elif IMPLEMENTATION == FULLY_EXPL_REG_VARS_AND_HALO_THREADS_1D_SHUFFLE
void
gen_kernel_body(const int curr_kernel)
{
  printf("const int RX = (STENCIL_WIDTH - 1)/2;");
  printf("const int3 vertexIdx = (int3){"
         "threadIdx.x + blockIdx.x * (blockDim.x - 2*RX) + start.x - RX,"
         "threadIdx.y + blockIdx.y * blockDim.y + start.y,"
         "threadIdx.z + blockIdx.z * blockDim.z + start.z,"
         "};");

  printf("if (vertexIdx.x >= end.x + RX) return;");

  /*
  printf("const int3 vertexIdx = (int3){"
         "threadIdx.x + blockIdx.x * blockDim.x + start.x,"
         "threadIdx.y + blockIdx.y * blockDim.y + start.y,"
         "threadIdx.z + blockIdx.z * blockDim.z + start.z,"
         "};");
         */
  printf("const int3 globalVertexIdx = (int3){"
         "d_multigpu_offset.x + vertexIdx.x,"
         "d_multigpu_offset.y + vertexIdx.y,"
         "d_multigpu_offset.z + vertexIdx.z,"
         "};");
  printf("const int3 globalGridN = d_mesh_info.int3_params[AC_global_grid_n];");
  printf("const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);");

  printf("(void)globalVertexIdx;"); // Silence unused warning
  printf("(void)globalGridN;");     // Silence unused warning

  for (int field = 0; field < NUM_FIELDS; ++field)
    printf("const auto f%d_prev = vba.out[%d][idx];", field, field);

  printf("const auto previous __attribute__((unused)) = [&](const Field field)"
         "{ switch (field) {");
  for (int field = 0; field < NUM_FIELDS; ++field)
    printf("case %d: { return f%d_prev; }", field, field);

  printf("default: return (AcReal)NAN;"
         "}");
  printf("};");
  printf("const auto write=[&](const Field field, const AcReal value)"
         "{ vba.out[field][idx] = value; };");

  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
      if (stencils_accessed[curr_kernel][field][stencil]) {
        printf("const AcReal* __restrict__ in%d = vba.in[%d];", field, field);
        break;
      }
    }
  }

#if 1 // 1.5D shuffle
  for (int height = 0; height < STENCIL_HEIGHT; ++height)
    printf("AcReal mid%d;", height);
  const int RX = (STENCIL_WIDTH - 1) / 2;
  const int RY = (STENCIL_HEIGHT - 1) / 2;
  const int RZ = (STENCIL_DEPTH - 1) / 2;

  // Prefetch stencil elements to local memory
  int cell_initialized[NUM_FIELDS][STENCIL_DEPTH][STENCIL_HEIGHT]
                      [STENCIL_WIDTH] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      // Prefetch midpoint to local memory
      for (int height = 0; height < STENCIL_HEIGHT; ++height)
        printf("mid%d = "
               "in%d[IDX(vertexIdx.x,vertexIdx.y + (%d), vertexIdx.z + (%d))];",
               height, field, height - RY, depth - RZ);
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width] &&
                !cell_initialized[field][depth][height][width]) {
              printf("const auto f%d_%d_%d_%d = ", //
                     field, depth, height, width);
              if (width < RX) {
                printf("__shfl_up_sync(0xffffffff, mid%d, %d);", height,
                       RX - width);
              }
              else if (width == RX) {
                printf("mid%d;", height);
              }
              else {
                printf("__shfl_down_sync(0xffffffff, mid%d, %d);", height,
                       width - RX);
              }
              cell_initialized[field][depth][height][width] = 1;
            }
          }
        }
      }
    }
  }
#else // 1D shuffle
  printf("AcReal mid;");
  const int RX = (STENCIL_WIDTH - 1) / 2;
  const int RY = (STENCIL_HEIGHT - 1) / 2;
  const int RZ = (STENCIL_DEPTH - 1) / 2;

  // Prefetch stencil elements to local memory
  int cell_initialized[NUM_FIELDS][STENCIL_DEPTH][STENCIL_HEIGHT]
                      [STENCIL_WIDTH] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        // Prefetch midpoint to local memory
        printf("mid = "
               "in%d[IDX(vertexIdx.x,vertexIdx.y + (%d), vertexIdx.z + (%d))];",
               field, height - RY, depth - RZ);
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width] &&
                !cell_initialized[field][depth][height][width]) {
              printf("const auto f%d_%d_%d_%d = ", //
                     field, depth, height, width);
              if (width < RX) {
                printf("__shfl_up_sync(0xffffffff, mid, %d);", RX - width);
              }
              else if (width == RX) {
                printf("mid;");
              }
              else {
                printf("__shfl_down_sync(0xffffffff, mid, %d);", width - RX);
              }
              cell_initialized[field][depth][height][width] = 1;
            }
          }
        }
      }
    }
  }
#endif

  // Prefetch stencil coefficients to local memory
  int coeff_initialized[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT]
                       [STENCIL_WIDTH] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

          int stencil_accessed = 0;
          for (int field = 0; field < NUM_FIELDS; ++field)
            stencil_accessed |= stencils_accessed[curr_kernel][field][stencil];
          if (!stencil_accessed)
            continue;

          if (stencils[stencil][depth][height][width] &&
              !coeff_initialized[stencil][depth][height][width]) {
            printf("const auto s%d_%d_%d_%d = ", //
                   stencil, depth, height, width);

            // CT const
            // printf("%s;", stencils[stencil][depth][height][width]);
            printf("stencils[%d][%d][%d][%d];", stencil, depth, height, width);

            coeff_initialized[stencil][depth][height][width] = 1;
          }
        }
      }
    }
  }

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
    for (int height = 0; height < STENCIL_HEIGHT; ++height) {
      for (int width = 0; width < STENCIL_WIDTH; ++width) {
        for (int field = 0; field < NUM_FIELDS; ++field) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("%s(s%d_%d_%d_%d*"
                       "f%d_%d_%d_%d);",
                       stencil_unary_ops[stencil], stencil, depth, height,
                       width, field, depth, height, width);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf( //
                    "%s(f%d_s%d,%s(s%d_%d_%d_%d*"
                    "f%d_%d_%d_%d));",
                    stencil_binary_ops[stencil], field, stencil,
                    stencil_unary_ops[stencil], stencil, depth, height, width,
                    field, depth, height, width);
              }
            }
          }
        }
      }
    }
  }

  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencil_initialized[field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }

  printf("if (threadIdx.x < RX) return;");
  printf("if (threadIdx.x >= blockDim.x - RX) return;");
}
#endif

int
main(int argc, char** argv)
{
  (void)vtxbuf_names; // Unused

  // Generate stencil definitions
  if (argc == 2 && !strcmp(argv[1], "-definitions")) {
    gen_stencil_definitions();
  }
  else if (argc == 2 && !strcmp(argv[1], "-mem-accesses")) {
    gen_stencil_accesses();
  }
  // Generate memory accesses for the DSL kernels
  else if (argc == 3) {
    const int curr_kernel = atoi(argv[2]);
    gen_kernel_body(curr_kernel);
  }
  else {
    fprintf(stderr, "Fatal error: invalid arguments passed to stencilgen.c");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
