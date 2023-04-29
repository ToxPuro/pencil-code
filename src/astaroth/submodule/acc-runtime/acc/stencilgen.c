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
#include <stdbool.h>
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

static void
gen_return_if_oob(void)
{
  printf("if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || "
         "vertexIdx.z >= end.z) { return; }");
}

static void
prefetch_output_elements_and_gen_prev_function(void)
{
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
}

void
gen_stencil_accesses(void)
{
  gen_kernel_prefix();
  gen_return_if_oob();
  prefetch_output_elements_and_gen_prev_function();

  printf("AcReal /*__restrict__*/ "
         "processed_stencils[NUM_FIELDS][NUM_STENCILS];");

  for (size_t i = 0; i < NUM_STENCILS; ++i)
    printf("const auto %s=[&](const auto field)"
           "{stencils_accessed[field][stencil_%s]=1;return AcReal(1.0);};",
           stencil_names[i], stencil_names[i]);
}

/** ct_const_weights: Compile-time constant weights
  If ct_const_weights = false, the stencil coeffs are fetched from constant
  memory at runtime
*/
/*
static void
prefetch_stencil_coeffs(const int curr_kernel, const bool ct_const_weights)
{
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

            if (ct_const_weights)
              printf("%s;", stencils[stencil][depth][height][width]);
            else
              printf("stencils[%d][%d][%d][%d];", stencil, depth, height,
                     width);

            coeff_initialized[stencil][depth][height][width] = 1;
          }
        }
      }
    }
  }
}

static void
prefetch_stencil_elements(const int curr_kernel)
{
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
#if !AC_USE_HIP
              printf("__ldg(&");
#endif
              printf("vba.in[%d][IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                     "vertexIdx.z+(%d))]",
                     field, -STENCIL_ORDER / 2 + width,
                     -STENCIL_ORDER / 2 + height, -STENCIL_ORDER / 2 + depth);
#if !AC_USE_HIP
              printf(")");
#endif
              printf(";");
              cell_initialized[field][depth][height][width] = 1;
            }
          }
        }
      }
    }
  }
}

static void
compute_stencil_ops(const int curr_kernel)
{
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
                printf("s%d_%d_%d_%d * %s(f%d_%d_%d_%d);", //
                       stencil, depth, height, width,
                       stencil_unary_ops[stencil], field, depth, height, width);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d,"
                       "s%d_%d_%d_%d * %s(f%d_%d_%d_%d)"
                       ");",
                       stencil_binary_ops[stencil], field, stencil, //
                       stencil, depth, height, width,               //
                       stencil_unary_ops[stencil], field, depth, height, width);
              }
            }
          }
        }
      }
    }
  }

  for (int field = 0; field < NUM_FIELDS; ++field)
    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil)
      if (stencil_initialized[field][stencil] !=
          stencils_accessed[curr_kernel][field][stencil])
        raise_error("stencil_initialized != stencil_accessed, this affects "
                    "gen_stencil_functions (stencil_accessed should be "
                    "replaced with stencil_initialized)");
}
*/

static void
gen_stencil_functions(const int curr_kernel)
{
  for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
    printf("const auto %s __attribute__((unused)) = [&](const auto field){",
           stencil_names[stencil]);
    printf("switch (field) {");
    for (int field = 0; field < NUM_FIELDS; ++field) {
      if (stencils_accessed[curr_kernel][field][stencil])
        printf("case %d: return f%d_s%d;", field, field, stencil);
    }
    printf("default: return (AcReal)NAN;");
    printf("}");
    printf("};");
  }
}

/** Supports 2.5D and 2D smem blocking (see `rolling_cache` switch) */
static void
prefetch_stencil_elems_to_smem_and_compute_stencil_ops(const int curr_kernel)
{
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x;");

  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "threadIdx.z + blockIdx.z * blockDim.z + start.z - "
         "(STENCIL_DEPTH-1)/2};");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {
    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {

      const bool rolling_cache = true;
      if (rolling_cache) {
        // 2.5D blocking with smem

        // Fetch from gmem
        printf("if (%d == 0 || threadIdx.z == blockDim.z - 1) {", depth);
        printf("for (int curr = sid; curr < sx * sy;"
               "curr += blockDim.x * blockDim.y) {");
        printf("const int i = curr %% sx;");
        printf("const int j = curr / sx;");
        printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ break; }");
        printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ break; }");
        printf("if (baseIdx.z + (%d) >= end.z + (STENCIL_DEPTH-1)/2){ break; }",
               depth);
        printf("smem[i + j * sx + ((threadIdx.z+%d)%%blockDim.z) * sx * sy] = ",
               depth);
        printf("vba.in[%d]"
               "[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + (%d))];",
               field, depth);
        printf("}");
        printf("}");
        printf("__syncthreads();");
      }
      else {
        // 2D blocking with smem
        printf("for (int curr = sid; curr < sx * sy;"
               "curr += blockDim.x * blockDim.y) {");
        printf("const int i = curr %% sx;");
        printf("const int j = curr / sx;");
        printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ break; }");
        printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ break; }");
        printf("if (baseIdx.z + (%d) >= end.z + (STENCIL_DEPTH-1)/2){ break; }",
               depth);
        printf("smem[i + j * sx + ((threadIdx.z+%d)%%blockDim.z) * sx * sy] = ",
               depth);
        printf("vba.in[%d]"
               "[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + (%d))];",
               field, depth);
        printf("}");
        printf("__syncthreads();");
      }

      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
        for (int width = 0; width < STENCIL_WIDTH; ++width) {
          for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {

            // Skip if the stencil is not used
            if (!stencils_accessed[curr_kernel][field][stencil])
              continue;

            if (stencils[stencil][depth][height][width]) {
              if (!stencil_initialized[field][stencil]) {
                printf("auto f%d_s%d = ", field, stencil);
                printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                       width);
                printf("%s(smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "((threadIdx.z+%d)%%blockDim.z) * sx * sy]);",
                       stencil_unary_ops[stencil], width, height, depth);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
                       stencil);
                printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                       width);
                printf("%s(smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "((threadIdx.z+%d)%%blockDim.z) * sx * sy])",
                       stencil_unary_ops[stencil], width, height, depth);
                printf(");");
              }
            }
          }
        }
      }
      printf("__syncthreads();");
    }
  }
}

/** Supports 3D smem blocking (see `rolling_cache` switch) */
static void
prefetch_stencil_elems_to_smem_3d_and_compute_stencil_ops(const int curr_kernel)
{
  printf("extern __shared__ AcReal smem[];");
  printf("const int sx = blockDim.x + STENCIL_WIDTH - 1;");
  printf("const int sy = blockDim.y + STENCIL_HEIGHT - 1;");
  printf("const int sz = blockDim.z + STENCIL_DEPTH - 1;");
  printf("const int sid = threadIdx.x + "
         "threadIdx.y * blockDim.x + "
         "threadIdx.z * blockDim.x * blockDim.y;");

  printf("const int3 baseIdx = (int3){"
         "blockIdx.x * blockDim.x + start.x - (STENCIL_WIDTH-1)/2,"
         "blockIdx.y * blockDim.y + start.y - (STENCIL_HEIGHT-1)/2,"
         "blockIdx.z * blockDim.z + start.z - (STENCIL_DEPTH-1)/2};");
  printf("const int tpb = blockDim.x * blockDim.y * blockDim.z;");

  int stencil_initialized[NUM_FIELDS][NUM_STENCILS] = {0};
  for (int field = 0; field < NUM_FIELDS; ++field) {

    printf("for (int curr = sid; curr < sx * sy * sz; curr += tpb) {");
    printf("const int i = curr %% sx;");
    printf("const int j = (curr %% (sx * sy)) / sx;");
    printf("const int k = curr / (sx * sy);");
    printf("if (baseIdx.x + i >= end.x + (STENCIL_WIDTH-1)/2){ break; }");
    printf("if (baseIdx.y + j >= end.y + (STENCIL_HEIGHT-1)/2){ break; }");
    printf("if (baseIdx.z + k >= end.z + (STENCIL_DEPTH-1)/2){ break; }");
    printf("smem[i + j * sx + k * sx * sy] = ");
#if !AC_USE_HIP
    printf("__ldg(&");
#endif
    printf("vba.in[%d]", field);
    printf("[IDX(baseIdx.x + i, baseIdx.y + j, baseIdx.z + k)]");
#if !AC_USE_HIP
    printf(")");
#endif
    printf(";");
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
                printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                       width);
                printf("%s(smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "(threadIdx.z + %d) * sx * sy]);",
                       stencil_unary_ops[stencil], width, height, depth);

                stencil_initialized[field][stencil] = 1;
              }
              else {
                printf("f%d_s%d = ", field, stencil);
                printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
                       stencil);
                printf("stencils[%d][%d][%d][%d] * ", stencil, depth, height,
                       width);
                printf("%s(smem[(threadIdx.x + %d) + "
                       "(threadIdx.y + %d) * sx + "
                       "(threadIdx.z + %d) * sx * sy])",
                       stencil_unary_ops[stencil], width, height, depth);
                printf(");");
              }
            }
          }
        }
      }
    }
    printf("__syncthreads();");
  }
}

void
gen_kernel_body(const int curr_kernel)
{
  switch (IMPLEMENTATION) {
  case IMPLICIT_CACHING: {
    gen_kernel_prefix();
    gen_return_if_oob();
    prefetch_output_elements_and_gen_prev_function();

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
                  printf("stencils[%d][%d][%d][%d] *", //
                         stencil, depth, height, width);
                  printf("%s(", stencil_unary_ops[stencil]);
#if !AC_USE_HIP
                  printf("__ldg(&");
#endif
                  printf("vba.in[%d]"
                         "[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                         "vertexIdx.z+(%d))])",
                         field, -STENCIL_ORDER / 2 + width,
                         -STENCIL_ORDER / 2 + height,
                         -STENCIL_ORDER / 2 + depth);
#if !AC_USE_HIP
                  printf(")");
#endif
                  printf(";");

                  stencil_initialized[field][stencil] = 1;
                }
                else {
                  printf("f%d_s%d = ", field, stencil);
                  printf("%s(f%d_s%d, ", stencil_binary_ops[stencil], field,
                         stencil);
                  printf("stencils[%d][%d][%d][%d] *", //
                         stencil, depth, height, width);
                  printf("%s(", stencil_unary_ops[stencil]);
#if !AC_USE_HIP
                  printf("__ldg(&");
#endif
                  printf("vba.in[%d]"
                         "[IDX(vertexIdx.x+(%d),vertexIdx.y+(%d), "
                         "vertexIdx.z+(%d))])",
                         field, -STENCIL_ORDER / 2 + width,
                         -STENCIL_ORDER / 2 + height,
                         -STENCIL_ORDER / 2 + depth);
#if !AC_USE_HIP
                  printf(")");
#endif
                  printf(");");
                }
              }
            }
          }
        }
      }
    }

    gen_stencil_functions(curr_kernel);
    /*
    gen_kernel_prefix();
    gen_return_if_oob();

    prefetch_output_elements_and_gen_prev_function();
    prefetch_stencil_elements(curr_kernel);
    prefetch_stencil_coeffs(curr_kernel, false);

    compute_stencil_ops(curr_kernel);
    gen_stencil_functions(curr_kernel);
    */

    return;
  }
  case EXPLICIT_CACHING: {
    gen_kernel_prefix(); // Note no bounds check

    prefetch_stencil_elems_to_smem_and_compute_stencil_ops(curr_kernel);
    gen_return_if_oob();

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function();
    return;
  }
  case EXPLICIT_CACHING_3D_BLOCKING: {
    gen_kernel_prefix(); // Note no bounds check

    prefetch_stencil_elems_to_smem_3d_and_compute_stencil_ops(curr_kernel);
    gen_return_if_oob();

    gen_stencil_functions(curr_kernel);
    prefetch_output_elements_and_gen_prev_function();
    return;
  }
  default: {
    fprintf(stderr,
            "Fatal error: invalid IMPLEMENTATION passed to stencilgen.c");
    return;
  }
  }
}

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
