# Building ACC runtime (incl. DSL files)

The DSL source files should have a postfix `*.ac` and there should be only one
`.ac` file per directory.

    * `mkdir build`

    * `cd build`

    * `cmake -DDSL_MODULE_DIR=<optional path to the dir containing DSL sources> ..`

    * `make -j`


## Debugging

As ACC is in active development, compiler bugs and cryptic error messages are
expected. In case of issues, please check the following files in
`acc-runtime/api` in the build directory.

Intermediate files:

1. `user_kernels.ac.pp_stage*`. The DSL file after a specific preprocessing stage.
1. `user_kernels.h.raw`. The raw generated CUDA kernels without formatting applied.

Final files:

1. `user_defines.h`. The project-wide defines generated with the DSL.
1. `user_declarations.h`. Forward declarations of user kernels.
1. `user_kernels.h`. The generated CUDA kernels.

To make inspecting the code easier, we recommend using an
autoformatting tool, for example, `clang-format` or GNU `indent`.


## Known issues

  * The final function call of a kernel gets sometimes dropped due to an unknown reason. For instance:
  ```
  Kernel kernel() {
    ...
    device_function(...) // This call is not present in `user_kernels.h`
  }
  ```
  If you're able to reproduce this, please create a bitbucket issue with the incorrectly translated DSL code.

# The Astaroth Domain-Specific Language

The Astaroth Domain-Specific Language (DSL) is a high-level GPGPU language
designed for improved productivity and performance in stencil computations. The
Astaroth DSL compiler (acc) is a source-to-source compiler, which converts
DSL kernels into CUDA/HIP kernels. The generated kernels provide performance
that is on-par with hand-tuned low-level GPGPU code in stencil computations.
Special care has been taken to ensure efficient code generation in use cases
encountered in computational physics, where there are multiple coupled fields,
which makes manual caching notoriously difficult.

The Astaroth DSL is based on the stream processing model, where an array of
instructions is executed on streams of data in parallel. A kernel is a small
GPU program, which defines the operations performed on a number of data streams.
In our case, data streams correspond to a vertices in a grid, similar to how
vertex shaders operate in graphics shading languages.

# Syntax

#### Comments and preprocessor directives
```
// This is a comment
#define    ZERO (0)   // Visible only in device code
hostdefine ONE  (1) // Visible in both device and host code
```

#### Variables
```
real var    // Explicit type declaration
real dconst // The type of device constants must be explicitly specified

var0 = 1    // The type of local variables can be left out (implicit typing)
var1 = 1.0  // Implicit precision (determined based on compilation flags)
var2 = 1.   // Trailing zero can be left out
var3 = 1e3  // E notation
var4 = 1.f  // Explicit single-precision
var5 = 0.1d // Explicit double-precision
var6 = "Hello"
```

> Note: Shadowing is not allowed, all identifiers within a scope must be unique

#### Arrays
```
int arr0 = 1, 2, 3 // The type of arrays must be explicitly specified
real arr1 = 1.0, 2.0, 3.0
len(arr1) // Length of an array
```

#### Casting
```
var7 = real(1)        // Cast
vec0 = real3(1, 2, 3) // Cast
```

#### Printing
```
// print is the same as `printf` in the C programming language
print("Hello from thread (%d, %d, %d)\n", vertexIdx.x, vertexIdx.y, vertexIdx.z)
```

#### Looping
```
int arr = 1, 2, 3
for var in arr {
    print("%d\n", var)
}

i = 0
while i < 3 {
    i += 1
}

for i in 0:10 { // Note: 10 is exclusive
  print("%d", i)
}
```

#### Functions
```
func(param) {
    print("%s", param)
}

func2(val) {
    return val
}

# Note `Kernel` type qualifier
Kernel func3() {
    func("Hello!")
}
```

> Note: Function parameters are **passed by constant reference**. Therefore input parameters **cannot be modified** and one may need to allocate temporary storage for intermediate values when performing more complex calculations.

> Note: Overloading is not allowed, all function identifiers must be unique

#### Stencils
```
// Format
<Optional reduction operation> Stencil <identifier> {
    [z][y][x] = coefficient,
    ...
}
// where [z][y][x] is the x/y/z offset from current position.

// For example,
Stencil example {
    [0][0][-1] = a,
    [0][0][0] = b
    [0][0][1] = c,
}

// Which is equivalent to
Sum Stencil example {
    ...
}

// and is calculated equivalently to
example(field) {
    return  a * field[IDX(vertexIdx.x - 1, vertexIdx.y, vertexIdx.z)] +
            b * field[IDX(vertexIdx.x,     vertexIdx.y, vertexIdx.z)] +
            c * field[IDX(vertexIdx.x + 1, vertexIdx.y, vertexIdx.z)]
}

By default, the binary operation for reducing `Stencil` elements is `Sum` (as above). Currently supported operations are `Sum` and `Max`. See the up-to-date list of the supported operations in `acc-runtime/acc/ac.y` rule `type_qualifier`.

// Real-world example
Max Stencil largest_neighbor {
    [1][0][0]  = 1,
    [-1][0][0] = 1,
    [0][1][0]  = 1,
    [0][-1][0] = 1,
    [0][0][1]  = 1,
    [0][0][-1] = 1,
}

Stencil derx {
    [0][0][-3] = -DER1_3,
    [0][0][-2] = -DER1_2,
    [0][0][-1] = -DER1_1,
    [0][0][1]  = DER1_1,
    [0][0][2]  = DER1_2,
    [0][0][3]  = DER1_3
}

Stencil dery {
    [0][-3][0] = -DER1_3,
    [0][-2][0] = -DER1_2,
    [0][-1][0] = ...
}

Stencil derz {
    [-3][0][0] = -DER1_3,
    [-2][0][0] = ...
}

gradient(field) {
    return real3(derx(field), dery(field), derz(field))
}
```

> Note: Stencil coefficients supplied in the DSL source must be compile-time constants. To set up coefficients at runtime, see [instructions below](#loading-and-storing-stencil-coefficients-at-runtime).

> Note: To reduce redundant communication or to enable larger stencils, the stencil order can be changed by modifying `static const size_t stencil_order = ...` in `acc-runtime/acc/codegen.c`. Modifying the stencil order with the DSL is currently not supported.


#### Fields

A `Field` is a scalar array that can be used in conjuction with `Stencil` operations. For convenience, a vector field can be constructed from three scalar fields by declaring them a `Field3` structure.
```
Field ux, uy, uz // Three scalar fields `ux`, `uy`, and `uz`
#define uu Field3(ux, uy, uz) // A vector field `uu` consisting of components `ux`, `uy`, and `uz`

Kernel kernel() {
    write(ux, derx(ux)) // Writes the x derivative of the field `ux` to the output buffer
}
```

#### Built-in variables and functions
```
// Variables
dim3 threadIdx       // Current thread index within a thread block (see CUDA docs)
dim3 blockIdx        // Current thread block index (see CUDA docs)
dim3 vertexIdx       // The current vertex index within a single device
dim3 globalVertexIdx // The current vertex index across multiple devices
dim3 globalGridN     // The total size of the computational domain (incl. all subdomains of all processes)

// Functions
void write(Field, real)  // Writes a real value to the output field at 'vertexIdx'
void print("int: %d", 0) // Printing. Uses the same syntax as printf() in C
real dot(real3, real3)   // Dot product
real3 cross(real3 a, real3 b) // Right-hand-side cross product a x b
size_t len(arr) // Returns the length of an array `arr`

// Trigonometric functions
exp
sin
cos
sqrt
fabs

// Advanced functions (should avoid, dangerous)
real previous(Field) // Returns the value in the output buffer. Read after write() results in undefined behaviour.

// Constants
real AC_REAL_PI // Value of pi using the same precision as `real`
```

> See astaroth/acc-runtime/acc/codegen.c, function `symboltable_reset` for an up-to-date list of all built-in symbols.

# Advanced

The input and output arrays can also be accessed without declaring a `Stencil` as follows.

```
Field field0

Kernel kernel() {
  // The example showcases two ways of accessing a field element without the Stencil structure
  a = FIELD_IN[field0][IDX(vertexIdx)] // Note that IDX() here accepts the 3D spatial index
  b = FIELD_OUT[field0][IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)] // And also individual index components
}
```

> Note: Accessing field elements using `FIELD_IN` and `FIELD_OUT` does not cache the reads and is significantly less efficient than using a `Stencil`.

# Interaction with the Astaroth Core and Utils libraries

## Loading and storing stencil coefficients at runtime

The stencil coefficients defined in the DSL syntax must be known at compile time for simplicity. However, the Astaroth Runtime API provides the functions `acLoadStencil` and `acStoreStencil` for loading/storing stencil coefficients at runtime. This is useful for, say, trying out different coefficients without the need for recompilation and for setting the coefficients programmatically if too cumbersome by hand.

See also the functions `acDeviceLoadStencil`, `acDeviceStoreStencil`, `acGridLoadStencil`, and `acGridStoreStencil` provided by the Astaroth Core library.


## Additional physics-specific API functions

To enable additional API functions in the Astaroth Core library for integration (`acIntegrate` function family) and MHD-specific tasks (automated testing, MHD samples), one must set `hostdefine AC_INTEGRATION_ENABLED (1)` in the DSL file. Note that if used in the DSL code, the hostdefine must not define anything that is not visible at compile-time. For example, `hostdefine R_PI (M_PI)`, where `M_PI` is defined is some host header, `M_PI` will not be visible in the DSL code and will result in a compilation error. Additionally, code such as `#if M_PI` will be always false in the DSL source if `M_PI` is not visible in the DSL file.

> Note: The extended API depends on several hardcoded fields and device constants. It is not recommended to enable it unless you work on the MHD sample case (`acc-runtime/samples/mhd`) or its derivatives.

## Stencil order

The stencil order can be set by the user by `hostdefine STENCIL_ORDER (x)`, where `x` is the total number of cells on both sides of the center point per axis. For example, a simple von Neumann stencil is of order 2.

> Note: The size of the halo surrounding the computational domain depends on `STENCIL_ORDER`.