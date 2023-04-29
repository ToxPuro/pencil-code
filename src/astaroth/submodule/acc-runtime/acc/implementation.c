#include "implementation.h"

typedef struct {
  size_t x, y, z;
} Volume;

static Volume
as_volume(const int x, const int y, const int z)
{
  return (Volume){as_size_t(x), as_size_t(y), as_size_t(z)};
}

static size_t
get_smem(const Volume tpb, const size_t stencil_order,
         const size_t bytes_per_elem)
{
  return 0;
}

static bool
is_valid_configuration(const Volume tpb, const Volume dim)
{
  return false;
}