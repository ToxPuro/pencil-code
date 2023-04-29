program pc
  use, intrinsic :: iso_c_binding
  implicit none

  include "astaroth.f90"

  type(AcMeshInfo) :: info
  type(c_ptr) :: device

  print *, "Num int params"
  print *, AC_NUM_INT_PARAMS

  ! Setup config
  info%int_params(AC_nx + 1) = 128
  info%int_params(AC_ny + 1) = 128
  info%int_params(AC_nz + 1) = 128
  call achostupdatebuiltinparams(info)

  call acdevicecreate(0, info, device)
  call acdeviceprintinfo(device)
  call acdevicedestroy(device)

end program
