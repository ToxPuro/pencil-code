! $Id$
!
!  Dummy module for MPI communication. This allows the code to run on a
!  single CPU.
!
module Mpicomm
!
  use Cdata
  use Cparam
!
  implicit none
!
  include 'mpicomm.h'
!
  interface mpirecv_logical
     module procedure mpirecv_logical_scl
     module procedure mpirecv_logical_arr
  endinterface
!
  interface mpirecv_real
    module procedure mpirecv_real_scl
    module procedure mpirecv_real_arr
    module procedure mpirecv_real_arr2
    module procedure mpirecv_real_arr3
    module procedure mpirecv_real_arr4
  endinterface
!
  interface mpirecv_int
    module procedure mpirecv_int_scl
    module procedure mpirecv_int_arr
  endinterface
!
  interface mpisend_logical
     module procedure mpisend_logical_scl
     module procedure mpisend_logical_arr
  endinterface
!
  interface mpisend_real
    module procedure mpisend_real_scl
    module procedure mpisend_real_arr
    module procedure mpisend_real_arr2
    module procedure mpisend_real_arr3
    module procedure mpisend_real_arr4
  endinterface
!
  interface mpisendrecv_real
    module procedure mpisendrecv_real_scl
    module procedure mpisendrecv_real_arr
    module procedure mpisendrecv_real_arr2
    module procedure mpisendrecv_real_arr3
    module procedure mpisendrecv_real_arr4
  endinterface
!
  interface mpisend_int
    module procedure mpisend_int_scl
    module procedure mpisend_int_arr
  endinterface
!
  interface mpibcast_logical
    module procedure mpibcast_logical_scl
    module procedure mpibcast_logical_arr
    module procedure mpibcast_logical_arr2
  endinterface
!
  interface mpibcast_int
    module procedure mpibcast_int_scl
    module procedure mpibcast_int_arr
  endinterface
!
  interface mpibcast_real
    module procedure mpibcast_real_scl
    module procedure mpibcast_real_arr
    module procedure mpibcast_real_arr2
    module procedure mpibcast_real_arr3
    module procedure mpibcast_real_arr4
  endinterface
!
  interface mpibcast_double
    module procedure mpibcast_double_scl
    module procedure mpibcast_double_arr
  endinterface
!
  interface mpibcast_cmplx
    module procedure mpibcast_cmplx_arr_sgl
  endinterface
!
  interface mpibcast_char
    module procedure mpibcast_char_scl
    module procedure mpibcast_char_arr
  endinterface
!
  interface mpireduce_sum_int
    module procedure mpireduce_sum_int_scl
    module procedure mpireduce_sum_int_arr
    module procedure mpireduce_sum_int_arr2
    module procedure mpireduce_sum_int_arr3
    module procedure mpireduce_sum_int_arr4
  endinterface
!
  interface mpireduce_sum
    module procedure mpireduce_sum_scl
    module procedure mpireduce_sum_arr
    module procedure mpireduce_sum_arr2
    module procedure mpireduce_sum_arr3
    module procedure mpireduce_sum_arr4
  endinterface
!
  interface mpireduce_sum_double
    module procedure mpireduce_sum_double_scl
    module procedure mpireduce_sum_double_arr
    module procedure mpireduce_sum_double_arr2
    module procedure mpireduce_sum_double_arr3
    module procedure mpireduce_sum_double_arr4
  endinterface
!
  interface mpireduce_max
    module procedure mpireduce_max_scl
    module procedure mpireduce_max_arr
  endinterface
!
  interface mpireduce_max_int
    module procedure mpireduce_max_scl_int
  endinterface
!
  interface mpiallreduce_sum
    module procedure mpiallreduce_sum_scl
    module procedure mpiallreduce_sum_arr
    module procedure mpiallreduce_sum_arr2
    module procedure mpiallreduce_sum_arr3
    module procedure mpiallreduce_sum_arr4
    module procedure mpiallreduce_sum_arr5
  endinterface
!
  interface mpiallreduce_sum_int
    module procedure mpiallreduce_sum_int_scl
    module procedure mpiallreduce_sum_int_arr
  endinterface
!
  interface mpiallreduce_max
    module procedure mpiallreduce_max_scl
    module procedure mpiallreduce_max_arr
  endinterface
!
  interface mpiallreduce_min_sgl
    module procedure mpiallreduce_min_scl_sgl
  endinterface
!
  interface mpiallreduce_min_dbl
    module procedure mpiallreduce_min_scl_dbl
  endinterface
!
  interface mpiallreduce_or
    module procedure mpiallreduce_or_scl
  endinterface
!
  interface mpireduce_min
    module procedure mpireduce_min_scl
    module procedure mpireduce_min_arr
  endinterface
!
  interface mpireduce_or
    module procedure mpireduce_or_scl
    module procedure mpireduce_or_arr
  endinterface
!
  interface mpireduce_and
    module procedure mpireduce_and_scl
    module procedure mpireduce_and_arr
  endinterface
!
  interface distribute_xy
    module procedure distribute_xy_0D
    module procedure distribute_xy_2D
    module procedure distribute_xy_3D
    module procedure distribute_xy_4D
  endinterface
!
  interface collect_xy
    module procedure collect_xy_0D
    module procedure collect_xy_2D
    module procedure collect_xy_3D
    module procedure collect_xy_4D
  endinterface
!
  interface distribute_z
    module procedure distribute_z_3D
    module procedure distribute_z_4D
  endinterface
!
  interface collect_z
    module procedure collect_z_3D
    module procedure collect_z_4D
  endinterface
!
  interface distribute_to_pencil_xy
    module procedure distribute_to_pencil_xy_2D
  endinterface
!
  interface collect_from_pencil_xy
    module procedure collect_from_pencil_xy_2D
  endinterface
!
  interface remap_to_pencil_y
    module procedure remap_to_pencil_y_1D
    module procedure remap_to_pencil_y_2D
    module procedure remap_to_pencil_y_3D
    module procedure remap_to_pencil_y_4D
  endinterface
!
  interface unmap_from_pencil_y
    module procedure unmap_from_pencil_y_1D
    module procedure unmap_from_pencil_y_2D
    module procedure unmap_from_pencil_y_3D
    module procedure unmap_from_pencil_y_4D
  endinterface
!
  interface remap_to_pencil_z
    module procedure remap_to_pencil_z_1D
    module procedure remap_to_pencil_z_2D
    module procedure remap_to_pencil_z_3D
    module procedure remap_to_pencil_z_4D
  endinterface
!
  interface unmap_from_pencil_z
    module procedure unmap_from_pencil_z_1D
    module procedure unmap_from_pencil_z_2D
    module procedure unmap_from_pencil_z_3D
    module procedure unmap_from_pencil_z_4D
  endinterface
!
  interface remap_to_pencil_xy
    module procedure remap_to_pencil_xy_2D
    module procedure remap_to_pencil_xy_3D
    module procedure remap_to_pencil_xy_4D
  endinterface
!
  interface unmap_from_pencil_xy
    module procedure unmap_from_pencil_xy_2D
    module procedure unmap_from_pencil_xy_3D
    module procedure unmap_from_pencil_xy_4D
  endinterface
!
  interface transp_pencil_xy
    module procedure transp_pencil_xy_2D
    module procedure transp_pencil_xy_3D
    module procedure transp_pencil_xy_4D
  endinterface
!
  interface remap_to_pencil_yz
    module procedure remap_to_pencil_yz_3D
    module procedure remap_to_pencil_yz_4D
  endinterface
!
  interface unmap_from_pencil_yz
    module procedure unmap_from_pencil_yz_3D
    module procedure unmap_from_pencil_yz_4D
  endinterface
!
  interface mpirecv_nonblock_real
    module procedure mpirecv_nonblock_real_arr
    module procedure mpirecv_nonblock_real_arr4
  endinterface
!
  interface mpisend_nonblock_real
    module procedure mpisend_nonblock_real_arr
    module procedure mpisend_nonblock_real_arr4
  endinterface
!
  interface mpirecv_nonblock_int
    module procedure mpirecv_nonblock_int_scl
    module procedure mpirecv_nonblock_int_arr
  endinterface
!
  interface mpisend_nonblock_int
    module procedure mpisend_nonblock_int_scl
    module procedure mpisend_nonblock_int_arr
  endinterface
!
  interface parallel_open
    module procedure parallel_open_ext
    module procedure parallel_open_int
  endinterface
!
  interface parallel_close
    module procedure parallel_close_ext
    module procedure parallel_close_int
  endinterface
!
!  interface mpigather_and_out
!    module procedure mpigather_and_out_real
!    module procedure mpigather_and_out_cmplx
!  endinterface
!
  integer :: mpi_precision,MPI_CMPLX
!
  contains
!***********************************************************************
    subroutine mpicomm_init()
!
!  29-jul-2010/anders: dummy
!
      mpi_precision = -1; MPI_CMPLX=-1
!
    endsubroutine mpicomm_init
!***********************************************************************
    subroutine initialize_mpicomm()
!
!  Make a quick consistency check.
!
      if (ncpus>1) then
        call stop_it('Inconsistency: MPICOMM=nompicomm, but ncpus>=2')
      endif
!
!  For a single CPU run, set processor to zero.
!
      lmpicomm = .false.
      iproc = 0
      lroot = .true.
      ipx = 0
      ipy = 0
      ipz = 0
      lfirst_proc_x = .true.
      lfirst_proc_y = .true.
      lfirst_proc_z = .true.
      lfirst_proc_xy = .true.
      lfirst_proc_yz = .true.
      lfirst_proc_xz = .true.
      lfirst_proc_xyz = .true.
      llast_proc_x = .true.
      llast_proc_y = .true.
      llast_proc_z = .true.
      llast_proc_xy = .true.
      llast_proc_yz = .true.
      llast_proc_xz = .true.
      llast_proc_xyz = .true.
      ylneigh = 0
      zlneigh = 0
      yuneigh = 0
      zuneigh = 0
!
    endsubroutine initialize_mpicomm
!***********************************************************************
    subroutine initiate_isendrcv_bdry(f,ivar1_opt,ivar2_opt)
!
!  For one processor, use periodic boundary conditions.
!  In this dummy routine this is done in finalize_isendrcv_bdry.
!
      real, dimension (mx,my,mz,mfarray) :: f
      integer, optional :: ivar1_opt, ivar2_opt
!
      if (ALWAYS_FALSE) print*, f, ivar1_opt, ivar2_opt
!
    endsubroutine initiate_isendrcv_bdry
!***********************************************************************
    subroutine finalize_isendrcv_bdry(f,ivar1_opt,ivar2_opt)
!
!  Apply boundary conditions.
!
      real, dimension (mx,my,mz,mfarray) :: f
      integer, optional :: ivar1_opt, ivar2_opt
!
      if (ALWAYS_FALSE) print*, f, ivar1_opt, ivar2_opt
!
    endsubroutine finalize_isendrcv_bdry
!***********************************************************************
    subroutine isendrcv_bdry_x(f,ivar1_opt,ivar2_opt)
!
!  Dummy
!
      real, dimension(:,:,:,:), intent(in) :: f
      integer, intent(in), optional :: ivar1_opt, ivar2_opt
!
      if (ALWAYS_FALSE) print *, f, ivar1_opt, ivar2_opt
!
    endsubroutine isendrcv_bdry_x
!***********************************************************************
    subroutine initiate_shearing(f,ivar1_opt,ivar2_opt)
!
      real, dimension (mx,my,mz,mfarray) :: f
      integer, optional :: ivar1_opt, ivar2_opt
!
      if (ALWAYS_FALSE) print*, f, ivar1_opt, ivar2_opt
!
    endsubroutine initiate_shearing
!***********************************************************************
    subroutine finalize_shearing(f,ivar1_opt,ivar2_opt)
!
!  Shear-periodic boundary conditions in x (using just one CPU).
!
      real, dimension (mx,my,mz,mfarray) :: f
      integer, optional :: ivar1_opt, ivar2_opt
!
      double precision :: deltay_dy, frac, c1, c2, c3, c4, c5, c6
      integer :: ivar1, ivar2, displs
!
      ivar1=1; ivar2=mcom
      if (present(ivar1_opt)) ivar1=ivar1_opt
      if (present(ivar2_opt)) ivar2=ivar2_opt
!
      if (nygrid==1) then ! Periodic boundary conditions.
        f( 1:l1-1,:,:,ivar1:ivar2) = f(l2i:l2,:,:,ivar1:ivar2)
        f(l2+1:mx,:,:,ivar1:ivar2) = f(l1:l1i,:,:,ivar1:ivar2)
      else
        deltay_dy=deltay/dy
        displs=int(deltay_dy)
        frac=deltay_dy-displs
        c1 = -          (frac+1.)*frac*(frac-1.)*(frac-2.)*(frac-3.)/120.
        c2 = +(frac+2.)          *frac*(frac-1.)*(frac-2.)*(frac-3.)/24.
        c3 = -(frac+2.)*(frac+1.)     *(frac-1.)*(frac-2.)*(frac-3.)/12.
        c4 = +(frac+2.)*(frac+1.)*frac          *(frac-2.)*(frac-3.)/12.
        c5 = -(frac+2.)*(frac+1.)*frac*(frac-1.)          *(frac-3.)/24.
        c6 = +(frac+2.)*(frac+1.)*frac*(frac-1.)*(frac-2.)          /120.
        f( 1:l1-1,m1:m2,:,ivar1:ivar2) = &
             c1*cshift(f(l2i:l2,m1:m2,:,ivar1:ivar2),-displs+2,2) &
            +c2*cshift(f(l2i:l2,m1:m2,:,ivar1:ivar2),-displs+1,2) &
            +c3*cshift(f(l2i:l2,m1:m2,:,ivar1:ivar2),-displs  ,2) &
            +c4*cshift(f(l2i:l2,m1:m2,:,ivar1:ivar2),-displs-1,2) &
            +c5*cshift(f(l2i:l2,m1:m2,:,ivar1:ivar2),-displs-2,2) &
            +c6*cshift(f(l2i:l2,m1:m2,:,ivar1:ivar2),-displs-3,2)
        f(l2+1:mx,m1:m2,:,ivar1:ivar2) = &
             c1*cshift(f(l1:l1i,m1:m2,:,ivar1:ivar2), displs-2,2) &
            +c2*cshift(f(l1:l1i,m1:m2,:,ivar1:ivar2), displs-1,2) &
            +c3*cshift(f(l1:l1i,m1:m2,:,ivar1:ivar2), displs  ,2) &
            +c4*cshift(f(l1:l1i,m1:m2,:,ivar1:ivar2), displs+1,2) &
            +c5*cshift(f(l1:l1i,m1:m2,:,ivar1:ivar2), displs+2,2) &
            +c6*cshift(f(l1:l1i,m1:m2,:,ivar1:ivar2), displs+3,2)
      endif
!
    endsubroutine finalize_shearing
!***********************************************************************
    subroutine radboundary_zx_recv(mrad,idir,Qrecv_zx)
!
      integer :: mrad,idir
      real, dimension(mx,mz) :: Qrecv_zx
!
      if (ALWAYS_FALSE) print*,mrad,idir,Qrecv_zx(1,1)
!
    endsubroutine radboundary_zx_recv
!***********************************************************************
    subroutine radboundary_xy_recv(nrad,idir,Qrecv_xy)
!
      integer :: nrad,idir
      real, dimension(mx,my) :: Qrecv_xy
!
      if (ALWAYS_FALSE) print*,nrad,idir,Qrecv_xy(1,1)
!
    endsubroutine radboundary_xy_recv
!***********************************************************************
    subroutine radboundary_zx_send(mrad,idir,Qsend_zx)
!
      integer :: mrad,idir
      real, dimension(mx,mz) :: Qsend_zx
!
      if (ALWAYS_FALSE) print*,mrad,idir,Qsend_zx(1,1)
!
    endsubroutine radboundary_zx_send
!***********************************************************************
    subroutine radboundary_xy_send(nrad,idir,Qsend_xy)
!
      integer :: nrad,idir
      real, dimension(mx,my) :: Qsend_xy
!
      if (ALWAYS_FALSE) print*,nrad,idir,Qsend_xy(1,1)
!
    endsubroutine radboundary_xy_send
!***********************************************************************
    subroutine radboundary_yz_sendrecv(lrad,idir,Qsend_yz,Qrecv_yz)
!
      integer :: lrad,idir
      real, dimension(my,mz) :: Qsend_yz,Qrecv_yz
!
      if (ALWAYS_FALSE) print*,lrad,idir,Qsend_yz(1,1),Qrecv_yz(1,1)
!
    endsubroutine radboundary_yz_sendrecv
!***********************************************************************
    subroutine radboundary_zx_sendrecv(mrad,idir,Qsend_zx,Qrecv_zx)
!
      integer :: mrad,idir
      real, dimension(mx,mz) :: Qsend_zx,Qrecv_zx
!
      if (ALWAYS_FALSE) print*,mrad,idir,Qsend_zx(1,1),Qrecv_zx(1,1)
!
    endsubroutine radboundary_zx_sendrecv
!***********************************************************************
    subroutine radboundary_yz_periodic_ray(Qrad_yz,tau_yz, &
                                           Qrad_yz_all,tau_yz_all)
!
!  Trivial counterpart of radboundary_yz_periodic_ray() from mpicomm.f90
!
!  17-nov-14/axel: adapted from radboundary_zx_periodic_ray
!
      real, dimension(ny,nz), intent(in) :: Qrad_yz,tau_yz
      real, dimension(ny,nz,0:nprocx-1) :: Qrad_yz_all,tau_yz_all
!
      Qrad_yz_all(:,:,ipx)=Qrad_yz
      tau_yz_all(:,:,ipx)=tau_yz
!
    endsubroutine radboundary_yz_periodic_ray
!***********************************************************************
    subroutine radboundary_zx_periodic_ray(Qrad_zx,tau_zx, &
                                           Qrad_zx_all,tau_zx_all)
!
!  Trivial counterpart of radboundary_zx_periodic_ray() from mpicomm.f90
!
!  19-jul-05/tobi: coded
!
      real, dimension(nx,nz), intent(in) :: Qrad_zx,tau_zx
      real, dimension(nx,nz,0:nprocy-1) :: Qrad_zx_all,tau_zx_all
!
      Qrad_zx_all(:,:,ipy)=Qrad_zx
      tau_zx_all(:,:,ipy)=tau_zx
!
    endsubroutine radboundary_zx_periodic_ray
!***********************************************************************
    subroutine mpirecv_logical_scl(bcast_array,proc_src,tag_id)
!
      logical :: bcast_array
      integer :: proc_src, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, proc_src, tag_id
!
    endsubroutine mpirecv_logical_scl
!***********************************************************************
    subroutine mpirecv_logical_arr(bcast_array,nbcast_array,proc_src,tag_id)
!
      integer :: nbcast_array
      logical, dimension(nbcast_array) :: bcast_array
      integer :: proc_src, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc_src, tag_id
!
    endsubroutine mpirecv_logical_arr
!***********************************************************************
    subroutine mpirecv_real_scl(bcast_array,proc_src,tag_id)
!
      real :: bcast_array
      integer :: proc_src, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array,proc_src, tag_id
!
    endsubroutine mpirecv_real_scl
!***********************************************************************
    subroutine mpirecv_real_arr(bcast_array,nbcast_array,proc_src,tag_id)
!
      integer :: nbcast_array
      real, dimension(nbcast_array) :: bcast_array
      integer :: proc_src, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc_src, tag_id
!
    endsubroutine mpirecv_real_arr
!***********************************************************************
    subroutine mpirecv_real_arr2(bcast_array,nbcast_array,proc_src,tag_id)
!
      integer, dimension(2) :: nbcast_array
      real, dimension(nbcast_array(1), nbcast_array(2)) :: bcast_array
      integer :: proc_src, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc_src, tag_id
!
    endsubroutine mpirecv_real_arr2
!***********************************************************************
    subroutine mpirecv_real_arr3(bcast_array,nb,proc_src,tag_id)
!
      integer, dimension(3) :: nb
      real, dimension(nb(1),nb(2),nb(3)) :: bcast_array
      integer :: proc_src, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, nb, proc_src, tag_id
!
    endsubroutine mpirecv_real_arr3
!***********************************************************************
    subroutine mpirecv_real_arr4(bcast_array,nb,proc_src,tag_id)
!
      integer, dimension(4) :: nb
      real, dimension(nb(1),nb(2),nb(3),nb(4)) :: bcast_array
      integer :: proc_src, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, nb, proc_src, tag_id
!
    endsubroutine mpirecv_real_arr4
!***********************************************************************
    subroutine mpirecv_int_scl(bcast_array,proc_src,tag_id)
!
      integer :: bcast_array
      integer :: proc_src, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, proc_src, tag_id
!
    endsubroutine mpirecv_int_scl
!***********************************************************************
    subroutine mpirecv_int_arr(bcast_array,nbcast_array,proc_src,tag_id)
!
      integer :: nbcast_array
      integer, dimension(nbcast_array) :: bcast_array
      integer :: proc_src, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc_src, tag_id
!
    endsubroutine mpirecv_int_arr
!***********************************************************************
    subroutine mpisend_logical_scl(bcast_array,proc_rec,tag_id)
!
      logical :: bcast_array
      integer :: proc_rec, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, proc_rec, tag_id
!
    endsubroutine mpisend_logical_scl
!***********************************************************************
    subroutine mpisend_logical_arr(bcast_array,nbcast_array,proc_rec,tag_id)
!
      integer :: nbcast_array
      logical, dimension(nbcast_array) :: bcast_array
      integer :: proc_rec, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc_rec, tag_id
!
    endsubroutine mpisend_logical_arr
!***********************************************************************
    subroutine mpisend_real_scl(bcast_array,proc_rec,tag_id)
!
      real :: bcast_array
      integer :: proc_rec, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, proc_rec, tag_id
!
    endsubroutine mpisend_real_scl
!***********************************************************************
    subroutine mpisend_real_arr(bcast_array,nbcast_array,proc_rec,tag_id)
!
      integer :: nbcast_array
      real, dimension(nbcast_array) :: bcast_array
      integer :: proc_rec, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc_rec, tag_id
!
    endsubroutine mpisend_real_arr
!***********************************************************************
    subroutine mpisend_real_arr2(bcast_array,nbcast_array,proc_rec,tag_id)
!
      integer, dimension(2) :: nbcast_array
      real, dimension(nbcast_array(1),nbcast_array(2)) :: bcast_array
      integer :: proc_rec, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc_rec, tag_id
!
    endsubroutine mpisend_real_arr2
!***********************************************************************
    subroutine mpisend_real_arr3(bcast_array,nb,proc_rec,tag_id)
!
      integer, dimension(3) :: nb
      real, dimension(nb(1),nb(2),nb(3)) :: bcast_array
      integer :: proc_rec, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, nb, proc_rec, tag_id
!
    endsubroutine mpisend_real_arr3
!***********************************************************************
    subroutine mpisend_real_arr4(bcast_array,nb,proc_rec,tag_id)
!
      integer, dimension(4) :: nb
      real, dimension(nb(1),nb(2),nb(3),nb(4)) :: bcast_array
      integer :: proc_rec, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, nb, proc_rec, tag_id
!
    endsubroutine mpisend_real_arr4
!***********************************************************************
    subroutine mpirecv_nonblock_real_arr(bcast_array,nbcast_array,proc_src,tag_id,ireq)
!
      integer :: nbcast_array
      real, dimension(nbcast_array) :: bcast_array
      integer :: proc_src, tag_id, ireq
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc_src, tag_id, ireq
!
    endsubroutine mpirecv_nonblock_real_arr
!***********************************************************************
    subroutine mpirecv_nonblock_real_arr4(bcast_array,nb,proc_src,ireq,tag_id)
!
      integer, dimension(4) :: nb
      real, dimension(nb(1),nb(2),nb(3),nb(4)) :: bcast_array
      integer :: proc_src, tag_id, ireq
!
      if (ALWAYS_FALSE) print*, bcast_array, nb, proc_src, tag_id, ireq
!
    endsubroutine mpirecv_nonblock_real_arr4
!***********************************************************************
    subroutine mpirecv_nonblock_int_scl(bcast_array,proc_src,ireq,tag_id)
!
      integer :: bcast_array
      integer :: proc_src, tag_id, ireq
!
      if (ALWAYS_FALSE) print*, bcast_array, proc_src, tag_id, ireq
!
    endsubroutine mpirecv_nonblock_int_scl
!***********************************************************************
    subroutine mpirecv_nonblock_int_arr(bcast_array,nbcast_array,proc_src,tag_id,ireq)
!
      integer :: nbcast_array
      integer, dimension(nbcast_array) :: bcast_array
      integer :: proc_src, tag_id, ireq
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc_src, tag_id, ireq
!
    endsubroutine mpirecv_nonblock_int_arr
!***********************************************************************
    subroutine mpisend_nonblock_real_arr(bcast_array,nbcast_array,proc_rec,tag_id,ireq)
!
      integer :: nbcast_array
      real, dimension(nbcast_array) :: bcast_array
      integer :: proc_rec, tag_id, ireq
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc_rec, tag_id, ireq
!
    endsubroutine mpisend_nonblock_real_arr
!***********************************************************************
    subroutine mpisend_nonblock_int_scl(bcast_array,proc_rec,ireq,tag_id)
!
      integer :: bcast_array
      integer :: proc_rec, tag_id, ireq
!
      if (ALWAYS_FALSE) print*, bcast_array, proc_rec, tag_id, ireq
!
    endsubroutine mpisend_nonblock_int_scl
!***********************************************************************
    subroutine mpisend_nonblock_int_arr(bcast_array,nbcast_array,proc_rec,tag_id,iref)
!
      integer :: nbcast_array
      integer, dimension(nbcast_array) :: bcast_array
      integer :: proc_rec, tag_id, iref
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc_rec, tag_id, iref
!
    endsubroutine mpisend_nonblock_int_arr
!***********************************************************************
    subroutine mpisend_nonblock_real_arr4(bcast_array,nb,proc_rec,ireq,tag_id)
!
      integer, dimension(4) :: nb
      real, dimension(nb(1),nb(2),nb(3),nb(4)) :: bcast_array
      integer :: proc_rec, tag_id, ireq
!
      if (ALWAYS_FALSE) print*, bcast_array, nb, proc_rec, tag_id, ireq
!
    endsubroutine mpisend_nonblock_real_arr4
!***********************************************************************
    subroutine mpisendrecv_real_scl(send_array,proc_dest,sendtag, &
      recv_array,proc_src,recvtag)

    real :: send_array, recv_array
    integer :: proc_src, proc_dest, sendtag, recvtag

    if (ALWAYS_FALSE) print*, sendtag, recvtag

    endsubroutine mpisendrecv_real_scl
!***********************************************************************
    subroutine mpisendrecv_real_arr(send_array,sendcnt,proc_dest,sendtag, &
      recv_array,proc_src,recvtag)

    integer :: sendcnt
    real, dimension(sendcnt) :: send_array
    real, dimension(sendcnt) :: recv_array
    integer :: proc_src, proc_dest, sendtag, recvtag

    if (ALWAYS_FALSE) print*, sendtag, recvtag

    endsubroutine mpisendrecv_real_arr
!***********************************************************************
    subroutine mpisendrecv_real_arr2(send_array,sendcnt_arr,proc_dest,sendtag, &
     recv_array,proc_src,recvtag)

    integer, dimension(2) :: sendcnt_arr
    real, dimension(sendcnt_arr(1),sendcnt_arr(2)) :: send_array
    real, dimension(sendcnt_arr(1),sendcnt_arr(2)) :: recv_array
    integer :: proc_src, proc_dest, sendtag, recvtag

    if (ALWAYS_FALSE) print*, sendtag, recvtag

    endsubroutine mpisendrecv_real_arr2
!***********************************************************************
    subroutine mpisendrecv_real_arr3(send_array,sendcnt_arr,proc_dest,sendtag, &
     recv_array,proc_src,recvtag)

    integer, dimension(3) :: sendcnt_arr
    real, dimension(sendcnt_arr(1),sendcnt_arr(2),sendcnt_arr(3)) :: send_array
    real, dimension(sendcnt_arr(1),sendcnt_arr(2),sendcnt_arr(3)) :: recv_array
    integer :: proc_src, proc_dest, sendtag, recvtag

    if (ALWAYS_FALSE) print*, sendtag, recvtag

    endsubroutine mpisendrecv_real_arr3
!***********************************************************************
    subroutine mpisendrecv_real_arr4(send_array,sendcnt_arr,proc_dest,sendtag, &
     recv_array,proc_src,recvtag)

    integer, dimension(4) :: sendcnt_arr
    real, dimension(sendcnt_arr(1),sendcnt_arr(2),sendcnt_arr(3), &
      sendcnt_arr(4)) :: send_array
    real, dimension(sendcnt_arr(1),sendcnt_arr(2),sendcnt_arr(3), &
      sendcnt_arr(4)) :: recv_array
    integer :: proc_src, proc_dest, sendtag, recvtag

    if (ALWAYS_FALSE) print*, sendtag, recvtag

    endsubroutine mpisendrecv_real_arr4
!***********************************************************************
    subroutine mpisend_int_scl(bcast_array,proc_rec,tag_id)
!
      integer :: bcast_array
      integer :: proc_rec, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, proc_rec, tag_id
!
    endsubroutine mpisend_int_scl
!***********************************************************************
    subroutine mpisend_int_arr(bcast_array,nbcast_array,proc_rec,tag_id)
!
      integer :: nbcast_array
      integer, dimension(nbcast_array) :: bcast_array
      integer :: proc_rec, tag_id
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc_rec, tag_id
!
    endsubroutine mpisend_int_arr
!***********************************************************************
    subroutine mpibcast_logical_scl(lbcast_array,proc)
!
      logical :: lbcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, lbcast_array, proc
!
    endsubroutine mpibcast_logical_scl
!***********************************************************************
    subroutine mpibcast_logical_arr(lbcast_array,nbcast_array,proc)
!
      integer :: nbcast_array
      logical, dimension(nbcast_array) :: lbcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, lbcast_array, nbcast_array, proc
!
    endsubroutine mpibcast_logical_arr
!***********************************************************************
    subroutine mpibcast_logical_arr2(bcast_array,nbcast_array,proc)
!
      integer, dimension(2) :: nbcast_array
      logical, dimension(nbcast_array(1),nbcast_array(2)) :: bcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc
!
    endsubroutine mpibcast_logical_arr2
!***********************************************************************
    subroutine mpibcast_int_scl(ibcast_array,proc)
!
      integer :: ibcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, ibcast_array,proc
!
    endsubroutine mpibcast_int_scl
!***********************************************************************
    subroutine mpibcast_int_arr(ibcast_array,nbcast_array,proc)
!
      integer :: nbcast_array
      integer, dimension(nbcast_array) :: ibcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, ibcast_array, nbcast_array, proc
!
    endsubroutine mpibcast_int_arr
!***********************************************************************
    subroutine mpibcast_real_scl(bcast_array,proc)
!
      real :: bcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, bcast_array, proc
!
    endsubroutine mpibcast_real_scl
!***********************************************************************
    subroutine mpibcast_real_arr(bcast_array,nbcast_array,proc)
!
      integer :: nbcast_array
      real, dimension(nbcast_array) :: bcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc
!
    endsubroutine mpibcast_real_arr
!***********************************************************************
    subroutine mpibcast_real_arr2(bcast_array,nbcast_array,proc)
!
      integer, dimension(2) :: nbcast_array
      real, dimension(nbcast_array(1),nbcast_array(2)) :: bcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc
!
    endsubroutine mpibcast_real_arr2
!***********************************************************************
    subroutine mpibcast_real_arr3(bcast_array,nb,proc)
!
      integer, dimension(3) :: nb
      real, dimension(nb(1),nb(2),nb(3)) :: bcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, bcast_array, nb, proc
!
    endsubroutine mpibcast_real_arr3
!***********************************************************************
    subroutine mpibcast_real_arr4(bcast_array,nb,proc)
!
      integer, dimension(4) :: nb
      real, dimension(nb(1),nb(2),nb(3),nb(4)) :: bcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, bcast_array, nb, proc
!
    endsubroutine mpibcast_real_arr4
!***********************************************************************
    subroutine mpibcast_double_scl(bcast_array,proc)
!
      double precision :: bcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, bcast_array,proc
!
    endsubroutine mpibcast_double_scl
!***********************************************************************
    subroutine mpibcast_double_arr(bcast_array,nbcast_array,proc)
!
      integer :: nbcast_array
      double precision, dimension(nbcast_array) :: bcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc
!
    endsubroutine mpibcast_double_arr
!***********************************************************************
    subroutine mpibcast_char_scl(cbcast_array,proc)
!
      character :: cbcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, cbcast_array,proc
!
    endsubroutine mpibcast_char_scl
!***********************************************************************
    subroutine mpibcast_char_arr(cbcast_array,nbcast_array,proc)
!
      integer :: nbcast_array
      character, dimension(nbcast_array) :: cbcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, cbcast_array, nbcast_array, proc
!
    endsubroutine mpibcast_char_arr
!***********************************************************************
    subroutine mpibcast_cmplx_arr_dbl(bcast_array,nbcast_array,proc)
!
!  Communicate real array between processors.
!
      integer :: nbcast_array
      complex(KIND=rkind8), dimension(nbcast_array) :: bcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc
!
    endsubroutine mpibcast_cmplx_arr_dbl
!***********************************************************************
    subroutine mpibcast_cmplx_arr_sgl(bcast_array,nbcast_array,proc)
!
!  Communicate real array between processors.
!
      integer :: nbcast_array
      complex, dimension(nbcast_array) :: bcast_array
      integer, optional :: proc
!
      if (ALWAYS_FALSE) print*, bcast_array, nbcast_array, proc
!
    endsubroutine mpibcast_cmplx_arr_sgl
!***********************************************************************
    subroutine mpiwait(bwait)
!
      integer :: bwait
!      
      if (ALWAYS_FALSE) print*,bwait
! 
   endsubroutine mpiwait
!***********************************************************************
    subroutine mpiallreduce_sum_scl(fsum_tmp,fsum,idir)
!
      real :: fsum_tmp, fsum
      integer, optional :: idir
!
      fsum=fsum_tmp
      if (present(idir).and.ALWAYS_FALSE) print*,idir
!
    endsubroutine mpiallreduce_sum_scl
!***********************************************************************
    subroutine mpiallreduce_sum_arr(fsum_tmp,fsum,nreduce,idir)
!
      integer :: nreduce
      real, dimension(nreduce) :: fsum_tmp, fsum
      integer, optional :: idir
!
      fsum=fsum_tmp
      if (present(idir).and.ALWAYS_FALSE) print*,idir
!
    endsubroutine mpiallreduce_sum_arr
!***********************************************************************
    subroutine mpiallreduce_sum_arr2(fsum_tmp,fsum,nreduce,idir)
!
      integer, dimension(2) :: nreduce
      real, dimension(nreduce(1),nreduce(2)) :: fsum_tmp, fsum
      integer, optional :: idir
!
      fsum=fsum_tmp
      if (present(idir).and.ALWAYS_FALSE) print*,idir
!
    endsubroutine mpiallreduce_sum_arr2
!***********************************************************************
    subroutine mpiallreduce_sum_arr3(fsum_tmp,fsum,nreduce,idir)
!
      integer, dimension(3) :: nreduce
      real, dimension(nreduce(1),nreduce(2),nreduce(3)) :: fsum_tmp, fsum
      integer, optional :: idir
!
      fsum=fsum_tmp
      if (present(idir).and.ALWAYS_FALSE) print*,idir
!
    endsubroutine mpiallreduce_sum_arr3
!***********************************************************************
    subroutine mpiallreduce_sum_arr4(fsum_tmp,fsum,nreduce,idir)
!
      integer, dimension(4) :: nreduce
      real, dimension(nreduce(1),nreduce(2),nreduce(3),nreduce(4)) :: fsum_tmp, fsum
      integer, optional :: idir
!
      fsum=fsum_tmp
      if (present(idir).and.ALWAYS_FALSE) print*,idir
!
    endsubroutine mpiallreduce_sum_arr4
!***********************************************************************
    subroutine mpiallreduce_sum_arr5(fsum_tmp,fsum,nreduce,idir)
!
      integer, dimension(5) :: nreduce
      real, dimension(nreduce(1),nreduce(2),nreduce(3),nreduce(4),nreduce(5)) :: fsum_tmp, fsum
      integer, optional :: idir
!
      fsum=fsum_tmp
      if (present(idir).and.ALWAYS_FALSE) print*,idir
!
    endsubroutine mpiallreduce_sum_arr5
!***********************************************************************
    subroutine mpiallreduce_sum_int_scl(fsum_tmp,fsum)
!
      integer :: fsum_tmp, fsum
!
      fsum=fsum_tmp
!
    endsubroutine mpiallreduce_sum_int_scl
!***********************************************************************
    subroutine mpiallreduce_sum_int_arr(fsum_tmp,fsum,nreduce)
!
      integer :: nreduce
      integer, dimension(nreduce) :: fsum_tmp, fsum
!
      fsum=fsum_tmp
!
    endsubroutine mpiallreduce_sum_int_arr
!***********************************************************************
    subroutine mpiallreduce_max_scl(fmax_tmp,fmax)
!
      real :: fmax_tmp, fmax
!
      fmax=fmax_tmp
!
    endsubroutine mpiallreduce_max_scl
!***********************************************************************
    subroutine mpiallreduce_max_arr(fmax_tmp,fmax,nreduce)
!
      integer :: nreduce
      real, dimension(nreduce) :: fmax_tmp, fmax
!
      fmax=fmax_tmp
!
    endsubroutine mpiallreduce_max_arr
!***********************************************************************
    subroutine mpiallreduce_min_scl_dbl(fmin_tmp,fmin)
!
      real(KIND=rkind8) :: fmin_tmp,fmin
!
      fmin=fmin_tmp
!
    endsubroutine mpiallreduce_min_scl_dbl
!***********************************************************************
    subroutine mpiallreduce_min_scl_sgl(fmin_tmp,fmin)
!
      real(KIND=rkind4) :: fmin_tmp,fmin
!
      fmin=fmin_tmp
!
    endsubroutine mpiallreduce_min_scl_sgl
!***********************************************************************
    subroutine mpiallreduce_or_scl(flor_tmp, flor)
!
      logical, intent(in) :: flor_tmp
      logical, intent(out) :: flor
!
      flor = flor_tmp
!
    endsubroutine mpiallreduce_or_scl
!***********************************************************************
    subroutine mpireduce_max_scl_int(fmax_tmp,fmax)
!
      integer :: fmax_tmp, fmax
!
      fmax=fmax_tmp
!
    endsubroutine mpireduce_max_scl_int
!***********************************************************************
    subroutine mpireduce_max_scl(fmax_tmp,fmax)
!
      real :: fmax_tmp, fmax
!
      fmax=fmax_tmp
!
    endsubroutine mpireduce_max_scl
!***********************************************************************
    subroutine mpireduce_max_arr(fmax_tmp,fmax,nreduce)
!
      integer :: nreduce
      real, dimension(nreduce) :: fmax_tmp, fmax
!
      fmax=fmax_tmp
!
    endsubroutine mpireduce_max_arr
!***********************************************************************
    subroutine mpireduce_min_scl(fmin_tmp,fmin)
!
      real :: fmin_tmp, fmin
!
      fmin=fmin_tmp
!
    endsubroutine mpireduce_min_scl
!***********************************************************************
    subroutine mpireduce_min_arr(fmin_tmp,fmin,nreduce)
!
      integer :: nreduce
      real, dimension(nreduce) :: fmin_tmp, fmin
!
      fmin=fmin_tmp
!
    endsubroutine mpireduce_min_arr
!***********************************************************************
    subroutine mpireduce_sum_int_scl(fsum_tmp,fsum)
!
      integer :: fsum_tmp,fsum
!
      fsum=fsum_tmp
!
    endsubroutine mpireduce_sum_int_scl
!***********************************************************************
    subroutine mpireduce_sum_int_arr(fsum_tmp,fsum,nreduce)
!
      integer :: nreduce
      integer, dimension(nreduce) :: fsum_tmp,fsum
!
      fsum=fsum_tmp
!
    endsubroutine mpireduce_sum_int_arr
!***********************************************************************
    subroutine mpireduce_sum_int_arr2(fsum_tmp,fsum,nreduce)
!
      integer, dimension(2) :: nreduce
      integer, dimension(nreduce(1),nreduce(2)) :: fsum_tmp,fsum
!
      fsum=fsum_tmp
!
    endsubroutine mpireduce_sum_int_arr2
!***********************************************************************
    subroutine mpireduce_sum_int_arr3(fsum_tmp,fsum,nreduce,idir)
!
      integer, dimension(3) :: nreduce
      integer, dimension(nreduce(1),nreduce(2),nreduce(3)) :: fsum_tmp,fsum
      integer, optional :: idir
!
      fsum=fsum_tmp
      if (present(idir).and.ALWAYS_FALSE) print*,idir
!
    endsubroutine mpireduce_sum_int_arr3
!***********************************************************************
    subroutine mpireduce_sum_int_arr4(fsum_tmp,fsum,nreduce)
!
      integer, dimension(4) :: nreduce
      integer, dimension(nreduce(1),nreduce(2),nreduce(3),nreduce(4)) :: fsum_tmp,fsum
!
      fsum=fsum_tmp
!
    endsubroutine mpireduce_sum_int_arr4
!***********************************************************************
    subroutine mpireduce_sum_scl(fsum_tmp,fsum,idir)
!
      real :: fsum_tmp,fsum
      integer, optional :: idir
!
      fsum=fsum_tmp
      if (present(idir).and.ALWAYS_FALSE) print*,idir
!
    endsubroutine mpireduce_sum_scl
!***********************************************************************
    subroutine mpireduce_sum_arr(fsum_tmp,fsum,nreduce,idir)
!
      integer :: nreduce
      real, dimension(nreduce) :: fsum_tmp,fsum
      integer, optional :: idir
!
      fsum=fsum_tmp
      if (present(idir).and.ALWAYS_FALSE) print*,idir
!
    endsubroutine mpireduce_sum_arr
!***********************************************************************
    subroutine mpireduce_sum_arr2(fsum_tmp,fsum,nreduce,idir)
!
      integer, dimension(2) :: nreduce
      real, dimension(nreduce(1),nreduce(2)) :: fsum_tmp,fsum
      integer, optional :: idir
!
      fsum=fsum_tmp
      if (present(idir).and.ALWAYS_FALSE) print*,idir
!
    endsubroutine mpireduce_sum_arr2
!***********************************************************************
    subroutine mpireduce_sum_arr3(fsum_tmp,fsum,nreduce,idir)
!
      integer, dimension(3) :: nreduce
      real, dimension(nreduce(1),nreduce(2),nreduce(3)) :: fsum_tmp,fsum
      integer, optional :: idir
!
      fsum=fsum_tmp
      if (present(idir).and.ALWAYS_FALSE) print*,idir
!
    endsubroutine mpireduce_sum_arr3
!***********************************************************************
    subroutine mpireduce_sum_arr4(fsum_tmp,fsum,nreduce,idir)
!
      integer, dimension(4) :: nreduce
      real, dimension(nreduce(1),nreduce(2),nreduce(3),nreduce(4)) :: fsum_tmp,fsum
      integer, optional :: idir
!
      fsum=fsum_tmp
      if (present(idir).and.ALWAYS_FALSE) print*,idir
!
    endsubroutine mpireduce_sum_arr4
!***********************************************************************
    subroutine mpireduce_sum_double_scl(dsum_tmp,dsum)
!
      double precision :: dsum_tmp,dsum
!
      dsum=dsum_tmp
!
    endsubroutine mpireduce_sum_double_scl
!***********************************************************************
    subroutine mpireduce_sum_double_arr(dsum_tmp,dsum,nreduce)
!
      integer :: nreduce
      double precision, dimension(nreduce) :: dsum_tmp,dsum
!
      dsum=dsum_tmp
!
    endsubroutine mpireduce_sum_double_arr
!***********************************************************************
    subroutine mpireduce_sum_double_arr2(dsum_tmp,dsum,nreduce)
!
      integer, dimension(2) :: nreduce
      double precision, dimension(nreduce(1),nreduce(2)) :: dsum_tmp,dsum
!
      dsum=dsum_tmp
!
    endsubroutine mpireduce_sum_double_arr2
!***********************************************************************
    subroutine mpireduce_sum_double_arr3(dsum_tmp,dsum,nreduce)
!
      integer, dimension(3) :: nreduce
      double precision, dimension(nreduce(1),nreduce(2),nreduce(3)) :: dsum_tmp,dsum
!
      dsum=dsum_tmp
!
    endsubroutine mpireduce_sum_double_arr3
!***********************************************************************
    subroutine mpireduce_sum_double_arr4(dsum_tmp,dsum,nreduce)
!
      integer, dimension(4) :: nreduce
      double precision, dimension(nreduce(1),nreduce(2),nreduce(3),nreduce(4)) :: dsum_tmp,dsum
!
      dsum=dsum_tmp
!
    endsubroutine mpireduce_sum_double_arr4
!***********************************************************************
    subroutine mpireduce_or_scl(flor_tmp,flor)
!
      logical :: flor_tmp, flor
!
      flor=flor_tmp
!
    endsubroutine mpireduce_or_scl
!***********************************************************************
    subroutine mpireduce_or_arr(flor_tmp,flor,nreduce)
!
      integer :: nreduce
      logical, dimension(nreduce) :: flor_tmp, flor
!
      flor=flor_tmp
!
    endsubroutine mpireduce_or_arr
!***********************************************************************
    subroutine mpireduce_and_scl(fland_tmp,fland)
!
      logical :: fland_tmp, fland
!
      fland=fland_tmp
!
    endsubroutine mpireduce_and_scl
!***********************************************************************
    subroutine mpireduce_and_arr(fland_tmp,fland,nreduce)
!
      integer :: nreduce
      logical, dimension(nreduce) :: fland_tmp, fland
!
      fland=fland_tmp
!
    endsubroutine mpireduce_and_arr
!***********************************************************************
    subroutine start_serialize()
!
    endsubroutine start_serialize
!***********************************************************************
    subroutine end_serialize()
!
    endsubroutine end_serialize
!***********************************************************************
    subroutine mpibarrier()
!
    endsubroutine mpibarrier
!***********************************************************************
    subroutine mpifinalize()
!
    endsubroutine mpifinalize
!***********************************************************************
    function mpiwtime()
!
!  Mimic the MPI_WTIME() timer function. On many machines, the
!  implementation through system_clock() will overflow after about 50
!  minutes, so MPI_WTIME() is better.
!
!   5-oct-2002/wolf: coded
!
      double precision :: mpiwtime
      integer :: count_rate,time
!
      call system_clock(COUNT_RATE=count_rate)
      call system_clock(COUNT=time)
!
      if (count_rate /= 0) then
        mpiwtime = (time*1.)/count_rate
      else                      ! occurs with ifc 6.0 after long (> 2h) runs
        mpiwtime = 0
      endif
!
    endfunction mpiwtime
!***********************************************************************
    function mpiwtick()
!
!  Mimic the MPI_WTICK() function for measuring timer resolution.
!
!   5-oct-2002/wolf: coded
!
      double precision :: mpiwtick
      integer :: count_rate
!
      call system_clock(COUNT_RATE=count_rate)
      if (count_rate /= 0) then
        mpiwtick = 1./count_rate
      else                      ! occurs with ifc 6.0 after long (> 2h) runs
        mpiwtick = 0
      endif
!
    endfunction mpiwtick
!***********************************************************************
    subroutine touch_file(fname)
!
!  touch file (used for code locking)
!  25-may-03/axel: coded
!  06-mar-07/wolf: moved here from sub.f90, so we can use it below
!
      character (len=*) :: fname
!
      if (lroot) then
        open(1,FILE=fname,STATUS='replace')
        close(1)
      endif
!
    endsubroutine touch_file
!***********************************************************************
    subroutine die_gracefully()
!
!  Stop... perform any necessary shutdown stuff.
!
!  29-jun-05/tony: coded
!
      call touch_file('ERROR')
!
      call mpifinalize
      STOP 1                    ! Return nonzero exit status
!
    endsubroutine die_gracefully
!***********************************************************************
    subroutine die_immediately()
!
!  Stop... perform any necessary shutdown stuff.
!
!  29-jun-05/tony: coded
!
      call touch_file('ERROR')
!
      STOP 2                    ! Return nonzero exit status
!
    endsubroutine die_immediately
!***********************************************************************
    subroutine stop_it(msg,code)
!
!  Print message and stop.
!
!  6-nov-01/wolf: coded
!  4-nov-11/MR: optional parameter for error code added
!
      use general, only: itoa
!
      character (len=*) :: msg
      integer, optional :: code
!
      if (lroot) then
        if (present(code)) then
          write(0,'(A,A)') 'STOPPED: ', msg, '. CODE: '//trim(itoa(code))
        else
          write(0,'(A,A)') 'STOPPED: ', msg
        endif
      endif
!
      call mpifinalize
      STOP 1                    ! Return nonzero exit status
!
    endsubroutine stop_it
!***********************************************************************
    subroutine stop_it_if_any(stop_flag,msg)
!
!  Conditionally print message and stop.
!
!  22-nov-04/wolf: coded
!
      logical :: stop_flag
      character (len=*) :: msg
!
      if (stop_flag) call stop_it(msg)
!
    endsubroutine stop_it_if_any
!***********************************************************************
    subroutine check_emergency_brake()
!
!  Check the lemergency_brake flag and stop with any provided
!  message if it is set.
!
!  29-jul-06/tony: coded
!
      if (lemergency_brake) call stop_it( &
            "Emergency brake activated. Check for error messages above.")
!
    endsubroutine check_emergency_brake
!***********************************************************************
    subroutine transp(a,var)
!
!  Doing a transpose (dummy version for single processor).
!
!   5-sep-02/axel: adapted from version in mpicomm.f90
!
      real, dimension(nx,ny,nz) :: a
      real, dimension(:,:), allocatable :: tmp
      character :: var
!
      integer :: m, n, iy, ibox
!
      if (ip<10) print*, 'transp for single processor'
!
!  Doing x-y transpose if var='y'
!
      if (var=='y') then
        if (nygrid/=1) then
!
          if (mod(nx,ny)/=0) then
            if (lroot) print*, 'transp: works only if nx is an integer '//&
                 'multiple of ny!'
            call stop_it('transp')
          endif
!
          allocate (tmp(nx,ny))
          do n=1,nz
            do ibox=0,nx/nygrid-1
              iy=ibox*ny
              tmp=transpose(a(iy+1:iy+ny,:,n))
              a(iy+1:iy+ny,:,n)=tmp
            enddo
          enddo
          deallocate (tmp)
!
        endif
!
!  Doing x-z transpose if var='z'
!
      elseif (var=='z') then
        if (nzgrid/=1) then
!
          if (nx/=nz) then
            if (lroot) print*, 'transp: works only for nx=nz!'
            call stop_it('transp')
          endif
!
          allocate (tmp(nx,nz))
          do m=1,ny
            tmp=transpose(a(:,m,:))
            a(:,m,:)=tmp
          enddo
          deallocate (tmp)
!
        endif
!
      endif
!
    endsubroutine transp
!***********************************************************************
    subroutine transp_xy(a)
!
!  Doing a transpose in x and y only
!  (dummy version for single processor)
!
!   5-oct-02/tobi: adapted from transp
!
      real, dimension(nx,ny), intent(inout) :: a
!
      real, dimension(:,:), allocatable :: tmp
      integer :: ibox,iy
!
      if (ny/=1) then
!
        if (mod(nx,ny)/=0) then
          call stop_it('transp: nxgrid must be an integer multiple of nygrid')
        endif
!
        allocate (tmp(ny,ny))
        do ibox=0,nxgrid/nygrid-1
          iy=ibox*ny
          tmp=transpose(a(iy+1:iy+ny,:)); a(iy+1:iy+ny,:)=tmp
        enddo
        deallocate (tmp)
!
      endif
!
    endsubroutine transp_xy
!***********************************************************************
    subroutine transp_xy_other(a)
!
!  Doing a transpose in x and y only
!  (dummy version for single processor)
!
!   5-oct-02/tobi: adapted from transp
!
      real, dimension(:,:), intent(inout) :: a
!
      real, dimension(:,:), allocatable :: tmp
      integer :: ibox,iy,ny_other,nx_other
      integer :: nxgrid_other,nygrid_other
!
      nx_other=size(a,1); ny_other=size(a,2)
      nxgrid_other=nx_other
      nygrid_other=ny_other*nprocy
!
      if (ny_other/=1) then
!
        if (mod(nx_other,ny_other)/=0) then
          call stop_it('transp: nxgrid must be an integer multiple of nygrid')
        endif
!
        allocate (tmp(ny_other,ny_other))
        do ibox=0,nxgrid_other/nygrid_other-1
          iy=ibox*ny_other
          tmp=transpose(a(iy+1:iy+ny_other,:)); a(iy+1:iy+ny_other,:)=tmp
        enddo
        deallocate (tmp)
!
      endif
!
    endsubroutine transp_xy_other
!***********************************************************************
    subroutine transp_other(a,var)
!
!  Doing a transpose in 3D
!  (dummy version for single processor)
!
!  08-may-08/wlad: adapted from transp
!
      real, dimension(:,:,:), intent(inout) :: a
      real, dimension(:,:), allocatable :: tmp
      character :: var
      integer :: ibox,iy,ny_other,nx_other,nz_other
      integer :: m,n,nxgrid_other,nygrid_other,nzgrid_other
!
      nx_other=size(a,1); ny_other=size(a,2) ; nz_other=size(a,3)
      nxgrid_other=nx_other
      nygrid_other=ny_other*nprocy
      nzgrid_other=nz_other*nprocz
!
      if (var=='y') then
!
        if (ny_other/=1) then
!
          if (mod(nx_other,ny_other)/=0) then
            call stop_it('transp_other: nxgrid must be an integer'//&
                 'multiple of nygrid')
          endif
!
          allocate (tmp(ny_other,ny_other))
          do ibox=0,nxgrid_other/nygrid_other-1
            iy=ibox*ny_other
            do n=1,nz_other
              tmp=transpose(a(iy+1:iy+ny_other,:,n))
              a(iy+1:iy+ny_other,:,n)=tmp
            enddo
          enddo
          deallocate (tmp)
!
        endif
      elseif (var=='z') then
        if (nzgrid_other/=1) then
!
          if (nx_other/=nz_other) then
            if (lroot) print*, &
                 'transp_other: works only for nx_grid=nz_grid!'
            call stop_it('transp_other')
          endif
!
          allocate (tmp(nx_other,nz_other))
          do m=1,ny_other
            tmp=transpose(a(:,m,:))
            a(:,m,:)=tmp
          enddo
          deallocate (tmp)
!
        endif
!
      endif
!
    endsubroutine transp_other
!***********************************************************************
    subroutine transp_xz(a,b)
!
!  Doing the transpose of information distributed on several processors.
!  This routine transposes 2D arrays in x and z only.
!
!  19-dec-06/anders: Adapted from transp
!
      real, dimension(:,:), intent(in) :: a
      real, dimension(:,:), intent(out) :: b
!
      b=transpose(a)
!
    endsubroutine transp_xz
!***********************************************************************
    subroutine transp_zx(b,a)
!
!  Doing the transpose of information distributed on several processors.
!  This routine transposes 2D arrays in x and z only.
!
!  19-dec-06/anders: Adapted from transp
!
      real, dimension(:,:), intent(in) :: b
      real, dimension(:,:), intent(out) :: a
!
      a=transpose(b)
!
    endsubroutine transp_zx
!***********************************************************************
    subroutine fill_zghostzones_3vec(vec,ivar)
!
!  Fills the upper and lower ghostzones for periodic BCs and a 3-vector vec.
!  ivar, ivar+1, ivar+2 indices of the variables vec corresponds to
!
!  20-oct-09/MR: coded
!
      real, dimension(mz,3), intent(inout) :: vec
      integer, intent(in)                  :: ivar
!
      integer :: j
!
      do j=1,3
        if ( bcz12(ivar+j-1,1)=='p' ) then
          vec(1:n1-1        ,j) = vec(n2i:n2,j)
          vec(n2+1:n2+nghost,j) = vec(n1:n1i,j)
        endif
      enddo
!
    endsubroutine fill_zghostzones_3vec
!***********************************************************************
    subroutine communicate_vect_field_ghosts(f,topbot,start_index)
!
!  Helper routine for communication of ghost cell values of a vector field.
!  Needed by potential field extrapolations, which only compute nx*ny arrays.
!  Can also be used for synchronization of changed uu values with ghost cells,
!  if the start_index parameter set to iux (default is iax).
!
!   8-oct-2006/tobi: Coded
!  28-dec-2010/Bourdin.KIS: extended to work for any 3D vector field data.
!
      real, dimension (mx,my,mz,mfarray), intent (inout) :: f
      character (len=3), intent (in) :: topbot
      integer, intent(in), optional :: start_index
!
      integer :: nn1,nn2,is,ie
!
      is = iax
      if (present (start_index)) is = start_index
      ie = is + 2
!
      nn1=-1
      nn2=-1
!
      select case (topbot)
        case ('bot'); nn1=1;  nn2=n1
        case ('top'); nn1=n2; nn2=mz
        case default; call stop_it("communicate_vect_field_ghosts: "//topbot//&
                                   " should be either `top' or `bot'")
      end select
!
!  Periodic boundaries in y
!
      f(l1:l2,   1:m1-1,nn1:nn2,is:ie) = f(l1:l2,m2i:m2 ,nn1:nn2,is:ie)
      f(l1:l2,m2+1:my  ,nn1:nn2,is:ie) = f(l1:l2, m1:m1i,nn1:nn2,is:ie)
!
!  Periodic boundaries in x
!
      f(   1:l1-1,:,nn1:nn2,is:ie) = f(l2i:l2 ,:,nn1:nn2,is:ie)
      f(l2+1:mx  ,:,nn1:nn2,is:ie) = f( l1:l1i,:,nn1:nn2,is:ie)
!
    endsubroutine communicate_vect_field_ghosts
!***********************************************************************
    subroutine communicate_xy_ghosts(data)
!
!  Helper routine for communication of ghost cells in horizontal direction.
!
!  11-apr-2011/Bourdin.KIS: adapted from communicate_vect_field_ghosts.
!
      real, dimension (mx,my), intent (inout) :: data
!
!  Periodic boundaries in y
!
      data(l1:l2,   1:m1-1) = data(l1:l2,m2i:m2 )
      data(l1:l2,m2+1:my  ) = data(l1:l2, m1:m1i)
!
!  Periodic boundaries in x
!
      data(   1:l1-1,:) = data(l2i:l2 ,:)
      data(l2+1:mx  ,:) = data( l1:l1i,:)
!
    endsubroutine communicate_xy_ghosts
!***********************************************************************
    subroutine sum_xy(in, out)
!
!  Sum up 0D data in the xy-plane and distribute back the sum.
!
!  19-jan-2011/Bourdin.KIS: coded
!
      real, intent(in) :: in
      real, intent(out) :: out
!
      out = in
!
    endsubroutine sum_xy
!***********************************************************************
    subroutine distribute_xy_0D(out, in, source_proc)
!
!  This routine distributes a scalar on the source processor
!  to all processors in the xy-plane.
!
!  25-jan-2012/Bourdin.KIS: coded
!
      real, intent(out) :: out
      real, intent(in), optional :: in
      integer, intent(in), optional :: source_proc
!
      if (present (in) .or. present (source_proc)) out = in
!
    endsubroutine distribute_xy_0D
!***********************************************************************
    subroutine distribute_xy_2D(out, in, source_proc)
!
!  This routine divides a large array of 2D data on the broadcaster processor
!  and distributes it to all processors in the xy-plane.
!
!  08-jan-2011/Bourdin.KIS: coded
!
      real, dimension(:,:), intent(out) :: out
      real, dimension(:,:), intent(in), optional :: in
      integer, intent(in), optional :: source_proc
!
      if (present (in) .or. present (source_proc)) out = in
!
    endsubroutine distribute_xy_2D
!***********************************************************************
    subroutine distribute_xy_3D(out, in, source_proc)
!
!  This routine divides a large array of 3D data on the broadcaster processor
!  and distributes it to all processors in the xy-plane.
!
!  08-jan-2011/Bourdin.KIS: coded
!
      real, dimension(:,:,:), intent(out) :: out
      real, dimension(:,:,:), intent(in), optional :: in
      integer, intent(in), optional :: source_proc
!
      if (present (in) .or. present (source_proc)) out = in
!
    endsubroutine distribute_xy_3D
!***********************************************************************
    subroutine distribute_xy_4D(out, in, source_proc)
!
!  This routine divides a large array of 4D data on the broadcaster processor
!  and distributes it to all processors in the xy-plane.
!
!  08-jan-2011/Bourdin.KIS: coded
!
      real, dimension(:,:,:,:), intent(out) :: out
      real, dimension(:,:,:,:), intent(in), optional :: in
      integer, intent(in), optional :: source_proc
!
      if (present (in) .or. present (source_proc)) out = in
!
    endsubroutine distribute_xy_4D
!***********************************************************************
    subroutine collect_xy_0D(in, out, dest_proc)
!
!  Collect 0D data from all processors in the xy-plane
!  and combine it into one large array on the collector processor.
!
!  08-jan-2011/Bourdin.KIS: coded
!
      real, intent(in) :: in
      real, dimension(:,:), intent(out), optional :: out
      integer, intent(in), optional :: dest_proc
!
      if (nprocx /= size (out, 1)) &
          call stop_it ('collect_xy_0D: output x dim must be nprocx')
      if (nprocy /= size (out, 2)) &
          call stop_it ('collect_xy_0D: output y dim must be nprocy')
!
      if (present (out) .or. present (dest_proc)) out = in
!
    endsubroutine collect_xy_0D
!***********************************************************************
    subroutine collect_xy_2D(in, out, dest_proc)
!
!  Collect 2D data from all processors in the xy-plane
!  and combine it into one large array on the collector processor.
!
!  08-jan-2011/Bourdin.KIS: coded
!
      real, dimension(:,:), intent(in) :: in
      real, dimension(:,:), intent(out), optional :: out
      integer, intent(in), optional :: dest_proc
!
      if (present (out) .or. present (dest_proc)) out = in
!
    endsubroutine collect_xy_2D
!***********************************************************************
    subroutine collect_xy_3D(in, out, dest_proc)
!
!  Collect 3D data from all processors in the xy-plane
!  and combine it into one large array on the collector processor.
!
!  08-jan-2011/Bourdin.KIS: coded
!
      real, dimension(:,:,:), intent(in) :: in
      real, dimension(:,:,:), intent(out), optional :: out
      integer, intent(in), optional :: dest_proc
!
      if (present (out) .or. present (dest_proc)) out = in
!
    endsubroutine collect_xy_3D
!***********************************************************************
    subroutine collect_xy_4D(in, out, dest_proc)
!
!  Collect 4D data from all processors in the xy-plane
!  and combine it into one large array on the collector processor.
!
!  08-jan-2011/Bourdin.KIS: coded
!
      real, dimension(:,:,:,:), intent(in) :: in
      real, dimension(:,:,:,:), intent(out), optional :: out
      integer, intent(in), optional :: dest_proc
!
      if (present (out) .or. present (dest_proc)) out = in
!
    endsubroutine collect_xy_4D
!***********************************************************************
    subroutine distribute_z_3D(out, in, source_proc)
!
!  This routine divides a large array of 3D data on the source processor
!  and distributes it to all processors in the z-direction.
!
!  09-mar-2011/Bourdin.KIS: coded
!
      real, dimension(:,:,:), intent(out) :: out
      real, dimension(:,:,:), intent(in), optional :: in
      integer, intent(in), optional :: source_proc
!
      if (present (in) .or. present (source_proc)) out = in
!
    endsubroutine distribute_z_3D
!***********************************************************************
    subroutine distribute_z_4D(out, in, source_proc)
!
!  This routine divides a large array of 4D data on the source processor
!  and distributes it to all processors in the z-direction.
!
!  09-mar-2011/Bourdin.KIS: coded
!
      real, dimension(:,:,:,:), intent(out) :: out
      real, dimension(:,:,:,:), intent(in), optional :: in
      integer, intent(in), optional :: source_proc
!
      if (present (in) .or. present (source_proc)) out = in
!
    endsubroutine distribute_z_4D
!***********************************************************************
    subroutine collect_z_3D(in, out, dest_proc)
!
!  Collect 3D data from all processors in the z-direction
!  and combine it into one large array on one destination processor.
!
!  09-mar-2011/Bourdin.KIS: coded
!
      real, dimension(:,:,:), intent(in) :: in
      real, dimension(:,:,:), intent(out), optional :: out
      integer, intent(in), optional :: dest_proc
!
      if (present (out) .or. present (dest_proc)) out = in
!
    endsubroutine collect_z_3D
!***********************************************************************
    subroutine collect_z_4D(in, out, dest_proc)
!
!  Collect 4D data from all processors in the z-direction
!  and combine it into one large array on one destination processor.
!
!  09-mar-2011/Bourdin.KIS: coded
!
      real, dimension(:,:,:,:), intent(in) :: in
      real, dimension(:,:,:,:), intent(out), optional :: out
      integer, intent(in), optional :: dest_proc
!
      if (present (out) .or. present (dest_proc)) out = in
!
    endsubroutine collect_z_4D
!***********************************************************************
    subroutine globalize_xy(in, out, dest_proc, source_pz)
!
!  Globalizes local 4D data first along the x, then along the y-direction to
!  the destination processor. The local data is supposed to include the ghost
!  cells. Inner ghost layers are cut away during the combination of the data.
!  'dest_proc' is the destination iproc number relative to the first processor
!  in the corresponding xy-plane (Default: 0, equals lfirst_proc_xy).
!
!  23-Apr-2012/Bourdin.KIS: adapted from non-torus-type globalize_xy
!
      real, dimension(:,:,:,:), intent(in) :: in
      real, dimension(:,:,:,:), intent(out), optional :: out
      integer, intent(in), optional :: dest_proc, source_pz
!
      if (present (dest_proc) .or. present (source_pz)) continue
      if (present (out)) out = in
!
    endsubroutine globalize_xy
!***********************************************************************
    subroutine localize_xy(out, in, source_proc, dest_pz)
!
!  Localizes global 4D data first along the y, then along the x-direction to
!  the destination processor. The global data is supposed to include the outer
!  ghost layers. The returned data will include inner ghost layers.
!  Inner ghost layers are cut away during the combination of the data.
!
!  23-Apr-2012/Bourdin.KIS: adapted from non-torus-type localize_xy
!
      real, dimension(:,:,:,:), intent(out) :: out
      real, dimension(:,:,:,:), intent(in), optional :: in
      integer, intent(in), optional :: source_proc, dest_pz
!
      if (present (source_proc) .or. present (dest_pz)) continue
      if (present (in)) out = in
!
    endsubroutine localize_xy
!***********************************************************************
    subroutine globalize_z(in, out, dest_proc)
!
!  Globalizes local 1D data in the z-direction to the destination processor.
!  The local data is supposed to include the ghost cells.
!  Inner ghost layers are cut away during the combination of the data.
!
!  13-aug-2011/Bourdin.KIS: coded
!
      real, dimension(:), intent(in) :: in
      real, dimension(:), intent(out), optional :: out
      integer, intent(in), optional :: dest_proc
!
      if (present (dest_proc)) continue
      if (present (out)) out = in
!
    endsubroutine globalize_z
!***********************************************************************
    subroutine localize_z(out, in, source_proc)
!
!  Localizes global 1D data to all processors along the z-direction.
!  The global data is supposed to include the outer ghost layers.
!  The returned data will include inner ghost layers.
!
!  13-aug-2011/Bourdin.KIS: coded
!
      real, dimension(:), intent(out) :: out
      real, dimension(:), intent(in), optional :: in
      integer, intent(in), optional :: source_proc
!
      if (present (source_proc)) continue
      if (present (in)) out = in
!
    endsubroutine localize_z
!***********************************************************************
    subroutine distribute_to_pencil_xy_2D(in, out)
!
!  Distribute 2D data to several processors and reform into pencil shape.
!  This routine divides global 2D data and distributes it in the xy-plane.
!
!  22-jul-2010/Bourdin.KIS: coded
!
      real, dimension(:,:), intent(in) :: in
      real, dimension(:,:), intent(out) :: out
!
      out = in
!
    endsubroutine distribute_to_pencil_xy_2D
!***********************************************************************
    subroutine collect_from_pencil_xy_2D(in, out)
!
!  Collect 2D data from several processors and combine into global shape.
!  This routine collects 2D pencil shaped data distributed in the xy-plane.
!
!  22-jul-2010/Bourdin.KIS: coded
!
      real, dimension(:,:), intent(in) :: in
      real, dimension(:,:), intent(out) :: out
!
      out = in
!
    endsubroutine collect_from_pencil_xy_2D
!***********************************************************************
    subroutine remap_to_pencil_x(in, out)
!
!  Remaps data distributed on several processors into pencil shape.
!  This routine remaps 1D arrays in x only for nprocx>1.
!
!   08-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:), intent(in) :: in
      real, dimension(:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_x
!***********************************************************************
    subroutine unmap_from_pencil_x(in, out)
!
!  Unmaps pencil shaped 1D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocx>1.
!
!  08-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:), intent(in) :: in
      real, dimension(:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_x
!***********************************************************************
    subroutine remap_to_pencil_y_1D(in, out)
!
!  Remaps data distributed on several processors into pencil shape.
!  This routine remaps 1D arrays in y only for nprocy>1.
!
!  08-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:), intent(in) :: in
      real, dimension(:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_y_1D
!***********************************************************************
    subroutine remap_to_pencil_y_2D(in, out)
!
!  Remaps data distributed on several processors into pencil shape.
!  This routine remaps 2D arrays in y only for nprocy>1.
!
!  08-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:,:), intent(in) :: in
      real, dimension(:,:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_y_2D
!***********************************************************************
    subroutine remap_to_pencil_y_3D(in, out)
!
!  Remaps data distributed on several processors into pencil shape.
!  This routine remaps 3D arrays in y only for nprocy>1.
!
!  08-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:), intent(in) :: in
      real, dimension(:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_y_3D
!***********************************************************************
    subroutine remap_to_pencil_y_4D(in, out)
!
!  Remaps data distributed on several processors into pencil shape.
!  This routine remaps 4D arrays in y only for nprocy>1.
!
!  08-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:,:), intent(in) :: in
      real, dimension(:,:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_y_4D
!***********************************************************************
    subroutine unmap_from_pencil_y_1D(in, out)
!
!  Unmaps pencil shaped 1D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocy>1.
!
!  08-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:), intent(in) :: in
      real, dimension(:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_y_1D
!***********************************************************************
    subroutine unmap_from_pencil_y_2D(in, out)
!
!  Unmaps pencil shaped 2D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocy>1.
!
!  08-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:,:), intent(in) :: in
      real, dimension(:,:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_y_2D
!***********************************************************************
    subroutine unmap_from_pencil_y_3D(in, out)
!
!  Unmaps pencil shaped 3D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocy>1.
!
!  08-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:), intent(in) :: in
      real, dimension(:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_y_3D
!***********************************************************************
    subroutine unmap_from_pencil_y_4D(in, out)
!
!  Unmaps pencil shaped 4D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocy>1.
!
!  08-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:,:), intent(in) :: in
      real, dimension(:,:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_y_4D
!***********************************************************************
    subroutine remap_to_pencil_z_1D(in, out)
!
!  Remaps data distributed on several processors into pencil shape.
!  This routine remaps 1D arrays in z only for nprocz>1.
!
!  13-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:), intent(in) :: in
      real, dimension(:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_z_1D
!***********************************************************************
    subroutine remap_to_pencil_z_2D(in, out)
!
!  Remaps data distributed on several processors into pencil shape.
!  This routine remaps 2D arrays in z only for nprocz>1.
!
!  13-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:,:), intent(in) :: in
      real, dimension(:,:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_z_2D
!***********************************************************************
    subroutine remap_to_pencil_z_3D(in, out)
!
!  Remaps data distributed on several processors into pencil shape.
!  This routine remaps 3D arrays in z only for nprocz>1.
!
!  13-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:), intent(in) :: in
      real, dimension(:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_z_3D
!***********************************************************************
    subroutine remap_to_pencil_z_4D(in, out)
!
!  Remaps data distributed on several processors into pencil shape.
!  This routine remaps 4D arrays in z only for nprocz>1.
!
!  13-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:,:), intent(in) :: in
      real, dimension(:,:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_z_4D
!***********************************************************************
    subroutine unmap_from_pencil_z_1D(in, out)
!
!  Unmaps pencil shaped 1D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocz>1.
!
!  13-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:), intent(in) :: in
      real, dimension(:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_z_1D
!***********************************************************************
    subroutine unmap_from_pencil_z_2D(in, out)
!
!  Unmaps pencil shaped 2D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocz>1.
!
!  13-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:,:), intent(in) :: in
      real, dimension(:,:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_z_2D
!***********************************************************************
    subroutine unmap_from_pencil_z_3D(in, out)
!
!  Unmaps pencil shaped 3D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocz>1.
!
!  13-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:), intent(in) :: in
      real, dimension(:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_z_3D
!***********************************************************************
    subroutine unmap_from_pencil_z_4D(in, out)
!
!  Unmaps pencil shaped 4D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocz>1.
!
!  13-dec-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:,:), intent(in) :: in
      real, dimension(:,:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_z_4D
!***********************************************************************
    subroutine remap_to_pencil_xy_2D(in, out)
!
!  Remaps data distributed on several processors into pencil shape.
!  This routine remaps 2D arrays in x and y only for nprocx>1.
!
!   4-jul-2010/Bourdin.KIS: coded
!
      real, dimension(:,:), intent(in) :: in
      real, dimension(:,:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_xy_2D
!***********************************************************************
    subroutine remap_to_pencil_xy_2D_other(in, out)
!
!  Remaps data distributed on several processors into pencil shape.
!  This routine remaps 2D arrays in x and y only for nprocx>1.
!
!   4-jul-2010/Bourdin.KIS: coded
!
      real, dimension(:,:), intent(in) :: in
      real, dimension(:,:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_xy_2D_other
!***********************************************************************
    subroutine remap_to_pencil_xy_3D(in, out)
!
!  Remaps data distributed on several processors into pencil shape.
!  This routine remaps 3D arrays in x and y only for nprocx>1.
!
!  14-jul-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:), intent(in) :: in
      real, dimension(:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_xy_3D
!***********************************************************************
    subroutine remap_to_pencil_xy_4D(in, out)
!
!  Remaps data distributed on several processors into pencil shape.
!  This routine remaps 4D arrays in x and y only for nprocx>1.
!
!  14-jul-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:,:), intent(in) :: in
      real, dimension(:,:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_xy_4D
!***********************************************************************
    subroutine unmap_from_pencil_xy_2D(in, out)
!
!  Unmaps pencil shaped 2D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocx>1.
!
!   4-jul-2010/Bourdin.KIS: coded
!
      real, dimension(:,:), intent(in) :: in
      real, dimension(:,:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_xy_2D
!***********************************************************************
    subroutine unmap_from_pencil_xy_2D_other(in, out)
!
!  Unmaps pencil shaped 2D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocx>1.
!
!   4-jul-2010/Bourdin.KIS: coded
!
      real, dimension(:,:), intent(in) :: in
      real, dimension(:,:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_xy_2D_other
!***********************************************************************
    subroutine unmap_from_pencil_xy_3D(in, out)
!
!  Unmaps pencil shaped 3D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocx>1.
!
!  14-jul-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:), intent(in) :: in
      real, dimension(:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_xy_3D
!***********************************************************************
    subroutine unmap_from_pencil_xy_4D(in, out)
!
!  Unmaps pencil shaped 4D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocx>1.
!
!  14-jul-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:,:), intent(in) :: in
      real, dimension(:,:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_xy_4D
!***********************************************************************
    subroutine transp_pencil_xy_2D(in, out)
!
!  Transpose 2D data distributed on several processors.
!  This routine transposes arrays in x and y only.
!  The data must be mapped in pencil shape, especially for nprocx>1.
!
!   4-jul-2010/Bourdin.KIS: coded, adapted parts of transp_xy
!
      real, dimension(:,:), intent(in) :: in
      real, dimension(:,:), intent(out) :: out
!
      out = transpose (in)
!
    endsubroutine transp_pencil_xy_2D
!***********************************************************************
    subroutine transp_pencil_xy_3D(in, out)
!
!  Transpose 3D data distributed on several processors.
!  This routine transposes arrays in x and y only.
!  The data must be mapped in pencil shape, especially for nprocx>1.
!
!  14-jul-2010/Bourdin.KIS: coded, adapted parts of transp_xy
!
      real, dimension(:,:,:), intent(in) :: in
      real, dimension(:,:,:), intent(out) :: out
!
      integer :: pos_z
!
      do pos_z = 1, size (in, 3)
        out(:,:,pos_z) = transpose (in(:,:,pos_z))
      enddo
!
    endsubroutine transp_pencil_xy_3D
!***********************************************************************
    subroutine transp_pencil_xy_4D(in, out)
!
!  Transpose 4D data distributed on several processors.
!  This routine transposes arrays in x and y only.
!  The data must be mapped in pencil shape, especially for nprocx>1.
!
!  14-jul-2010/Bourdin.KIS: coded, adapted parts of transp_xy
!
      real, dimension(:,:,:,:), intent(in) :: in
      real, dimension(:,:,:,:), intent(out) :: out
!
      integer :: pos_z, pos_a
!
      do pos_z = 1, size (in, 3)
        do pos_a = 1, size (in, 4)
          out(:,:,pos_z,pos_a) = transpose (in(:,:,pos_z,pos_a))
        enddo
      enddo
!
    endsubroutine transp_pencil_xy_4D
!***********************************************************************
    subroutine remap_to_pencil_yz_3D(in, out)
!
!  Remaps data distributed on several processors into z-pencil shape.
!  This routine remaps 3D arrays in y and z only for nprocz>1.
!
!  27-oct-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:), intent(in) :: in
      real, dimension(:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_yz_3D
!***********************************************************************
    subroutine remap_to_pencil_yz_4D(in, out)
!
!  Remaps data distributed on several processors into z-pencil shape.
!  This routine remaps 4D arrays in y and z only for nprocz>1.
!
!  27-oct-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:,:), intent(in) :: in
      real, dimension(:,:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine remap_to_pencil_yz_4D
!***********************************************************************
    subroutine unmap_from_pencil_yz_3D(in, out)
!
!  Unmaps z-pencil shaped 3D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocz>1.
!
!  27-oct-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:), intent(in) :: in
      real, dimension(:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_yz_3D
!***********************************************************************
    subroutine unmap_from_pencil_yz_4D(in, out)
!
!  Unmaps z-pencil shaped 4D data distributed on several processors back to normal shape.
!  This routine is the inverse of the remap function for nprocz>1.
!
!  27-oct-2010/Bourdin.KIS: coded
!
      real, dimension(:,:,:,:), intent(in) :: in
      real, dimension(:,:,:,:), intent(out) :: out
!
      out = in
!
    endsubroutine unmap_from_pencil_yz_4D
!***********************************************************************
    subroutine y2x(a,xi,zj,zproc_no,ay)
!
!  Load the y dimension of an array in a 1-d array.
!
!  21-mar-2011/axel: adapted from z2x
!
      real, dimension(nx,ny,nz), intent(in) :: a
      real, dimension(ny), intent(out) :: ay
      integer, intent(in) :: xi,zj,zproc_no
!
      ay(:)=a(xi,:,zj)
      if (ALWAYS_FALSE) print*,zproc_no
!
    endsubroutine y2x
!***********************************************************************
    subroutine z2x(a,xi,yj,yproc_no,az)
!
!  Load the z dimension of an array in a 1-d array.
!
!  1-july-2008: dhruba
!
      real, dimension(nx,ny,nz), intent(in) :: a
      real, dimension(nz), intent(out) :: az
      integer, intent(in) :: xi,yj,yproc_no
!
      az(:)=a(xi,yj,:)
      if (ALWAYS_FALSE) print*,yproc_no
!
    endsubroutine z2x
!***********************************************************************
    subroutine parallel_open_ext(unit,file,form,nitems)
!
!  Read a global file.
!
!  18-mar-10/Bourdin.KIS: implemented
!
      integer :: unit
      character (len=*) :: file
      character (len=*), optional :: form
      integer, optional :: nitems
!
      logical :: exists
!
      if (present(nitems)) nitems=0
!
!  Test if file exists.
!
      inquire(FILE=file,exist=exists)
      if (.not. exists) call stop_it('parallel_open: file not found "'//trim(file)//'"')
!
!  Open file.
!
      if (present(form)) then
        open(unit, FILE=file, FORM=form, STATUS='old')
      else
        open(unit, FILE=file, STATUS='old')
      endif
!
    endsubroutine parallel_open_ext
!***********************************************************************
    subroutine parallel_open_int(unit,file,form,nitems)
!
!  Read a global file into buffer unit.
!
!  12-jan-15/MR: implemented
!
      character(len=:), allocatable :: unit
      character(len=*) :: file
      character(len=*), optional :: form
      integer, optional :: nitems
!
      integer :: ni
      character(LEN=labellen) :: message
!
      ni=read_infile(file,unit,message)
      if (ni<0) call stop_it_if_any(.true.,message)

      if (present(nitems)) nitems=ni
!
    endsubroutine parallel_open_int
!***********************************************************************
    subroutine parallel_close_ext(unit)
!
!  Close a file unit opened by parallel_open.
!
!  18-mar-10/Bourdin.KIS: implemented
!
      integer :: unit
!
      close(unit)
!
    endsubroutine parallel_close_ext
!***********************************************************************
    subroutine parallel_close_int(unit)
!
!  "Close" an internal file by deallocating it.
!
!  12-jan-15/MR: implemented
!
      character(len=:), allocatable :: unit
!
      if (allocated(unit)) deallocate(unit)
!
    endsubroutine parallel_close_int
!***********************************************************************
    subroutine mpigather_xy( sendbuf, recvbuf, lpz )
!
!  21-dec-10/MR: coded
!
      real, dimension(nxgrid,ny)     :: sendbuf
      real, dimension(nxgrid,nygrid) :: recvbuf
      integer                        :: lpz
!
      recvbuf(:,1:ny) = sendbuf
!
      if (ALWAYS_FALSE) print*,lpz
!
    endsubroutine mpigather_xy
!***********************************************************************
    subroutine mpigather_z(sendbuf,recvbuf,n1,lproc)
!
!  21-dec-10/MR: coded
!  20-apr-11/MR: buffer dimensions corrected
!
      integer,                    intent(in)  :: n1
      real, dimension(n1,nz)    , intent(in)  :: sendbuf
      real, dimension(n1,nzgrid), intent(out) :: recvbuf
      integer, optional,          intent(in)  :: lproc
!
      recvbuf(:,1:nz) = sendbuf
!
      if (ALWAYS_FALSE) print*,n1,present(lproc)
!
    endsubroutine mpigather_z
!***********************************************************************
    subroutine mpigather_and_out_real( sendbuf, unit, ltransp, kxrange, kyrange, zrange )
!
!  21-dec-10/MR: coded
!  06-apr-11/MR: optional parameters kxrange, kyrange, zrange for selective output added
!  10-may-11/MR: modified into real and complex flavors
!  20-mar-15/MR: made potentially big arrays sendbuf* assumed-shape
!
      use General, only: write_by_ranges_2d_real, write_by_ranges_2d_cmplx
!
      implicit none
!
      integer,                              intent(in) :: unit
      real,    dimension(:,:,:),            intent(in) :: sendbuf
      complex, dimension(:,:,:,:),          intent(in) :: sendbuf_cmplx
      logical,                    optional, intent(in) :: ltransp
      integer, dimension(3,*),    optional, intent(in) :: kxrange, kyrange,zrange
!
      integer :: ncomp,k,kl,ic
      logical :: ltrans, lcomplex
      integer, dimension(3,nk_max) :: kxrangel,kyrangel
      integer, dimension(3,nz_max) :: zrangel
!
      lcomplex = .false.
      ncomp = 1
      goto 1
!
      entry mpigather_and_out_cmplx( sendbuf_cmplx, unit, ltransp, kxrange, kyrange, zrange )
      ncomp = size(sendbuf_cmplx,4)
      lcomplex = .true.
!
   1  if ( .not.present(ltransp) ) then
        ltrans=.false.
      else
        ltrans=ltransp
      endif
!
      if ( .not.present(kxrange) ) then
        kxrangel = 0
        kxrangel(:,1) = (/1,nxgrid,1/)
      else
        kxrangel=kxrange(:,1:nk_max)
      endif
!
      if ( .not.present(kyrange) ) then
        kyrangel = 0
        kyrangel(:,1) = (/1,nygrid,1/)
      else
        kyrangel=kyrange(:,1:nk_max)
      endif
!
      if ( .not.present(zrange) ) then
        zrangel = 0
        zrangel(:,1) = (/1,nzgrid,1/)
      else
        zrangel=zrange(:,1:nz_max)
      endif
!
      do ic=1,ncomp
        do k=1,nz_max
          if ( zrangel(1,k) > 0 ) then
            do kl=zrangel(1,k),zrangel(2,k),zrangel(3,k)
              if ( lcomplex ) then
                call write_by_ranges_2d_cmplx( 1, sendbuf_cmplx(:,:,kl,ic), kxrangel, kyrangel, ltrans )
              else
                call write_by_ranges_2d_real( 1, sendbuf(:,:,kl), kxrangel, kyrangel, ltrans )
              endif
            enddo
          endif
        enddo
      enddo
!
      if (ALWAYS_FALSE) print*,unit,present(ltransp)
!
    endsubroutine mpigather_and_out_real
!***********************************************************************
    subroutine mpimerge_1d(vector,nk,idir)
!
!  21-dec-10/MR: coded
!
      integer,             intent(in)    :: nk
      real, dimension(nk), intent(inout) :: vector
      integer, optional,   intent(in)    :: idir
!
      if (ALWAYS_FALSE) print*,vector,nk,present(idir)
!
      return
!
    endsubroutine mpimerge_1d
!***********************************************************************
  logical function report_clean_output(flag, message)
!
    logical,             intent(IN)  :: flag
    character (LEN=120), intent(OUT) :: message
!
    message = ''
    report_clean_output = .false.
!
    if (ALWAYS_FALSE) print*,flag,message
!
  end function report_clean_output
!***********************************************************************
    function read_infile(file,buffer,message) result(ni)
!
!  Primary reading of a global file
!
!  11-jan-15/MR: outsourced from true_parallel_open
!
      use Cdata, only: comment_char
      use File_io, only: file_size

      character(len=*) :: file,message
      character(len=:), allocatable :: buffer
      integer :: ni
!
      character(len=4096) :: linebuf          ! fixed length problematic

      integer, parameter :: unit=1
      integer :: bytes,inda,inda2,ind,indc,lenbuf,ios
      logical :: exists,l0

      ni=-1
!
!  Test if file exists.
!
        inquire(FILE=file,exist=exists)
        if (.not. exists) then
          message='read_infile: file not found "'//trim(file)//'"'
          return
        endif
        bytes=file_size(file)
        if (bytes < 0) then
          message='read_infile: could not determine file size"'//trim(file)//'"'
          return
        elseif (bytes == 0) then
          message='read_infile: file is empty "'//trim(file)//'"'
          return
        endif
!
!  Allocate temporary memory.
!
        allocate(character(len=bytes) :: buffer)
!
!  Read file content into buffer.
!
        open(unit, FILE=file, STATUS='old')

        l0=.true.; buffer=' '; ni=0

        do
          read(unit,'(a)',iostat=ios) linebuf
          if (ios<0) exit

          linebuf=adjustl(linebuf)
!
          inda=index(linebuf,"'")
          ind=index(linebuf,'!'); indc=index(linebuf,comment_char)

          if (inda>0) then
            inda2=index(linebuf(inda+1:),"'")+inda
            if (inda2==inda) inda2=len(linebuf)+1
            if (ind>inda.and.ind<inda2) ind=0
            if (indc>inda.and.indc<inda2) indc=0
          endif

          if (indc>0) ind=min(max(ind,1),indc)
!
          if (ind==0) then
            ind=len(trim(linebuf))
          else
            ind=ind-1
            if (ind>0) ind=len(trim(linebuf(1:ind)))
          endif
!
          if (ind==0) then        ! is a comment or empty line -> skip
            cycle
          elseif (l0) then
            buffer=linebuf(1:ind)
            lenbuf=ind
            l0=.false.
          else
            buffer=buffer(1:lenbuf)//' '//linebuf(1:ind)
            lenbuf=lenbuf+ind+1
          endif
          ni=ni+1

        enddo

        close(unit)
!
    endfunction read_infile
!***********************************************************************
endmodule Mpicomm
