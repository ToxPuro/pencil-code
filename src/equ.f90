! $Id$
!
!  subroutines in the chosen set of physics modules.
!
module Equ
!
  use Cdata
  use Messages
  use Boundcond
  use Mpicomm
  use Grid, only: calc_pencils_grid, get_grid_mn
!
  implicit none
!
  public :: pde, debug_imn_arrays, initialize_pencils
  public :: impose_floors_ceilings
!
  private
  real, pointer, dimension (:) :: p_fname,p_fname_keep
  real, pointer, dimension (:,:) :: p_fnamer,p_fname_sound
  integer, pointer, dimension (:,:) :: p_ncountsz
  real, pointer, dimension (:,:,:) :: p_fnamex,p_fnamey,p_fnamez,p_fnamexy,p_fnamexz
  real, pointer, dimension(:,:,:,:) :: p_fnamerz
  real, pointer, dimension(:) :: p_dt1_max
!
  contains
!***********************************************************************
    include 'pencil_init.inc' ! defines subroutine initialize_pencils()
!***********************************************************************
    subroutine pde(f,df,p)
!
!  Call the different evolution equations.
!
!  10-sep-01/axel: coded
!  12-may-12/MR: call of density_before_boundary added for boussinesq;
!                moved call of timing after call of anelastic_after_mn
!  26-aug-13/MR: added call of diagnostic for imaginary parts
!   9-jun-15/MR: call of gravity_after_boundary added
!  24-sep-16/MR: added offset manipulation for second derivatives in complete one-sided fornulation.
!   5-jan-17/MR: removed mn-offset manipulation
!  14-feb-17/MR: adaptations for use of GPU kernels in calculating the rhss of the pde
!
      use Chiral
      use Chemistry
      use Density
      use Detonate, only: detonate_before_boundary
      use Diagnostics
      use Dustdensity
      use Energy
      use EquationOfState
      use Forcing, only: forcing_after_boundary
!                         
! To check ghost cell consistency, please uncomment the following line:
!     use Ghost_check, only: check_ghosts_consistency
      use General, only: ioptest, loptest
      use GhostFold, only: fold_df, fold_df_3points
      use Gpu
      use Gravity
      use Hydro
      use Interstellar, only: interstellar_before_boundary
      use Magnetic
      use Hypervisc_strict, only: hyperviscosity_strict
      use Hyperresi_strict, only: hyperresistivity_strict
      use NeutralDensity, only: neutraldensity_after_boundary
      use NSCBC
      use Particles_main
      use Poisson
      use Pscalar
      use PointMasses
      use Polymer
      use Radiation
      use Selfgravity
      use Shear
      use Shock, only: shock_before_boundary, calc_shock_profile_simple
      use Solid_Cells, only: update_solid_cells, dsolid_dt_integrate
      use Special, only: special_before_boundary,special_after_boundary
      use Sub
      use Testfield
      use Testflow
      use Testscalar
      use Viscosity, only: viscosity_after_boundary
      use Grid, only: coarsegrid_interp
!
      real, dimension (mx,my,mz,mfarray) :: f
      real, dimension (mx,my,mz,mvar) :: df
      type (pencil_case) :: p
!
      intent(inout):: f       ! inout due to lshift_datacube_x,
                              ! density floor, or velocity ceiling
      intent(out)  :: df,p
!
      logical :: early_finalize
      real, dimension(1)  :: mass_per_proc
      integer :: iv
!
!  Print statements when they are first executed.
!
      headtt = headt .and. lfirst .and. lroot
!
      if (headtt.or.ldebug) print*,'pde: ENTER'
      if (headtt) call svn_id( &
           "$Id$")
!
!  Initialize counter for calculating and communicating print results.
!  Do diagnostics only in the first of the 3 (=itorder) substeps.
!
      ldiagnos   =lfirst.and.lout
      l1davgfirst=lfirst.and.l1davg
      l2davgfirst=lfirst.and.l2davg
!
!  Derived diagnostics switches.
!
      l1dphiavg=lcylinder_in_a_box.and.l1davgfirst
!
!  For chemistry with LSODE
!
      lchemonly=.false.
!
!  Record times for diagnostic and 2d average output.
!
      if (ldiagnos   ) tdiagnos  =t ! (diagnostics are for THIS time)
      if (l1davgfirst) t1ddiagnos=t ! (1-D averages are for THIS time)
      if (l2davgfirst)  then
        t2davgfirst=t ! (2-D averages are for THIS time)
!
!  [AB: Isn't it true that not all 2-D averages use rcyl_mn?
!  lwrite_phiaverages=T is required, and perhaps only that.]
!  [BD: add also the z_mn dependency]
!
        lpencil(i_rcyl_mn)=.true.
        lpencil(i_z_mn)=.true.
      endif
!
!  Shift entire data cube by one grid point at the beginning of each
!  time-step. Useful for smearing out possible x-dependent numerical
!  diffusion, e.g. in a linear shear flow.
!
      if (lfirst .and. lshift_datacube_x) then
        call boundconds_x(f)
        do  n=n1,n2; do m=m1,m2
          f(:,m,n,:)=cshift(f(:,m,n,:),1,1)
        enddo; enddo
      endif
!
!  Need to finalize communication early either for test purposes, or
!  when radiation transfer of global ionization is calculated.
!  This could in principle be avoided (but it not worth it now)
!
      early_finalize=  test_nonblocking.or. &
                     leos_ionization.or.lradiation_ray.or. &
                     lhyperviscosity_strict.or.lhyperresistivity_strict.or. &
                     ltestscalar.or.ltestfield.or.ltestflow.or. &
                     lparticles_spin.or.lsolid_cells.or. &
                     lchemistry.or.lweno_transport .or. lbfield .or. & 
!                     lslope_limit_diff .or. lvisc_smag .or. &
                     lvisc_smag .or. &
                     lyinyang .or. &   !!!
                     ncoarse>1 .or. lgpu 
                     !.or. lgpu
                     ! & .or. lgpu
!
!  Write crash snapshots to the hard disc if the time-step is very low.
!  The user must have set crash_file_dtmin_factor>0.0 in &run_pars for
!  this to be done.
!
      if (crash_file_dtmin_factor > 0.0) call output_crash_files(f)
!
!  For debugging purposes impose minimum or maximum value on certain variables.
!
      call impose_floors_ceilings(f)   !MR: too early, f modifications come below
!
!  Apply global boundary conditions to particle positions and communicate
!  migrating particles between the processors.
!
      if (lparticles) call particles_boundconds(f)
      if (lpointmasses) call boundconds_pointmasses
!
!  Calculate the potential of the self gravity. Must be done before
!  communication in order to be able to take the gradient of the potential
!  later.
!
      call calc_selfpotential(f)
!
!  Check for dust grain mass interval overflows
!  (should consider having possibility for all modules to fiddle with the
!   f array before boundary conditions are sent)
!
      if (.not. lchemistry) then
        if (ldustdensity) call null_dust_vars(f)
        if (ldustdensity .and. lmdvar .and. lfirst) call redist_mdbins(f)
      endif
!
!  Call "before_boundary" hooks (for f array precalculation)
!
      if (linterstellar) call interstellar_before_boundary(f)
      if (ldensity.or.lboussinesq) call density_before_boundary(f)
      if (lhydro.or.lhydro_kinematic) call hydro_before_boundary(f)
      if (lmagnetic)     call magnetic_before_boundary(f)
                         call energy_before_boundary(f)
      if (lshear)        call shear_before_boundary(f)
      if (lchiral)       call chiral_before_boundary(f)
      if (lspecial)      call special_before_boundary(f)
      if (ltestflow)     call testflow_before_boundary(f)
      if (ltestfield)    call testfield_before_boundary(f)
      if (lparticles)    call particles_before_boundary(f)
      if (lpscalar)      call pscalar_before_boundary(f)
      if (ldetonate)     call detonate_before_boundary(f)
      if (lchemistry)    call chemistry_before_boundary(f)
      if (lparticles.and.lspecial) call particles_special_bfre_bdary(f)
      if (lshock)        call shock_before_boundary(f)
!
!  Prepare x-ghost zones; required before f-array communication
!  AND shock calculation
!
      call boundconds_x(f)
!
!  Initiate (non-blocking) communication and do boundary conditions.
!  Required order:
!  1. x-boundaries (x-ghost zones will be communicated) - done above
!  2. communication
!  3. y- and z-boundaries
!
      if (nghost>0) then
        if (ldebug) print*,'pde: before initiate_isendrcv_bdry'
        call initiate_isendrcv_bdry(f)
        if (early_finalize) then
          call finalize_isendrcv_bdry(f)
          if (lcoarse) call coarsegrid_interp(f)   ! after boundconds_x???
          call boundconds_y(f)
          call boundconds_z(f)
        endif
      endif
!
! update solid cell "ghost points". This must be done in order to get the
! correct boundary layer close to the solid geometry, i.e. no-slip conditions.
!
      call update_solid_cells(f)
!
!  For sixth order momentum-conserving, symmetric hyperviscosity with positive
!  definite heating rate we need to precalculate the viscosity term. The
!  restivitity term for sixth order hyperresistivity with positive definite
!  heating rate must also be precalculated.
!
      if (lhyperviscosity_strict)   call hyperviscosity_strict(f)
      if (lhyperresistivity_strict) call hyperresistivity_strict(f)
!
!  Dynamically set the (hyper-)diffusion coefficients
!
      if (ldynamical_diffusion) call set_dyndiff_coeff(f)
!
!  Calculate the characteristic velocity
!  for slope limited diffusion
!
!      if (lslope_limit_diff.and.llast) then
!        f(2:mx-2,2:my-2,2:mz-2,iFF_char_c)=0.
!print*,'vor magnetic:', maxval(f(2:mx-2,2:my-2,2:mz-2,iFF_char_c))
!        call update_char_vel_energy(f)
!        call update_char_vel_magnetic(f)
!        call update_char_vel_hydro(f)
        !call update_char_vel_density(f)
        !f(2:mx-2,2:my-2,2:mz-2,iFF_char_c)=sqrt(f(2:mx-2,2:my-2,2:mz-2,iFF_char_c))
!  JW: for hydro it is done without sqrt
        !if (ldiagnos) print*, 'max(char_c)=', maxval(f(2:mx-2,2:my-2,2:mz-2,iFF_char_c))
!      endif
!
!  For calculating the pressure gradient directly from the pressure (which is
!  derived from the basic thermodynamical variables), we need to fill in the
!  pressure in the f array.
!
      call fill_farray_pressure(f)
!
!  Set inverse timestep to zero before entering loop over m and n.
!  If we want to have a logarithmic time advance, we want set this here
!  as the maximum. All other routines can then still make it shorter.
!
      if (lfirst.and.ldt) then
        if (dtmax/=0.0) then
          if (lfractional_tstep_advance) then
            dt1_max=1./(dt_incr*t)
          else
            dt1_max=1./dtmax
          endif
        else
          dt1_max=0.0
        endif
      endif
!
!  Calculate ionization degree (needed for thermodynamics)
!  Radiation transport along rays. If lsingle_ray, then this
!  is only used for visualization and only needed when lvideo
!  (but this is decided in radtransfer itself)
!
      if (leos_ionization.or.leos_temperature_ionization) call ioncalc(f)
      if (lradiation_ray) call radtransfer(f)
!
!  Calculate shock profile (simple).
!
      if (lshock) call calc_shock_profile_simple(f)
!
!  Call "after" hooks (for f array precalculation). This may imply
!  calculating averages (some of which may only be required for certain
!  settings in hydro of the testfield procedure (only when lsoca=.false.),
!  for example. They used to be or are still called hydro_after_boundary etc,
!  and will soon be renamed to hydro_after_boundary.
!
!  Important to note that the processor boundaries are not full updated 
!  at this point, even if the name 'after_boundary' suggesting this.
!  Use early_finalize in this case.
!  MR+joern+axel, 8.10.2015
!
      call timing('pde','before "after_boundary" calls')
!
      if (lhydro)                 call hydro_after_boundary(f)
      if (lviscosity)             call viscosity_after_boundary(f)
      if (lmagnetic)              call magnetic_after_boundary(f)
      if (lenergy)                call energy_after_boundary(f)
      if (lgrav)                  call gravity_after_boundary(f)
      if (lforcing)               call forcing_after_boundary(f)
      if (lpolymer)               call calc_polymer_after_boundary(f)
      if (ltestscalar)            call testscalar_after_boundary(f)
      if (ltestfield)             call testfield_after_boundary(f)
!AB: quick fix
      !if (ltestfield)             call testfield_after_boundary(f,p)
      if (ldensity)               call density_after_boundary(f)
      if (lneutraldensity)        call neutraldensity_after_boundary(f)
      if (ltestflow)              call calc_ltestflow_nonlin_terms(f,df)  ! should not use df!
      if (lspecial)               call special_after_boundary(f)
!
!  Calculate quantities for a chemical mixture. This is done after
!  communication has finalized since many of the arrays set up here
!  are not communicated, and in this subroutine also ghost zones are calculated.
!
      if (lchemistry .and. ldensity) call calc_for_chem_mixture(f)
!
      call timing('pde','after "after_boundary" calls')
!
      print*, 'rhs check:',itsub,':',iproc,':',f(10,10,10,1:8)
      ! write(80+iproc,*) 'itsub=', itsub

      ! write(88+iproc,'(12(f11.6,1x))') f(l1:l2,m1:m2,n1:n2,1:8)
      if (lgpu) then
        print*,"l1 etc on pencil code", l1,l2,m1,m2,n1,n2
        call rhs_gpu(f,itsub,early_finalize)
        if (ldiagnos.or.l1davgfirst.or.l1dphiavg.or.l2davgfirst) then
          call copy_farray_from_GPU(f)
          call calc_all_module_diagnostics(f,p)
        endif
      else
        call rhs_cpu(f,df,p,mass_per_proc,early_finalize)
      endif
!
!  Finish the job for the anelastic approximation
!
      if (lanelastic) call anelastic_after_mn(f,p,df,mass_per_proc)
!
      call timing('pde','after the end of the mn_loop')
!
!  Integrate diagnostics related to solid cells (e.g. drag and lift).
!
      if (lsolid_cells) call dsolid_dt_integrate
!
!  Calculate the gradient of the potential if there is room allocated in the
!  f-array.
!
      if (igpotselfx/=0) then
        call initiate_isendrcv_bdry(f,igpotselfx,igpotselfz)
        call finalize_isendrcv_bdry(f,igpotselfx,igpotselfz)
        call boundconds_x(f,igpotselfx,igpotselfz)
        call boundconds_y(f,igpotselfx,igpotselfz)
        call boundconds_z(f,igpotselfx,igpotselfz)
      endif
!
!  Change df and dfp according to the chosen particle modules.
!
      if (lparticles) then
        call particles_pde_blocks(f,df)
        call particles_pde(f,df)
      endif
!
      if (lpointmasses) call pointmasses_pde(f,df)
!
!  Electron inertia: our df(:,:,:,iax:iaz) so far is
!  (1 - l_e^2\Laplace) daa, thus to get the true daa, we need to invert
!  that operator.
!  [wd-aug-2007: This should be replaced by the more general stuff with the
!   Poisson solver (so l_e can be non-constant), so at some point, we can
!   remove/replace this]
!
!      if (lelectron_inertia .and. inertial_length/=0.) then
!        do iv = iax,iaz
!          call inverse_laplacian_semispectral(df(:,:,:,iv), H=linertial_2)
!        enddo
!        df(:,:,:,iax:iaz) = -df(:,:,:,iax:iaz) * linertial_2
!      endif
!
!  Take care of flux-limited diffusion
!  This is now commented out, because we always use radiation_ray instead.
!
!--   if (lradiation_fld) f(:,:,:,idd)=DFF_new
!
!  Fold df from first ghost zone into main df.
!
      if (lfold_df) then
        if (lhydro .and. (.not. lpscalar) .and. (.not. lchemistry)) then
          call fold_df(df,iux,iuz)
        else
          call fold_df(df,iux,mvar)
        endif
      endif
      if (lfold_df_3points) call fold_df_3points(df,iux,mvar)
!
!  -------------------------------------------------------------
!  NO CALLS MODIFYING DF BEYOND THIS POINT (APART FROM FREEZING)
!  -------------------------------------------------------------
!
!  Freezing must be done after the full (m,n) loop, as df may be modified
!  outside of the considered pencil.
!
      call freeze(df,p)

!  Boundary treatment of the df-array.
!
!  This is a way to impose (time-
!  dependent) boundary conditions by solving a so-called characteristic
!  form of the fluid equations on the boundaries, as opposed to setting
!  actual values of the variables in the f-array. The method is called
!  Navier-Stokes characteristic boundary conditions (NSCBC).
!
!  The treatment should be done after the y-z-loop, but before the Runge-
!  Kutta solver adds to the f-array.
!
      if (lnscbc) call nscbc_boundtreat(f,df)
!
!  0-D Diagnostics.
!
      if (ldiagnos) then
        call diagnostic(fname,nname)
        call diagnostic(fname_keep,nname,lcomplex=.true.)
      endif
!
!  1-D diagnostics.
!
      if (l1davgfirst) then
        if (lwrite_xyaverages) call xyaverages_z
        if (lwrite_xzaverages) call xzaverages_y
        if (lwrite_yzaverages) call yzaverages_x
      endif
      if (l1dphiavg) call phizaverages_r
!
!  2-D averages.
!
      if (l2davgfirst) then
        if (lwrite_yaverages)   call yaverages_xz
        if (lwrite_zaverages)   call zaverages_xy
        if (lwrite_phiaverages) call phiaverages_rz
      endif
!
!  Note: zaverages_xy are also needed if bmx and bmy are to be calculated
!  (of course, yaverages_xz does not need to be calculated for that).
!
      if (.not.l2davgfirst.and.ldiagnos.and.ldiagnos_need_zaverages) then
        if (lwrite_zaverages) call zaverages_xy
      endif
!
!  Calculate mean fields and diagnostics related to mean fields.
!
      if (ldiagnos) then
        if (lmagnetic) call calc_mfield
        if (lhydro)    call calc_mflow
        if (lpscalar)  call calc_mpscalar
      endif
!
!  Calculate rhoccm and cc2m (this requires that these are set in print.in).
!  Broadcast result to other processors. This is needed for calculating PDFs.
!
!      if (idiag_rhoccm/=0) then
!        if (iproc==0) rhoccm=fname(idiag_rhoccm)
!        call mpibcast_real(rhoccm)
!      endif
!
!      if (idiag_cc2m/=0) then
!        if (iproc==0) cc2m=fname(idiag_cc2m)
!        call mpibcast_real(cc2m)
!      endif
!
!      if (idiag_gcc2m/=0) then
!        if (iproc==0) gcc2m=fname(idiag_gcc2m)
!        call mpibcast_real(gcc2m)
!      endif
!
!  Reset lwrite_prof.
!
      lwrite_prof=.false.
!
    endsubroutine pde
!****************************************************************************
    subroutine calc_all_module_diagnostics(f,p)
!
!  Calculates module diagnostics (so far only density, energy, hydro, magnetic)
!
!  10-sep-2019/MR: coded
!
      use Density, only: calc_diagnostics_density
      use Energy, only: calc_diagnostics_energy
      use Hydro, only: calc_diagnostics_hydro
      use Magnetic, only: calc_diagnostics_magnetic
      use Forcing, only: calc_diagnostics_forcing

      real, dimension (mx,my,mz,mfarray),intent(INOUT) :: f
      type (pencil_case)                ,intent(INOUT) :: p

      integer :: imn

      lfirstpoint=.true.
      do imn=1,nyz

        n=nn(imn)
        m=mm(imn)
!
!  Skip points not belonging to coarse grid.
!
        lcoarse_mn=lcoarse.and.mexts(1)<=m.and.m<=mexts(2)
        if (lcoarse_mn) then
          lcoarse_mn=lcoarse_mn.and.ninds(0,m,n)>0
          if (ninds(0,m,n)<=0) cycle
        endif

        call calc_all_pencils(f,p)
        call calc_diagnostics_density(f,p)
        call calc_diagnostics_energy(f,p)
        call calc_diagnostics_hydro(f,p)
        call calc_diagnostics_magnetic(f,p)
        if (lforcing_cont) call calc_diagnostics_forcing(p)

        lfirstpoint=.false.

      enddo

    endsubroutine calc_all_module_diagnostics
!****************************************************************************
    subroutine calc_all_pencils(f,p)

      use Diagnostics, only: calc_phiavg_profile
      use BorderProfiles, only: calc_pencils_borderprofiles
      use Chiral
      use Chemistry
      use Cosmicray
      use CosmicrayFlux
      use Density
      use Dustvelocity
      use Dustdensity
      use Energy
      use EquationOfState
      use Forcing, only: calc_pencils_forcing
      use Gravity
      use Heatflux
      use Hydro
      use Lorenz_gauge
      use Magnetic
      use NeutralDensity
      use NeutralVelocity
      use Particles_main
      use Pscalar
      use PointMasses
      use Polymer
      use Radiation
      use Selfgravity
      use Shear
      use Shock, only: calc_pencils_shock, calc_shock_profile, &
                       calc_shock_profile_simple
      use Solid_Cells, only: update_solid_cells_pencil
      use Special, only: calc_pencils_special, dspecial_dt
      use Ascalar
      use Testfield
      use Testflow
      use Testscalar
      use Viscosity, only: calc_pencils_viscosity
      use omp_lib

      real, dimension (mx,my,mz,mfarray),intent(INOUT) :: f
      type (pencil_case)                ,intent(INOUT) :: p
!
!  The solid cells may have to be updated at the beginning of every
!  pencil calculation.
!
        call update_solid_cells_pencil(f)
!
!  Grid spacing. In case of equidistant grid and cartesian coordinates
!  this is calculated before the (m,n) loop.
!
        if (.not. lcartesian_coords .or. .not.all(lequidist)) call get_grid_mn
!
!  Calculate grid/geometry related pencils.
!
        call calc_pencils_grid(p)
!
!  Calculate profile for phi-averages if needed.
!
        if (((l2davgfirst.and.lwrite_phiaverages )  .or. &
             (l1dphiavg  .and.lwrite_phizaverages)) .and. &
            (lcylinder_in_a_box.or.lsphere_in_a_box)) call calc_phiavg_profile(p)
!
!  Calculate pencils for the pencil_case.
!  Note: some no-modules (e.g. nohydro) also calculate some pencils,
!  so it would be wrong to check for lhydro etc in such cases.
! DM : in the formulation of lambda effect due to Kitchanov and Olemski,
! DM : we need to have dsdr to calculate lambda. Hence, the way it is done now,
! DM : we need to have energy pencils calculated before viscosity pencils.
! DM : This is *bad* practice and must be corrected later.
! To check ghost cell consistency, please uncomment the following 2 lines:
!       if (.not. lpencil_check_at_work .and. necessary(imn)) &
!       call check_ghosts_consistency (f, 'before calc_pencils_*')
                              call calc_pencils_hydro(f,p)

 
                              call calc_pencils_density(f,p)
        if (lpscalar)         call calc_pencils_pscalar(f,p)
        if (lascalar)         call calc_pencils_ascalar(f,p)
                              call calc_pencils_eos(f,p)
        if (lshock)           call calc_pencils_shock(f,p)
        if (lchemistry)       call calc_pencils_chemistry(f,p)
                              call calc_pencils_energy(f,p)
        if (lviscosity)       call calc_pencils_viscosity(f,p)
        if (lforcing_cont)    call calc_pencils_forcing(f,p)
        if (llorenz_gauge)    call calc_pencils_lorenz_gauge(f,p)
        if (lmagnetic)        call calc_pencils_magnetic(f,p)
        if (lpolymer)         call calc_pencils_polymer(f,p)
        if (lgrav)            call calc_pencils_gravity(f,p)
        if (lselfgravity)     call calc_pencils_selfgravity(f,p)
        if (ldustvelocity)    call calc_pencils_dustvelocity(f,p)
        if (ldustdensity)     call calc_pencils_dustdensity(f,p)
        if (lneutralvelocity) call calc_pencils_neutralvelocity(f,p)
        if (lneutraldensity)  call calc_pencils_neutraldensity(f,p)
        if (lcosmicray)       call calc_pencils_cosmicray(f,p)
        if (lcosmicrayflux)   call calc_pencils_cosmicrayflux(f,p)
        if (lchiral)          call calc_pencils_chiral(f,p)
        if (lradiation)       call calc_pencils_radiation(f,p)
        if (lshear)           call calc_pencils_shear(f,p)
        if (lborder_profiles) call calc_pencils_borderprofiles(f,p)
        if (lpointmasses)     call calc_pencils_pointmasses(f,p)
        if (lparticles)       call particles_calc_pencils(f,p)
        if (lspecial)         call calc_pencils_special(f,p)
        if (lheatflux)        call calc_pencils_heatflux(f,p)

    endsubroutine calc_all_pencils
!***********************************************************************
    subroutine rhs_cpu(f,df,p,mass_per_proc,early_finalize)
!
!  Calculates rhss of the PDEs.
!
!  14-feb-17/MR: Carved out from pde.
!  21-feb-17/MR: Moved all module-specific estimators of the (inverse) possible timestep 
!                to the individual modules.
!  30-mar-23/TP: Added multithreading through OpenMP declaratives and making needed variables threadprivate
!                The main loop is also splitten into two parts in order to better overlap communication and computation
!
      use Chiral
      use Chemistry
      use Cosmicray
      use CosmicrayFlux
      use Density
      use Diagnostics
      use Dustvelocity
      use Dustdensity
      use Energy
      use EquationOfState
      use Forcing, only: calc_pencils_forcing, calc_diagnostics_forcing
      use GhostFold, only: fold_df, fold_df_3points
      use Gravity
      use Heatflux
      use Hydro
      use Lorenz_gauge
      use Magnetic
      use NeutralDensity
      use NeutralVelocity
      use Particles_main
      use Pscalar
      use PointMasses
      use Polymer
      use Radiation
      use Selfgravity
      use Shear
      use Solid_Cells
      use Shock, only: calc_pencils_shock, calc_shock_profile, &
                       calc_shock_profile_simple
      use Special, only: calc_pencils_special, dspecial_dt
      use Sub, only: sum_mn
      use Ascalar
      use Testfield
      use Testflow
      use Testscalar
      use Viscosity, only: calc_pencils_viscosity
!$    use Omp_lib

      real, dimension (mx,my,mz,mfarray),intent(INOUT) :: f
      real, dimension (mx,my,mz,mvar)   ,intent(OUT  ) :: df
      type (pencil_case)                ,intent(INOUT) :: p
      real, dimension(1)                ,intent(INOUT) :: mass_per_proc
      logical                           ,intent(IN   ) :: early_finalize

      real, dimension (nx,3) :: df_iuu_pencil
      logical :: lcommunicate
      !TP: Done because imn needs to be threadprivate and iterator can't be threadprivate in omp do
      integer :: i
!$    integer :: num_omp_ranks, omp_rank
!
!
      lfirstpoint=.true.
      lcommunicate=.not.early_finalize 

!$ call OMP_set_num_threads(10)
!$ call init_reduc_pointers()
!$ call copy_module_variables_in()
!$omp parallel firstprivate(p,df_iuu_pencil) 
!$omp master
 if (lcommunicate .and. necessary_imn<nyz) then
                call finalize_isendrcv_bdry(f)
                call boundconds_y(f)
                call boundconds_z(f)
                lcommunicate=.false.
 else
    lcommunicate=.false.
        endif
!$omp end master
!$omp do 
    do i=1,necessary_imn-1
    !omp critical
        call rhs_loop(f,df,p,mass_per_proc,early_finalize, nyz, i, df_iuu_pencil)
    !omp end critical
    enddo 
!$omp end do nowait

!TP wait for communication to be done
    do while(lcommunicate)
    enddo
!$omp do 
    do i=necessary_imn,nyz
    !omp critical
        call rhs_loop(f,df,p,mass_per_proc,early_finalize, nyz, i, df_iuu_pencil)
!omp end critical
    enddo 
!$omp end do 
!
!$ call prep_finalize_thread_diagnos()
!$omp critical
!$  if(OMP_get_thread_num() /= 0) call thread_reductions()
!$omp end critical
!$omp end parallel
!
      if (ltime_integrals.and.llast) then
        if (lhydro) call update_for_time_integrals_hydro
      endif
!
    endsubroutine rhs_cpu
!***********************************************************************
    subroutine rhs_loop(f,df,p,mass_per_proc, early_finalize,nyz,i, df_iuu_pencil)
!  30-mar-23/TP: Carved from rhs_cpu. Represents one iteration of the main loop in rhs_cpu. 
!                Done since this eases multithreading debugging and eases splitting the loop into separate parts
      use Diagnostics
      use Chiral
      use Chemistry
      use Cosmicray
      use CosmicrayFlux
      use Density
      use Diagnostics, only: initialize_diagnostic_arrays, prep_finalize_thread_diagnos
      use Dustvelocity
      use Dustdensity
      use Energy
      use EquationOfState
      use Forcing, only: calc_pencils_forcing, calc_diagnostics_forcing
      use GhostFold, only: fold_df, fold_df_3points
      use Gravity
      use Heatflux
      use Hydro
      use Lorenz_gauge
      use Magnetic
      use NeutralDensity
      use NeutralVelocity
      use Particles_main
      use Pscalar
      use PointMasses
      use Polymer
      use Radiation
      use Selfgravity
      use Shear
      use Shock, only: calc_pencils_shock, calc_shock_profile, &
                       calc_shock_profile_simple
      use Solid_Cells, only: update_solid_cells_pencil, dsolid_dt
      use Special, only: calc_pencils_special, dspecial_dt
      use Sub, only: sum_mn
      use Ascalar
      use Testfield
      use Testflow
      use Testscalar
      use Viscosity, only: calc_pencils_viscosity
!$    use omp_lib

            real, dimension (mx,my,mz,mfarray),intent(INOUT) :: f
      real, dimension (mx,my,mz,mvar)   ,intent(OUT  ) :: df
      type (pencil_case)                ,intent(INOUT) :: p
      real, dimension(1)                ,intent(INOUT) :: mass_per_proc
      logical                           ,intent(IN   ) :: early_finalize

      integer :: nyz
      integer :: i
      real, dimension (nx,3) :: df_iuu_pencil
        imn = i
        n=nn(imn)
        m=mm(imn)

        !if (imn_array(m,n)==0) cycle
!  Skip points not belonging to coarse grid.
!
        lcoarse_mn=lcoarse.and.mexts(1)<=m.and.m<=mexts(2)
        if (lcoarse_mn) then
          lcoarse_mn=lcoarse_mn.and.ninds(0,m,n)>0
          if (ninds(0,m,n)<=0) return 
        endif
!omp critical
!
!  Store the velocity part of df array in a temporary array
!  while solving the anelastic case.
!
!$      if (.false.) &
        call timing('pde','before lanelastic',mnloop=.true.)

        if (lanelastic) then
          df_iuu_pencil = df(l1:l2,m,n,iux:iuz)
          df(l1:l2,m,n,iux:iuz)=0.0
        endif
!
!        if (loptimise_ders) der_call_count=0 !DERCOUNT
!
!  Make sure all ghost points are set.
!
!$      if (.false.) &
        call timing('pde','finished boundconds_z',mnloop=.true.)
!
!  For each pencil, accumulate through the different modules
!  advec_XX and diffus_XX, which are essentially the inverse
!  advective and diffusive timestep for that module.
!  (note: advec2 is an inverse _squared_ timestep)
!  Note that advec_cs2 should always be initialized when leos.
!
        if (lfirst.and.ldt.and.(.not.ldt_paronly)) then
          advec_cs2=0.0
          maxadvec=0.
          if (lenergy.or.ldensity.or.lmagnetic.or.lradiation.or.lneutralvelocity.or.lcosmicray.or. &
              (ltestfield_z.and.iuutest>0)) &
            advec2=0.
          if (ldensity.or.lviscosity.or.lmagnetic.or.lenergy) &
            advec2_hypermesh=0.0
          maxdiffus=0.
          maxdiffus2=0.
          maxdiffus3=0.
          maxsrc=0.
        endif

        call calc_all_pencils(f,p)       
!
!  --------------------------------------------------------
!  NO CALLS MODIFYING PENCIL_CASE PENCILS BEYOND THIS POINT
!  --------------------------------------------------------
!
!  hydro, density, and entropy evolution
!  Note that pressure gradient is added in denergy_dt of noentropy to momentum,
!  even if lentropy=.false.
!
        
        call duu_dt(f,df,p)
        call dlnrho_dt(f,df,p)
        call denergy_dt(f,df,p)
!
!  Magnetic field evolution
!
        if (lmagnetic) call daa_dt(f,df,p)
!
!  Lorenz gauge evolution
!
        if (llorenz_gauge) call dlorenz_gauge_dt(f,df,p)
!
!  Polymer evolution
!
        if (lpolymer) call dpoly_dt(f,df,p)
!
!  Testscalar evolution
!
        if (ltestscalar) call dcctest_dt(f,df,p)
!
!  Testfield evolution
!
        if (ltestfield) call daatest_dt(f,df,p)
!
!  Testflow evolution
!
        if (ltestflow) call duutest_dt(f,df,p)
!
!  Passive scalar evolution
!
        if (lpscalar) call dlncc_dt(f,df,p)
!
!  Supersaturation evolution
        
        if (lascalar) call dacc_dt(f,df,p)
!
!  Dust evolution
!
        if (ldustvelocity) call duud_dt(f,df,p)
        if (ldustdensity) call dndmd_dt(f,df,p)
!
!  Neutral evolution
!
        if (lneutraldensity) call dlnrhon_dt(f,df,p)
        if (lneutralvelocity) call duun_dt(f,df,p)
!
!  Add gravity, if present
!
        if (lgrav) then
          if (lhydro.or.ldustvelocity.or.lneutralvelocity) &
               call duu_dt_grav(f,df,p)
        endif
!
!  Self-gravity
!
        if (lselfgravity) call duu_dt_selfgrav(f,df,p)
!
!  Cosmic ray energy density
!
        if (lcosmicray) call decr_dt(f,df,p)
!
!  Cosmic ray flux
!
        if (lcosmicrayflux) call dfcr_dt(f,df,p)
!
!  Chirality of left and right handed aminoacids
!
        if (lchiral) call dXY_chiral_dt(f,df,p)
!
!  Evolution of radiative energy
!
        if (lradiation_fld) call de_dt(f,df,p)
!
!  Evolution of chemical species
!
        if (lchemistry) call dchemistry_dt(f,df,p)
!
!  Evolution of heatflux vector
!
        if (lheatflux) call dheatflux_dt(f,df,p)
!
!  Continuous forcing diagonstics.
!
        if (lforcing_cont) call calc_diagnostics_forcing(p)
!
!  Add and extra 'special' physics
        if (lspecial) call dspecial_dt(f,df,p)
        
!
!  Add radiative cooling and radiative pressure (for ray method)
!
        if (lradiation_ray.and.lenergy) then
          call radiative_cooling(f,df,p)
          call radiative_pressure(f,df,p)
        endif
!
!  Find diagnostics related to solid cells (e.g. drag and lift).
!  Integrating to the full result is done after loops over m and n.
!
        if (lsolid_cells) call dsolid_dt(f,df,p)
!
!  Add shear if present
!
        if (lshear) call shearing(f,df,p)
!
        if (lparticles) call particles_pde_pencil(f,df,p)
!
        if (lpointmasses) call pointmasses_pde_pencil(f,df,p)
        
!if(ldiagnos .and. fname(12) > 0.) print*,fname(12)
!  Call diagnostics that involves the full right hand side
!  This must be done at the end of all calls that might modify df.
!
        if (ldiagnos) then
          if (lhydro) call df_diagnos_hydro(df,p)
          if (lmagnetic) call df_diagnos_magnetic(df,p)
        endif
!
!  General phiaverage quantities -- useful for debugging.
!  MR: Results are constant in time, so why here?
!
        if (l2davgfirst) then
          call phisum_mn_name_rz(p%rcyl_mn,idiag_rcylmphi)
          call phisum_mn_name_rz(p%phi_mn,idiag_phimphi)
          call phisum_mn_name_rz(p%z_mn,idiag_zmphi)
          call phisum_mn_name_rz(p%r_mn,idiag_rmphi)
        endif
!
!  Do the time integrations here, before the pencils are overwritten.
!
        if (ltime_integrals.and.llast) then
          if (lhydro)    call time_integrals_hydro(f,p)
          if (lmagnetic) call time_integrals_magnetic(f,p)
        endif

        call set_dt1_max(p)
!
!  Diagnostics showing how close to advective and diffusive time steps we are
!
        if (lfirst.and.ldt.and.(.not.ldt_paronly).and.ldiagnos) then
          if (idiag_dtv/=0) call max_mn_name(maxadvec/cdt,idiag_dtv,l_dt=.true.)
          if (idiag_dtdiffus/=0) call max_mn_name(maxdiffus/cdtv,idiag_dtdiffus,l_dt=.true.)
          if (idiag_dtdiffus2/=0) call max_mn_name(maxdiffus2/cdtv2,idiag_dtdiffus2,l_dt=.true.)
          if (idiag_dtdiffus3/=0) call max_mn_name(maxdiffus3/cdtv3,idiag_dtdiffus3,l_dt=.true.)
!
!  Regular and hyperdiffusive mesh Reynolds numbers
!
          if (idiag_Rmesh/=0) call max_mn_name(pi_1*maxadvec/(maxdiffus+tini),idiag_Rmesh)
          if (idiag_Rmesh3/=0) call max_mn_name(pi5_1*maxadvec/(maxdiffus3+tini),idiag_Rmesh3)
          call max_mn_name(maxadvec,idiag_maxadvec)
        endif
         
!
!  Display derivative info
!
!debug   if (loptimise_ders.and.lout) then                         !DERCOUNT
!debug     do iv=1,nvar                                            !DERCOUNT
!debug     do ider=1,8                                             !DERCOUNT
!debug     do j=1,3                                                !DERCOUNT
!debug     do k=1,3                                                !DERCOUNT
!debug       if (der_call_count(iv,ider,j,k) > 1) then             !DERCOUNT
!debug         print*,'DERCOUNT: '//varname(iv)//' derivative ', & !DERCOUNT
!debug                                                 ider,j,k, & !DERCOUNT
!debug                                               ' called ', & !DERCOUNT
!debug                              der_call_count(iv,ider,j,k), & !DERCOUNT
!debug                                                  'times!'   !DERCOUNT
!debug       endif                                                 !DERCOUNT
!debug     enddo                                                   !DERCOUNT
!debug     enddo                                                   !DERCOUNT
!debug     enddo                                                   !DERCOUNT
!debug     enddo                                                   !DERCOUNT
!debug     if (maxval(der_call_count)>1) call fatal_error( &        !DERCOUNT
!debug      'pde','ONE OR MORE DERIVATIVES HAS BEEN DOUBLE CALLED') !DERCOUNT
!debug   endif
!
!  Fill in the rhs of the poisson equation and restore the df(:,:,:,iuu) array
!  for anelastic case
!
        if (lanelastic) then
!          call calc_pencils_density(f,p)
          f(l1:l2,m,n,irhs)   = p%rho*df(l1:l2,m,n,iuu)
          f(l1:l2,m,n,irhs+1) = p%rho*df(l1:l2,m,n,iuu+1)
          f(l1:l2,m,n,irhs+2) = p%rho*df(l1:l2,m,n,iuu+2)
          df(l1:l2,m,n,iux:iuz) = df_iuu_pencil + df(l1:l2,m,n,iux:iuz)
          call sum_mn(p%rho,mass_per_proc(1))
        endif
!$      if (.false.) &
        call timing('pde','end of mn loop',mnloop=.true.)
!
!  End of loops over m and n.
!
        headtt=.false.
        lfirstpoint=.false.
!omp end critical
 
    endsubroutine rhs_loop
!***********************************************************************
    subroutine debug_imn_arrays
!
!  For debug purposes: writes out the mm, nn, and necessary arrays.
!
!  23-nov-02/axel: coded
!
      open(1,file=trim(directory)//'/imn_arrays.dat')
      do imn=1,nyz
        if (necessary(imn)) write(1,'(a)') '----necessary=.true.----'
        write(1,'(4i6)') imn,mm(imn),nn(imn)
      enddo
      close(1)
!
    endsubroutine debug_imn_arrays
!***********************************************************************
    subroutine output_crash_files(f)
!
!  Write crash snapshots when time-step is low.
!
!  15-aug-2007/anders: coded
!
      use Snapshot
!
      real, dimension(mx,my,mz,mfarray) :: f
!
      integer, save :: icrash=0
      character (len=10) :: filename
      character (len=1) :: icrash_string
!
      if ( (it>1) .and. lfirst .and. (dt<=crash_file_dtmin_factor*dtmin) ) then
        write(icrash_string, fmt='(i1)') icrash
        filename='crash'//icrash_string//'.dat'
        call wsnap(filename,f,mvar_io,ENUM=.false.)
        if (lroot) then
          print*, 'Time-step is very low - writing '//trim(filename)
          print*, '(it, itsub=', it, itsub, ')'
          print*, '(t, dt=', t, dt, ')'
        endif
!
!  Next crash index, cycling from 0-9 to avoid excessive writing of
!  snapshots to the hard disc.
!
        icrash=icrash+1
        icrash=mod(icrash,10)
      endif
!
    endsubroutine output_crash_files
!***********************************************************************
    subroutine set_dyndiff_coeff(f)
!
!  Set dynamical diffusion coefficients.
!
!  18-jul-14/ccyang: coded.
!  03-apr-16/ccyang: add switch ldyndiff_useumax
!
      use Density,   only: dynamical_diffusion
      use Energy,    only: dynamical_thermal_diffusion
      use Magnetic,  only: dynamical_resistivity
      use Sub,       only: find_max_fvec, find_rms_fvec
      use Viscosity, only: dynamical_viscosity
!
      real, dimension(mx,my,mz,mfarray), intent(in) :: f
!
      real :: uc
!
!  Find the characteristic speed, either the absolute maximum or the rms.
!
      if (ldyndiff_useumax) then
        uc = find_max_fvec(f, iuu)
      else
        uc = find_rms_fvec(f, iuu)
      endif
!
!  Ask each module to set the diffusion coefficients.
!
      if (ldensity)                      call dynamical_diffusion(uc)
      if (lmagnetic .and. .not. lbfield) call dynamical_resistivity(uc)
      if (lenergy)                       call dynamical_thermal_diffusion(uc)
      if (lviscosity)                    call dynamical_viscosity(uc)
!
    endsubroutine set_dyndiff_coeff
!***********************************************************************
    subroutine impose_floors_ceilings(f)
!
!  Impose floors or ceilings for implemented fields.
!
!  20-oct-14/ccyang: modularized from pde.
!
      use Cosmicray, only: impose_ecr_floor
      use Density, only: impose_density_floor,impose_density_ceiling
      use Dustdensity, only: impose_dustdensity_floor
      use Energy, only: impose_energy_floor
      use Hydro, only: impose_velocity_ceiling
!
      real, dimension(mx,my,mz,mfarray), intent(inout) :: f
!
      call impose_density_floor(f)
      call impose_density_ceiling(f)
      call impose_velocity_ceiling(f)
      call impose_energy_floor(f)
      call impose_dustdensity_floor(f)
      call impose_ecr_floor(f)
!
    endsubroutine impose_floors_ceilings
!***********************************************************************
    subroutine freeze(df,p)
!
      use Sub, only: quintic_step
      use Solid_Cells, only: freeze_solid_cells
!$    use omp_lib

      real, dimension(mx,my,mz,mvar), intent(inout) :: df
      type (pencil_case) :: p

      logical, dimension(npencils) :: lpenc_loc
      real, dimension(nx) :: pfreeze
      integer :: imn,iv,i
!
      !if (lgpu) call freeze_gpu

      lpenc_loc=.false. 
      if (lcylinder_in_a_box.or.lcylindrical_coords) then 
        lpenc_loc(i_rcyl_mn)=.true.
      else
        lpenc_loc(i_r_mn)=.true.
      endif
!
      headtt = headt .and. lfirst .and. lroot
!
!$omp parallel firstprivate(p,pfreeze,iv)
!$omp do 
!
      do i=1,nyz
        imn = i
        n=nn(imn)
        m=mm(imn)
        !if (imn_array(m,n)==0) cycle
!
!  Recalculate grid/geometry related pencils. The r_mn and rcyl_mn are requested
!  in pencil_criteria_grid. Unfortunately we need to recalculate them here.
!
        if (any(lfreeze_varext).or.any(lfreeze_varint)) then

          call calc_pencils_grid(p,lpenc_loc)
!
!  Set df=0 for r_mn<rfreeze_int.
!
          if (any(lfreeze_varint)) then
            if (headtt) print*, 'pde: freezing variables for r < ', rfreeze_int, &
                                ' : ', lfreeze_varint
            if (lcylinder_in_a_box.or.lcylindrical_coords) then
              if (wfreeze_int==0.0) then
                where (p%rcyl_mn<=rfreeze_int)
                  pfreeze=0.0
                elsewhere  
                  pfreeze=1.0
                endwhere  
              else
                pfreeze=quintic_step(p%rcyl_mn,rfreeze_int,wfreeze_int,SHIFT=fshift_int)
              endif
            else
              if (wfreeze_int==0.0) then
                where (p%r_mn<=rfreeze_int)
                  pfreeze=0.0
                elsewhere 
                  pfreeze=1.0
                endwhere  
              else
                pfreeze=quintic_step(p%r_mn,rfreeze_int,wfreeze_int,SHIFT=fshift_int)
              endif
            endif
!
            do iv=1,nvar
              if (lfreeze_varint(iv)) df(l1:l2,m,n,iv)=pfreeze*df(l1:l2,m,n,iv)
            enddo
!
          endif
!
!  Set df=0 for r_mn>r_ext.
!
          if (any(lfreeze_varext)) then
            if (headtt) print*, 'pde: freezing variables for r > ', rfreeze_ext, &
                                ' : ', lfreeze_varext
            if (lcylinder_in_a_box.or.lcylindrical_coords) then
              if (wfreeze_ext==0.0) then
                where (p%rcyl_mn>=rfreeze_ext)
                  pfreeze=0.0
                elsewhere  
                  pfreeze=1.0
                endwhere  
              else
                pfreeze=1.0-quintic_step(p%rcyl_mn,rfreeze_ext,wfreeze_ext,SHIFT=fshift_ext)
              endif
            else
              if (wfreeze_ext==0.0) then
                where (p%r_mn>=rfreeze_ext)
                  pfreeze=0.0
                elsewhere 
                  pfreeze=1.0
                endwhere  
              else
                pfreeze=1.0-quintic_step(p%r_mn,rfreeze_ext,wfreeze_ext,SHIFT=fshift_ext)
              endif
            endif
!
            do iv=1,nvar
              if (lfreeze_varext(iv)) df(l1:l2,m,n,iv) = pfreeze*df(l1:l2,m,n,iv)
            enddo
          endif
        endif
!
!  Set df=0 inside square.
!
        if (any(lfreeze_varsquare)) then
          if (headtt) print*, 'pde: freezing variables inside square : ',lfreeze_varsquare
          pfreeze=1.0-quintic_step(x(l1:l2),xfreeze_square,wfreeze,SHIFT=-1.0)*&
                      quintic_step(spread(y(m),1,nx),yfreeze_square,-wfreeze,SHIFT=-1.0)
!
          do iv=1,nvar
            if (lfreeze_varsquare(iv)) df(l1:l2,m,n,iv) = pfreeze*df(l1:l2,m,n,iv)
          enddo
        endif
!
!  Freeze components of variables in boundary slice if specified by boundary
!  condition 'f'
!
!  Freezing boundary conditions in x.
!
        if (lfrozen_bcs_x) then ! are there any frozen vars at all?
!
!  Only need to do this on left/right-most
!  processor and in left/right--most pencils
!
          if (lfirst_proc_x) where (lfrozen_bot_var_x(1:nvar)) df(l1,m,n,1:nvar) = 0.
          if (llast_proc_x) where (lfrozen_top_var_x(1:nvar)) df(l2,m,n,1:nvar) = 0.
!
        endif
!
!  Freezing boundary conditions in y.
!
        if (lfrozen_bcs_y) then ! are there any frozen vars at all?
!
!  Only need to do this on bottom/top-most
!  processor and in bottom/top-most pencils.
!
          if (lfirst_proc_y .and. (m == m1)) then
            do iv=1,nvar
              if (lfrozen_bot_var_y(iv)) df(l1:l2,m,n,iv) = 0.
            enddo
          endif
          if (llast_proc_y .and. (m == m2)) then
            do iv=1,nvar
              if (lfrozen_top_var_y(iv)) df(l1:l2,m,n,iv) = 0.
            enddo
          endif
        endif
!
!  Freezing boundary conditions in z.
!
        if (lfrozen_bcs_z) then ! are there any frozen vars at all?
!
!  Only need to do this on bottom/top-most
!  processor and in bottom/top-most pencils.
!
          if (lfirst_proc_z .and. (n == n1)) then
            do iv=1,nvar
              if (lfrozen_bot_var_z(iv)) df(l1:l2,m,n,iv) = 0.
            enddo
          endif
          if (llast_proc_z .and. (n == n2)) then
            do iv=1,nvar
              if (lfrozen_top_var_z(iv)) df(l1:l2,m,n,iv) = 0.
            enddo
          endif
        endif
!
!  Set df=0 for all solid cells.
!
        call freeze_solid_cells(df)
!
        headtt=.false.
!
      enddo
!
!$omp end do
!$omp end parallel

    endsubroutine freeze
!***********************************************************************
    subroutine set_dt1_max(p)
!
!  In max_mn maximum values of u^2 (etc) are determined sucessively
!  va2 is set in magnetic (or nomagnetic)
!  In rms_mn sum of all u^2 (etc) is accumulated
!  Calculate maximum advection speed for timestep; needs to be done at
!  the first substep of each time step
!  Note that we are (currently) accumulating the maximum value,
!  not the maximum squared!
!
!  The dimension of the run ndim (=0, 1, 2, or 3) enters the viscous time step.
!  This has to do with the term on the diagonal, cdtv depends on order of scheme
!
      use General, only: notanumber

      type (pencil_case), intent(IN) :: p

      real, dimension(nx) :: dt1_max_loc, dt1_advec, dt1_diffus, dt1_src
      real :: dt1_preac

      if (lfirst.and.ldt.and.(.not.ldt_paronly)) then
!
!  sum or maximum of the advection terms?
!  (lmaxadvec_sum=.false. by default)
!
        advec2=advec2+advec_cs2
        if (lenergy.or.ldensity.or.lmagnetic.or.lradiation.or.lneutralvelocity.or.lcosmicray.or. &
            (ltestfield_z.and.iuutest>0)) maxadvec=maxadvec+sqrt(advec2)

        if (ldensity.or.lhydro.or.lmagnetic.or.lenergy) maxadvec=maxadvec+sqrt(advec2_hypermesh)
!
!  Time step constraints from each module. (At the moment, magnetic and testfield use the same variable.)
!  cdt, cdtv, and cdtc are empirical non-dimensional coefficients.
!
!  Timestep constraint from advective terms.
!
        dt1_advec = maxadvec/cdt
!
!  Timestep constraint from diffusive terms.
!
        dt1_diffus = maxdiffus/cdtv + maxdiffus2/cdtv2 + maxdiffus3/cdtv3
!
!  Timestep constraint from source terms.
!
        dt1_src = maxsrc/cdtsrc
!
!  Timestep combination from advection, diffusion and "source". 
!
        dt1_max_loc = sqrt(dt1_advec**2 + dt1_diffus**2 + dt1_src**2)
! write(88,*)  'imn,threadnum,dt1_max=',imn,omp_rank,dt1_max_loc,dt1_advec + dt1_diffus + dt1_src
!
!  time step constraint from the coagulation kernel
!
        if (ldustdensity) dt1_max_loc = max(dt1_max_loc,reac_dust/cdtc)
!
!  time step constraint from speed of chemical reactions
!
        if (lchemistry .and. .not.llsode) then
!           dt1_preac= reac_pchem/cdtc
          dt1_max_loc = max(dt1_max_loc,reac_chem/cdtc)
        endif
!
!  time step constraint from relaxation time of polymer
!
        if (lpolymer) dt1_max_loc = max(dt1_max_loc,1./(trelax_poly*cdt_poly))
!
!  Exclude the frozen zones from the time-step calculation.
!
        if (any(lfreeze_varint)) then
          if (lcylinder_in_a_box.or.lcylindrical_coords) then
            where (p%rcyl_mn<=rfreeze_int) 
              dt1_max_loc=0.; maxadvec=0.; maxdiffus=0.; maxdiffus2=0.; maxdiffus3=0.
            endwhere
          else
            where (p%r_mn<=rfreeze_int) 
              dt1_max_loc=0.; maxadvec=0.; maxdiffus=0.; maxdiffus2=0.; maxdiffus3=0.
            endwhere
          endif
        endif
!
        if (any(lfreeze_varext)) then
          if (lcylinder_in_a_box.or.lcylindrical_coords) then
            where (p%rcyl_mn>=rfreeze_ext)
              dt1_max_loc=0.; maxadvec=0.; maxdiffus=0.; maxdiffus2=0.; maxdiffus3=0.
            endwhere
          else
            where (p%r_mn>=rfreeze_ext)
              dt1_max_loc=0.; maxadvec=0.; maxdiffus=0.; maxdiffus2=0.; maxdiffus3=0.
            endwhere
          endif
        endif
!  MR: the next correct? freezes *outside* square
        if (any(lfreeze_varsquare).and.y(m)>yfreeze_square) then
          where (x(l1:l2)>xfreeze_square)
            dt1_max_loc=0.; maxadvec=0.; maxdiffus=0.; maxdiffus2=0.; maxdiffus3=0.
          endwhere
        endif

        dt1_max=max(dt1_max,dt1_max_loc)
!
!  Check for NaNs in the advection time-step.
!
        if (notanumber(maxadvec)) then
          print*, 'pde: maxadvec contains a NaN at iproc=', iproc_world
          if (lenergy) print*, 'advec_cs2  =',advec_cs2
          call fatal_error_local('pde','')
        endif
      endif

    endsubroutine set_dt1_max
!***********************************************************************
    subroutine init_reduc_pointers()

!  30-mar-23/TP: Function for initialization of pointers used in thread_reductions. Called in rhs_cpu 
      use Solid_Cells
      use omp_lib

      p_fname => fname
      p_fname_keep => fname_keep
      p_fnamer => fnamer
      p_fname_sound => fname_sound
      p_fnamex => fnamex
      p_fnamey => fnamey
      p_fnamez => fnamez
      p_fnamexy => fnamexy
      p_fnamexz => fnamexz
      p_fnamerz => fnamerz
      p_dt1_max => dt1_max
      p_ncountsz => ncountsz
      if(lsolid_cells) call solid_cells_init_reduc_pointers()
    endsubroutine init_reduc_pointers
!***********************************************************************
    subroutine thread_reductions
  
!  30-mar-23/TP: Function for reducing accumulated variables inside rhs_cpu loop. Needed for multithreading since the threads compute them separately.
    use Solid_Cells
    use omp_lib

    integer :: imn
     
      if(ldiagnos .and. allocated(fname)) then
          do imn=1,size(fname)
          if(any(inds_max_diags == imn) .and. fname(imn) /= 0.) then
              p_fname(imn) = max(p_fname(imn),fname(imn))
          else if(any(inds_sum_diags == imn)) then
              p_fname(imn) = p_fname(imn) + fname(imn)
          endif
          enddo
        endif
     if(l1davgfirst) then
          if(allocated(fnamex)) p_fnamex = p_fnamex + fnamex
          if(allocated(fnamey)) p_fnamey = p_fnamey + fnamey
          if(allocated(fnamez)) p_fnamez = p_fnamez + fnamez
          if(allocated(fnamer)) p_fnamer = p_fnamer + fnamer
      endif
      if(l2davgfirst) then
          if(allocated(fnamexy)) p_fnamexy = p_fnamexy + fnamexy
          if(allocated(fnamexz)) p_fnamexz = p_fnamexz + fnamexz
          if(allocated(fnamerz)) p_fnamerz = p_fnamerz + fnamerz
      endif
      ! fname_keep ??
      if(allocated(fname_keep)) p_fname_keep = p_fname_keep + fname_keep
      if(allocated(fname_sound)) p_fname_sound = p_fname_sound + fname_sound
      if(allocated(ncountsz)) p_ncountsz = p_ncountsz + ncountsz
      p_dt1_max = max(p_dt1_max,dt1_max)
      if(lsolid_cells) call solid_cells_thread_reductions
    endsubroutine thread_reductions
!***********************************************************************
    subroutine copy_module_variables_in()

!  30-mar-23/TP: Function needed for intelligent way to copyin threadprivate variables that are needed in rhs_cpu.
!                The current version is copying in way more than needed, this is really for not having to worry about while debugging TODO: educe the number of variables
    use Diagnostics
      use Chiral
      use Chemistry
      use Cosmicray
      use CosmicrayFlux
      use Density
      use Diagnostics
      use Dustvelocity
      use Deriv
      use Dustdensity
      use Energy
      use EquationOfState
      use Forcing, only: calc_pencils_forcing, calc_diagnostics_forcing
      use GhostFold, only: fold_df, fold_df_3points
      use Gravity
      use Heatflux
      use Hydro
      use Lorenz_gauge
      use Magnetic
      use NeutralDensity
      use NeutralVelocity
      use Particles_main
      !use Particles_cdata
      use Pscalar
      use PointMasses
      use Polymer
      use Radiation
      use Selfgravity
      use Shear
      use Shock, only: calc_pencils_shock, calc_shock_profile, &
                       calc_shock_profile_simple
      use Solid_Cells
      use Special, only: calc_pencils_special, dspecial_dt
      use Sub, only: sum_mn
      use Ascalar
      use Testfield
      use Testflow
      use Testscalar
      use Viscosity, only: calc_pencils_viscosity
!$    use omp_lib

    if(lentropy) call entropy_copy_in()
    ! if(lparticles) then
    !   !$omp parallel copyin(kshepherd,kneighbour,ipar)
    !   !$omp end parallel
    ! endif
    if(lsolid_cells) call copyin_solid_cells()
    if(ldustvelocity) call dust_velocity_copy_in()
    if(lhydro) call copyin_hydro()
    call eos_copy_in()
    !$omp parallel copyin(imn,lcoarse_mn,advec_cs2,maxadvec,advec2,advec2_hypermesh,lfirstpoint,fname,fnamer,fname_sound,&
    !$omp fnamex,fnamey,fnamez,fnamexz,fnamerz,headtt,fname_half,fname_keep,fnamexy,dt1_max,inds_sum_diags,inds_max_diags, &
    !$omp maxdiffus,maxdiffus2,maxdiffus3,maxsrc,costh,nvar,naux,naux_com,nglobal,iux,iu0x,iaxtest, iaztest, reac_dust,&
    !$omp dxyz_2,fweight,dxyz_4,dxyz_6,dVol,dxmax_pencil,dxmin_pencil,drcyl,dline_1,lequidist,x0,y0,z0,Lx,Ly,Lz,alpha_ts,&
    !$omp beta_ts,lchemonly,lsnap_down,lwrite_sound,lslope_limit_diff,leos_idealgas,ilnrho,irho,mm,nn,necessary,&
    !$omp imn_array,lpencil_check_at_work,nnamerz,ncoords_sound,tdiagnos,t1ddiagnos,t2davgfirst,sound_coords_list,ncountsz,&
    !$omp cform_sound,cnamerz,ldiagnos,l2davgfirst,l1davgfirst,l1dphiavg,errormsg,&
    !$omp mailaddress,aux_var,aux_count,lisotropic_advection,&
    !$omp lgravz,lgravr,seed,iuud,iudx,iudy,iudz,ind,imd,idc,idcj,&
    !$omp bcx,bcy,bcz,bcx12,bcy12,bcz12,x,dx_1,dx_tilde,xprim,y,z,dx,dy,r_mn,r1_mn,&
    !$omp r2_mn,sinth,sin1th,sin2th,cotth,sinph,cosph,cos1th,tanth,rcyl_mn,rcyl_mn1,&
    !$omp rcyl_mn2,xyz_star,nt,ldownsampling,ip,lgravr_gas,&
    !$omp lgravr_neutrals,lgravr_dust,iuxt,iuyt,iuzt,ivisc_forcx,ivisc_forcy,&
    !$omp ivisc_forcz,i_adv_derx,i_adv_dery,iuu_flucx,iuu_sphr,headt,varname,&
    !$omp lfirst,ltime_integrals,deltay,iapn,test_nonblocking,trelax_poly,ip11,ip21,ip31,&
    !$omp iss, iss_run_aver)
    !here removed one's varname
    !copyin(maxdiffus,maxdiffus2,maxdiffus3,maxsrc) 
    !copyin(der2_coef0,der2_coef2)
    !omp copyin(dt1_max,imn,lcoarse_mn,advec_cs2,maxadvec,advec2,advec2_hypermesh,maxdiffus,maxdiffus2,maxdiffus3,maxsrc,dt1_max,lfirstpoint,llastpoint,iaztest, reac_dust, dxyz_2)
    !omp parallel copyin(llastpoint,lfirstpoint,fname,fnamer,fname_sound,fnamex,fnamey,fnamez,fnamexz,fnamerz,headtt,fname_half,fname_keep,fnamexy,dt1_max,inds_sum_diags,inds_max_diags)
    !$omp end parallel
    endsubroutine copy_module_variables_in
!***********************************************************************
endmodule Equ
