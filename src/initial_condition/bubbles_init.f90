! $Id$
!
! Initial condition for bubbles in a stratified medium.
! Set bubble size, bubble density, surrounding medium density,
! temperatures, stratification and magnetic field.
!
!** AUTOMATIC CPARAM.INC GENERATION ****************************
! Declare (for generation of cparam.inc) the number of f array
! variables and auxiliary variables added by this module
!
! CPARAM logical, parameter :: linitial_condition = .true.
!
!***************************************************************

module InitialCondition
!
  use Cparam
  use Cdata
  use General, only: keep_compiler_quiet
  use Messages

!
  implicit none
!
  include '../initial_condition.h'
!
  ! r_b = radius of the bubble
  ! [xyz]_b = position of the bubble
  ! rho_b = density of the bubble
  ! T_b = temperature of the bubble
  ! rho_m_0 = density of the medium at z = 0
  ! z_0 > 0 (see comment below)
  ! T_0 = temperature of the medium at z = 0
  ! profile = stratification profile, either 'polytropic' (including adiabatic) or 'isothermal'
  ! lam = exponent for tanh smoothing profile for the density and temperature
  ! passive_scalar = determines if a passive scalar at the bubble should be set
  ! n_smooth = exponent of the smoothing function for the vector potential
  ! b_field = shape of the magnetic field ('abc' or 'spheromak')
  ! tau = parapeter tau for the spheromak field
  ! k_aa = wave vector of the initial magnetic vector potential in terms of the bubble radius
  ! ampl = amplitude of the initial magnetic vector potential
  
  real :: r_b = 1.0
  real :: x_b = 0, y_b = 0, z_b = 0
  real :: rho_b = 0.1, T_b = 5.0
  real :: rho_m_0 = 1.0, T_0 = 1.0
  real :: z_0 = 1.0  ! Note that R/mu = 0.4 in some cases. Consider this with z_0 = R/(mu g) T_0.
  character (len=labellen) :: profile = 'isothermal'
  real :: lam = 20
  integer :: passive_scalar = 0, n_smooth = 2
  character (len=labellen) :: b_field = 'abc'
  real :: tau
  real :: k_aa = 1., ampl = 1., asym_factor = 1., sigma_b = 1.
  
  namelist /initial_condition_pars/ &
      r_b, x_b, y_b, z_b, rho_b, T_b, rho_m_0, T_0, z_0, lam, profile, passive_scalar, b_field, tau, k_aa, &
      ampl, asym_factor, sigma_b, n_smooth
!
  contains
!***********************************************************************
    subroutine register_initial_condition()
!
!  Register variables associated with this module; likely none.
!
!  20-may-11/simon: coded
!
      if (lroot) call svn_id( &
           "$Id$")
!
    endsubroutine register_initial_condition
!***********************************************************************
    subroutine initialize_initial_condition(f)
!
!  Initialize any module variables which are parameter dependent.
!
!  07-may-09/wlad: coded
!
      real, dimension (mx,my,mz,mfarray) :: f
!
      call keep_compiler_quiet(f)
!
    endsubroutine initialize_initial_condition
!***********************************************************************
    subroutine initial_condition_uu(f)
!
!  Initialize the velocity field.
!
!  07-may-09/wlad: coded
!
      real, dimension (mx,my,mz,mfarray), intent(inout) :: f
!
!  SAMPLE IMPLEMENTATION
!
      call keep_compiler_quiet(f)
!
    endsubroutine initial_condition_uu
!***********************************************************************
    subroutine initial_condition_lnrho(f)
!
!  Initialize logarithmic density. init_lnrho 
!  will take care of converting it to linear 
!  density if you use ldensity_nolog
!
!  07-sep-11/simon: coded
!
      use SharedVariables
      use EquationOfState
      
      real, dimension (mx,my,mz,mfarray), intent(inout) :: f
      integer :: l, m, n
      real :: log_rho_b, log_rho_m_0
      real :: r, z_tilde, log_rho_ex  ! working variables for smoothing
!
    ! initialize the density of the medium and the bubble
    log_rho_b = log(rho_b)
    log_rho_m_0 = log(rho_m_0)

    do n = n1, n2, 1
      do m = m1, m2, 1
        do l = l1, l2, 1
          ! check if this point lies in the bubble
          if (((x(l) - x_b)**2 + (y(m) - y_b)**2 + (z(n) - z_b)**2) .le. r_b**2) then
            ! reduce the shocks at the start of the simulation by reducing the
            ! density gradient
             r = sqrt((x(l) - x_b)**2 + (y(m) - y_b)**2 + (z(n) - z_b)**2)
             z_tilde = z_b + (z(n)-z_b)*r_b/r

             if (profile == 'isothermal') then
                 log_rho_ex = log_rho_m_0 - z(n)/z_0
                 f(l,m,n,ilnrho) = log_rho_b*(1-tanh((r-r_b+3*r_b/lam)/r_b*lam))/2 + &
                                   log_rho_ex*(1+tanh((r-r_b+3*r_b/lam)/r_b*lam))/2
             endif

             if (profile == 'polytropic') then
                 log_rho_ex = log_rho_m_0 + log(1-(gamma-1)/gamma*z_tilde/z_0) / (gamma-1)
                 f(l,m,n,ilnrho) = log_rho_b*(1-tanh((r-r_b+3*r_b/lam)/r_b*lam))/2 + &
                                   log_rho_ex*(1+tanh((r-r_b+3*r_b/lam)/r_b*lam))/2
             endif
           
!            f(l,m,n,ilnrho) = log_rho_b
          else
             if (profile == 'isothermal') then
                 f(l,m,n,ilnrho) = log_rho_m_0 - z(n)/z_0
             endif
             if (profile == 'polytropic') then
                 f(l,m,n,ilnrho) = log_rho_m_0 + log(1-(gamma-1)/gamma*z(n)/z_0) / (gamma-1)
             endif
          endif
        enddo
      enddo
    enddo    

    endsubroutine initial_condition_lnrho
!********************************************************************
    subroutine initial_condition_aa(f)
!
!  Initialize entropy.
!
!  07-sep-11/simon: coded
!
!       use SharedVariables
!       use EquationOfState
      use Poisson
      use Sub

      real, dimension (mx,my,mz,mfarray) :: f      
      integer :: l, j, ju
      real :: log_T_b, log_T_0
      real :: j0, j1, R, R2d, theta, gg, gg_p, BB_r, BB_theta, BB_phi ! auxiliary variables
      real, dimension (nx,ny,nz,3) :: jj, tmpJ  ! This is phi for poisson.f90
!
    ! initialize the magnetic field inside the bubble
    if (b_field == 'abc') then
      do n = n1, n2, 1
        do m = m1, m2, 1
          do l = l1, l2, 1
            ! check if this point lies in the bubble
            if (((x(l) - x_b)**2 + (y(m) - y_b)**2 + (z(n) - z_b)**2) .le. r_b**2) then
              f(l,m,n,iax) = ampl * (cos((y(m)-y_b)*k_aa) + sin((z(n)-z_b)*k_aa))
              f(l,m,n,iay) = ampl * (cos((z(n)-z_b)*k_aa) + sin((x(l)-x_b)*k_aa))
              f(l,m,n,iaz) = ampl * (cos((x(l)-x_b)*k_aa) + sin((y(m)-y_b)*k_aa))
              f(l,m,n,iax:iaz) = f(l,m,n,iax:iaz) * &
                                 (1-(sqrt((x(l)-x_b)**2+(y(m)-y_b)**2+(z(n)-z_b)**2)/r_b)**n_smooth)
            endif
          enddo
        enddo
      enddo
    endif
!
    if (b_field == 'spheromak') then
      tau = tau
      do n = n1, n2, 1
        do m = m1, m2, 1
          do l = l1, l2, 1
            R = sqrt((x(l) - x_b)**2 + (y(m) - y_b)**2 + (z(n) - z_b)**2)
            R2d = sqrt((x(l) - x_b)**2 + (y(m) - y_b)**2)
            theta = atan2(R2d, (z(n)-z_b))

            if (R .le. r_b) then
              ! Define the primary fields g(\alpha r) and g'(\alpha, r)
              gg = (R/r_b)**2 - 3/(tau*sin(tau))*(sin(R/r_b*tau)/(R/r_b*tau) - cos(R/r_b*tau))
              gg_p = 2*R/r_b/tau - 3/(tau*sin(tau))*(-sin(R/r_b*tau)/(R/r_b*tau)**2 + &
                     cos(R/r_b*tau)/(R/r_b*tau) + sin(R/r_b*tau))

              ! Construct the magnetic field.
              BB_r = 2*ampl*gg/(R/r_b*tau)**2*cos(theta)
              BB_theta = -ampl*gg_p/(R/r_b*tau)*sin(theta)
              BB_phi = ampl*gg/(R/r_b*tau)*sin(theta)
              f(l,m,n,iax) = BB_r*(x(l)-x_b)/R + BB_theta*(x(l)-x_b)*(z(n)-z_b)/(R2d*R) - BB_phi*(y(m)-y_b)/R2d
              f(l,m,n,iay) = BB_r*(y(m)-y_b)/R + BB_theta*(y(m)-y_b)*(z(n)-z_b)/(R2d*R) + BB_phi*(x(l)-x_b)/R2d
              f(l,m,n,iaz) = BB_r*(z(n)-z_b)/R - BB_theta*R2d/R
!             R = sqrt((x(l) - x_b)**2 + (y(m) - y_b)**2 + (z(n) - z_b)**2)
!             ! check if this point lies in the bubble
!             if (R .le. r_b) then
!               ! Compute some auxiliary variables.
!               j0 = bessel_j0(lam_bb*R)
!               j1 = bessel_j1(lam_bb*R)
!               R2d = sqrt((x(l) - x_b)**2 + (y(m) - y_b)**2)
!               ! terms Psi grad(Phi)
!               f(l,m,n,iax) = j1*R2d/R**2 * (-y(m)+y_b)
!               f(l,m,n,iay) = j1*R2d/R**2 * (x(l)-x_b)
!               ! terms grad(Psi)xgrad(Phi)
!               f(l,m,n,iax) = f(l,m,n,iax) - j1*R2d/R**4 * (-(x(l)-x_b)*(z(n)-z_b)) + &
!                              (j0-j1/lam_bb/R)*R2d/R**3*lam_bb * (-(x(l)-x_b)*(z(n)-z_b))
!               f(l,m,n,iay) = f(l,m,n,iay) - j1*R2d/R**4 * (-(y(m)-y_b)*(z(n)-z_b)) + &
!                              (j0-j1/lam_bb/R)*R2d/R**3*lam_bb * (-(y(m)-y_b)*(z(n)-z_b))
!               f(l,m,n,iaz) = f(l,m,n,iaz) + 2*j1/R**2*R2d - j1*R2d**3/R**4 + (j0-j1/lam_bb/R)*R2d**3/R**3*lam_bb
!               ! multiply by common factor
!               f(l,m,n,iax:iaz) = 2*pi*r_b*ampl * f(l,m,n,iax:iaz) * &
!                                  (1-(R/r_b)**n_smooth)
            endif
          enddo
        enddo
      enddo

      
!  Compute curl(B) = J for the Poisson solver
      do m=m1,m2
         do n=n1,n2
            call curl(f,iaa,jj(:,m-nghost,n-nghost,:))
         enddo
      enddo
      tmpJ = -jj
!  Use the Poisson solver to solve \nabla^2 A = -J for A
      do j=1,3
        call inverse_laplacian(tmpJ(:,:,:,j))
      enddo
      
!  Overwrite the f-array with the correct vector potential A
      do j=1,3
          ju=iaa-1+j
          f(l1:l2,m1:m2,n1:n2,ju) = tmpJ(:,:,:,j)
      enddo
      
    endif
!      
    endsubroutine initial_condition_aa
!***********************************************************************
    subroutine initial_condition_ss(f)
!
!  Initialize entropy.
!
!  07-sep-11/simon: coded
!
      use SharedVariables
      use EquationOfState

      real, dimension (mx,my,mz,mfarray), intent(inout) :: f      
      integer :: l, m, n
      real :: log_T_b, log_T_0
      real :: r, z_tilde, log_T_ex  ! working variables for smoothing
!
    ! initialize the temperature of the medium and the bubble
    log_T_b = log(T_b)
    log_T_0 = log(T_0)
    do n = n1, n2, 1
      do m = m1, m2, 1
        do l = l1, l2, 1
          ! check if this point lies in the bubble
          if (((x(l) - x_b)**2 + (y(m) - y_b)**2 + (z(n) - z_b)**2) .le. r_b**2) then
            ! reduce the shocks at the start of the simulation by reducing the
            ! temperature gradient
             r = sqrt((x(l) - x_b)**2 + (y(m) - y_b)**2 + (z(n) - z_b)**2)
             z_tilde = z_b + (z(n)-z_b)*r_b/r

             log_T_ex = log_T_0 + log(1-(gamma-1)/gamma*z_tilde/z_0)
             f(l,m,n,ilnTT) = log_T_b*(1-tanh((r-r_b+3*r_b/lam)/r_b*lam))/2 + log_T_ex*(1+tanh((r-r_b+3*r_b/lam)/r_b*lam))/2

!            f(l,m,n,ilnTT) = log_T_b
          else
            f(l,m,n,ilnTT) = log_T_0 + log(1-(gamma-1)/gamma*z(n)/z_0)
          endif
        enddo
      enddo
    enddo
!      
    endsubroutine initial_condition_ss
!***********************************************************************
    subroutine initial_condition_lncc(f)
!
!  Initialize entropy.
!
!  07-sep-11/simon: coded
!
      real, dimension (mx,my,mz,mfarray), intent(inout) :: f      
      integer :: l, m, n
!
    ! initialize the temperature of the medium and the bubble
    if (passive_scalar == 1) then
        do n = n1, n2, 1
            do m = m1, m2, 1
                do l = l1, l2, 1
                    ! check if this point lies in the bubble
                    if (((x(l) - x_b)**2 + (y(m) - y_b)**2 + (z(n) - z_b)**2) .le. r_b**2) then
                        f(l,m,n,ilncc) = 1.
                    else
                        f(l,m,n,ilncc) = 0.
                    endif
                enddo
            enddo
        enddo
    endif
!      
    endsubroutine initial_condition_lncc
!***********************************************************************
    subroutine read_initial_condition_pars(iostat)
!
      use File_io, only: parallel_unit
!
      integer, intent(out) :: iostat
!
      read(parallel_unit, NML=initial_condition_pars, IOSTAT=iostat)
!
    endsubroutine read_initial_condition_pars
!***********************************************************************
    subroutine write_initial_condition_pars(unit)
!
      integer, intent(in) :: unit
!
      write(unit, NML=initial_condition_pars)
!
    endsubroutine write_initial_condition_pars
!***********************************************************************
!
!********************************************************************
!************        DO NOT DELETE THE FOLLOWING       **************
!********************************************************************
!**  This is an automatically generated include file that creates  **
!**  copies dummy routines from noinitial_condition.f90 for any    **
!**  InitialCondition routines not implemented in this file        **
!**                                                                **
    include '../initial_condition_dummies.inc'
!********************************************************************
endmodule InitialCondition
