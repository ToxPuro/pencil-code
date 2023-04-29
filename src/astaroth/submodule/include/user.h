// clang-format off
//#ifdef PENCIL_ASTAROTH

#if AC_DOUBLE_PRECISION == 1
  #define DOUBLE_PRECISION
#endif
  //#include "../../../cparam.inc_c.h"
  #include "/homeappl/home/mreinhar/git/pencil-code/samples/gputest/src/cparam_c.h"
  #include "/homeappl/home/mreinhar/git/pencil-code/samples/gputest/src/cdata_c.h"
  #define STENCIL_ORDER (2*NGHOST)

  #include "/homeappl/home/mreinhar/git/pencil-code/samples/gputest/src/astaroth/PC_moduleflags.h"

  #define CONFIG_PATH

  #define USER_PROVIDED_DEFINES

//#endif
// clang-format on
