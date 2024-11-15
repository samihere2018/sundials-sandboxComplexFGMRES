/*-----------------------------------------------------------------
 * Programmer(s): Mustafa Aggul @ SMU
 *---------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2024, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 *---------------------------------------------------------------
 * Example problem:
 * 
 * In this problem, we employ existing NVectors for vectors with 
 * complex entries: even-numbered indices store real components 
 * while we allocate the odd-numbered indices for their imaginary 
 * counterparts. 
 * The length of vectors is doubled this way, which means storing 
 * a complex vector of size N requires a real NVector of length 2*N.
 * 
 * For example: [1+i  2-2i] is represented as [1 1 2 -2].
 * 
 * The following is a simple complex-valued IVP
 *     dy/dt = y*t + 2i
 * for t in the interval [0.0, 1.0], with initial condition: y=1+2i.
 *
 * This program solves the problem with the Ark method.
 * Output is printed every 0.025 increment of time (40 total).
 * Run statistics (optional outputs) are printed at the end.
 *-----------------------------------------------------------------*/

/* Header files */
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "nvector_serialcomplex.h"
#include <arkode/arkode_arkstep.h>         /* prototypes for ARKStep fcts., consts */
#include <nvector/nvector_serial.h>        /* serial N_Vector types, fcts., macros */
#include <sunmatrix/sunmatrix_dense.h>     /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_sptfqmr.h>   /* access to dense SUNLinearSolver      */
#include <sundials/sundials_types.h>       /* definition of type sunrealtype          */

#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif

/* User-supplied Functions Called by the Solver */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);
static int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

/* Private function to check function return values */
static int check_flag(void *flagvalue, const char *funcname, int opt);

/* Main Program */
int main()
{
  /* general problem parameters */
  sunrealtype T0 = SUN_RCONST(0.0);            /* initial time */
  sunrealtype Tf = SUN_RCONST(1.0);            /* final time */
  sunrealtype dTout = SUN_RCONST(0.025);       /* time between outputs */
  sunindextype NEQ = 2;                 /* number of dependent vars. */
  sunrealtype reltol = SUN_RCONST(1.0e-6);     /* tolerances */
  sunrealtype abstol = SUN_RCONST(1.0e-10);
  sunrealtype some_constant = SUN_RCONST(1.0); /* Constant problem parameter */

  /* general problem variables */
  int flag;                       /* reusable error-checking flag */
  N_Vector y = NULL;              /* empty vector for storing solution */
  SUNMatrix A = NULL;             /* empty matrix for linear solver */
  SUNLinearSolver LS = NULL;      /* empty linear solver object */
  void *arkode_mem = NULL;        /* empty ARKode memory structure */
  FILE *UFID;
  sunrealtype t, tout;
  long int nst, nst_a, nfe, nfi, nsetups, nje, nfeLS, nni, ncfn, netf;

  /* Create the SUNDIALS context object for this simulation */
  SUNContext ctx;
  flag = SUNContext_Create(SUN_COMM_NULL, &ctx);
  if (check_flag(&flag, "SUNContext_Create", 1)) return 1;

  /* Initial diagnostics output */
  printf("\nComplex ODE test problem:\n");
  printf("   constant problem parameter = %"GSYM"\n",    some_constant);
  printf("   reltol = %.1"ESYM"\n",  reltol);
  printf("   abstol = %.1"ESYM"\n\n",abstol);

  /* Initialize data structures */
  y = N_VNew_Serial(NEQ, ctx);          /* Create serial vector for solution */
  if (check_flag((void *)y, "N_VNew_Serial", 0)) return 1;

double complex initial_condition  = 1 + 2*I;

NV_Ith_S(y, 0) = creal(initial_condition); /* Specify initial condition */
NV_Ith_S(y, 1) = cimag(initial_condition); /* Specify initial condition */

  // N_VConst(SUN_RCONST(0.0), y);        /* Specify initial condition */

  /* Call ARKStepCreate to initialize the ARK timestepper module and
     specify the right-hand side function in y'=f(t,y), the inital time
     T0, and the initial dependent variable vector y.  Note: since this
     problem is fully implicit, we set f_E to NULL and f_I to f. */
  arkode_mem = ARKStepCreate(NULL, f, T0, y, ctx);
  if (check_flag((void *)arkode_mem, "ARKStepCreate", 0)) return 1;

  /* Set routines */
  flag = ARKStepSetUserData(arkode_mem, (void *) &some_constant);  /* Pass lamda to user functions */
  if (check_flag(&flag, "ARKStepSetUserData", 1)) return 1;
  flag = ARKStepSStolerances(arkode_mem, reltol, abstol);  /* Specify tolerances */
  if (check_flag(&flag, "ARKStepSStolerances", 1)) return 1;

  /* Initialize dense matrix data structure and solver */
  A = SUNDenseMatrix(NEQ, NEQ, ctx);
  if (check_flag((void *)A, "SUNDenseMatrix", 0)) return 1;
  LS = SUNLinSol_SPTFQMR(y, SUN_PREC_NONE, 10, ctx);
  if (check_flag((void *)LS, "SUNLinSol_Dense", 0)) return 1;

  /* Linear solver interface */
  flag = ARKStepSetLinearSolver(arkode_mem, LS, A);        /* Attach matrix and linear solver */
  if (check_flag(&flag, "ARKStepSetLinearSolver", 1)) return 1;
  flag = ARKStepSetJacFn(arkode_mem, Jac);                 /* Set Jacobian routine */
  if (check_flag(&flag, "ARKStepSetJacFn", 1)) return 1;

  /* Specify linearly implicit RHS, with non-time-dependent Jacobian */
  flag = ARKStepSetLinear(arkode_mem, 0);
  if (check_flag(&flag, "ARKStepSetLinear", 1)) return 1;

  /* Open output stream for results, output comment line */
  UFID = fopen("solution.txt","w");
  fprintf(UFID,"#  t             u\n");
  fprintf(UFID, "-----------------------------\n");

  /* output initial condition to disk */
    fprintf(UFID," %.3"FSYM" | " "%.5"FSYM " + " "%.5" FSYM "i\n", T0, NV_Ith_S(y, 0), NV_Ith_S(y, 1));

  /* Main time-stepping loop: calls ARKStepEvolve to perform the integration, then
     prints results.  Stops when the final time has been reached */
  t = T0;
  tout = T0+dTout;
  printf("        t              u        \n");
  printf("   -----------------------------\n");
  while (Tf - t > 1.0e-15) {

    flag = ARKStepEvolve(arkode_mem, tout, y, &t, ARK_NORMAL);      /* call integrator */
    if (check_flag(&flag, "ARKStepEvolve", 1)) break;
    printf("  %10.6" FSYM "%10.6" FSYM " + " "%1.6" FSYM "i\n", t, NV_Ith_S(y, 0), NV_Ith_S(y, 1)); /* access/print solution */

    fprintf(UFID," %.3"FSYM" | " "%.5"FSYM " + " "%.5" FSYM "i\n", t, NV_Ith_S(y, 0), NV_Ith_S(y, 1));

    if (flag >= 0) {                                         /* successful solve: update time */
      tout += dTout;
      tout = (tout > Tf) ? Tf : tout;
    } else {                                                 /* unsuccessful solve: break */
      fprintf(stderr,"Solver failure, stopping integration\n");
      break;
    }
  }
  printf("   ---------------------\n");
  fclose(UFID);

  /* Get/print some final statistics on how the solve progressed */
  flag = ARKStepGetNumSteps(arkode_mem, &nst);
  check_flag(&flag, "ARKStepGetNumSteps", 1);
  flag = ARKStepGetNumStepAttempts(arkode_mem, &nst_a);
  check_flag(&flag, "ARKStepGetNumStepAttempts", 1);
  flag = ARKStepGetNumRhsEvals(arkode_mem, &nfe, &nfi);
  check_flag(&flag, "ARKStepGetNumRhsEvals", 1);
  flag = ARKStepGetNumLinSolvSetups(arkode_mem, &nsetups);
  check_flag(&flag, "ARKStepGetNumLinSolvSetups", 1);
  flag = ARKStepGetNumErrTestFails(arkode_mem, &netf);
  check_flag(&flag, "ARKStepGetNumErrTestFails", 1);
  flag = ARKStepGetNumNonlinSolvIters(arkode_mem, &nni);
  check_flag(&flag, "ARKStepGetNumNonlinSolvIters", 1);
  flag = ARKStepGetNumNonlinSolvConvFails(arkode_mem, &ncfn);
  check_flag(&flag, "ARKStepGetNumNonlinSolvConvFails", 1);
  flag = ARKStepGetNumJacEvals(arkode_mem, &nje);
  check_flag(&flag, "ARKStepGetNumJacEvals", 1);
  flag = ARKStepGetNumLinRhsEvals(arkode_mem, &nfeLS);
  check_flag(&flag, "ARKStepGetNumLinRhsEvals", 1);

  printf("\nFinal Solver Statistics:\n");
  printf("   Internal solver steps = %li (attempted = %li)\n", nst, nst_a);
  printf("   Total RHS evals:  Fe = %li,  Fi = %li\n", nfe, nfi);
  printf("   Total linear solver setups = %li\n", nsetups);
  printf("   Total RHS evals for setting up the linear system = %li\n", nfeLS);
  printf("   Total number of Jacobian evaluations = %li\n", nje);
  printf("   Total number of Newton iterations = %li\n", nni);
  printf("   Total number of linear solver convergence failures = %li\n", ncfn);
  printf("   Total number of error test failures = %li\n\n", netf);

  /* Clean up and return */
  N_VDestroy(y);            /* Free y vector */
  ARKStepFree(&arkode_mem); /* Free integrator memory */
  SUNLinSolFree(LS);        /* Free linear solver */
  SUNMatDestroy(A);         /* Free A matrix */
  SUNContext_Free(&ctx);    /* Free context */

  return flag;
}

/*-------------------------------
 * Functions called by the solver
 *-------------------------------*/

/* f routine to compute the ODE RHS function f(t,y). */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  sunrealtype *rdata = (sunrealtype *) user_data;   /* cast user_data to sunrealtype */
  sunrealtype some_constant = rdata[0];          /* set shortcut for problem parameter */
  sunrealtype cr = NV_Ith_S(y,0);                /* access current solution value */
  sunrealtype ci = NV_Ith_S(y,1);                /* access current solution value */

  /* fill in the RHS function: "NV_Ith_S" accesses the 0th entry of ydot */
  NV_Ith_S(ydot,0) = cr * t;
  NV_Ith_S(ydot,1) = ci * t + 2;

  return 0;                                   /* return with success */
}

/* Jacobian routine to compute J(t,y) = df/dy. */
static int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  sunrealtype *rdata = (sunrealtype *) user_data;   /* cast user_data to sunrealtype */
  sunrealtype lamda = rdata[0];                  /* set shortcut for stiffness parameter */
  sunrealtype *Jdata = SUNDenseMatrix_Data(J);

  /* Fill in Jacobian of f: set the first entry of the data array to set the (0,0) entry */
  Jdata[0] = t;
  Jdata[1] = t;

  return 0;                                   /* return with success */
}

/*-------------------------------
 * Private helper functions
 *-------------------------------*/

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer
    opt == 1 means SUNDIALS function returns a flag so check if
             flag >= 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer
*/
static int check_flag(void *flagvalue, const char *funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1; }

  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
              funcname, *errflag);
      return 1; }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1; }

  return 0;
}

/*---- end of file ----*/
