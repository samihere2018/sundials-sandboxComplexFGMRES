/*-----------------------------------------------------------------
 * Programmer(s): Mustafa Aggul @ SMU
 *---------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2023, Lawrence Livermore National Security
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
 * The following test simulates a complex-valued system of ODEs.
 * 
 * This is an ODE system with 2 components, Y = [u,v],
 * satisfying the equations,
 *    du/dt = -2u - v
 *    dv/dt = +u - 2v
 * for t in the interval [0.0, 1.0], with initial conditions
 * Y0 = [2,i].
 *
 * This program solves the problem with the DIRK method, using a
 * Newton iteration with the SUNDENSE dense linear solver, and a
 * user-supplied Jacobian routine.
 *-----------------------------------------------------------------*/

/* Header files */
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <arkode/arkode_arkstep.h>      /* prototypes for ARKStep fcts., consts */
#include <nvector/nvector_serial.h>     /* serial N_Vector types, fcts., macros */
#include <sunmatrix/sunmatrix_dense.h>  /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h>  /* access to dense SUNLinearSolver      */
#include <sundials/sundials_types.h>    /* def. of type 'sunrealtype' */

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
static int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

/* Private function to check function return values */
static int check_flag(void *flagvalue, const char *funcname, int opt);

/* Main Program */
int main()
{
  /* general problem parameters */
  sunrealtype T0 = SUN_RCONST(0.0);         /* initial time */
  sunrealtype Tf = SUN_RCONST(1.0);       /* final time */
  sunrealtype dTout = SUN_RCONST(0.01);     /* time between outputs */
  sunindextype NEQ = 4;              /* number of dependent vars. */
  int Nt = (int) ceil(Tf/dTout);     /* number of output times */
  sunrealtype reltol = 1.0e-6;          /* tolerances */
  sunrealtype abstol = 1.0e-10;
  sunrealtype u0, v0;

  /* general problem variables */
  int flag;                      /* reusable error-checking flag */
  N_Vector y = NULL;             /* empty vector for storing solution */
  SUNMatrix A = NULL;            /* empty matrix for solver */
  SUNLinearSolver LS = NULL;     /* empty linear solver object */
  void *arkode_mem = NULL;       /* empty ARKode memory structure */
  FILE *UFID;
  sunrealtype t, tout;
  int iout;
  long int nst, nst_a, nfe, nfi, nsetups, nje, nfeLS, nni, nnf, ncfn, netf;

  /* Create the SUNDIALS context object for this simulation */
  SUNContext ctx;
  flag = SUNContext_Create(SUN_COMM_NULL, &ctx);
  if (check_flag(&flag, "SUNContext_Create", 1)) return 1;

  /* Initial problem output */
  printf("\n Complex-valued Sytem of ODE test problem:\n");
  printf("   initial conditions:  u0 = %"GSYM",  v0 = %"GSYM"\n",u0,v0);
  printf("   reltol = %.1"ESYM",  abstol = %.1"ESYM"\n\n",reltol,abstol);

  y = N_VNew_Serial(NEQ, ctx);           /* Create serial vector for solution */
  if (check_flag((void *)y, "N_VNew_Serial", 0)) return 1;
  
  
  NV_Ith_S(y,0) = 2.0;               /* Set initial conditions */
  NV_Ith_S(y,1) = 0.0;
  NV_Ith_S(y,2) = 0.0;               
  NV_Ith_S(y,3) = 1.0;

  /* Call ARKStepCreate to initialize the ARK timestepper module and
     specify the right-hand side function in y'=f(t,y), the inital time
     T0, and the initial dependent variable vector y.  Note: since this
     problem is fully implicit, we set f_E to NULL and f_I to f. */
  arkode_mem = ARKStepCreate(NULL, f, T0, y, ctx);
  if (check_flag((void *)arkode_mem, "ARKStepCreate", 0)) return 1;

  /* Set routines */
  // flag = ARKStepSetUserData(arkode_mem, (void *) rdata);     /* Pass rdata to user functions */
  // if (check_flag(&flag, "ARKStepSetUserData", 1)) return 1;
  flag = ARKStepSStolerances(arkode_mem, reltol, abstol);    /* Specify tolerances */
  if (check_flag(&flag, "ARKStepSStolerances", 1)) return 1;
  flag = ARKStepSetInterpolantType(arkode_mem, ARK_INTERP_LAGRANGE);  /* Specify stiff interpolant */
  if (check_flag(&flag, "ARKStepSetInterpolantType", 1)) return 1;
  flag = ARKStepSetDeduceImplicitRhs(arkode_mem, 1);  /* Avoid eval of f after stage */
  if (check_flag(&flag, "ARKStepSetDeduceImplicitRhs", 1)) return 1;

  /* Initialize dense matrix data structure and solver */
  A = SUNDenseMatrix(NEQ, NEQ, ctx);
  if (check_flag((void *)A, "SUNDenseMatrix", 0)) return 1;
  LS = SUNLinSol_Dense(y, A, ctx);
  if (check_flag((void *)LS, "SUNLinSol_Dense", 0)) return 1;

  /* Linear solver interface */
  flag = ARKStepSetLinearSolver(arkode_mem, LS, A);        /* Attach matrix and linear solver */
  if (check_flag(&flag, "ARKStepSetLinearSolver", 1)) return 1;
  flag = ARKStepSetJacFn(arkode_mem, Jac);                 /* Set Jacobian routine */
  if (check_flag(&flag, "ARKStepSetJacFn", 1)) return 1;

  /* Open output stream for results, output comment line */
  UFID = fopen("solution.txt","w");
  fprintf(UFID,"# t u v\n");
  fprintf(UFID," %.3"FSYM" | " "%.5"FSYM " + " "%.5" FSYM "i  " "%.5"FSYM " + " "%.5" FSYM "i\n", 
          T0, NV_Ith_S(y, 0), NV_Ith_S(y, 1), NV_Ith_S(y, 2), NV_Ith_S(y, 3));


  /* Main time-stepping loop: calls ARKStepEvolve to perform the integration, then
     prints results.  Stops when the final time has been reached */
  t = T0;
  tout = T0+dTout;
  printf("        t           u           v             \n");
  printf("   -------------------------------------------\n");
  printf(" %.3"FSYM" | " "%.5"FSYM " + " "%.5" FSYM "i  " "%.5"FSYM " + " "%.5" FSYM "i\n", 
           t, NV_Ith_S(y, 0), NV_Ith_S(y, 1), NV_Ith_S(y, 2), NV_Ith_S(y, 3));

  for (iout=0; iout<Nt; iout++) {

    flag = ARKStepEvolve(arkode_mem, tout, y, &t, ARK_NORMAL);      /* call integrator */
    if (check_flag(&flag, "ARKStepEvolve", 1)) break;
    printf(" %.3"FSYM" | " "%.5"FSYM " + " "%.5" FSYM "i  " "%.5"FSYM " + " "%.5" FSYM "i\n", 
           t, NV_Ith_S(y, 0), NV_Ith_S(y, 1), NV_Ith_S(y, 2), NV_Ith_S(y, 3));
    fprintf(UFID," %.3"FSYM" | " "%.5"FSYM " + " "%.5" FSYM "i  " "%.5"FSYM " + " "%.5" FSYM "i\n", 
           t, NV_Ith_S(y, 0), NV_Ith_S(y, 1), NV_Ith_S(y, 2), NV_Ith_S(y, 3));
    if (flag >= 0) {                                         /* successful solve: update time */
      tout += dTout;
      tout = (tout > Tf) ? Tf : tout;
    } else {                                                 /* unsuccessful solve: break */
      fprintf(stderr,"Solver failure, stopping integration\n");
      break;
    }
  }
  printf("   -------------------------------------------\n");
  fclose(UFID);

  /* Print some final statistics */
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
  flag = ARKStepGetNumStepSolveFails(arkode_mem, &ncfn);
  check_flag(&flag, "ARKStepGetNumStepSolveFails", 1);
  flag = ARKStepGetNumNonlinSolvIters(arkode_mem, &nni);
  check_flag(&flag, "ARKStepGetNumNonlinSolvIters", 1);
  flag = ARKStepGetNumNonlinSolvConvFails(arkode_mem, &nnf);
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
  printf("   Total number of nonlinear solver convergence failures = %li\n", nnf);
  printf("   Total number of error test failures = %li\n", netf);
  printf("   Total number of failed steps from solver failure = %li\n", ncfn);

  /* Clean up and return with successful completion */
  N_VDestroy(y);               /* Free y vector */
  ARKStepFree(&arkode_mem);    /* Free integrator memory */
  SUNLinSolFree(LS);           /* Free linear solver */
  SUNMatDestroy(A);            /* Free A matrix */
  SUNContext_Free(&ctx);       /* Free context */

  return 0;
}

/*-------------------------------
 * Functions called by the solver
 *-------------------------------*/

/* f routine to compute the ODE RHS function f(t,y). */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  sunrealtype *rdata = (sunrealtype *) user_data;   /* cast user_data to sunrealtype */
  sunrealtype ru = NV_Ith_S(y,0);                 /* access solution values */
  sunrealtype iu = NV_Ith_S(y,1);
  sunrealtype rv = NV_Ith_S(y,2);
  sunrealtype iv = NV_Ith_S(y,3);

  /* fill in the RHS function */
  NV_Ith_S(ydot,0) = -2.0*ru - 1.0*rv;
  NV_Ith_S(ydot,1) = -2.0*iu - 1.0*iv;
  NV_Ith_S(ydot,2) = +1.0*ru - 2.0*rv;
  NV_Ith_S(ydot,3) = +1.0*iu - 2.0*iv;

  return 0;                                  /* Return with success */
}

/* Jacobian routine to compute J(t,y) = df/dy. */
static int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
               N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  sunrealtype *rdata = (sunrealtype *) user_data;   /* cast user_data to sunrealtype */

  /* fill in the Jacobian via SUNDenseMatrix macro, SM_ELEMENT_D (see sunmatrix_dense.h) */
  SM_ELEMENT_D(J,0,0) = -2.0;
  SM_ELEMENT_D(J,0,1) = -0.0;
  SM_ELEMENT_D(J,0,0) = -1.0;
  SM_ELEMENT_D(J,0,1) = -0.0;

  SM_ELEMENT_D(J,1,0) = -0.0;
  SM_ELEMENT_D(J,1,1) = -2.0;
  SM_ELEMENT_D(J,1,0) = -0.0;
  SM_ELEMENT_D(J,1,1) = -1.0;

  SM_ELEMENT_D(J,2,0) = +1.0;
  SM_ELEMENT_D(J,2,1) = -0.0;
  SM_ELEMENT_D(J,2,0) = -2.0;
  SM_ELEMENT_D(J,2,1) = -0.0;

  SM_ELEMENT_D(J,3,0) = -0.0;
  SM_ELEMENT_D(J,3,1) = +1.0;
  SM_ELEMENT_D(J,3,0) = -0.0;
  SM_ELEMENT_D(J,3,1) = -2.0;

  return 0;                                   /* Return with success */
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
