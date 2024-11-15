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
 * This program solves the problem with the ERK method.
 * Output is printed every 0.025 increment of time (40 total).
 * Run statistics (optional outputs) are printed at the end.
 *-----------------------------------------------------------------*/

/* Header files */
#include <arkode/arkode_erkstep.h> /* prototypes for ERKStep fcts., consts */
#include <math.h>
#include <complex.h>
#include <nvector/nvector_serial.h> /* serial N_Vector types, fcts., macros */ 
#include <stdio.h>
#include <sundials/sundials_math.h>  /* def. of SUNRsqrt, etc. */
#include <sundials/sundials_types.h> /* def. of type 'sunrealtype' */

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
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);

/* Private function to check function return values */
static int check_flag(void* flagvalue, const char* funcname, int opt);

/* Main Program */
int main(void)
{
  /* general problem parameters */
  sunrealtype T0     = SUN_RCONST(0.0);  /* initial time */
  sunrealtype Tf     = SUN_RCONST(1.0);  /* final time */
  sunrealtype dTout  = SUN_RCONST(0.025);/* time between outputs */
  sunindextype NEQ   = 2;                /* number of dependent vars. */
  sunrealtype reltol = 1.0e-6;           /* tolerances */
  sunrealtype abstol = 1.0e-10;

  /* general problem variables */
  int flag;                /* reusable error-checking flag */
  N_Vector y       = NULL; /* empty vector for storing solution */
  void* arkode_mem = NULL; /* empty ARKode memory structure */
  FILE *UFID, *FID;
  sunrealtype t, tout;

  /* Create the SUNDIALS context object for this simulation */
  SUNContext ctx;
  flag = SUNContext_Create(SUN_COMM_NULL, &ctx);
  if (check_flag(&flag, "SUNContext_Create", 1)) { return 1; }

  /* Initial problem output */
  printf("\nComplex ODE test problem:\n");
  printf("   reltol = %.1" ESYM "\n", reltol);
  printf("   abstol = %.1" ESYM "\n\n", abstol);

  /* Initialize data structures */
  y = N_VNew_Serial(NEQ, ctx); /* Create serial vector for solution */
  if (check_flag((void*)y, "N_VNew_Serial", 0)) { return 1; }
  
double complex initial_condition = 1 + 2*I;

  NV_Ith_S(y, 0) = creal(initial_condition); /* Specify initial condition */
  NV_Ith_S(y, 1) = cimag(initial_condition); /* Specify initial condition */

  /* Call ERKStepCreate to initialize the ERK timestepper module and
     specify the right-hand side function in y'=f(t,y), the inital time
     T0, and the initial dependent variable vector y. */
  arkode_mem = ERKStepCreate(f, T0, y, ctx);
  if (check_flag((void*)arkode_mem, "ERKStepCreate", 0)) { return 1; }

  /* Specify tolerances */
  flag = ERKStepSStolerances(arkode_mem, reltol, abstol);
  if (check_flag(&flag, "ERKStepSStolerances", 1)) { return 1; }

  /* Open output stream for results, output comment line */
  UFID = fopen("solution.txt", "w");
  fprintf(UFID,"#  t             u\n");
  fprintf(UFID, "-----------------------------\n");

  /* output initial condition to disk */
    fprintf(UFID," %.3"FSYM" | " "%.5"FSYM " + " "%.5" FSYM "i\n", T0, NV_Ith_S(y, 0), NV_Ith_S(y, 1));

  /* Main time-stepping loop: calls ERKStepEvolve to perform the integration, then
     prints results.  Stops when the final time has been reached */
  t    = T0;
  tout = T0 + dTout;
  printf("        t              u        \n");
  printf("   -----------------------------\n");
  while (Tf - t > 1.0e-15)
  {
    flag = ERKStepEvolve(arkode_mem, tout, y, &t, ARK_NORMAL); /* call integrator */
    if (check_flag(&flag, "ERKStepEvolve", 1)) { break; }
    printf("  %10.6" FSYM "%10.6" FSYM " + " "%1.6" FSYM "i\n", t, NV_Ith_S(y, 0), NV_Ith_S(y, 1)); /* access/print solution */
    fprintf(UFID," %.3"FSYM" | " "%.5"FSYM " + " "%.5" FSYM "i\n", t, NV_Ith_S(y, 0), NV_Ith_S(y, 1));
    if (flag >= 0)
    { /* successful solve: update time */
      tout += dTout;
      tout = (tout > Tf) ? Tf : tout;
    }
    else
    { /* unsuccessful solve: break */
      fprintf(stderr, "Solver failure, stopping integration\n");
      break;
    }
  }
  printf("   ---------------------\n");
  fclose(UFID);

  /* Print final statistics */
  printf("\nFinal Statistics:\n");
  flag = ERKStepPrintAllStats(arkode_mem, stdout, SUN_OUTPUTFORMAT_TABLE);

  /* Print final statistics to a file in CSV format */
  FID  = fopen("ark_analytic_nonlin_stats.csv", "w");
  flag = ERKStepPrintAllStats(arkode_mem, FID, SUN_OUTPUTFORMAT_CSV);
  fclose(FID);

  /* Clean up and return with successful completion */
  N_VDestroy(y);            /* Free y vector */
  ERKStepFree(&arkode_mem); /* Free integrator memory */
  SUNContext_Free(&ctx);    /* Free context */

  return 0;
}

/*-------------------------------
 * Functions called by the solver
 *-------------------------------*/

/* f routine to compute the ODE RHS function f(t,y). */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
  NV_Ith_S(ydot, 0) = t * NV_Ith_S(y, 0);
  NV_Ith_S(ydot, 1) = t * NV_Ith_S(y, 1) + 2;
  return 0;
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
static int check_flag(void* flagvalue, const char* funcname, int opt)
{
  int* errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL)
  {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1;
  }

  /* Check if flag < 0 */
  else if (opt == 1)
  {
    errflag = (int*)flagvalue;
    if (*errflag < 0)
    {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
              funcname, *errflag);
      return 1;
    }
  }

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL)
  {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return 1;
  }

  return 0;
}

/*---- end of file ----*/