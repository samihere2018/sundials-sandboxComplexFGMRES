/* -----------------------------------------------------------------
 * Programmer(s): Mustafa Aggul @ SMU
 * Edited by Sylvia Amihere @ SMU
 * -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2024, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------
 * These test functions check some components of a complex-valued
 * SUNLINEARSOLVER module implementation (for more thorough tests,
 * see the main SUNDIALS repository, inside examples/sunlinsol/).
 * -----------------------------------------------------------------
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "nvector_serialcomplex.h"
#include "sunlinsol_spfgmrcomplex.h"
// #include "sundials_iterativecomplex.h" //Amihere
// #include "sundials_iterativecomplex_impl.h" //Amihere


#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif

/* constants */
#define ZERO     SUN_RCONST(0.0)
#define ONE      SUN_RCONST(1.0)
#define FIVE     SUN_RCONST(5.0)
#define THOUSAND SUN_RCONST(1000.0)

#define SOMECOMPLEXNUMBERd    (2.0 + 5.0*I)
#define SOMECOMPLEXNUMBERup   (-1.0 + 2.0*I)
#define SOMECOMPLEXNUMBERlow  (3.0 - 4.0*I)

/* user data structure */
typedef struct
{
  sunindextype N;       /* problem size */
  N_Vector d;           /* matrix diagonal */
  N_Vector s1;          /* scaling vector */
  N_Vector s2;          /* scaling vector */
  suncomplextype up;    /* nondiagonal entries of the matrix */
  suncomplextype low;  /* nondiagonal entries of the matrix */
} UserData;

/* private functions */
/*    matrix-vector product  */
int ATimes(void* ProbData, N_Vector v, N_Vector z);
/*    preconditioner setup */
int PSetup(void* ProbData);
/*    preconditioner solve */
int PSolve(void* ProbData, N_Vector r, N_Vector z, sunrealtype tol, int lr);
/*    checks function return values  */
static int check_flag(void* flagvalue, const char* funcname, int opt);
/*    uniform random number generator in [0,1] */
static sunrealtype urand(void);
static int check_vector(N_Vector X, N_Vector Y, sunrealtype tol);

/* global copy of the problem size (for check_vector routine) */
sunindextype problem_size;


/* ----------------------------------------------------------------------
 * SUNLinSol_SPTFQMR Linear Solver Testing Routine
 *
 * We construct a tridiagonal matrix Ahat, a random solution xhat,
 * and a corresponding rhs vector bhat = Ahat*xhat, such that each
 * of these is unit-less.  To test row/column scaling, we use the
 * matrix A = S1-inverse Ahat S2, rhs vector b = S1-inverse bhat,
 * and solution vector x = (S2-inverse) xhat; hence the linear
 * system has rows scaled by S1-inverse and columns scaled by S2,
 * where S1 and S2 are the diagonal matrices with entries from the
 * vectors s1 and s2, the 'scaling' vectors supplied to SPTFQMR
 * having strictly positive entries.  When this is combined with
 * preconditioning, assume that Phat is the desired preconditioner
 * for Ahat, then our preconditioning matrix P \approx A should be
 *    left prec:  P-inverse \approx S1-inverse Ahat-inverse S1
 *    right prec:  P-inverse \approx S2-inverse Ahat-inverse S2.
 * Here we use a diagonal preconditioner D, so the S*-inverse
 * and S* in the product cancel one another.
 * --------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
  int fails    = 0;    /* counter for test failures */
  int passfail = 0;    /* overall pass/fail flag    */
  SUNLinearSolver LS;  /* linear solver object      */
  SUNMatrix A = NULL;  /* matrix object             */
  N_Vector xhat, x, b; /* test vectors              */
  UserData ProbData;   /* problem data structure    */
  int pretype, maxl, print_timing, failure, gstype; //Amihere
  sunindextype i;
  suncomplextype* vecdata;
  double tol;
  SUNContext sunctx;

  if (SUNContext_Create(SUN_COMM_NULL, &sunctx))
  {
    printf("ERROR: SUNContext_Create failed\n");
    return (-1);
  }

  /* check inputs: local problem size */
  if (argc < 5)
  {
    printf("ERROR: FOUR (5) Inputs required:\n");
    printf("  Problem size should be >0\n");
    printf("  Gram-Schmidt orthogonalisation type should be 1 (Modified) or 2 (Classical)\n");
    printf("  Preconditioning type should be 1 (LEFT) or 2 (RIGHT)\n");
    printf("  Maximum Krylov subspace dimension should be >0\n");
    printf("  Solver tolerance should be >0\n");
    return 1;
  }
  ProbData.N   = (sunindextype)atol(argv[1]);
  problem_size = ProbData.N;
  if (ProbData.N <= 0)
  {
    printf("ERROR: Problem size must be a positive integer\n");
    return 1;
  }

  //Amihere - type of Gram Schmidt orthogonalization Process
  gstype = atoi(argv[2]);
  if (gstype == 1) { gstype = SUN_MODIFIED_GS; }
  else if (gstype == 2) { gstype = SUN_CLASSICAL_GS; }
  else
  {
    printf("ERROR: Gram-Schmidt process type must be either 1 or 2\n");
    return 1;
  }

  pretype = atoi(argv[3]);
  if (pretype == 1) { pretype = SUN_PREC_LEFT; }
  else if (pretype == 2) { pretype = SUN_PREC_RIGHT; }
  else
  {
    printf("ERROR: Preconditioning type must be either 1 or 2\n");
    return 1;
  }

  maxl = atoi(argv[4]);
  if (maxl <= 0)
  {
    printf(
      "ERROR: Maximum Krylov subspace dimension must be a positive integer\n");
    return 1;
  }
  tol = atof(argv[5]);
  if (tol <= ZERO)
  {
    printf("ERROR: Solver tolerance must be a positive real number\n");
    return 1;
  }

  printf("\nCustom linear solver test:\n");
  printf("  Problem size = %ld\n", (long int)ProbData.N);
  printf("  Gram-Schmidt Process = %i\n", gstype);
  printf("  Preconditioning type = %i\n", pretype);
  printf("  Maximum Krylov subspace dimension = %i\n", maxl);
  printf("  Solver Tolerance = %g\n", tol);

  /* Create vectors */
  x = N_VNew_SComplex(ProbData.N, sunctx);
  if (check_flag(x, "N_VNew_SComplex", 0)) { return 1; }
  xhat = N_VClone_SComplex(x);
  if (check_flag(xhat, "N_VClone_SComplex", 0)) { return 1; }
  b = N_VClone_SComplex(x);
  if (check_flag(b, "N_VClone_SComplex", 0)) { return 1; }
  ProbData.d = N_VClone_SComplex(x);
  if (check_flag(ProbData.d, "N_VClone_SComplex", 0)) { return 1; }
  ProbData.s1 = N_VClone_SComplex(x);
  if (check_flag(ProbData.s1, "N_VClone_SComplex", 0)) { return 1; }
  ProbData.s2 = N_VClone_SComplex(x);
  if (check_flag(ProbData.s2, "N_VClone_SComplex", 0)) { return 1; }
  ProbData.up = SOMECOMPLEXNUMBERup;
  ProbData.low = SOMECOMPLEXNUMBERlow;

  /* Fill xhat vector with 2,3,4 ... */
  vecdata = N_VGetArrayPointer_SComplex(xhat);
  for (i = 0; i < ProbData.N; i++) { vecdata[i] = 1.0 + (suncomplextype)i; }

  /* Fill Jacobi vector with matrix diagonal */
  N_VConst_SComplex(SOMECOMPLEXNUMBERd, ProbData.d);

  /* Create Custom linear solver */
  LS = SUNLinSol_SPFGMRComplex(x, pretype, maxl, sunctx);
  // LS = SUNLinSol_SComplex(x, pretype, gstype, maxl, sunctx); //Amihere

  /* Test GetType */
  if (SUNLinSolGetType_SPFGMRComplex(LS) != SUNLINEARSOLVER_ITERATIVE)
  {
    printf(">>> FAILED test -- SUNLinSolGetType \n");
    fails++;
  }
  else { printf("    PASSED test -- SUNLinSolGetType \n");}

  /* Test GetID */
  // if (SUNLinSolGetID_SPFGMRComplex(LS) != SUNLINEARSOLVER_CUSTOM)
  // {
  //   printf(">>> FAILED test -- SUNLinSolGetID \n");
  //   fails++;
  // }
  // else { printf("    PASSED test -- SUNLinSolGetID \n"); }

  /* Test SetATimes */
  failure = SUNLinSolSetATimes_SPFGMRComplex(LS, &ProbData, ATimes);
  if (failure)
  {
    printf(">>> FAILED test -- SUNLinSolSetATimes returned %d \n", failure);
    fails++;
  }
  else { printf("    PASSED test -- SUNLinSolSetATimes \n"); }

  /* Test SetPreconditioner */
  failure = SUNLinSolSetPreconditioner_SPFGMRComplex(LS, &ProbData, PSetup, PSolve);
  if (failure)
  {
    printf(">>> FAILED test -- SUNLinSolSetPreconditioner returned %d \n", failure);
    fails++;
  }
  else { printf("    PASSED test -- SUNLinSolSetPreconditioner \n"); }

  /* Test SetScalingVectors */
  failure = SUNLinSolSetScalingVectors_SPFGMRComplex(LS, ProbData.s1, ProbData.s2);
  if (failure)
  {
    printf(">>> FAILED test -- SUNLinSolSetScalingVectors returned %d \n", failure);
    fails++;
  }
  else { printf("    PASSED test -- SUNLinSolSetScalingVectors \n"); }

  /* Test SetZeroGuess */
  failure = SUNLinSolSetZeroGuess_SPFGMRComplex(LS, SUNTRUE);
  if (failure)
  {
    printf(">>> FAILED test -- SUNLinSolSetZeroGuess_SComplex returned %d \n", failure);
    fails++;
  }
  else { printf("    PASSED test -- SUNLinSolSetZeroGuess_SComplex \n"); }

  failure = SUNLinSolSetZeroGuess_SPFGMRComplex(LS, SUNFALSE);
  if (failure)
  {
    printf(">>> FAILED test -- SUNLinSolSetZeroGuess_SComplex returned %d \n", failure);
    fails++;
  }
  else { printf("    PASSED test -- SUNLinSolSetZeroGuess_SComplex \n"); }

  /* Test Initialize */
  if (SUNLinSolInitialize_SPFGMRComplex(LS))
  { 
    printf(">>> FAILED test -- SUNLinSolInitialize_SComplex check \n");
    fails++;
  }
  else { printf("    PASSED test -- SUNLinSolInitialize_SComplex \n"); }


  /*** Poisson-like solve w/ scaled rows (Jacobi preconditioning) ***/

  /* set scaling vectors */
  vecdata = N_VGetArrayPointer_SComplex(ProbData.s1);
  for (i = 0; i < ProbData.N; i++) { vecdata[i] = ONE + THOUSAND * urand(); }

  vecdata = N_VGetArrayPointer_SComplex(ProbData.s2);
  for (i = 0; i < ProbData.N; i++) { vecdata[i] = FIVE + THOUSAND * urand(); }

  /* Fill x vector with scaled version */
  N_VDiv_SComplex(xhat, ProbData.s2, x);

  /* Fill b vector with result of matrix-vector product */
  fails = ATimes(&ProbData, x, b);
  if (check_flag(&fails, "ATimes", 1)) { return 1; }

  /* Run tests with this setup */
  failure = SUNLinSol_SPFGMRComplex_SetPrecType(LS, pretype);
  if (failure) { printf(">>> FAILED test -- SUNLinSolSetPrecType_SComplex check \n"); }
  else { printf("    PASSED test -- SUNLinSol_SetPrecType \n"); }

  //Amihere
  failure = SUNLinSol_SPFGMRComplex_SetGSType(LS, gstype);
  if (failure) { printf(">>> FAILED test -- SUNLinSolSetGSType_SComplex check \n"); }
  else { printf("    PASSED test -- SUNLinSol_SetGSType \n"); }

  failure = SUNLinSolSetup_SPFGMRComplex(LS, A);
  if (failure)
  {
    printf(">>> FAILED test -- SUNLinSolSetup_SComplex check \n");
    return (1);
  }
  else { printf("    PASSED test -- SUNLinSolSetup_SComplex \n"); }

  N_Vector y = N_VClone_SComplex(x);
  N_VConst_SComplex(ZERO, y);
  failure = SUNLinSolSetZeroGuess_SPFGMRComplex(LS, SUNTRUE);
  if (failure)
  {
    printf(">>> FAILED test -- SUNLinSolSetZeroGuess_SComplex returned %d \n", failure);
    N_VDestroy_SComplex(y);
    return (1);
  }

  failure = SUNLinSolSolve_SPFGMRComplex(LS, A, y, b, tol);
  if (failure)
  {
    printf(">>> FAILED test -- SUNLinSolSolve_SComplex returned %d \n", failure);
    N_VDestroy_SComplex(y);
    return (1);
  }

  failure = check_vector(x, y, 10.0 * tol);
  if (failure)
  {
    printf(">>> FAILED test -- SUNLinSolSolve_SComplex check \n");
    N_VDestroy_SComplex(y);
    return (1);
  }
  else
  { printf("    PASSED test -- SUNLinSolSolve_SComplex \n"); }
  N_VDestroy_SComplex(y);

  sunindextype lastflag = SUNLinSolLastFlag_SPFGMRComplex(LS);
  printf("    PASSED test -- SUNLinSolLastFlag_SComplex (%ld) \n", (long int)lastflag);


  int numiters = SUNLinSolNumIters_SPFGMRComplex(LS);
  printf("    PASSED test -- SUNLinSolNumIters_SComplex (%d) \n", numiters);

  double resnorm = (double) SUNLinSolResNorm_SPFGMRComplex(LS);
  if (resnorm < ZERO)
  {
    printf(">>> FAILED test -- SUNLinSolResNorm_SComplex returned %g \n", resnorm);
    return (1);
  }
  else { printf("    PASSED test -- SUNLinSolResNorm_SComplex\n"); }

  N_Vector resid = SUNLinSolResid_SPFGMRComplex(LS);
  if (resid == NULL)
  {
    printf(">>> FAILED test -- SUNLinSolResid_SComplex returned NULL N_Vector \n");
    return (1);
  }
  else { printf("    PASSED test -- SUNLinSolResid_SComplex\n"); }

  /* Print result */
  if (fails)
  {
    printf("FAIL: MySUNLinSol module, failed %i tests\n\n", fails);
    passfail += 1;
  }
  else { printf("SUCCESS: MySUNLinSol module, passed all tests\n\n"); }

  /* Free solver and vectors */
  SUNLinSolFree_SPFGMRComplex(LS);
  N_VDestroy_SComplex(x);
  N_VDestroy_SComplex(xhat);
  N_VDestroy_SComplex(b);
  N_VDestroy_SComplex(ProbData.d);
  N_VDestroy_SComplex(ProbData.s1);
  N_VDestroy_SComplex(ProbData.s2);
  SUNContext_Free(&sunctx);

  return (passfail);
}

/* ----------------------------------------------------------------------
 * Private helper functions
 * --------------------------------------------------------------------*/

/* matrix-vector product  */
int ATimes(void* Data, N_Vector v_vec, N_Vector z_vec)
{
  /* local variables */
  suncomplextype *v, *z, *s1, *s2, *diag, up, low;
  sunindextype i, N;
  UserData* ProbData;

  /* access user data structure and vector data */
  ProbData = (UserData*)Data;
  v        = N_VGetArrayPointer_SComplex(v_vec);
  if (check_flag(v, "N_VGetArrayPointer_SComplex", 0)) { return 1; }
  z = N_VGetArrayPointer_SComplex(z_vec);
  if (check_flag(z, "N_VGetArrayPointer_SComplex", 0)) { return 1; }
  s1 = N_VGetArrayPointer_SComplex(ProbData->s1);
  if (check_flag(s1, "N_VGetArrayPointer_SComplex", 0)) { return 1; }
  s2 = N_VGetArrayPointer_SComplex(ProbData->s2);
  if (check_flag(s2, "N_VGetArrayPointer_SComplex", 0)) { return 1; }
  N = ProbData->N;
  up = ProbData->up;
  low = ProbData->low;
  diag = N_VGetArrayPointer_SComplex(ProbData->d);
  if (check_flag(diag, "N_VGetArrayPointer_SComplex", 0)) { return 1; }

  /* perform product at the left domain boundary (note: v is zero at the boundary)*/
  z[0] = (diag[0] * v[0] * s2[0] - v[1] * s2[1] * up) / s1[0];

  /* iterate through interior of the domain, performing product */
  for (i = 1; i < N - 1; i++)
  {
    z[i] = (-v[i - 1] * s2[i - 1] * low + diag[i] * v[i] * s2[i] - v[i + 1] * s2[i + 1] * up) /
           s1[i];
  }

  /* perform product at the right domain boundary (note: v is zero at the boundary)*/
  z[N - 1] = (-v[N - 2] * s2[N - 2] * low + diag[N - 1] * v[N - 1] * s2[N - 1]) / s1[N - 1];

  /* return with success */
  return 0;
}

/* preconditioner setup -- nothing to do here since everything is already stored */
int PSetup(void* Data) { return 0; }

/* preconditioner solve */
int PSolve(void* Data, N_Vector r_vec, N_Vector z_vec, sunrealtype tol, int lr)
{
  /* local variables */
  suncomplextype *r, *z, *d;
  sunindextype i;
  UserData* ProbData;

  /* access user data structure and vector data */
  ProbData = (UserData*)Data;
  r        = N_VGetArrayPointer_SComplex(r_vec);
  if (check_flag(r, "N_VGetArrayPointer_SComplex", 0)) { return 1; }
  z = N_VGetArrayPointer_SComplex(z_vec);
  if (check_flag(z, "N_VGetArrayPointer_SComplex", 0)) { return 1; }
  d = N_VGetArrayPointer_SComplex(ProbData->d);
  if (check_flag(d, "N_VGetArrayPointer_SComplex", 0)) { return 1; }

  /* iterate through domain, performing Jacobi solve */
  for (i = 0; i < ProbData->N; i++) { z[i] = r[i] / d[i]; }

  /* return with success */
  return 0;
}

/* uniform random number generator */
static sunrealtype urand(void)
{
  return ((sunrealtype)rand() / (sunrealtype)RAND_MAX);
}

/* Check function return value based on "opt" input:
     0:  function allocates memory so check for NULL pointer
     1:  function returns a flag so check for flag != 0 */
static int check_flag(void* flagvalue, const char* funcname, int opt)
{
  int* errflag;

  /* Check if function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL)
  {
    fprintf(stderr, "\nERROR: %s() failed - returned NULL pointer\n\n", funcname);
    return 1;
  }

  /* Check if flag != 0 */
  if (opt == 1)
  {
    errflag = (int*)flagvalue;
    if (*errflag != 0)
    {
      fprintf(stderr, "\nERROR: %s() failed with flag = %d\n\n", funcname,
              *errflag);
      return 1;
    }
  }

  return 0;
}

int SUNCCompare(suncomplextype a, suncomplextype b, sunrealtype tol)
{
  return (cabs(a - b) > tol) ? (1) : (0);
}

/* ----------------------------------------------------------------------
 * Implementation-specific 'check' routines
 * --------------------------------------------------------------------*/
int check_vector(N_Vector X, N_Vector Y, sunrealtype tol)
{
  int failure = 0;
  sunindextype i;
  suncomplextype *Xdata, *Ydata;
  sunrealtype maxerr;

  Xdata = N_VGetArrayPointer_SComplex(X);
  Ydata = N_VGetArrayPointer_SComplex(Y);

  /* check vector data */
  for (i = 0; i < problem_size; i++)
  {
    failure += SUNCCompare(Xdata[i], Ydata[i], tol);
  }

  if (failure > ZERO)
  {
    maxerr = ZERO;
    for (i = 0; i < problem_size; i++)
    {
      maxerr = SUNMAX(SUNCabs(Xdata[i] - Ydata[i]) / SUNCabs(Xdata[i]), maxerr);
    }
    printf("check err failure: maxerr = %" GSYM " (tol = %" GSYM ")\n", maxerr,
           tol);
    return (1);
  }
  else { return (0); }
}
