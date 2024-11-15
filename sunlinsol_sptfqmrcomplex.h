/* -----------------------------------------------------------------
 * Programmer(s): Mustafa Aggul @ SMU
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
 * This is the implementation file for a complex-valued SUNLinearSolver
 * (note that this is just SUNDIALS' SPTFQMR-Complex solver).
 * -----------------------------------------------------------------*/

#ifndef _CS_SUNLINSOL_H
#define _CS_SUNLINSOL_H

#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include "nvector_serialcomplex.h"
#include <sundials/sundials_math.h>

#ifdef __cplusplus /* wrapper to enable C++ usage */
extern "C" {
#endif

/* Default solver parameters */
#define CS_MAXL_DEFAULT 5

/* ------------------------------------------
 * Custom Implementation of SUNLinearSolver
 * ------------------------------------------ */

struct _CSSUNLinearSolverContent
{
  int maxl;
  int pretype;
  sunbooleantype zeroguess;
  int numiters;
  sunrealtype resnorm;
  int last_flag;

  SUNATimesFn ATimes;
  void* ATData;
  SUNPSetupFn Psetup;
  SUNPSolveFn Psolve;
  void* PData;

  N_Vector s1;
  N_Vector s2;
  N_Vector r_star;
  N_Vector q;
  N_Vector d;
  N_Vector v;
  N_Vector p;
  N_Vector* r;
  N_Vector u;
  N_Vector vtemp1;
  N_Vector vtemp2;
  N_Vector vtemp3;
};

typedef struct _CSSUNLinearSolverContent* CSSUNLinearSolverContent;

/* -------------------------------------
 * Exported Functions
 * -------------------------------------- */

SUNDIALS_EXPORT SUNLinearSolver SUNLinSol_SComplex(N_Vector y, int pretype,
                                               int maxl, SUNContext sunctx);
SUNDIALS_EXPORT SUNErrCode SUNLinSolSetPrecType_SComplex(SUNLinearSolver S,
                                                     int pretype);
SUNDIALS_EXPORT SUNErrCode SUNLinSolSetMaxl_SComplex(SUNLinearSolver S, int maxl);
SUNDIALS_EXPORT SUNLinearSolver_Type SUNLinSolGetType_SComplex(SUNLinearSolver S);
SUNDIALS_EXPORT SUNLinearSolver_ID SUNLinSolGetID_SComplex(SUNLinearSolver S);
SUNDIALS_EXPORT SUNErrCode SUNLinSolInitialize_SComplex(SUNLinearSolver S);
SUNDIALS_EXPORT SUNErrCode SUNLinSolSetATimes_SComplex(SUNLinearSolver S,
                                                   void* A_data,
                                                   SUNATimesFn ATimes);
SUNDIALS_EXPORT SUNErrCode SUNLinSolSetPreconditioner_SComplex(SUNLinearSolver S,
                                                           void* P_data,
                                                           SUNPSetupFn Pset,
                                                           SUNPSolveFn Psol);
SUNDIALS_EXPORT SUNErrCode SUNLinSolSetScalingVectors_SComplex(SUNLinearSolver S,
                                                           N_Vector s1,
                                                           N_Vector s2);
SUNDIALS_EXPORT SUNErrCode SUNLinSolSetZeroGuess_SComplex(SUNLinearSolver S,
                                                      sunbooleantype onoff);
SUNDIALS_EXPORT int SUNLinSolSetup_SComplex(SUNLinearSolver S, SUNMatrix A);
SUNDIALS_EXPORT int SUNLinSolSolve_SComplex(SUNLinearSolver S, SUNMatrix A,
                                        N_Vector x, N_Vector b,
                                        sunrealtype tol);
SUNDIALS_EXPORT int SUNLinSolNumIters_SComplex(SUNLinearSolver S);
SUNDIALS_EXPORT sunrealtype SUNLinSolResNorm_SComplex(SUNLinearSolver S);
SUNDIALS_EXPORT N_Vector SUNLinSolResid_SComplex(SUNLinearSolver S);
SUNDIALS_EXPORT sunindextype SUNLinSolLastFlag_SComplex(SUNLinearSolver S);
SUNDIALS_EXPORT SUNErrCode SUNLinSolSpace_SComplex(SUNLinearSolver S,
                                               long int* lenrwLS,
                                               long int* leniwLS);
SUNDIALS_EXPORT SUNErrCode SUNLinSolFree_SComplex(SUNLinearSolver S);

#ifdef __cplusplus
}
#endif

#endif

/*---- end of file ----*/
