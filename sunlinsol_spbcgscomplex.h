/*
 * -----------------------------------------------------------------
 * Programmer(s): Daniel Reynolds @ SMU
 * Edited by Sylvia Amihere @ SMU
 * Based on code sundials_spbcgs.h by: Peter Brown and
 *     Aaron Collier @ LLNL
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
 * -----------------------------------------------------------------------
 * This is the header file for the complex-valued SPBCGS implementation 
 * of the SUNLINSOL module, SUNLINSOL_SPBCGSComplex. The SPBCGS algorithm 
 * is based on the Scaled Preconditioned Bi-CG-Stabilized method.
 *
 * Note:
 *   - The definition of the generic SUNLinearSolver structure can
 *     be found in the header file sundials_linearsolver.h.
 * ----------------------------------------------------------------------
 */

#ifndef _SUNLINSOL_SPBCGSComplex_H 
#define _SUNLINSOL_SPBCGSComplex_H

#include <stdio.h>
#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include "nvector_serialcomplex.h"
#include <sundials/sundials_math.h>

#ifdef __cplusplus /* wrapper to enable C++ usage */
extern "C" {
#endif

/* Default SPBCGS solver parameters */
#define SUNSPBCGSComplex_MAXL_DEFAULT 5 

/* --------------------------------------------------------
 * Complex-valued SPBCGS Implementation of SUNLinearSolver
 * -------------------------------------------------------- */

struct _SUNLinearSolverContent_SPBCGSComplex
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
  N_Vector r;
  N_Vector r_star;
  N_Vector p;
  N_Vector q;
  N_Vector u;
  N_Vector Ap;
  N_Vector vtemp;
};

typedef struct _SUNLinearSolverContent_SPBCGSComplex* SUNLinearSolverContent_SPBCGSComplex;

/* ---------------------------------------
 *Exported Functions for SUNLINSOL_SPBCGS
 * --------------------------------------- */

SUNDIALS_EXPORT SUNLinearSolver SUNLinSol_SPBCGSComplex(N_Vector y, int pretype,
                                                 int maxl, SUNContext sunctx);
SUNDIALS_EXPORT SUNErrCode SUNLinSol_SPBCGSComplex_SetPrecType(SUNLinearSolver S,
                                                       int pretype);
SUNDIALS_EXPORT SUNErrCode SUNLinSol_SPBCGSComplex_SetMaxl(SUNLinearSolver S, int maxl);
SUNDIALS_EXPORT SUNLinearSolver_Type SUNLinSolGetType_SPBCGSComplex(SUNLinearSolver S);
SUNDIALS_EXPORT SUNLinearSolver_ID SUNLinSolGetID_SPBCGSComplex(SUNLinearSolver S);
SUNDIALS_EXPORT SUNErrCode SUNLinSolInitialize_SPBCGSComplex(SUNLinearSolver S);
SUNDIALS_EXPORT SUNErrCode SUNLinSolSetATimes_SPBCGSComplex(SUNLinearSolver S,
                                                     void* A_data,
                                                     SUNATimesFn ATimes);
SUNDIALS_EXPORT SUNErrCode SUNLinSolSetPreconditioner_SPBCGSComplex(SUNLinearSolver S,
                                                             void* P_data,
                                                             SUNPSetupFn Pset,
                                                             SUNPSolveFn Psol);
SUNDIALS_EXPORT SUNErrCode SUNLinSolSetScalingVectors_SPBCGSComplex(SUNLinearSolver S,
                                                             N_Vector s1,
                                                             N_Vector s2);
SUNDIALS_EXPORT SUNErrCode SUNLinSolSetZeroGuess_SPBCGSComplex(SUNLinearSolver S,
                                                        sunbooleantype onoff);
SUNDIALS_EXPORT int SUNLinSolSetup_SPBCGSComplex(SUNLinearSolver S, SUNMatrix A);
SUNDIALS_EXPORT int SUNLinSolSolve_SPBCGSComplex(SUNLinearSolver S, SUNMatrix A,
                                          N_Vector x, N_Vector b,
                                          sunrealtype tol);
SUNDIALS_EXPORT int SUNLinSolNumIters_SPBCGSComplex(SUNLinearSolver S);
SUNDIALS_EXPORT sunrealtype SUNLinSolResNorm_SPBCGSComplex(SUNLinearSolver S);
SUNDIALS_EXPORT N_Vector SUNLinSolResid_SPBCGSComplex(SUNLinearSolver S);
SUNDIALS_EXPORT sunindextype SUNLinSolLastFlag_SPBCGSComplex(SUNLinearSolver S);
SUNDIALS_EXPORT SUNErrCode SUNLinSolSpace_SPBCGSComplex(SUNLinearSolver S,
                                                 long int* lenrwLS,
                                                 long int* leniwLS);
SUNDIALS_EXPORT SUNErrCode SUNLinSolFree_SPBCGSComplex(SUNLinearSolver S);

#ifdef __cplusplus
}
#endif

#endif
