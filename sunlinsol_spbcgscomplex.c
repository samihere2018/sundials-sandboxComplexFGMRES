/* -----------------------------------------------------------------
 * Programmer(s): Daniel Reynolds @ SMU
 * Edited by Sylvia Amihere @ SMU
 * Based on sundials_spbcgs.c code, written by Peter Brown and
 *                Aaron Collier @ LLNL
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
 * This is the implementation file for the SPBCGS implementation of
 * the SUNLINSOL package.
 * -----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>

// #include <sundials/priv/sundials_errors_impl.h>
// #include <sundials/sundials_math.h>
#include "sunlinsol_spbcgscomplex.h"

// #include "sundials/sundials_errors.h"
// #include "sundials_logger_impl.h"
// #include "sundials_macros.h"

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)

/*
 * -----------------------------------------------------------------
 * SPBCGS solver structure accessibility macros:
 * -----------------------------------------------------------------
 */

#define SPBCGSComplex_CONTENT(S) ((SUNLinearSolverContent_SPBCGSComplex)(S->content))
#define PRETYPE(S)        (SPBCGSComplex_CONTENT(S)->pretype)
#define LASTFLAG(S)       (SPBCGSComplex_CONTENT(S)->last_flag)

/*
 * -----------------------------------------------------------------
 * exported functions
 * -----------------------------------------------------------------
 */

/* ----------------------------------------------------------------------------
 * Function to create a new SPBCGS linear solver
 */

SUNLinearSolver SUNLinSol_SPBCGSComplex(N_Vector y, int pretype, int maxl,
                                 SUNContext sunctx)
{
  // SUNFunctionBegin(sunctx);
  SUNLinearSolver S;
  SUNLinearSolverContent_SPBCGSComplex content;

  /* check for legal pretype and maxl values; if illegal use defaults */
  if ((pretype != SUN_PREC_NONE) && (pretype != SUN_PREC_LEFT) &&
      (pretype != SUN_PREC_RIGHT) && (pretype != SUN_PREC_BOTH))
  {
    pretype = SUN_PREC_NONE;
  }
  if (maxl <= 0) { maxl = SUNSPBCGSComplex_MAXL_DEFAULT; }

  // /* check that the supplied N_Vector supports all requisite operations */
  // SUNAssertNull((y->ops->nvclone) && (y->ops->nvdestroy) &&
  //                 (y->ops->nvlinearsum) && (y->ops->nvprod) &&
  //                 (y->ops->nvdiv) && (y->ops->nvscale) && (y->ops->nvdotprod),
  //               SUN_ERR_ARG_INCOMPATIBLE);

  /* Create linear solver */
  S = NULL;
  S = SUNLinSolNewEmpty(sunctx);
  // SUNCheckLastErrNull();

  /* Attach operations */
  S->ops->gettype           = SUNLinSolGetType_SPBCGSComplex;
  S->ops->getid             = SUNLinSolGetID_SPBCGSComplex;
  S->ops->setatimes         = SUNLinSolSetATimes_SPBCGSComplex;
  S->ops->setpreconditioner = SUNLinSolSetPreconditioner_SPBCGSComplex;
  S->ops->setscalingvectors = SUNLinSolSetScalingVectors_SPBCGSComplex;
  S->ops->setzeroguess      = SUNLinSolSetZeroGuess_SPBCGSComplex;
  S->ops->initialize        = SUNLinSolInitialize_SPBCGSComplex;
  S->ops->setup             = SUNLinSolSetup_SPBCGSComplex;
  S->ops->solve             = SUNLinSolSolve_SPBCGSComplex;
  S->ops->numiters          = SUNLinSolNumIters_SPBCGSComplex;
  S->ops->resnorm           = SUNLinSolResNorm_SPBCGSComplex;
  S->ops->resid             = SUNLinSolResid_SPBCGSComplex;
  S->ops->lastflag          = SUNLinSolLastFlag_SPBCGSComplex;
  S->ops->space             = SUNLinSolSpace_SPBCGSComplex;
  S->ops->free              = SUNLinSolFree_SPBCGSComplex;

  /* Create content */
  content = NULL;
  content = (SUNLinearSolverContent_SPBCGSComplex)malloc(sizeof *content);
  // SUNAssertNull(content, SUN_ERR_MALLOC_FAIL);

  /* Attach content */
  S->content = content;

  /* Fill content */
  content->last_flag = 0;
  content->maxl      = maxl;
  content->pretype   = pretype;
  content->zeroguess = SUNFALSE;
  content->numiters  = 0;
  content->resnorm   = ZERO;
  content->r_star    = NULL;
  content->r         = NULL;
  content->p         = NULL;
  content->q         = NULL;
  content->u         = NULL;
  content->Ap        = NULL;
  content->vtemp     = NULL;
  content->s1        = NULL;
  content->s2        = NULL;
  content->ATimes    = NULL;
  content->ATData    = NULL;
  content->Psetup    = NULL;
  content->Psolve    = NULL;
  content->PData     = NULL;

  /* Allocate content */
  content->r_star = N_VClone_SComplex(y);
  // SUNCheckLastErrNull();

  content->r = N_VClone_SComplex(y);
  // SUNCheckLastErrNull();

  content->p = N_VClone_SComplex(y);
  // SUNCheckLastErrNull();

  content->q = N_VClone_SComplex(y);
  // SUNCheckLastErrNull();

  content->u = N_VClone_SComplex(y);
  // SUNCheckLastErrNull();

  content->Ap = N_VClone_SComplex(y);
  // SUNCheckLastErrNull();

  content->vtemp = N_VClone_SComplex(y);
  // SUNCheckLastErrNull();

  return (S);
}

/* ----------------------------------------------------------------------------
 * Function to set the type of preconditioning for SPBCGS to use
 */

SUNErrCode SUNLinSol_SPBCGSComplex_SetPrecType(SUNLinearSolver S, int pretype)
{
  // SUNFunctionBegin(S->sunctx);
  // /* Check for legal pretype */
  // SUNAssert((pretype == SUN_PREC_NONE) || (pretype == SUN_PREC_LEFT) ||
  //             (pretype == SUN_PREC_RIGHT) || (pretype == SUN_PREC_BOTH),
  //           SUN_ERR_ARG_CORRUPT);

  /* Set pretype */
  PRETYPE(S) = pretype;
  return SUN_SUCCESS;
}

/* ----------------------------------------------------------------------------
 * Function to set the maximum number of iterations for SPBCGS to use
 */

SUNErrCode SUNLinSol_SPBCGSComplex_SetMaxl(SUNLinearSolver S, int maxl)
{
  // SUNFunctionBegin(S->sunctx);

  /* Check for legal pretype */
  if (maxl <= 0) { maxl = SUNSPBCGSComplex_MAXL_DEFAULT; }

  /* Set pretype */
  SPBCGSComplex_CONTENT(S)->maxl = maxl;
  return SUN_SUCCESS;
}

/*
 * -----------------------------------------------------------------
 * implementation of linear solver operations
 * -----------------------------------------------------------------
 */

SUNLinearSolver_Type SUNLinSolGetType_SPBCGSComplex( SUNLinearSolver S)
{
  return (SUNLINEARSOLVER_ITERATIVE);
}

SUNLinearSolver_ID SUNLinSolGetID_SPBCGSComplex(SUNLinearSolver S)
{
  return (SUNLINEARSOLVER_SPBCGS);
}

SUNErrCode SUNLinSolInitialize_SPBCGSComplex(SUNLinearSolver S)
{
  // SUNFunctionBegin(S->sunctx);

  if (SPBCGSComplex_CONTENT(S)->maxl <= 0)
  {
    SPBCGSComplex_CONTENT(S)->maxl = SUNSPBCGSComplex_MAXL_DEFAULT;
  }

  // SUNAssert(SPBCGSComplex_CONTENT(S)->ATimes, SUN_ERR_ARG_CORRUPT);

  if ((PRETYPE(S) != SUN_PREC_LEFT) && (PRETYPE(S) != SUN_PREC_RIGHT) &&
      (PRETYPE(S) != SUN_PREC_BOTH))
  {
    PRETYPE(S) = SUN_PREC_NONE;
  }

  // SUNAssert((PRETYPE(S) == SUN_PREC_NONE) || SPBCGS_CONTENT(S)->Psolve,
  //           SUN_ERR_ARG_CORRUPT);

  /* no additional memory to allocate */

  /* return with success */
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetATimes_SPBCGSComplex(SUNLinearSolver S, void* ATData,
                                     SUNATimesFn ATimes)
{
  // SUNFunctionBegin(S->sunctx);
  /* set function pointers to integrator-supplied ATimes routine
     and data, and return with success */
  SPBCGSComplex_CONTENT(S)->ATimes = ATimes;
  SPBCGSComplex_CONTENT(S)->ATData = ATData;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetPreconditioner_SPBCGSComplex(SUNLinearSolver S, void* PData,
                                             SUNPSetupFn Psetup,
                                             SUNPSolveFn Psolve)
{
  // SUNFunctionBegin(S->sunctx);
  /* set function pointers to integrator-supplied Psetup and PSolve
     routines and data, and return with success */
  SPBCGSComplex_CONTENT(S)->Psetup = Psetup;
  SPBCGSComplex_CONTENT(S)->Psolve = Psolve;
  SPBCGSComplex_CONTENT(S)->PData  = PData;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetScalingVectors_SPBCGSComplex(SUNLinearSolver S, N_Vector s1,
                                             N_Vector s2)
{
  // SUNFunctionBegin(S->sunctx);
  /* set N_Vector pointers to integrator-supplied scaling vectors,
     and return with success */
  SPBCGSComplex_CONTENT(S)->s1 = s1;
  SPBCGSComplex_CONTENT(S)->s2 = s2;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetZeroGuess_SPBCGSComplex(SUNLinearSolver S, sunbooleantype onoff)
{
  // SUNFunctionBegin(S->sunctx);
  /* set flag indicating a zero initial guess */
  SPBCGSComplex_CONTENT(S)->zeroguess = onoff;
  return SUN_SUCCESS;
}

int SUNLinSolSetup_SPBCGSComplex(SUNLinearSolver S, SUNMatrix A)
{
  // SUNFunctionBegin(S->sunctx);

  int status;
  SUNPSetupFn Psetup;
  void* PData;

  /* Set shortcuts to SPBCGS memory structures */
  Psetup = SPBCGSComplex_CONTENT(S)->Psetup;
  PData  = SPBCGSComplex_CONTENT(S)->PData;

  /* no solver-specific setup is required, but if user-supplied
     Psetup routine exists, call that here */
  if (Psetup != NULL)
  {
    status = Psetup(PData);
    if (status != 0)
    {
      LASTFLAG(S) = (status < 0) ? SUNLS_PSET_FAIL_UNREC : SUNLS_PSET_FAIL_REC;
      return (LASTFLAG(S));
    }
  }

  /* return with success */
  LASTFLAG(S) = SUN_SUCCESS;
  return (LASTFLAG(S));
}

int SUNLinSolSolve_SPBCGSComplex(SUNLinearSolver S, SUNMatrix A,
                          N_Vector x, N_Vector b, sunrealtype delta)
{
  // SUNFunctionBegin(S->sunctx);

  /* local data and shortcut variables */
  suncomplextype alpha, beta, omega, beta_denom, beta_num;
  sunrealtype omega_denom, r_norm, rho;
  N_Vector r_star, r, p, q, u, Ap, vtemp;
  sunbooleantype preOnLeft, preOnRight, scale_x, scale_b, converged;
  sunbooleantype* zeroguess;
  int l, l_max;
  void *A_data, *P_data;
  N_Vector sx, sb;
  SUNATimesFn atimes;
  SUNPSolveFn psolve;
  sunrealtype* res_norm;
  int* nli;
  int status;

  /* local variables for fused vector operations */
  suncomplextype cv[3];
  N_Vector Xv[3];

  /* Make local shortcuts to solver variables. */
  l_max     = SPBCGSComplex_CONTENT(S)->maxl;
  r_star    = SPBCGSComplex_CONTENT(S)->r_star;
  r         = SPBCGSComplex_CONTENT(S)->r;
  p         = SPBCGSComplex_CONTENT(S)->p;
  q         = SPBCGSComplex_CONTENT(S)->q;
  u         = SPBCGSComplex_CONTENT(S)->u;
  Ap        = SPBCGSComplex_CONTENT(S)->Ap;
  vtemp     = SPBCGSComplex_CONTENT(S)->vtemp;
  sb        = SPBCGSComplex_CONTENT(S)->s1;
  sx        = SPBCGSComplex_CONTENT(S)->s2;
  A_data    = SPBCGSComplex_CONTENT(S)->ATData;
  P_data    = SPBCGSComplex_CONTENT(S)->PData;
  atimes    = SPBCGSComplex_CONTENT(S)->ATimes;
  psolve    = SPBCGSComplex_CONTENT(S)->Psolve;
  zeroguess = &(SPBCGSComplex_CONTENT(S)->zeroguess);
  nli       = &(SPBCGSComplex_CONTENT(S)->numiters);
  res_norm  = &(SPBCGSComplex_CONTENT(S)->resnorm);

  /* Initialize counters and convergence flag */
  *nli      = 0;
  converged = SUNFALSE;

  /* set sunbooleantype flags for internal solver options */
  preOnLeft = ((PRETYPE(S) == SUN_PREC_LEFT) || (PRETYPE(S) == SUN_PREC_BOTH));
  preOnRight = ((PRETYPE(S) == SUN_PREC_RIGHT) || (PRETYPE(S) == SUN_PREC_BOTH));
  scale_x = (sx != NULL);
  scale_b = (sb != NULL);

  /* Check for unsupported use case */
  if (preOnRight && !(*zeroguess))
  {
    *zeroguess  = SUNFALSE;
    LASTFLAG(S) = SUN_ERR_ARG_INCOMPATIBLE;
    return SUN_ERR_ARG_INCOMPATIBLE;
  }

  /* Check if Atimes function has been set */
  // SUNAssert(atimes, SUN_ERR_ARG_CORRUPT);

  /* If preconditioning, check if psolve has been set */
  // SUNAssert(!(preOnLeft || preOnRight) || psolve, SUN_ERR_ARG_CORRUPT);

  /* Set r_star to initial (unscaled) residual r_0 = b - A*x_0 */

  if (*zeroguess)
  {
    N_VScale_SComplex(ONE, b, r_star);
    // SUNCheckLastErr();
  }
  else
  {
    status = atimes(A_data, x, r_star);
    if (status != 0)
    {
      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = (status < 0) ? SUNLS_ATIMES_FAIL_UNREC
                                 : SUNLS_ATIMES_FAIL_REC;
      return (LASTFLAG(S));
    }
    N_VLinearSum(ONE, b, -ONE, r_star, r_star);
    // SUNCheckLastErr();
  }

  /* Apply left preconditioner and b-scaling to r_star = r_0 */

  if (preOnLeft)
  {
    status = psolve(P_data, r_star, r, delta, SUN_PREC_LEFT);
    if (status != 0)
    {
      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                 : SUNLS_PSOLVE_FAIL_REC;
      return (LASTFLAG(S));
    }
  }
  else
  {
    N_VScale_SComplex(ONE, r_star, r);
    // SUNCheckLastErr();
  }

  if (scale_b)
  {
    N_VProd_SComplex(sb, r, r_star);
    // SUNCheckLastErr();
  }
  else
  {
    N_VScale_SComplex(ONE, r, r_star);
    // SUNCheckLastErr();
  }

  /* Initialize beta_denom to the dot product of r0 with r0 */

  beta_denom = N_VDotProd_SComplex(r_star, r_star);
  // SUNCheckLastErr();

  /* Set r_norm to L2 norm of r_star = sb P1_inv r_0, and
     return if small */

  // *res_norm = r_norm = rho = SUNRsqrt(beta_denom); 
  *res_norm = r_norm = rho = SUNRsqrt((sunrealtype)N_VDotProd_SComplex(r_star, r_star));

// #if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_INFO
//   SUNLogger_QueueMsg(S->sunctx->logger, SUN_LOGLEVEL_INFO,
//                      "SUNLinSolSolve_SPBCGS", "initial-residual",
//                      "nli = %li, resnorm = %.16g", (long int)0, *res_norm);
// #endif

  if (r_norm <= delta)
  {
    *zeroguess  = SUNFALSE;
    LASTFLAG(S) = SUN_SUCCESS;
    return (LASTFLAG(S));
  }

  /* Copy r_star to r and p */

  N_VScale_SComplex(ONE, r_star, r);
  // SUNCheckLastErr();
  N_VScale_SComplex(ONE, r_star, p);
  // SUNCheckLastErr();

  /* Set x = sx x if non-zero guess */
  if (scale_x && !(*zeroguess))
  {
    N_VProd_SComplex(sx, x, x);
    // SUNCheckLastErr();
  }

  /* Begin main iteration loop */

  for (l = 0; l < l_max; l++)
  {
    (*nli)++;

    /* Generate Ap = A-tilde p, where A-tilde = sb P1_inv A P2_inv sx_inv */

    /*   Apply x-scaling: vtemp = sx_inv p */

    if (scale_x)
    {
      N_VDiv_SComplex(p, sx, vtemp);
      // SUNCheckLastErr();
    }
    else
    {
      N_VScale_SComplex(ONE, p, vtemp);
      // SUNCheckLastErr();
    }

    /*   Apply right preconditioner: vtemp = P2_inv sx_inv p */

    if (preOnRight)
    {
      N_VScale_SComplex(ONE, vtemp, Ap);
      // SUNCheckLastErr();
      status = psolve(P_data, Ap, vtemp, delta, SUN_PREC_RIGHT);
      if (status != 0)
      {
        *zeroguess  = SUNFALSE;
        LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                   : SUNLS_PSOLVE_FAIL_REC;
        return (LASTFLAG(S));
      }
    }


    /*   Apply A: Ap = A P2_inv sx_inv p */

    status = atimes(A_data, vtemp, Ap);
    if (status != 0)
    {
      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = (status < 0) ? SUNLS_ATIMES_FAIL_UNREC
                                 : SUNLS_ATIMES_FAIL_REC;
      return (LASTFLAG(S));
    }


    /*   Apply left preconditioner: vtemp = P1_inv A P2_inv sx_inv p */

    if (preOnLeft)
    {
      status = psolve(P_data, Ap, vtemp, delta, SUN_PREC_LEFT);
      if (status != 0)
      {
        *zeroguess  = SUNFALSE;
        LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                   : SUNLS_PSOLVE_FAIL_REC;
        return (LASTFLAG(S));
      }
    }
    else
    {
      N_VScale_SComplex(ONE, Ap, vtemp);
      // SUNCheckLastErr();
    }


    /*   Apply b-scaling: Ap = sb P1_inv A P2_inv sx_inv p */

    if (scale_b)
    {
      N_VProd_SComplex(sb, vtemp, Ap);
      // SUNCheckLastErr();
    }
    else
    {
      N_VScale_SComplex(ONE, vtemp, Ap);
      // SUNCheckLastErr();
    }

    /* Calculate alpha = <r,r_star>/<Ap,r_star> */

    // AMIHERE: ORIGINALLY, THIS SHOULD BE dot(r_star, Ap) BUT THE DEFNITION OF INNER PRODUCT IS THE OPPOSITE IN MATLAB. 
    // HENCE, IN MATLAB dot(r_star, Ap) WORKS BUT NOT dot(Ap, r_star).
    alpha = N_VDotProd_SComplex(r_star, Ap); 
    // SUNCheckLastErr();
    alpha = beta_denom / alpha;

    /* Update q = r - alpha*Ap = r - alpha*(sb P1_inv A P2_inv sx_inv p) */

    N_VLinearSum_SComplex(ONE, r, -alpha, Ap, q);
    // SUNCheckLastErr();

    /* Generate u = A-tilde q */

    /*   Apply x-scaling: vtemp = sx_inv q */

    if (scale_x)
    {
      N_VDiv_SComplex(q, sx, vtemp);
      // SUNCheckLastErr();
    }
    else
    {
      N_VScale_SComplex(ONE, q, vtemp);
      // SUNCheckLastErr();
    }



    /*   Apply right preconditioner: vtemp = P2_inv sx_inv q */

    if (preOnRight)
    {
      N_VScale_SComplex(ONE, vtemp, u);
      // SUNCheckLastErr();
      status = psolve(P_data, u, vtemp, delta, SUN_PREC_RIGHT);
      if (status != 0)
      {
        *zeroguess  = SUNFALSE;
        LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                   : SUNLS_PSOLVE_FAIL_REC;
        return (LASTFLAG(S));
      }
    }

    /*   Apply A: u = A P2_inv sx_inv u */

    status = atimes(A_data, vtemp, u);
    if (status != 0)
    {
      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = (status < 0) ? SUNLS_ATIMES_FAIL_UNREC
                                 : SUNLS_ATIMES_FAIL_REC;
      return (LASTFLAG(S));
    }


    /*   Apply left preconditioner: vtemp = P1_inv A P2_inv sx_inv p */

    if (preOnLeft)
    {
      status = psolve(P_data, u, vtemp, delta, SUN_PREC_LEFT);
      if (status != 0)
      {
        *zeroguess  = SUNFALSE;
        LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                   : SUNLS_PSOLVE_FAIL_REC;
        return (LASTFLAG(S));
      }
    }
    else
    {
      N_VScale_SComplex(ONE, u, vtemp);
      // SUNCheckLastErr();
    }

    /*   Apply b-scaling: u = sb P1_inv A P2_inv sx_inv u */

    if (scale_b)
    {
      N_VProd_SComplex(sb, vtemp, u);
      // SUNCheckLastErr();
    }
    else
    {
      N_VScale_SComplex(ONE, vtemp, u);
      // SUNCheckLastErr();
    }

    /* Calculate omega = <u,q>/<u,u> */

    omega_denom = N_VDotProd_SComplex(u, u);
    // SUNCheckLastErr();
    if (omega_denom == ZERO) { omega_denom = ONE; }

    // AMIHERE: ORIGINALLY, THIS SHOULD BE dot(u, q) BUT THE DEFNITION OF INNER PRODUCT IS THE OPPOSITE IN MATLAB. 
    // HENCE, IN MATLAB dot(u, q) WORKS BUT NOT dot(q, u).
    omega = N_VDotProd_SComplex( u, q);
    // SUNCheckLastErr();
    omega /= omega_denom;

    /* Update x = x + alpha*p + omega*q */
    if (l == 0 && *zeroguess)
    {
      N_VLinearSum_SComplex(alpha, p, omega, q, x);
      // SUNCheckLastErr();
    }
    else
    {
      cv[0] = ONE;
      Xv[0] = x;

      cv[1] = alpha;
      Xv[1] = p;

      cv[2] = omega;
      Xv[2] = q;

      // SUNCheckCall(N_VLinearCombination_SComplex(3, cv, Xv, x));
      N_VLinearCombination_SComplex(3, cv, Xv, x);
    }

    /* Update the residual r = q - omega*u */

    N_VLinearSum_SComplex(ONE, q, -omega, u, r);
    // SUNCheckLastErr();

    /* Set rho = norm(r) and check convergence */

    *res_norm = rho = SUNRsqrt((sunrealtype)N_VDotProd_SComplex(r, r));
    // SUNCheckLastErr();

// #if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_INFO
//     SUNLogger_QueueMsg(S->sunctx->logger, SUN_LOGLEVEL_INFO,
//                        "SUNLinSolSolve_SPBCGS", "iterate-residual",
//                        "nli = %li, resnorm = %.16g", (long int)0, *res_norm);
// #endif

    if (rho <= delta)
    {
      converged = SUNTRUE;
      break;
    }

    /* Not yet converged, continue iteration */
    /* Update beta = <rnew,r_star> / <rold,r_start> * alpha / omega */


    // AMIHERE: ORIGINALLY, THIS SHOULD BE dot(r_star, r) BUT THE DEFNITION OF INNER PRODUCT IS THE OPPOSITE IN MATLAB. 
    // HENCE, IN MATLAB dot(r_star, r) WORKS BUT NOT dot(r, r_star).
    beta_num = N_VDotProd_SComplex(r_star, r);
    // SUNCheckLastErr();
    beta = ((beta_num / beta_denom) * (alpha / omega));

    /* Update p = r + beta*(p - omega*Ap) = beta*p - beta*omega*Ap + r */
    cv[0] = beta;
    Xv[0] = p;

    cv[1] = -alpha * (beta_num / beta_denom);
    //  cv[1] =  -beta * (omega); 
    Xv[1] = Ap;

    cv[2] = ONE;
    Xv[2] = r;

    // SUNCheckCall(N_VLinearCombination_SComplex(3, cv, Xv, p));
    N_VLinearCombination_SComplex(3, cv, Xv, p);


    /* update beta_denom for next iteration */
    beta_denom = beta_num;
  }

  /* Main loop finished */

  if ((converged == SUNTRUE) || (rho < r_norm))
  {
    /* Apply the x-scaling and right preconditioner: x = P2_inv sx_inv x */

    if (scale_x)
    {
      N_VDiv_SComplex(x, sx, x);
      // SUNCheckLastErr();
    }
    if (preOnRight)
    {
      status = psolve(P_data, x, vtemp, delta, SUN_PREC_RIGHT);
      if (status != 0)
      {
        *zeroguess  = SUNFALSE;
        LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                   : SUNLS_PSOLVE_FAIL_REC;
        return (LASTFLAG(S));
      }
      N_VScale_SComplex(ONE, vtemp, x);
      // SUNCheckLastErr();
    }

    *zeroguess = SUNFALSE;
    if (converged == SUNTRUE) { LASTFLAG(S) = SUN_SUCCESS; }
    else { LASTFLAG(S) = SUNLS_RES_REDUCED; }
    return (LASTFLAG(S));
  }
  else
  {
    *zeroguess  = SUNFALSE;
    LASTFLAG(S) = SUNLS_CONV_FAIL;
    return (LASTFLAG(S));
  }
}

int SUNLinSolNumIters_SPBCGSComplex(SUNLinearSolver S)
{
  /* return the stored 'numiters' value */
  return (SPBCGSComplex_CONTENT(S)->numiters);
}

sunrealtype SUNLinSolResNorm_SPBCGSComplex(SUNLinearSolver S)
{
  /* return the stored 'resnorm' value */
  return (SPBCGSComplex_CONTENT(S)->resnorm);
}

N_Vector SUNLinSolResid_SPBCGSComplex(SUNLinearSolver S)
{
  /* return the stored 'r' vector */
  return (SPBCGSComplex_CONTENT(S)->r);
}

sunindextype SUNLinSolLastFlag_SPBCGSComplex(SUNLinearSolver S)
{
  /* return the stored 'last_flag' value */
  return (LASTFLAG(S));
}

SUNErrCode SUNLinSolSpace_SPBCGSComplex(SUNLinearSolver S, long int* lenrwLS,
                                 long int* leniwLS)
{
  // SUNFunctionBegin(S->sunctx);
  sunindextype liw1, lrw1;
  if (SPBCGSComplex_CONTENT(S)->vtemp->ops->nvspace)
  {
    N_VSpace(SPBCGSComplex_CONTENT(S)->vtemp, &lrw1, &liw1);
    // SUNCheckLastErr();
  }
  else { lrw1 = liw1 = 0; }
  *lenrwLS = lrw1 * 9;
  *leniwLS = liw1 * 9;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolFree_SPBCGSComplex(SUNLinearSolver S)
{
  if (S->content)
  {
    /* delete items from within the content structure */
    if (SPBCGSComplex_CONTENT(S)->r_star)
    {
      N_VDestroy_SComplex(SPBCGSComplex_CONTENT(S)->r_star);
      SPBCGSComplex_CONTENT(S)->r_star = NULL;
    }
    if (SPBCGSComplex_CONTENT(S)->r)
    {
      N_VDestroy_SComplex(SPBCGSComplex_CONTENT(S)->r);
      SPBCGSComplex_CONTENT(S)->r = NULL;
    }
    if (SPBCGSComplex_CONTENT(S)->p)
    {
      N_VDestroy_SComplex(SPBCGSComplex_CONTENT(S)->p);
      SPBCGSComplex_CONTENT(S)->p = NULL;
    }
    if (SPBCGSComplex_CONTENT(S)->q)
    {
      N_VDestroy_SComplex(SPBCGSComplex_CONTENT(S)->q);
      SPBCGSComplex_CONTENT(S)->q = NULL;
    }
    if (SPBCGSComplex_CONTENT(S)->u)
    {
      N_VDestroy_SComplex(SPBCGSComplex_CONTENT(S)->u);
      SPBCGSComplex_CONTENT(S)->u = NULL;
    }
    if (SPBCGSComplex_CONTENT(S)->Ap)
    {
      N_VDestroy_SComplex(SPBCGSComplex_CONTENT(S)->Ap);
      SPBCGSComplex_CONTENT(S)->Ap = NULL;
    }
    if (SPBCGSComplex_CONTENT(S)->vtemp)
    {
      N_VDestroy_SComplex(SPBCGSComplex_CONTENT(S)->vtemp);
      SPBCGSComplex_CONTENT(S)->vtemp = NULL;
    }
    free(S->content);
    S->content = NULL;
  }
  if (S->ops)
  {
    free(S->ops);
    S->ops = NULL;
  }
  free(S);
  S = NULL;
  return SUN_SUCCESS;
}
