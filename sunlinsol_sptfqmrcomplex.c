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

#include <stdio.h>
#include <stdlib.h>
#include "sunlinsol_sptfqmrcomplex.h"

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)

/*
 * -----------------------------------------------------------------
 * solver structure accessibility macros:
 * -----------------------------------------------------------------
 */

#define CS_CONTENT(S) ((CSSUNLinearSolverContent)(S->content))
#define LASTFLAG(S)   (CS_CONTENT(S)->last_flag)

/*
 * -----------------------------------------------------------------
 * exported functions
 * -----------------------------------------------------------------
 */

/* ----------------------------------------------------------------------------
 * Function to create a new linear solver
 */

SUNLinearSolver SUNLinSol_SComplex(N_Vector y, int pretype, int maxl,
                               SUNContext sunctx)
{
  SUNLinearSolver S;
  CSSUNLinearSolverContent content;

  /* check for legal pretype and maxl values; if illegal use defaults */
  if ((pretype != SUN_PREC_NONE) && (pretype != SUN_PREC_LEFT) &&
      (pretype != SUN_PREC_RIGHT) && (pretype != SUN_PREC_BOTH))
  {
    pretype = SUN_PREC_NONE;
  }
  if (maxl <= 0) { maxl = CS_MAXL_DEFAULT; }

  /* Create linear solver */
  S = NULL;
  S = SUNLinSolNewEmpty(sunctx);
  if (S == NULL) { return NULL; }

  /* Attach operations */
  S->ops->gettype           = SUNLinSolGetType_SComplex;
  S->ops->getid             = SUNLinSolGetID_SComplex;
  S->ops->setatimes         = SUNLinSolSetATimes_SComplex;
  S->ops->setpreconditioner = SUNLinSolSetPreconditioner_SComplex;
  S->ops->setscalingvectors = SUNLinSolSetScalingVectors_SComplex;
  S->ops->setzeroguess      = SUNLinSolSetZeroGuess_SComplex;
  S->ops->initialize        = SUNLinSolInitialize_SComplex;
  S->ops->setup             = SUNLinSolSetup_SComplex;
  S->ops->solve             = SUNLinSolSolve_SComplex;
  S->ops->numiters          = SUNLinSolNumIters_SComplex;
  S->ops->resnorm           = SUNLinSolResNorm_SComplex;
  S->ops->resid             = SUNLinSolResid_SComplex;
  S->ops->lastflag          = SUNLinSolLastFlag_SComplex;
  S->ops->space             = SUNLinSolSpace_SComplex;
  S->ops->free              = SUNLinSolFree_SComplex;

  /* Create content */
  content = NULL;
  content = (CSSUNLinearSolverContent)malloc(sizeof *content);
  if (content == NULL) { return NULL; }

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
  content->q         = NULL;
  content->d         = NULL;
  content->v         = NULL;
  content->p         = NULL;
  content->r         = NULL;
  content->u         = NULL;
  content->vtemp1    = NULL;
  content->vtemp2    = NULL;
  content->vtemp3    = NULL;
  content->s1        = NULL;
  content->s2        = NULL;
  content->ATimes    = NULL;
  content->ATData    = NULL;
  content->Psetup    = NULL;
  content->Psolve    = NULL;
  content->PData     = NULL;

  /* Allocate content */
  content->r_star = N_VClone_SComplex(y);
  if (content->r_star == NULL) { return NULL; }
  content->q = N_VClone_SComplex(y);
  if (content->q == NULL) { return NULL; }
  content->d = N_VClone_SComplex(y);
  if (content->d == NULL) { return NULL; }
  content->v = N_VClone_SComplex(y);
  if (content->v == NULL) { return NULL; }
  content->p = N_VClone_SComplex(y);
  if (content->p == NULL) { return NULL; }
  content->r = N_VCloneVectorArray(2, y);
  if (content->r == NULL) { return NULL; }
  content->u = N_VClone_SComplex(y);
  if (content->u == NULL) { return NULL; }
  content->vtemp1 = N_VClone_SComplex(y);
  if (content->vtemp1 == NULL) { return NULL; }
  content->vtemp2 = N_VClone_SComplex(y);
  if (content->vtemp2 == NULL) { return NULL; }
  content->vtemp3 = N_VClone_SComplex(y);
  if (content->vtemp3 == NULL) { return NULL; }

  return (S);
}

/* ----------------------------------------------------------------------------
 * Function to set the type of preconditioning for SPTFQMR to use
 */

SUNErrCode SUNLinSolSetPrecType_SComplex(SUNLinearSolver S, int pretype)
{
  /* Set pretype */
  CS_CONTENT(S)->pretype = pretype;
  return SUN_SUCCESS;
}

/* ----------------------------------------------------------------------------
 * Function to set the maximum number of iterations for SPTFQMR to use
 */

SUNErrCode SUNLinSolSetMaxl_SComplex(SUNLinearSolver S, int maxl)
{
  /* Check for legal pretype */
  if (maxl <= 0) { maxl = CS_MAXL_DEFAULT; }

  /* Set pretype */
  CS_CONTENT(S)->maxl = maxl;
  return SUN_SUCCESS;
}

/*
 * -----------------------------------------------------------------
 * implementation of linear solver operations
 * -----------------------------------------------------------------
 */

SUNLinearSolver_Type SUNLinSolGetType_SComplex(SUNLinearSolver S)
{
  return (SUNLINEARSOLVER_ITERATIVE);
}

SUNLinearSolver_ID SUNLinSolGetID_SComplex(SUNLinearSolver S)
{
  return (SUNLINEARSOLVER_CUSTOM);
}

SUNErrCode SUNLinSolInitialize_SComplex(SUNLinearSolver S)
{
  CSSUNLinearSolverContent content;

  /* set shortcut to SPTFQMR memory structure */
  content = CS_CONTENT(S);

  /* ensure valid options */
  if (content->maxl <= 0) { content->maxl = CS_MAXL_DEFAULT; }
  if (content->ATimes == NULL) { return SUN_ERR_ARG_CORRUPT; }

  if ((content->pretype != SUN_PREC_LEFT) &&
      (content->pretype != SUN_PREC_RIGHT) && (content->pretype != SUN_PREC_BOTH))
  {
    content->pretype = SUN_PREC_NONE;
  }

  /* no additional memory to allocate */

  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetATimes_SComplex(SUNLinearSolver S, void* ATData,
                                   SUNATimesFn ATimes)
{
  /* set function pointers to integrator-supplied ATimes routine
     and data, and return with success */
  CS_CONTENT(S)->ATimes = ATimes;
  CS_CONTENT(S)->ATData = ATData;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetPreconditioner_SComplex(SUNLinearSolver S, void* PData,
                                           SUNPSetupFn Psetup,
                                           SUNPSolveFn Psolve)
{
  /* set function pointers to integrator-supplied Psetup and PSolve
     routines and data, and return with success */
  CS_CONTENT(S)->Psetup = Psetup;
  CS_CONTENT(S)->Psolve = Psolve;
  CS_CONTENT(S)->PData  = PData;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetScalingVectors_SComplex(SUNLinearSolver S, N_Vector s1,
                                           N_Vector s2)
{
  /* set N_Vector pointers to integrator-supplied scaling vectors,
     and return with success */
  CS_CONTENT(S)->s1 = s1;
  CS_CONTENT(S)->s2 = s2;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetZeroGuess_SComplex(SUNLinearSolver S, sunbooleantype onoff)
{
  /* set flag indicating a zero initial guess */
  CS_CONTENT(S)->zeroguess = onoff;
  return SUN_SUCCESS;
}

int SUNLinSolSetup_SComplex(SUNLinearSolver S, SUNMatrix A)
{
  int status = SUN_SUCCESS;
  SUNPSetupFn Psetup;
  void* PData;

  /* Set shortcuts to SPTFQMR memory structures */
  Psetup = CS_CONTENT(S)->Psetup;
  PData  = CS_CONTENT(S)->PData;

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
  return SUN_SUCCESS;
}

int SUNLinSolSolve_SComplex(SUNLinearSolver S, SUNMatrix A, N_Vector x,
                        N_Vector b, sunrealtype delta)
{
  /* local data and shortcut variables */
  sunrealtype tau, c, v_bar, omega;
  suncomplextype alpha, beta, eta, sigma;
  suncomplextype rho[2];

  sunrealtype r_init_norm, r_curr_norm;
  sunrealtype temp_val;
  sunbooleantype preOnLeft, preOnRight, scale_x, scale_b, converged, b_ok;
  sunbooleantype* zeroguess;
  int n, m, l_max;
  void *A_data, *P_data;
  SUNATimesFn atimes;
  SUNPSolveFn psolve;
  sunrealtype* res_norm;
  int* nli;
  N_Vector sx, sb, r_star, q, d, v, p, *r, u, vtemp1, vtemp2, vtemp3;
  suncomplextype cv[3];
  N_Vector Xv[3];
  int status = SUN_SUCCESS;

  /* Make local shorcuts to solver variables. */
  l_max     = CS_CONTENT(S)->maxl;
  r_star    = CS_CONTENT(S)->r_star;
  q         = CS_CONTENT(S)->q;
  d         = CS_CONTENT(S)->d;
  v         = CS_CONTENT(S)->v;
  p         = CS_CONTENT(S)->p;
  r         = CS_CONTENT(S)->r;
  u         = CS_CONTENT(S)->u;
  vtemp1    = CS_CONTENT(S)->vtemp1;
  vtemp2    = CS_CONTENT(S)->vtemp2;
  vtemp3    = CS_CONTENT(S)->vtemp3;
  sb        = CS_CONTENT(S)->s1;
  sx        = CS_CONTENT(S)->s2;
  A_data    = CS_CONTENT(S)->ATData;
  P_data    = CS_CONTENT(S)->PData;
  atimes    = CS_CONTENT(S)->ATimes;
  psolve    = CS_CONTENT(S)->Psolve;
  zeroguess = &(CS_CONTENT(S)->zeroguess);
  nli       = &(CS_CONTENT(S)->numiters);
  res_norm  = &(CS_CONTENT(S)->resnorm);

  /* Initialize counters and convergence flag */
  temp_val = r_curr_norm = -ONE;
  *nli                   = 0;
  converged              = SUNFALSE;
  b_ok                   = SUNFALSE;

  /* set sunbooleantype flags for internal solver options */
  preOnLeft  = ((CS_CONTENT(S)->pretype == SUN_PREC_LEFT) ||
               (CS_CONTENT(S)->pretype == SUN_PREC_BOTH));
  preOnRight = ((CS_CONTENT(S)->pretype == SUN_PREC_RIGHT) ||
                (CS_CONTENT(S)->pretype == SUN_PREC_BOTH));
  scale_x    = (sx != NULL);
  scale_b    = (sb != NULL);

  /* Check for unsupported use case */
  if (preOnRight && !(*zeroguess))
  {
    *zeroguess  = SUNFALSE;
    LASTFLAG(S) = SUN_ERR_ARG_INCOMPATIBLE;
    return SUN_ERR_ARG_INCOMPATIBLE;
  }

  /* Check if Atimes function has been set */
  if (atimes == NULL) { return SUN_ERR_ARG_CORRUPT; }

  /* If preconditioning, check if psolve has been set */
  if ((preOnLeft || preOnRight) && (psolve == NULL))
  { return SUN_ERR_ARG_CORRUPT; }

  /* Set r_star to initial (unscaled) residual r_star = r_0 = b - A*x_0 */
  /* NOTE: if x == 0 then just set residual to b and continue */
  if (*zeroguess) { N_VScale_SComplex(ONE, b, r_star); }
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
    N_VLinearSum_SComplex(ONE, b, -ONE, r_star, r_star);
  }

  /* Apply left preconditioner and b-scaling to r_star (or really just r_0) */
  if (preOnLeft)
  {
    status = psolve(P_data, r_star, vtemp1, delta, SUN_PREC_LEFT);

    if (status != 0)
    {
      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                 : SUNLS_PSOLVE_FAIL_REC;
      return (LASTFLAG(S));
    }
  }
  else { N_VScale_SComplex(ONE, r_star, vtemp1); }

  if (scale_b) { N_VProd_SComplex(sb, vtemp1, r_star); }
  else { N_VScale_SComplex(ONE, vtemp1, r_star); }

  /* Initialize rho[0] */
  /* NOTE: initialized here to reduce number of computations - avoid need
           to compute r_star^T*r_star twice, and avoid needlessly squaring
           values */
  rho[0] = N_VDotProd_SComplex(r_star, r_star);

  /* Compute norm of initial residual (r_0) to see if we really need
     to do anything */
  *res_norm = r_init_norm = (sunrealtype)SUNCsqrt(rho[0]);

  if (r_init_norm <= delta)
  {
    *zeroguess  = SUNFALSE;
    LASTFLAG(S) = SUN_SUCCESS;
    return (LASTFLAG(S));
  }

  /* Set v = A*r_0 (preconditioned and scaled) */
  if (scale_x) { N_VDiv_SComplex(r_star, sx, vtemp1); }
  else { N_VScale_SComplex(ONE, r_star, vtemp1); }

  if (preOnRight)
  {
    N_VScale_SComplex(ONE, vtemp1, v);
    status = psolve(P_data, v, vtemp1, delta, SUN_PREC_RIGHT);
    if (status != 0)
    {
      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                 : SUNLS_PSOLVE_FAIL_REC;
      return (LASTFLAG(S));
    }
  }

  status = atimes(A_data, vtemp1, v);
  if (status != 0)
  {
    *zeroguess = SUNFALSE;
    LASTFLAG(S) = (status < 0) ? SUNLS_ATIMES_FAIL_UNREC : SUNLS_ATIMES_FAIL_REC;
    return (LASTFLAG(S));
  }

  if (preOnLeft)
  {
    status = psolve(P_data, v, vtemp1, delta, SUN_PREC_LEFT);
    if (status != 0)
    {
      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                 : SUNLS_PSOLVE_FAIL_REC;
      return (LASTFLAG(S));
    }
  }
  else { N_VScale_SComplex(ONE, v, vtemp1); }

  if (scale_b) { N_VProd_SComplex(sb, vtemp1, v); }
  else { N_VScale_SComplex(ONE, vtemp1, v); }

  /* Initialize remaining variables */
  N_VScale_SComplex(ONE, r_star, r[0]);
  N_VScale_SComplex(ONE, r_star, u);
  N_VScale_SComplex(ONE, r_star, p);
  N_VConst_SComplex(ZERO, d);

  /* Set x = sx x if non-zero guess */
  if (scale_x && !(*zeroguess)) { N_VProd_SComplex(sx, x, x); }

  tau   = r_init_norm;
  v_bar = eta = ZERO;

  /* START outer loop */
  for (n = 0; n < l_max; ++n)
  {
    /* Increment linear iteration counter */
    (*nli)++;

    /* sigma = r_star^T*v */
    sigma = N_VDotProd_SComplex(r_star, v); //Amihere

    /* alpha = rho[0]/sigma */
    alpha = rho[0] / sigma;

    /* q = u-alpha*v */
    N_VLinearSum_SComplex(ONE, u, -alpha, v, q);

    /* r[1] = r[0]-alpha*A*(u+q) */
    N_VLinearSum_SComplex(ONE, u, ONE, q, r[1]);
    if (scale_x) { N_VDiv_SComplex(r[1], sx, r[1]); }

    if (preOnRight)
    {
      N_VScale_SComplex(ONE, r[1], vtemp1);
      status = psolve(P_data, vtemp1, r[1], delta, SUN_PREC_RIGHT);
      if (status != 0)
      {
        *zeroguess  = SUNFALSE;
        LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                   : SUNLS_PSOLVE_FAIL_REC;
        return (LASTFLAG(S));
      }
    }

    status = atimes(A_data, r[1], vtemp1);
    if (status != 0)
    {
      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = (status < 0) ? SUNLS_ATIMES_FAIL_UNREC
                                 : SUNLS_ATIMES_FAIL_REC;
      return (LASTFLAG(S));
    }

    if (preOnLeft)
    {
      status = psolve(P_data, vtemp1, r[1], delta, SUN_PREC_LEFT);
      if (status != 0)
      {
        *zeroguess  = SUNFALSE;
        LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                   : SUNLS_PSOLVE_FAIL_REC;
        return (LASTFLAG(S));
      }
    }
    else { N_VScale_SComplex(ONE, vtemp1, r[1]); }

    if (scale_b) { N_VProd_SComplex(sb, r[1], vtemp1); }
    else { N_VScale_SComplex(ONE, r[1], vtemp1); }
    N_VLinearSum_SComplex(ONE, r[0], -alpha, vtemp1, r[1]);

    /* START inner loop */
    for (m = 0; m < 2; ++m)
    {
      /* d = [*]+(v_bar^2*eta/alpha)*d */
      /* NOTES:
       *   (1) [*] = u if m == 0, and q if m == 1
       *   (2) using temp_val reduces the number of required computations
       *       if the inner loop is executed twice
       */
      if (m == 0)
      {
        temp_val = (sunrealtype)N_VDotProd_SComplex(r[1], r[1]);
        temp_val = SUNRsqrt(temp_val);
        omega    = (sunrealtype)N_VDotProd_SComplex(r[0], r[0]);
        omega = SUNRsqrt(SUNRsqrt(omega) * temp_val);
        N_VLinearSum_SComplex(ONE, u, SUNSQR(v_bar) * eta / alpha, d, d);
      }
      else
      {
        omega = temp_val;
        N_VLinearSum_SComplex(ONE, q, SUNSQR(v_bar) * eta / alpha, d, d);
      }

      /* v_bar = omega/tau */
      v_bar = omega / tau;

      /* c = (1+v_bar^2)^(-1/2) */
      c = ONE / SUNRsqrt(ONE + SUNSQR(v_bar));

      /* tau = tau*v_bar*c */
      tau = tau * v_bar * c;

      /* eta = c^2*alpha */
      eta = SUNSQR(c) * alpha;

      /* x = x+eta*d */
      if (n == 0 && m == 0 && *zeroguess) { N_VScale_SComplex(eta, d, x); }
      else { N_VLinearSum_SComplex(ONE, x, eta, d, x); }

      /* Check for convergence... */
      /* NOTE: just use approximation to norm of residual, if possible */
      *res_norm = r_curr_norm = tau * SUNRsqrt(m + 1);

      /* Exit inner loop if iteration has converged based upon approximation
         to norm of current residual */
      if (r_curr_norm <= delta)
      {
        converged = SUNTRUE;
        break;
      }

      /* Decide if actual norm of residual vector should be computed */
      /* NOTES:
       *   (1) if r_curr_norm > delta, then check if actual residual norm
       *       is OK (recall we first compute an approximation)
       *   (2) if r_curr_norm >= r_init_norm and m == 1 and n == l_max, then
       *       compute actual residual norm to see if the iteration can be
       *       saved
       *   (3) the scaled and preconditioned right-hand side of the given
       *       linear system (denoted by b) is only computed once, and the
       *       result is stored in vtemp3 so it can be reused - reduces the
       *       number of psovles if using left preconditioning
       */
      if ((r_curr_norm > delta) ||
          (r_curr_norm >= r_init_norm && m == 1 && n == l_max))
      {
        /* Compute norm of residual ||b-A*x||_2 (preconditioned and scaled) */
        if (scale_x) { N_VDiv_SComplex(x, sx, vtemp1); }
        else { N_VScale_SComplex(ONE, x, vtemp1); }

        if (preOnRight)
        {
          status = psolve(P_data, vtemp1, vtemp2, delta, SUN_PREC_RIGHT);
          if (status != 0)
          {
            *zeroguess  = SUNFALSE;
            LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                       : SUNLS_PSOLVE_FAIL_UNREC;
            return (LASTFLAG(S));
          }
          N_VScale_SComplex(ONE, vtemp2, vtemp1);
        }

        status = atimes(A_data, vtemp1, vtemp2);
        if (status != 0)
        {
          *zeroguess  = SUNFALSE;
          LASTFLAG(S) = (status < 0) ? SUNLS_ATIMES_FAIL_UNREC
                                     : SUNLS_ATIMES_FAIL_REC;
          return (LASTFLAG(S));
        }

        if (preOnLeft)
        {
          status = psolve(P_data, vtemp2, vtemp1, delta, SUN_PREC_LEFT);
          if (status != 0)
          {
            *zeroguess  = SUNFALSE;
            LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                       : SUNLS_PSOLVE_FAIL_REC;
            return (LASTFLAG(S));
          }
        }
        else { N_VScale_SComplex(ONE, vtemp2, vtemp1); }

        if (scale_b) { N_VProd_SComplex(sb, vtemp1, vtemp2); }
        else { N_VScale_SComplex(ONE, vtemp1, vtemp2); }

        /* Only precondition and scale b once (result saved for reuse) */
        if (!b_ok)
        {
          b_ok = SUNTRUE;
          if (preOnLeft)
          {
            status = psolve(P_data, b, vtemp3, delta, SUN_PREC_LEFT);
            if (status != 0)
            {
              *zeroguess  = SUNFALSE;
              LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                         : SUNLS_PSOLVE_FAIL_REC;
              return (LASTFLAG(S));
            }
          }
          else { N_VScale_SComplex(ONE, b, vtemp3); }

          if (scale_b) { N_VProd_SComplex(sb, vtemp3, vtemp3); }
        }
        N_VLinearSum_SComplex(ONE, vtemp3, -ONE, vtemp2, vtemp1);
        r_curr_norm = (sunrealtype)N_VDotProd_SComplex(vtemp1, vtemp1);
        *res_norm = r_curr_norm = SUNRsqrt(r_curr_norm);

        /* Exit inner loop if inequality condition is satisfied
           (meaning exit if we have converged) */
        if (r_curr_norm <= delta)
        {
          converged = SUNTRUE;
          break;
        }
      }

    } /* END inner loop */

    /* If converged, then exit outer loop as well */
    if (converged == SUNTRUE) { break; }

    /* rho[1] = r_star^T*r_[1] */
    rho[1] = N_VDotProd_SComplex(r_star, r[1]); //Amihere

    /* beta = rho[1]/rho[0] */
    beta = rho[1] / rho[0];

    /* u = r[1]+beta*q */
    N_VLinearSum_SComplex(ONE, r[1], beta, q, u);

    /* p = u+beta*(q+beta*p) = beta*beta*p + beta*q + u */
    cv[0] = SUNSQR(beta);
    Xv[0] = p;

    cv[1] = beta;
    Xv[1] = q;

    cv[2] = ONE;
    Xv[2] = u;

    N_VLinearCombination_SComplex(3, cv, Xv, p);

    /* v = A*p */
    if (scale_x) { N_VDiv_SComplex(p, sx, vtemp1); }
    else { N_VScale_SComplex(ONE, p, vtemp1); }

    if (preOnRight)
    {
      N_VScale_SComplex(ONE, vtemp1, v);
      status = psolve(P_data, v, vtemp1, delta, SUN_PREC_RIGHT);
      if (status != 0)
      {
        *zeroguess  = SUNFALSE;
        LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                   : SUNLS_PSOLVE_FAIL_REC;
        return (LASTFLAG(S));
      }
    }

    status = atimes(A_data, vtemp1, v);
    if (status != 0)
    {
      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = (status < 0) ? SUNLS_ATIMES_FAIL_UNREC
                                 : SUNLS_ATIMES_FAIL_REC;
      return (LASTFLAG(S));
    }

    if (preOnLeft)
    {
      status = psolve(P_data, v, vtemp1, delta, SUN_PREC_LEFT);
      if (status != 0)
      {
        *zeroguess  = SUNFALSE;
        LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                   : SUNLS_PSOLVE_FAIL_REC;
        return (LASTFLAG(S));
      }
    }
    else { N_VScale_SComplex(ONE, v, vtemp1); }

    if (scale_b) { N_VProd_SComplex(sb, vtemp1, v); }
    else { N_VScale_SComplex(ONE, vtemp1, v); }

    /* Shift variable values */
    /* NOTE: reduces storage requirements */
    N_VScale_SComplex(ONE, r[1], r[0]);
    rho[0] = rho[1];

  } /* END outer loop */

  /* Determine return value */
  /* If iteration converged or residual was reduced, then return current iterate
   * (x) */
  if ((converged == SUNTRUE) || (r_curr_norm < r_init_norm))
  {
    if (scale_x) { N_VDiv_SComplex(x, sx, x); }

    if (preOnRight)
    {
      status = psolve(P_data, x, vtemp1, delta, SUN_PREC_RIGHT);
      if (status != 0)
      {
        *zeroguess  = SUNFALSE;
        LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                   : SUNLS_PSOLVE_FAIL_UNREC;
        return (LASTFLAG(S));
      }
      N_VScale_SComplex(ONE, vtemp1, x);
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

int SUNLinSolNumIters_SComplex(SUNLinearSolver S)
{
  return (CS_CONTENT(S)->numiters);
}

sunrealtype SUNLinSolResNorm_SComplex(SUNLinearSolver S)
{
  return (CS_CONTENT(S)->resnorm);
}

N_Vector SUNLinSolResid_SComplex(SUNLinearSolver S)
{
  return (CS_CONTENT(S)->vtemp1);
}

sunindextype SUNLinSolLastFlag_SComplex(SUNLinearSolver S)
{
  return (LASTFLAG(S));
}

SUNErrCode SUNLinSolSpace_SComplex(SUNLinearSolver S, long int* lenrwLS,
                               long int* leniwLS)
{
  sunindextype liw1, lrw1;
  if (CS_CONTENT(S)->vtemp1->ops->nvspace)
  {
    N_VSpace_SComplex(CS_CONTENT(S)->vtemp1, &lrw1, &liw1);
  }
  else { lrw1 = liw1 = 0; }
  *lenrwLS = lrw1 * 11;
  *leniwLS = liw1 * 11;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolFree_SComplex(SUNLinearSolver S)
{
  if (S == NULL) { return SUN_SUCCESS; }

  if (S->content)
  {
    /* delete items from within the content structure */
    if (CS_CONTENT(S)->r_star)
    {
      N_VDestroy_SComplex(CS_CONTENT(S)->r_star);
      CS_CONTENT(S)->r_star = NULL;
    }
    if (CS_CONTENT(S)->q)
    {
      N_VDestroy_SComplex(CS_CONTENT(S)->q);
      CS_CONTENT(S)->q = NULL;
    }
    if (CS_CONTENT(S)->d)
    {
      N_VDestroy_SComplex(CS_CONTENT(S)->d);
      CS_CONTENT(S)->d = NULL;
    }
    if (CS_CONTENT(S)->v)
    {
      N_VDestroy_SComplex(CS_CONTENT(S)->v);
      CS_CONTENT(S)->v = NULL;
    }
    if (CS_CONTENT(S)->p)
    {
      N_VDestroy_SComplex(CS_CONTENT(S)->p);
      CS_CONTENT(S)->p = NULL;
    }
    if (CS_CONTENT(S)->r)
    {
      N_VDestroyVectorArray(CS_CONTENT(S)->r, 2);
      CS_CONTENT(S)->r = NULL;
    }
    if (CS_CONTENT(S)->u)
    {
      N_VDestroy_SComplex(CS_CONTENT(S)->u);
      CS_CONTENT(S)->u = NULL;
    }
    if (CS_CONTENT(S)->vtemp1)
    {
      N_VDestroy_SComplex(CS_CONTENT(S)->vtemp1);
      CS_CONTENT(S)->vtemp1 = NULL;
    }
    if (CS_CONTENT(S)->vtemp2)
    {
      N_VDestroy_SComplex(CS_CONTENT(S)->vtemp2);
      CS_CONTENT(S)->vtemp2 = NULL;
    }
    if (CS_CONTENT(S)->vtemp3)
    {
      N_VDestroy_SComplex(CS_CONTENT(S)->vtemp3);
      CS_CONTENT(S)->vtemp3 = NULL;
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