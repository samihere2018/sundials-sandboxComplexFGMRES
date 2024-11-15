/* -----------------------------------------------------------------
 * Programmer(s): Daniel Reynolds @ SMU
 * Based on sundials_spgmr.c code, written by Scott D. Cohen,
 *                Alan C. Hindmarsh and Radu Serban @ LLNL
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
 * This is the implementation file for the SPGMR implementation of
 * the SUNLINSOL package.
 * -----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>

// #include <sundials/priv/sundials_errors_impl.h>
// #include <sundials/sundials_math.h>
#include "sunlinsol_spgmrcomplex.h"

// #include "sundials_logger_impl.h"
// #include "sundials_macros.h"

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)

SUNErrCode SUNModifiedGSComplex(N_Vector* v, suncomplextype** h, int k, int p,
                         suncomplextype* new_vk_norm);
SUNErrCode SUNClassicalGSComplex(N_Vector* v, suncomplextype** h, int k, int p,
                          suncomplextype* new_vk_norm, suncomplextype* stemp,
                          N_Vector* vtemp);
int SUNQRfactComplex(int n, suncomplextype** h, suncomplextype* q, int job);
int SUNQRsolComplex(int n, suncomplextype** h, suncomplextype* q, suncomplextype* b);

/*
 * -----------------------------------------------------------------
 * SPGMR solver structure accessibility macros:
 * -----------------------------------------------------------------
 */

#define SPGMRComplex_CONTENT(S) ((SUNLinearSolverContent_SPGMRComplex)(S->content))
#define LASTFLAG(S)      (SPGMRComplex_CONTENT(S)->last_flag)

/*
 * -----------------------------------------------------------------
 * exported functions
 * -----------------------------------------------------------------
 */

/* ----------------------------------------------------------------------------
 * Function to create a new SPGMR linear solver
 */

SUNLinearSolver SUNLinSol_SPGMRComplex(N_Vector y, int pretype, int maxl,
                                SUNContext sunctx)
{
  // SUNFunctionBegin(sunctx);
  SUNLinearSolver S;
  SUNLinearSolverContent_SPGMRComplex content;

  /* check for legal pretype and maxl values; if illegal use defaults */
  if ((pretype != SUN_PREC_NONE) && (pretype != SUN_PREC_LEFT) &&
      (pretype != SUN_PREC_RIGHT) && (pretype != SUN_PREC_BOTH))
  {
    pretype = SUN_PREC_NONE;
  }
  if (maxl <= 0) { maxl = SUNSPGMRComplex_MAXL_DEFAULT; }

  // /* check that the supplied N_Vector supports all requisite operations */
  // SUNAssertNull((y->ops->nvclone) && (y->ops->nvdestroy) &&
  //                 (y->ops->nvlinearsum) && (y->ops->nvconst) && (y->ops->nvprod) &&
  //                 (y->ops->nvdiv) && (y->ops->nvscale) && (y->ops->nvdotprod),
  //               SUN_ERR_ARG_OUTOFRANGE);

  /* Create linear solver */
  S = NULL;
  S = SUNLinSolNewEmpty(sunctx);
  // SUNCheckLastErrNull();

  /* Attach operations */
  S->ops->gettype           = SUNLinSolGetType_SPGMRComplex;
  S->ops->getid             = SUNLinSolGetID_SPGMRComplex;
  S->ops->setatimes         = SUNLinSolSetATimes_SPGMRComplex;
  S->ops->setpreconditioner = SUNLinSolSetPreconditioner_SPGMRComplex;
  S->ops->setscalingvectors = SUNLinSolSetScalingVectors_SPGMRComplex;
  S->ops->setzeroguess      = SUNLinSolSetZeroGuess_SPGMRComplex;
  S->ops->initialize        = SUNLinSolInitialize_SPGMRComplex;
  S->ops->setup             = SUNLinSolSetup_SPGMRComplex;
  S->ops->solve             = SUNLinSolSolve_SPGMRComplex;
  S->ops->numiters          = SUNLinSolNumIters_SPGMRComplex;
  S->ops->resnorm           = SUNLinSolResNorm_SPGMRComplex;
  S->ops->resid             = SUNLinSolResid_SPGMRComplex;
  S->ops->lastflag          = SUNLinSolLastFlag_SPGMRComplex;
  S->ops->space             = SUNLinSolSpace_SPGMRComplex;
  S->ops->free              = SUNLinSolFree_SPGMRComplex;

  /* Create content */
  content = NULL;
  content = (SUNLinearSolverContent_SPGMRComplex)malloc(sizeof *content);
  // SUNAssertNull(content, SUN_ERR_MALLOC_FAIL);

  /* Attach content */
  S->content = content;

  /* Fill content */
  content->last_flag    = 0;
  content->maxl         = maxl;
  content->pretype      = pretype;
  content->gstype       = SUNSPGMRComplex_GSTYPE_DEFAULT;
  content->max_restarts = SUNSPGMRComplex_MAXRS_DEFAULT;
  content->zeroguess    = SUNFALSE;
  content->numiters     = 0;
  content->resnorm      = ZERO;
  content->xcor         = NULL;
  content->vtemp        = NULL;
  content->s1           = NULL;
  content->s2           = NULL;
  content->ATimes       = NULL;
  content->ATData       = NULL;
  content->Psetup       = NULL;
  content->Psolve       = NULL;
  content->PData        = NULL;
  content->V            = NULL;
  content->Hes          = NULL;
  content->givens       = NULL;
  content->yg           = NULL;
  content->cv           = NULL;
  content->Xv           = NULL;

  /* Allocate content */
  content->xcor = N_VClone_SComplex(y);
  // SUNCheckLastErrNull();
  content->vtemp = N_VClone_SComplex(y);
  // SUNCheckLastErrNull();

  return (S);
}

/* ----------------------------------------------------------------------------
 * Function to set the type of preconditioning for SPGMR to use
 */

SUNErrCode SUNLinSol_SPGMRComplex_SetPrecType(SUNLinearSolver S, int pretype)
{
  // SUNFunctionBegin(S->sunctx);
  // /* Check for legal pretype */
  // SUNAssert((pretype == SUN_PREC_NONE) || (pretype == SUN_PREC_LEFT) ||
  //             (pretype == SUN_PREC_RIGHT) || (pretype == SUN_PREC_BOTH),
  //           SUN_ERR_ARG_OUTOFRANGE);

  /* Set pretype */
  SPGMRComplex_CONTENT(S)->pretype = pretype;
  return SUN_SUCCESS;
}

/* ----------------------------------------------------------------------------
 * Function to set the type of Gram-Schmidt orthogonalization for SPGMR to use
 */

SUNErrCode SUNLinSol_SPGMRComplex_SetGSType(SUNLinearSolver S, int gstype)
{
  // SUNFunctionBegin(S->sunctx);
  /* Check for legal gstype */
  // SUNAssert(gstype == SUN_MODIFIED_GS || gstype == SUN_CLASSICAL_GS,
  //           SUN_ERR_ARG_OUTOFRANGE);

  /* Set pretype */
  SPGMRComplex_CONTENT(S)->gstype = gstype;
  return SUN_SUCCESS;
}

/* ----------------------------------------------------------------------------
 * Function to set the maximum number of GMRES restarts to allow
 */

SUNErrCode SUNLinSol_SPGMRComplex_SetMaxRestarts(SUNLinearSolver S, int maxrs)
{
  /* Illegal maxrs implies use of default value */
  if (maxrs < 0) { maxrs = SUNSPGMRComplex_MAXRS_DEFAULT; }

  /* Set max_restarts */
  SPGMRComplex_CONTENT(S)->max_restarts = maxrs;
  return SUN_SUCCESS;
}

/*
 * -----------------------------------------------------------------
 * implementation of linear solver operations
 * -----------------------------------------------------------------
 */

SUNLinearSolver_Type SUNLinSolGetType_SPGMRComplex( SUNLinearSolver S)
{
  return (SUNLINEARSOLVER_ITERATIVE);
}

SUNLinearSolver_ID SUNLinSolGetID_SPGMRComplex( SUNLinearSolver S)
{
  return (SUNLINEARSOLVER_SPGMR);
}

SUNErrCode SUNLinSolInitialize_SPGMRComplex(SUNLinearSolver S)
{
  int k;
  SUNLinearSolverContent_SPGMRComplex content;
  // SUNFunctionBegin(S->sunctx);

  /* set shortcut to SPGMR memory structure */
  content = SPGMRComplex_CONTENT(S);

  /* ensure valid options */
  if (content->max_restarts < 0)
  {
    content->max_restarts = SUNSPGMRComplex_MAXRS_DEFAULT;
  }

  // SUNAssert(content->ATimes, SUN_ERR_ARG_CORRUPT);

  if ((content->pretype != SUN_PREC_LEFT) &&
      (content->pretype != SUN_PREC_RIGHT) && (content->pretype != SUN_PREC_BOTH))
  {
    content->pretype = SUN_PREC_NONE;
  }

  // SUNAssert((content->pretype == SUN_PREC_NONE) || (content->Psolve != NULL),
  //           SUN_ERR_ARG_CORRUPT);

  /* allocate solver-specific memory (where the size depends on the
     choice of maxl) here */

  /*   Krylov subspace vectors */
  if (content->V == NULL)
  {
    content->V = N_VCloneVectorArray(content->maxl + 1, content->vtemp);
    // SUNCheckLastErr();
  }

  /*   Hessenberg matrix Hes */
  if (content->Hes == NULL)
  {
    content->Hes =
      // (sunrealtype**)malloc((content->maxl + 1) * sizeof(sunrealtype*));
      (suncomplextype**)malloc((content->maxl + 1) * sizeof(suncomplextype*)); //Amihere
    // SUNAssert(content->Hes, SUN_ERR_MALLOC_FAIL);

    for (k = 0; k <= content->maxl; k++)
    {
      content->Hes[k] = NULL;
      // content->Hes[k] = (sunrealtype*)malloc(content->maxl * sizeof(sunrealtype));
      content->Hes[k] = (suncomplextype*)malloc(content->maxl * sizeof(suncomplextype)); //Amihere
      // SUNAssert(content->Hes[k], SUN_ERR_MALLOC_FAIL);
    }
  }

  /*   Givens rotation components */
  if (content->givens == NULL)
  {
    content->givens =
      // (sunrealtype*)malloc(2 * content->maxl * sizeof(sunrealtype));
      (suncomplextype*)malloc(2 * content->maxl * sizeof(suncomplextype)); //Amihere
    // SUNAssert(content->givens, SUN_ERR_MALLOC_FAIL);
  }

  /*    y and g vectors */
  if (content->yg == NULL)
  {
    // content->yg = (sunrealtype*)malloc((content->maxl + 1) * sizeof(sunrealtype));
    content->yg = (suncomplextype*)malloc((content->maxl + 1) * sizeof(suncomplextype)); //Amihere
    // SUNAssert(content->yg, SUN_ERR_MALLOC_FAIL);
  }

  /*    cv vector for fused vector ops */
  if (content->cv == NULL)
  {
    // content->cv = (sunrealtype*)malloc((content->maxl + 1) * sizeof(sunrealtype));
    content->cv = (suncomplextype*)malloc((content->maxl + 1) * sizeof(suncomplextype)); //Amihere
    // SUNAssert(content->cv, SUN_ERR_MALLOC_FAIL);
  }

  /*    Xv vector for fused vector ops */
  if (content->Xv == NULL)
  {
    content->Xv = (N_Vector*)malloc((content->maxl + 1) * sizeof(N_Vector));
    // SUNAssert(content->Xv, SUN_ERR_MALLOC_FAIL);
  }

  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetATimes_SPGMRComplex(SUNLinearSolver S, void* ATData,
                                    SUNATimesFn ATimes)
{
  /* set function pointers to integrator-supplied ATimes routine
     and data, and return with success */
  SPGMRComplex_CONTENT(S)->ATimes = ATimes;
  SPGMRComplex_CONTENT(S)->ATData = ATData;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetPreconditioner_SPGMRComplex(SUNLinearSolver S, void* PData,
                                            SUNPSetupFn Psetup,
                                            SUNPSolveFn Psolve)
{
  /* set function pointers to integrator-supplied Psetup and PSolve
     routines and data, and return with success */
  SPGMRComplex_CONTENT(S)->Psetup = Psetup;
  SPGMRComplex_CONTENT(S)->Psolve = Psolve;
  SPGMRComplex_CONTENT(S)->PData  = PData;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetScalingVectors_SPGMRComplex(SUNLinearSolver S, N_Vector s1,
                                            N_Vector s2)
{
  /* set N_Vector pointers to integrator-supplied scaling vectors,
     and return with success */
  SPGMRComplex_CONTENT(S)->s1 = s1;
  SPGMRComplex_CONTENT(S)->s2 = s2;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetZeroGuess_SPGMRComplex(SUNLinearSolver S, sunbooleantype onff)
{
  /* set flag indicating a zero initial guess */
  SPGMRComplex_CONTENT(S)->zeroguess = onff;
  return SUN_SUCCESS;
}

int SUNLinSolSetup_SPGMRComplex(SUNLinearSolver S, SUNMatrix A)
{
  // SUNFunctionBegin(S->sunctx);

  int status = SUN_SUCCESS;

  /* Set shortcuts to SPGMR memory structures */
  SUNPSetupFn Psetup = SPGMRComplex_CONTENT(S)->Psetup;
  void* PData        = SPGMRComplex_CONTENT(S)->PData;

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

int SUNLinSolSolve_SPGMRComplex(SUNLinearSolver S, SUNMatrix A,
                         N_Vector x, N_Vector b, sunrealtype delta)
{
  // SUNFunctionBegin(S->sunctx);

  /* local data and shortcut variables */
  N_Vector *V, xcor, vtemp, s1, s2;
  // sunrealtype **Hes, *givens, *yg, *res_norm;
  suncomplextype **Hes, *givens, *yg;
  sunrealtype *res_norm;
  sunrealtype beta, rotation_product, r_norm, s_product, rho;
  sunbooleantype preOnLeft, preOnRight, scale2, scale1, converged;
  sunbooleantype* zeroguess;
  int i, j, k, l, l_plus_1, l_max, krydim, ntries, max_restarts, gstype;
  int* nli;
  suncomplextype resAns; //Amihere
  sunrealtype normV; //AMihere
  int kk, lk, lks; //Amihere
  void *A_data, *P_data;
  SUNATimesFn atimes;
  SUNPSolveFn psolve;
  // sunrealtype* cv;
  suncomplextype* cv;
  N_Vector* Xv;
  int status;

  /* Initialize some variables */
  l_plus_1 = 0;
  krydim   = 0;

  /* Make local shorcuts to solver variables. */
  l_max        = SPGMRComplex_CONTENT(S)->maxl;
  max_restarts = SPGMRComplex_CONTENT(S)->max_restarts;
  gstype       = SPGMRComplex_CONTENT(S)->gstype;
  V            = SPGMRComplex_CONTENT(S)->V;
  Hes          = SPGMRComplex_CONTENT(S)->Hes;
  givens       = SPGMRComplex_CONTENT(S)->givens;
  xcor         = SPGMRComplex_CONTENT(S)->xcor;
  yg           = SPGMRComplex_CONTENT(S)->yg;
  vtemp        = SPGMRComplex_CONTENT(S)->vtemp;
  s1           = SPGMRComplex_CONTENT(S)->s1;
  s2           = SPGMRComplex_CONTENT(S)->s2;
  A_data       = SPGMRComplex_CONTENT(S)->ATData;
  P_data       = SPGMRComplex_CONTENT(S)->PData;
  atimes       = SPGMRComplex_CONTENT(S)->ATimes;
  psolve       = SPGMRComplex_CONTENT(S)->Psolve;
  zeroguess    = &(SPGMRComplex_CONTENT(S)->zeroguess);
  nli          = &(SPGMRComplex_CONTENT(S)->numiters);
  res_norm     = &(SPGMRComplex_CONTENT(S)->resnorm);
  cv           = SPGMRComplex_CONTENT(S)->cv;
  Xv           = SPGMRComplex_CONTENT(S)->Xv;

  /* Initialize counters and convergence flag */
  *nli      = 0;
  converged = SUNFALSE;

  /* Set sunbooleantype flags for internal solver options */
  preOnLeft  = ((SPGMRComplex_CONTENT(S)->pretype == SUN_PREC_LEFT) ||
               (SPGMRComplex_CONTENT(S)->pretype == SUN_PREC_BOTH));
  preOnRight = ((SPGMRComplex_CONTENT(S)->pretype == SUN_PREC_RIGHT) ||
                (SPGMRComplex_CONTENT(S)->pretype == SUN_PREC_BOTH));
  scale1     = (s1 != NULL);
  scale2     = (s2 != NULL);

  /* Check if Atimes function has been set */
  // SUNAssert(atimes, SUN_ERR_ARG_CORRUPT);

  /* If preconditioning, check if psolve has been set */
  // SUNAssert(!(preOnLeft || preOnRight) || psolve, SUN_ERR_ARG_CORRUPT);

  /* Set vtemp and V[0] to initial (unscaled) residual r_0 = b - A*x_0 */
  if (*zeroguess)
  {
    N_VScale_SComplex(ONE, b, vtemp);
    // SUNCheckLastErr();
  }
  else
  {
    status = atimes(A_data, x, vtemp);
    if (status != 0)
    {
      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = (status < 0) ? SUNLS_ATIMES_FAIL_UNREC
                                 : SUNLS_ATIMES_FAIL_REC;
      return (LASTFLAG(S));
    }
    N_VLinearSum_SComplex(ONE, b, -ONE, vtemp, vtemp);
    // SUNCheckLastErr();
  }
  N_VScale_SComplex(ONE, vtemp, V[0]);
  // SUNCheckLastErr();

  /* Apply left preconditioner and left scaling to V[0] = r_0 */
  if (preOnLeft)
  {
    status = psolve(P_data, V[0], vtemp, delta, SUN_PREC_LEFT);
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
    N_VScale_SComplex(ONE, V[0], vtemp);
    // SUNCheckLastErr();
  }

  if (scale1)
  {
    N_VProd_SComplex(s1, vtemp, V[0]);
    // SUNCheckLastErr();
  }
  else
  {
    N_VScale_SComplex(ONE, vtemp, V[0]);
    // SUNCheckLastErr();
  }

  /* Set r_norm = beta to L2 norm of V[0] = s1 P1_inv r_0, and
     return if small  */
  r_norm = N_VDotProd_SComplex(V[0], V[0]);
  // SUNCheckLastErr();
  *res_norm = r_norm = beta = SUNRsqrt(r_norm);

// #if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_INFO
//   SUNLogger_QueueMsg(S->sunctx->logger, SUN_LOGLEVEL_INFO,
//                      "SUNLinSolSolve_SPGMR", "initial-residual",
//                      "nli = %li, resnorm = %.16g", (long int)0, *res_norm);
// #endif

  if (r_norm <= delta)
  {
    *zeroguess  = SUNFALSE;
    LASTFLAG(S) = SUN_SUCCESS;
    return (LASTFLAG(S));
  }

  /* Initialize rho to avoid compiler warning message */
  rho = beta;

  /* Set xcor = 0 */
  N_VConst_SComplex(ZERO, xcor);
  // SUNCheckLastErr();

  /* Begin outer iterations: up to (max_restarts + 1) attempts */
  for (ntries = 0; ntries <= max_restarts; ntries++)
  {
    /* Initialize the Hessenberg matrix Hes and Givens rotation
       product.  Normalize the initial vector V[0] */
    for (i = 0; i <= l_max; i++)
    {
      for (j = 0; j < l_max; j++) { Hes[i][j] = ZERO; }
    }

    rotation_product = ONE;
    N_VScale_SComplex(ONE / r_norm, V[0], V[0]);
    // SUNCheckLastErr();
    // printf("malx %d\n", l_max);

    /* Inner loop: generate Krylov sequence and Arnoldi basis */
    for (l = 0; l < l_max; l++)
    {
      (*nli)++;
      krydim = l_plus_1 = l + 1;

      /* Generate A-tilde V[l], where A-tilde = s1 P1_inv A P2_inv s2_inv */

      /*   Apply right scaling: vtemp = s2_inv V[l] */
      if (scale2)
      {
        N_VDiv_SComplex(V[l], s2, vtemp);
        // SUNCheckLastErr();
      }
      else
      {
        N_VScale_SComplex(ONE, V[l], vtemp);
        // SUNCheckLastErr();
      }

      /*   Apply right preconditioner: vtemp = P2_inv s2_inv V[l] */
      if (preOnRight)
      {
        N_VScale_SComplex(ONE, vtemp, V[l_plus_1]);
        // SUNCheckLastErr();
        status = psolve(P_data, V[l_plus_1], vtemp, delta, SUN_PREC_RIGHT);
        if (status != 0)
        {
          *zeroguess  = SUNFALSE;
          LASTFLAG(S) = (status < 0) ? SUNLS_PSOLVE_FAIL_UNREC
                                     : SUNLS_PSOLVE_FAIL_REC;
          return (LASTFLAG(S));
        }
      }

      /* Apply A: V[l+1] = A P2_inv s2_inv V[l] */
      status = atimes(A_data, vtemp, V[l_plus_1]);
      if (status != 0)
      {
        *zeroguess  = SUNFALSE;
        LASTFLAG(S) = (status < 0) ? SUNLS_ATIMES_FAIL_UNREC
                                   : SUNLS_ATIMES_FAIL_REC;
        return (LASTFLAG(S));
      }

      /* Apply left preconditioning: vtemp = P1_inv A P2_inv s2_inv V[l] */
      if (preOnLeft)
      {
        status = psolve(P_data, V[l_plus_1], vtemp, delta, SUN_PREC_LEFT);
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
        N_VScale_SComplex(ONE, V[l_plus_1], vtemp);
        // SUNCheckLastErr();
      }

      /* Apply left scaling: V[l+1] = s1 P1_inv A P2_inv s2_inv V[l] */
      if (scale1)
      {
        N_VProd_SComplex(s1, vtemp, V[l_plus_1]);
        // SUNCheckLastErr();
      }
      else
      {
        N_VScale_SComplex(ONE, vtemp, V[l_plus_1]);
        // SUNCheckLastErr();
      }

      /*  Orthogonalize V[l+1] against previous V[i]: V[l+1] = w_tilde */
      if (gstype == SUN_CLASSICAL_GS)
      {
        // SUNCheckCall(
        //   SUNClassicalGSComplex(V, Hes, l_plus_1, l_max, &(Hes[l_plus_1][l]), cv, Xv));
        SUNClassicalGSComplex(V, Hes, l_plus_1, l_max, &(Hes[l_plus_1][l]), cv, Xv);
      }
      else
      {
        // SUNCheckCall(SUNModifiedGSComplex(V, Hes, l_plus_1, l_max, &(Hes[l_plus_1][l])));
        SUNModifiedGSComplex(V, Hes, l_plus_1, l_max, &(Hes[l_plus_1][l]));
      }

      //Amihere: debugging
      // normV  = SUNRsqrt((sunrealtype)N_VDotProd_SComplex(V[l], V[l]));
      // // V[l] = (1.0 / normV) * V[l];
      // N_VScale_SComplex(ONE / normV, V[l], V[l]);
      // if (l==0){
      //   resAns = N_VDotProd_SComplex(V[l], V[l]);
      //   printf("Same: col1: %d and col2: %d are othorgonal, result: %f +i%f \n", l, l, creal(resAns), cimag(resAns));
      // } else {
      //   // lks = l;
      //   for (lk = 0; lk <= l; lk++) {
      //     for (kk = 0; kk <= l; kk++) {
      //       resAns = N_VDotProd_SComplex(V[lk], V[kk]);
      //       printf("col1: %d, col2: %d, result: %f +i%f \n", lk, kk, creal(resAns), cimag(resAns));
      //       // if (SUNRabs(creal(resAns == 0)) && SUNRabs(cimag(resAns==0))) {
      //       // if (SUNRabs(resAns == 0)) {
      //       //   printf("col1: %d and col2: %d are othorgonal, result: %f +i%f \n", lk, kk, creal(resAns), cimag(resAns));
      //       // } else{
      //       //   printf("col1: %d and col2: %d are not othorgonal, result: %f +i%f \n", lk, kk, creal(resAns), cimag(resAns));
      //       // }
      //     }
      //   }
      // }


      /*  Update the QR factorization of Hes */
      if (SUNQRfactComplex(krydim, Hes, givens, l) != 0)
      {
        *zeroguess  = SUNFALSE;
        LASTFLAG(S) = SUNLS_QRFACT_FAIL;
        return (LASTFLAG(S));
      }

      /*  Update residual norm estimate; break if convergence test passes */
      rotation_product *= givens[2 * l + 1];
      *res_norm = rho = SUNRabs(rotation_product * r_norm);

// #if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_INFO
//       SUNLogger_QueueMsg(S->sunctx->logger, SUN_LOGLEVEL_INFO,
//                          "SUNLinSolSolve_SPGMR", "iterate-residual",
//                          "nli = %li, resnorm = %.16g", (long int)*nli, *res_norm);
// #endif

      if (rho <= delta)
      {
        converged = SUNTRUE;
        break;
      }

      /* Normalize V[l+1] with norm value from the Gram-Schmidt routine */
      N_VScale_SComplex(ONE / Hes[l_plus_1][l], V[l_plus_1], V[l_plus_1]);
      // SUNCheckLastErr();
    }
    //Amihere: main
    // printf("H at row %d, column %d is: %f + i%f\n", 0, 0, creal(Hes[0][0]), cimag(Hes[0][0]));
    // printf("H at row %d, column %d is: %f + i%f\n", 0, 1, creal(Hes[0][1]), cimag(Hes[0][1]));
    // printf("H at row %d, column %d is: %f + i%f\n", 0, 2, creal(Hes[0][2]), cimag(Hes[0][2]));
    // printf("H at row %d, column %d is: %f + i%f\n", 1, 1, creal(Hes[1][1]), cimag(Hes[1][1]));
    // printf("H at row %d, column %d is: %f + i%f\n", 1, 2, creal(Hes[1][2]), cimag(Hes[1][2]));
    // printf("H at row %d, column %d is: %f + i%f\n", 2, 2, creal(Hes[2][2]), cimag(Hes[2][2]));
    // //Amihere: others
    // printf("H at row %d, column %d is: %f + i%f\n", 1, 0, creal(Hes[1][0]), cimag(Hes[1][0]));
    // printf("H at row %d, column %d is: %f + i%f\n", 2, 0, creal(Hes[2][0]), cimag(Hes[2][0]));
    // printf("H at row %d, column %d is: %f + i%f\n", 2, 1, creal(Hes[2][1]), cimag(Hes[2][1]));


    // // Amihere: debugging 
    // for (lk = 0; lk < l_max; lk++) {
    //   for (kk = 0; kk < l_max; kk++) {
    //     resAns = N_VDotProd_SComplex(V[lk], V[kk]);
    //     printf("col1: %d, col2: %d, result: %f +i%f \n", lk, kk, creal(resAns), cimag(resAns));
    //     // printf("H at row %d, column %d is: %f + i%f\n", lk, kk, creal(Hes[lk][kk]), cimag(Hes[lk][kk]));
    //     // printf("lmax:%d\n", l_max);
    //     // if (creal(resAns == 0) && cimag(resAns==0)) {
    //     //   printf("col1: %d and col2: %d are othorgonal, result: %f +i%f \n", l, k, creal(resAns), cimag(resAns));
    //     // } else{
    //     //   printf("col1: %d and col2: %d are not othorgonal, result: %f +i%f \n", l, k, creal(resAns), cimag(resAns));
    //     // }
    //   }
    // }

    /* Inner loop is done.  Compute the new correction vector xcor */

    /*   Construct g, then solve for y */
    yg[0] = r_norm;
    for (i = 1; i <= krydim; i++) { yg[i] = ZERO; }
    if (SUNQRsolComplex(krydim, Hes, givens, yg) != 0)
    {
      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = SUNLS_QRSOL_FAIL;
      return (LASTFLAG(S));
    }

    /*   Add correction vector V_l y to xcor */
    cv[0] = ONE;
    Xv[0] = xcor;

    for (k = 0; k < krydim; k++)
    {
      cv[k + 1] = yg[k];
      Xv[k + 1] = V[k];
    }
    // SUNCheckCall(N_VLinearCombination_SComplex(krydim + 1, cv, Xv, xcor));
    N_VLinearCombination_SComplex(krydim + 1, cv, Xv, xcor);

    /* If converged, construct the final solution vector x and return */
    if (converged)
    {
      /* Apply right scaling and right precond.: vtemp = P2_inv s2_inv xcor */
      if (scale2)
      {
        N_VDiv_SComplex(xcor, s2, xcor);
        // SUNCheckLastErr();
      }

      if (preOnRight)
      {
        status = psolve(P_data, xcor, vtemp, delta, SUN_PREC_RIGHT);
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
        N_VScale_SComplex(ONE, xcor, vtemp);
        // SUNCheckLastErr();
      }

      /* Add vtemp to initial x to get final solution x, and return */
      if (*zeroguess)
      {
        N_VScale_SComplex(ONE, vtemp, x);
        // SUNCheckLastErr();
      }
      else
      {
        N_VLinearSum_SComplex(ONE, x, ONE, vtemp, x);
        // SUNCheckLastErr();
      }

      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = SUN_SUCCESS;
      return (LASTFLAG(S));
    }

    /* Not yet converged; if allowed, prepare for restart */
    if (ntries == max_restarts) { break; }

    /* Construct last column of Q in yg */
    s_product = ONE;
    for (i = krydim; i > 0; i--)
    {
      yg[i] = s_product * givens[2 * i - 2];
      s_product *= givens[2 * i - 1];
    }
    yg[0] = s_product;

    /* Scale r_norm and yg */
    r_norm *= s_product;
    for (i = 0; i <= krydim; i++) { yg[i] *= r_norm; }
    r_norm = SUNRabs(r_norm);

    /* Multiply yg by V_(krydim+1) to get last residual vector; restart */
    for (k = 0; k <= krydim; k++)
    {
      cv[k] = yg[k];
      Xv[k] = V[k];
    }
    // SUNCheckCall(N_VLinearCombination_SComplex(krydim + 1, cv, Xv, V[0]));
    N_VLinearCombination_SComplex(krydim + 1, cv, Xv, V[0]);
  }

  /* Failed to converge, even after allowed restarts.
     If the residual norm was reduced below its initial value, compute
     and return x anyway.  Otherwise return failure flag. */
  if (rho < beta)
  {
    /* Apply right scaling and right precond.: vtemp = P2_inv s2_inv xcor */
    if (scale2)
    {
      N_VDiv_SComplex(xcor, s2, xcor);
      // SUNCheckLastErr();
    }

    if (preOnRight)
    {
      status = psolve(P_data, xcor, vtemp, delta, SUN_PREC_RIGHT);
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
      N_VScale_SComplex(ONE, xcor, vtemp);
      // SUNCheckLastErr();
    }

    /* Add vtemp to initial x to get final solution x, and return */
    if (*zeroguess)
    {
      N_VScale_SComplex(ONE, vtemp, x);
      // SUNCheckLastErr();
    }
    else
    {
      N_VLinearSum_SComplex(ONE, x, ONE, vtemp, x);
      // SUNCheckLastErr();
    }

    *zeroguess  = SUNFALSE;
    LASTFLAG(S) = SUNLS_RES_REDUCED;
    return (LASTFLAG(S));
  }

  *zeroguess  = SUNFALSE;
  LASTFLAG(S) = SUNLS_CONV_FAIL;
  return (LASTFLAG(S));
}

int SUNLinSolNumIters_SPGMRComplex(SUNLinearSolver S)
{
  return (SPGMRComplex_CONTENT(S)->numiters);
}

sunrealtype SUNLinSolResNorm_SPGMRComplex(SUNLinearSolver S)
{
  return (SPGMRComplex_CONTENT(S)->resnorm);
}

N_Vector SUNLinSolResid_SPGMRComplex(SUNLinearSolver S)
{
  return (SPGMRComplex_CONTENT(S)->vtemp);
}

sunindextype SUNLinSolLastFlag_SPGMRComplex(SUNLinearSolver S)
{
  return (LASTFLAG(S));
}

SUNErrCode SUNLinSolSpace_SPGMRComplex(SUNLinearSolver S, long int* lenrwLS,
                                long int* leniwLS)
{
  // SUNFunctionBegin(S->sunctx);
  int maxl;
  sunindextype liw1, lrw1;
  maxl = SPGMRComplex_CONTENT(S)->maxl;
  if (SPGMRComplex_CONTENT(S)->vtemp->ops->nvspace)
  {
    N_VSpace(SPGMRComplex_CONTENT(S)->vtemp, &lrw1, &liw1);
    // SUNCheckLastErr();
  }
  else { lrw1 = liw1 = 0; }
  *lenrwLS = lrw1 * (maxl + 5) + maxl * (maxl + 5) + 2;
  *leniwLS = liw1 * (maxl + 5);
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolFree_SPGMRComplex(SUNLinearSolver S)
{
  int k;

  if (S->content)
  {
    /* delete items from within the content structure */
    if (SPGMRComplex_CONTENT(S)->xcor)
    {
      N_VDestroy_SComplex(SPGMRComplex_CONTENT(S)->xcor);
      SPGMRComplex_CONTENT(S)->xcor = NULL;
    }
    if (SPGMRComplex_CONTENT(S)->vtemp)
    {
      N_VDestroy_SComplex(SPGMRComplex_CONTENT(S)->vtemp);
      SPGMRComplex_CONTENT(S)->vtemp = NULL;
    }
    if (SPGMRComplex_CONTENT(S)->V)
    {
      N_VDestroyVectorArray(SPGMRComplex_CONTENT(S)->V, SPGMRComplex_CONTENT(S)->maxl + 1);
      SPGMRComplex_CONTENT(S)->V = NULL;
    }
    if (SPGMRComplex_CONTENT(S)->Hes)
    {
      for (k = 0; k <= SPGMRComplex_CONTENT(S)->maxl; k++)
      {
        if (SPGMRComplex_CONTENT(S)->Hes[k])
        {
          free(SPGMRComplex_CONTENT(S)->Hes[k]);
          SPGMRComplex_CONTENT(S)->Hes[k] = NULL;
        }
      }
      free(SPGMRComplex_CONTENT(S)->Hes);
      SPGMRComplex_CONTENT(S)->Hes = NULL;
    }
    if (SPGMRComplex_CONTENT(S)->givens)
    {
      free(SPGMRComplex_CONTENT(S)->givens);
      SPGMRComplex_CONTENT(S)->givens = NULL;
    }
    if (SPGMRComplex_CONTENT(S)->yg)
    {
      free(SPGMRComplex_CONTENT(S)->yg);
      SPGMRComplex_CONTENT(S)->yg = NULL;
    }
    if (SPGMRComplex_CONTENT(S)->cv)
    {
      free(SPGMRComplex_CONTENT(S)->cv);
      SPGMRComplex_CONTENT(S)->cv = NULL;
    }
    if (SPGMRComplex_CONTENT(S)->Xv)
    {
      free(SPGMRComplex_CONTENT(S)->Xv);
      SPGMRComplex_CONTENT(S)->Xv = NULL;
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
