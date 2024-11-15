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
 * These test functions check some components of a complex NVECTOR
 * module implementation (for more thorough tests, see the main
 * SUNDIALS repository, inside examples/nvector/).
 * -----------------------------------------------------------------*/

#include <sundials/sundials_config.h>
#include "nvector_serialcomplex.h"
#include <stdio.h>
#include <stdlib.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_nvector.h>

#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM ".17f"
#endif

#define ZERO      SUN_RCONST(0.0)
#define HALF      SUN_RCONST(0.5)
#define ONE       SUN_RCONST(1.0)
#define TWO       SUN_RCONST(2.0)
#define NEG_HALF  SUN_RCONST(-0.5)
#define NEG_ONE   SUN_RCONST(-1.0)
#define NEG_TWO   SUN_RCONST(-2.0)

#define SUN_CCONST(x)     x

#define ZEROi         SUN_CCONST(0.0*I)
#define HALFi         SUN_RCONST(0.5*I)
#define ONEi          SUN_CCONST(1.0*I)
#define TWOi          SUN_CCONST(2.0*I)
#define NEG_HALFi     SUN_RCONST(-0.5*I)
#define NEG_ONEi      SUN_CCONST(-1.0*I)
#define NEG_TWOi      SUN_CCONST(-2.0*I)

/* ----------------------------------------------------------------------
 * Implementation specific utility functions for vector tests
 * --------------------------------------------------------------------*/
int SUNCCompare(suncomplextype a , suncomplextype b)
{
  return (cabs(a - b) > 1.0e-16 /* must be modified with precision*/) ? (1) : (0);
}

int check_ans(suncomplextype ans, N_Vector X, sunindextype local_length)
{
  int failure = 0;
  sunindextype i;
  suncomplextype* Xdata;

  Xdata = N_VGetArrayPointer_SComplex(X);

  /* check vector data */
  for (i = 0; i < local_length; i++) { failure += SUNCCompare(Xdata[i], ans); }

  return (failure > ZERO) ? (1) : (0);
}

sunbooleantype has_data(N_Vector X)
{
  /* check if data array is non-null */
  return (N_VGetArrayPointer_SComplex(X) == NULL) ? SUNFALSE : SUNTRUE;
}

void set_element_range(N_Vector X, sunindextype is, sunindextype ie,
                       sunrealtype val)
{
  sunindextype i;

  /* set elements [is,ie] of the data array */
  suncomplextype* xd = N_VGetArrayPointer_SComplex(X);
  for (i = is; i <= ie; i++) { xd[i] = val; }
}

void set_element(N_Vector X, sunindextype i, sunrealtype val)
{
  /* set i-th element of data array */
  set_element_range(X, i, i, val);
}

sunrealtype get_element(N_Vector X, sunindextype i)
{
  /* get i-th element of data array */
  return NV_Ith_CS(X, i);
}



/* ----------------------------------------------------------------------
 * Main NVector Testing Routine
 * --------------------------------------------------------------------*/
int main(int argc, char* argv[])
{
  int fails = 0;             /* counter for test failures */
  int retval;                /* function return value     */
  sunindextype length;       /* vector length             */
  N_Vector V, W, X, Y, Z;    /* test vectors              */
  SUNContext sunctx = NULL;

  if (SUNContext_Create(SUN_COMM_NULL, &sunctx))
  {
    printf("ERROR: SUNContext_Create failed\n");
    return -1;
  }

  /* check input and set vector length */
  if (argc < 2)
  {
    printf("ERROR: ONE (1) Input required: vector length \n");
    return (-1);
  }

  length = (sunindextype)atol(argv[1]);
  if (length <= 0)
  {
    printf("ERROR: length of vector must be a positive integer \n");
    return (-1);
  }

  printf("Testing custom N_Vector \n");
  printf("Vector length %ld \n", (long int)length);

  /* Create new vectors */
  V = N_VNewEmpty_SComplex(length, sunctx);
  if (V == NULL)
  {
    printf("FAIL: Unable to create a new empty vector \n\n");
    return (1);
  }
  N_VDestroy_SComplex(V);

  X = N_VNew_SComplex(length, sunctx);
  if (X == NULL)
  {
    printf("FAIL: Unable to create a new vector \n\n");
    return (1);
  }

  /* Check vector ID */
  if (N_VGetVectorID_SComplex(X) != SUNDIALS_NVEC_CUSTOM)
  {
    printf(">>> FAILED test -- N_VGetVectorID \n");
    printf("    Unrecognized vector type %d \n \n", N_VGetVectorID(X));
    fails += 1;
  }
  else { printf("PASSED test -- N_VGetVectorID \n"); }

  /* Check vector length */
  sunindextype Xlength = N_VGetLength_SComplex(X);
  N_VConst_SComplex(ONE, X);
  sunindextype Xlength2 = (sunindextype)N_VDotProd_SComplex(X, X);
  if (Xlength != Xlength2)
  {
    printf(">>> FAILED test -- N_VGetLength_SComplex (%li != %li)\n",
           (long int)Xlength, (long int)Xlength2);
    fails += 1;
  }
  else { printf("PASSED test -- N_VGetLength_SComplex\n"); }

  /* Check vector length with a fully imaginary complex dot product*/
  N_VConst_SComplex(ONEi, X);
  sunindextype Xlength3 = (sunindextype)N_VDotProd_SComplex(X, X);
  if (Xlength != Xlength3)
  {
    printf(">>> FAILED test -- N_VGetLength_SComplex (%li != %li)\n",
           (long int)Xlength, (long int)Xlength3);
    fails += 1;
  }
  else { printf("PASSED test -- N_VGetLength_SComplex\n"); }


  /* Check vector length with a complex dot product*/
  N_VConst_SComplex(ONE + ONEi, X);
  sunindextype Xlength4 = (sunindextype)(N_VDotProd_SComplex(X, X)/2.0);
  if (Xlength != Xlength4)
  {
    printf(">>> FAILED test -- N_VGetLength_SComplex (%li != %li)\n",
           (long int)Xlength, (long int)Xlength4);
    fails += 1;
  }
  else { printf("PASSED test -- N_VGetLength_SComplex\n"); }


  /* Test N_VClone_SComplex */
  W = N_VClone_SComplex(X);

  /*   check cloned vector */
  if (W == NULL)
  {
    printf(">>> FAILED test -- N_VClone_SComplex \n");
    printf("    After N_VClone_SComplex, X == NULL \n\n");
    fails += 1;
  }

  /*   check cloned vector data */
  if (!has_data(W))
  {
    printf(">>> FAILED test -- N_VClone_SComplex \n");
    printf("    Vector data == NULL \n\n");
    N_VDestroy_SComplex(W);
    fails += 1;
  }
  else
  {
    N_VConst_SComplex(ONE, W);
    if (check_ans(ONE, W, length))
    {
      printf(">>> FAILED test -- N_VClone_SComplex \n");
      printf("    Failed N_VClone_SComplex check \n\n");
      N_VDestroy_SComplex(W);
      fails += 1;
    }
    else
    { printf("PASSED test -- N_VClone_SComplex \n"); }
    N_VDestroy_SComplex(W);
  }

  /* Clone additional vectors for testing */
  Y = N_VClone_SComplex(X);
  if (Y == NULL)
  {
    N_VDestroy_SComplex(W);
    N_VDestroy_SComplex(X);
    printf("FAIL: Unable to create a new vector \n\n");
    return (1);
  }

  Z = N_VClone_SComplex(X);
  if (Z == NULL)
  {
    N_VDestroy_SComplex(W);
    N_VDestroy_SComplex(X);
    N_VDestroy_SComplex(Y);
    printf("FAIL: Unable to create a new vector \n\n");
    return (1);
  }

  /* Standard vector operation tests */
  printf("\nTesting standard vector operations:\n\n");

  /* Test N_VConst_SComplex: fill vector data with zeros to prevent passing
     in the case where the input vector is a vector of ones */
  set_element_range(X, 0, length - 1, ZERO);
  N_VConst_SComplex((ONE + TWOi), X);
  if (check_ans((ONE + TWOi), X, length))
  {
    printf(">>> FAILED test -- N_VConst_SComplex \n");
    fails += 1;
  }
  else { printf("PASSED test -- N_VConst_SComplex \n"); }

  /* Test N_VLinearSum_SComplex */

  /*   Case 1a: y = x + y, (Vaxpy Case 1) */
  N_VConst_SComplex((    ONE + HALFi), X);
  N_VConst_SComplex((NEG_TWO - ONEi), Y);
  N_VLinearSum_SComplex(TWO, X, ONE, Y, Y);
  if (check_ans(ZERO, Y, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 1a \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 1a \n"); }

  /*   Case 1b: y = -x + y, (Vaxpy Case 2) */
  N_VConst_SComplex((    ONE + HALFi), X);
  N_VConst_SComplex((NEG_ONE + ONEi), Y);
  N_VLinearSum_SComplex(NEG_ONE, X, ONE, Y, Y);
  if (check_ans(NEG_TWO + HALFi, Y, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 1b \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 1b \n"); }

  /*   Case 1c: y = ax + y, (Vaxpy Case 3) */
  N_VConst_SComplex(TWO + TWOi, X);
  N_VConst_SComplex(NEG_TWO - HALFi, Y);
  N_VLinearSum_SComplex(HALF, X, ONE, Y, Y);
  if (check_ans(NEG_ONE + HALFi, Y, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 1c \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 1c \n"); }

  /*   Case 2a: x = x + y, (Vaxpy Case 1) */
  N_VConst_SComplex(TWO + TWOi, X);
  N_VConst_SComplex(NEG_TWO - HALFi, Y);
  N_VLinearSum_SComplex(ONE, X, ONE, Y, X);
  if (check_ans(ONEi + HALFi, X, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 2a \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 2a \n"); }

  /*   Case 2b: x = x - y, (Vaxpy Case 2)*/
  N_VConst_SComplex(NEG_ONE + TWOi, X);
  N_VConst_SComplex(NEG_TWO + HALFi, Y);
  N_VLinearSum_SComplex(ONE, X, NEG_ONE, Y, X);
  if (check_ans(ONE + ONEi + HALFi, X, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 2b \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 2b \n"); }

  /*   Case 2c: x = x + by, (Vaxpy Case 3) */
  N_VConst_SComplex(TWO + TWOi, X);
  N_VConst_SComplex(NEG_HALF + NEG_HALFi, Y);
  N_VLinearSum_SComplex(ONE, X, TWO, Y, X);
  if (check_ans(ONE + ONEi, X, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 2c \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 2c \n"); }

  /*   Case 3: z = x + y, (VSum) */
  N_VConst_SComplex(ONE + NEG_TWOi, X);
  N_VConst_SComplex(NEG_ONE + ONEi, Y);
  N_VConst_SComplex(ZERO, Z);
  N_VLinearSum_SComplex(ONE, X, ONE, Y, Z);
  if (check_ans(NEG_ONEi, Z, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 3 \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 3 \n"); }

  /*   Case 4a: z = x - y, (VDiff) */
  N_VConst_SComplex(ONE + TWOi, X);
  N_VConst_SComplex(NEG_ONE + ONEi, Y);
  N_VConst_SComplex(ZERO, Z);
  N_VLinearSum_SComplex(ONE, X, NEG_ONE, Y, Z);
  if (check_ans(TWO + ONEi, Z, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 4a \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 4a \n"); }

  /*   Case 4b: z = -x + y, (VDiff) */
  N_VConst_SComplex(ONE + TWOi, X);
  N_VConst_SComplex(HALF + ONEi, Y);
  N_VConst_SComplex(ZERO, Z);
  N_VLinearSum_SComplex(NEG_ONE, X, ONE, Y, Z);
  if (check_ans(NEG_HALF + NEG_ONEi, Z, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 4b \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 4b \n"); }

  /*   Case 5a: z = x + by, (VLin1) */
  N_VConst_SComplex(HALFi, X);
  N_VConst_SComplex(ONE + NEG_HALFi, Y);
  N_VConst_SComplex(ZERO, Z);
  N_VLinearSum_SComplex(ONE, X, TWO, Y, Z);
  if (check_ans(TWO + NEG_HALFi, Z, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 5a \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 5a \n"); }

  /*   Case 5b: z = ax + y, (VLin1) */
  N_VConst_SComplex(ONE + NEG_HALFi, X);
  N_VConst_SComplex(NEG_ONE - NEG_TWOi, Y);
  N_VConst_SComplex(ZERO, Z);
  N_VLinearSum_SComplex(TWO, X, ONE, Y, Z);
  if (check_ans(ONE + ONEi, Z, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 5b \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 5b \n"); }

  /*   Case 6a: z = -x + by, (VLin2) */
  N_VConst_SComplex(ONE + NEG_HALFi, X);
  N_VConst_SComplex(NEG_ONE + TWOi, Y);
  N_VConst_SComplex(ZERO, Z);
  N_VLinearSum_SComplex(NEG_ONE, X, HALF, Y, Z);
  if (check_ans(NEG_TWO + HALF + ONEi + HALFi, Z, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 6a \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 6a \n"); }

  /*   Case 6b: z = ax - y, (VLin2) */
  N_VConst_SComplex(ONE + NEG_HALFi, X);
  N_VConst_SComplex(ONE + NEG_TWOi, Y);
  N_VConst_SComplex(ZERO, Z);
  N_VLinearSum_SComplex(TWO, X, NEG_ONE, Y, Z);
  if (check_ans(ONE + ONEi, Z, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 6b \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 6b \n"); }

  /*   Case 7: z = a(x + y), (VScaleSum) */
  N_VConst_SComplex(HALF + ONEi + HALFi, X);
  N_VConst_SComplex(NEG_ONE + NEG_HALFi, Y);
  N_VConst_SComplex(ZERO, Z);
  N_VLinearSum_SComplex(TWO, X, TWO, Y, Z);
  if (check_ans(NEG_ONE + TWOi, Z, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 7 \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 7 \n"); }

  /*   Case 8: z = a(x - y), (VScaleDiff) */
  N_VConst_SComplex(ONE + HALF - HALFi, X);
  N_VConst_SComplex(HALF + HALFi, Y);
  N_VConst_SComplex(ZERO, Z);
  N_VLinearSum_SComplex(TWO, X, NEG_TWO, Y, Z);
  if (check_ans(TWO - TWOi, Z, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 8 \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 8 \n"); }

  /*   Case 9: z = ax + by, All Other Cases */
  N_VConst_SComplex(ONE + NEG_TWOi, X);
  N_VConst_SComplex(HALF + HALFi, Y);
  N_VConst_SComplex(ZERO, Z);
  N_VLinearSum_SComplex(HALF, X, TWO, Y, Z);
  if (check_ans((ONE + HALF), Z, length))
  {
    printf(">>> FAILED test -- N_VLinearSum_SComplex Case 9 \n");
    fails++;
  }
  else { printf("PASSED test -- N_VLinearSum_SComplex Case 9 \n"); }


  /* Test N_VScale_SComplex */

  /*   Case 1: x = cx, VScaleBy */
  N_VConst_SComplex(HALF + HALFi, X);
  N_VScale_SComplex(TWO, X, X);
  if (check_ans(ONE + ONEi, X, length))
  {
    printf(">>> FAILED test -- N_VScale_SComplex Case 1 \n");
    fails++;
  }
  else { printf("PASSED test -- N_VScale_SComplex Case 1 \n"); }

  /*   Case 2: z = x, VCopy */
  N_VConst_SComplex(NEG_ONE + ONEi, X);
  N_VConst_SComplex(ZERO, Z);
  N_VScale_SComplex(ONE, X, Z);
  if (check_ans(NEG_ONE + ONEi, Z, length))
  {
    printf(">>> FAILED test -- N_VScale_SComplex Case 2\n");
    fails++;
  }
  else { printf("PASSED test -- N_VScale_SComplex Case 2 \n"); }

  /*   Case 3: z = -x, VNeg */
  N_VConst_SComplex(NEG_ONE + ONEi, X);
  N_VConst_SComplex(ZERO, Z);
  N_VScale_SComplex(NEG_ONE, X, Z);
  if (check_ans(ONE + NEG_ONEi, Z, length))
  {
    printf(">>> FAILED test -- N_VScale_SComplex Case 3 \n");
    fails++;
  }
  else { printf("PASSED test -- N_VScale_SComplex Case 3 \n"); }

  /*   Case 4: z = cx, All other cases */
  N_VConst_SComplex(NEG_HALF + ONEi, X);
  N_VConst_SComplex(ZERO, Z);
  N_VScale_SComplex(TWO, X, Z);
  if (check_ans(NEG_ONE + TWOi, Z, length))
  {
    printf(">>> FAILED test -- N_VScale_SComplex Case 4 \n");
    fails++;
  }
  else { printf("PASSED test -- N_VScale_SComplex Case 4 \n"); }

  /* Test N_VDotProd_SComplex */
  N_VConst_SComplex(TWO - HALFi, X);
  N_VConst_SComplex(ONE + ONEi, Y);
  sunindextype global_length = N_VGetLength_SComplex(X);
  sunindextype ans = N_VDotProd_SComplex(X, Y);

  /* ans should equal global vector length */
  if (SUNRCompare(ans, ((suncomplextype)global_length)*(ONE + HALF + NEG_TWOi + NEG_HALFi)))
  {
    printf(">>> FAILED test -- N_VDotProd_SComplex \n");
    fails++;
  }
  else { printf("PASSED test -- N_VDotProd_SComplex \n"); }

  /* Print result */
  if (fails) { printf("FAIL: NVector module failed %i tests \n\n", fails); }
  else { printf("SUCCESS: NVector module passed all tests \n\n"); }

  return (fails);
}
