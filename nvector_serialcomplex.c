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
 * This is the implementation file for a complex N_Vector
 * (note that this template just implements the serial N_Vector).
 * -----------------------------------------------------------------*/

#include "nvector_serialcomplex.h"
#include <stdio.h>
#include <stdlib.h>
#include <sundials/sundials_core.h>
#include "sundials/sundials_errors.h"

#define ZERO   SUN_RCONST(0.0) // check every occurance!
#define ONE    SUN_RCONST(1.0) // check every occurance!

/* Private functions for special cases of vector operations */
static void VCopy_SComplex(N_Vector x, N_Vector z);             /* z=x       */
static void VSum_SComplex(N_Vector x, N_Vector y, N_Vector z);  /* z=x+y     */
static void VDiff_SComplex(N_Vector x, N_Vector y, N_Vector z); /* z=x-y     */
static void VNeg_SComplex(N_Vector x, N_Vector z);              /* z=-x      */
static void VScaleSum_SComplex(suncomplextype c, N_Vector x, N_Vector y,
                           N_Vector z); /* z=c(x+y)  */
static void VScaleDiff_SComplex(suncomplextype c, N_Vector x, N_Vector y,
                            N_Vector z); /* z=c(x-y)  */
static void VLin1_SComplex(suncomplextype a, N_Vector x, N_Vector y,
                       N_Vector z); /* z=ax+y    */
static void VLin2_SComplex(suncomplextype a, N_Vector x, N_Vector y,
                       N_Vector z);                            /* z=ax-y    */
static void Vaxpy_SComplex(suncomplextype a, N_Vector x, N_Vector y); /* y <- ax+y */
static void VScaleBy_SComplex(suncomplextype a, N_Vector x);          /* x <- ax   */

/* Private functions for special cases of vector array operations */
static void VSumVectorArray_SComplex(int nvec, N_Vector* X, N_Vector* Y,
                                 N_Vector* Z); /* Z=X+Y     */
static void VDiffVectorArray_SComplex(int nvec, N_Vector* X, N_Vector* Y,
                                  N_Vector* Z); /* Z=X-Y     */
static void VScaleSumVectorArray_SComplex(int nvec, suncomplextype c, N_Vector* X,
                                      N_Vector* Y, N_Vector* Z); /* Z=c(X+Y)  */
static void VScaleDiffVectorArray_SComplex(int nvec, suncomplextype c, N_Vector* X,
                                       N_Vector* Y, N_Vector* Z); /* Z=c(X-Y)  */
static void VLin1VectorArray_SComplex(int nvec, suncomplextype a, N_Vector* X,
                                  N_Vector* Y, N_Vector* Z); /* Z=aX+Y    */
static void VLin2VectorArray_SComplex(int nvec, suncomplextype a, N_Vector* X,
                                  N_Vector* Y, N_Vector* Z); /* Z=aX-Y    */
static void VaxpyVectorArray_SComplex(int nvec, suncomplextype a, N_Vector* X,
                                  N_Vector* Y); /* Y <- aX+Y */

/*
 * -----------------------------------------------------------------
 * exported functions
 * -----------------------------------------------------------------
 */

/* ----------------------------------------------------------------------------
 * Function to create a new empty serial vector
 */

N_Vector N_VNewEmpty_SComplex(sunindextype length, SUNContext sunctx)
{

  N_Vector v;
  CS_NVectorContent content;

  if (length < 0) { return NULL; }

  /* Create an empty vector object */
  v = NULL;
  v = N_VNewEmpty(sunctx);

  /* Attach operations */

  /* constructors, destructors, and utility operations */
  v->ops->nvgetvectorid     = N_VGetVectorID_SComplex;
  v->ops->nvclone           = N_VClone_SComplex;
  v->ops->nvcloneempty      = N_VCloneEmpty_SComplex;
  v->ops->nvdestroy         = N_VDestroy_SComplex;
  v->ops->nvspace           = N_VSpace_SComplex;
  
  v->ops->nvgetlength       = N_VGetLength_SComplex;
  v->ops->nvgetlocallength  = N_VGetLength_SComplex;

  /* standard vector operations */
  // v->ops->nvlinearsum    = N_VLinearSum_SComplex;
  v->ops->nvlinearsum    = N_VLinearSum_Real;
  
  // v->ops->nvconst        = N_VConst_SComplex;
  v->ops->nvconst        = N_VConst_Real;

  v->ops->nvprod         = N_VProd_SComplex;
  v->ops->nvdiv          = N_VDiv_SComplex;
  
  // v->ops->nvscale        = N_VScale_SComplex;
  v->ops->nvscale        = N_VScale_Real;
  
  v->ops->nvabs          = N_VAbs_SComplex;
  v->ops->nvinv          = N_VInv_SComplex;

  // v->ops->nvaddconst     = N_VAddConst_SComplex;
  v->ops->nvaddconst     = N_VAddConst_Real;

  // v->ops->nvdotprod      = N_VDotProd_SComplex;
  v->ops->nvdotprod      = N_VDotProd_Real;

  v->ops->nvmaxnorm      = N_VMaxNorm_SComplex;
  v->ops->nvwrmsnorm     = N_VWrmsNorm_SComplex;
  v->ops->nvwl2norm      = N_VWL2Norm_SComplex;
  v->ops->nvl1norm       = N_VL1Norm_SComplex;

  /* fused and vector array operations are disabled (NULL) by default */

  /* local reduction operations */
  // v->ops->nvdotprodlocal     = N_VDotProd_SComplex;
  v->ops->nvdotprodlocal     = N_VDotProd_Real;

  v->ops->nvmaxnormlocal     = N_VMaxNorm_SComplex;
  v->ops->nvl1normlocal      = N_VL1Norm_SComplex;
  v->ops->nvwsqrsumlocal     = N_VWSqrSumLocal_SComplex;

  /* single buffer reduction operations */
  // v->ops->nvdotprodmultilocal = N_VDotProdMulti_SComplex;
  v->ops->nvdotprodmultilocal = N_VDotProdMulti_Real;

  /* XBraid interface operations */
  v->ops->nvbufsize   = N_VBufSize_SComplex;
  v->ops->nvbufpack   = N_VBufPack_SComplex;
  v->ops->nvbufunpack = N_VBufUnpack_SComplex;

  /* debugging functions */
  v->ops->nvprint     = N_VPrint_SComplex;
  v->ops->nvprintfile = N_VPrintFile_SComplex;

  /* Create content */
  content = NULL;
  content = (CS_NVectorContent)malloc(sizeof *content);
  if (content == NULL) { return NULL; }

  /* Attach content */
  v->content = content;

  /* Initialize content */
  content->length   = length;
  content->own_data = SUNFALSE;
  content->complex_data = NULL;

  return (v);
}

/* ----------------------------------------------------------------------------
 * Function to create a new serial vector
 */

N_Vector N_VNew_SComplex(sunindextype length, SUNContext sunctx)
{
  N_Vector v;
  suncomplextype* complex_data;

  if (length < 0) { return NULL; }

  v = NULL;
  v = N_VNewEmpty_SComplex(length, sunctx);
  if (v == NULL) { return NULL; }

  /* Create data */
  complex_data = NULL;
  if (length > 0)
  {
    complex_data = (suncomplextype*)malloc(length * sizeof(suncomplextype));
    if (complex_data == NULL) { return NULL; }
  }

  /* Attach data */
  NV_OWN_DATA_CS(v) = SUNTRUE;
  NV_COMPLEX_DATA_CS(v) = complex_data;

  return (v);
}

/* ----------------------------------------------------------------------------
 * Function to create a serial N_Vector with user data component
 */

N_Vector N_VMake_SComplex(sunindextype length, suncomplextype* v_data,
                        SUNContext sunctx)
{
  N_Vector v;

  if (length < 0) { return NULL; }

  v = NULL;
  v = N_VNewEmpty_SComplex(length, sunctx);
  if (v == NULL) { return NULL; }

  if (length > 0)
  {
    /* Attach data */
    NV_OWN_DATA_CS(v) = SUNFALSE;
    NV_COMPLEX_DATA_CS(v) = v_data;
  }

  return (v);
}

/* ----------------------------------------------------------------------------
 * Function to return number of vector elements
 */
sunindextype N_VGetLength_SComplex(N_Vector v) { return NV_LENGTH_CS(v); }

/* ----------------------------------------------------------------------------
 * Function to print the a serial vector to stdout
 */

void N_VPrint_SComplex(N_Vector x)
{
  N_VPrintFile_SComplex(x, stdout);
}

/* ----------------------------------------------------------------------------
 * Function to print the a serial vector to outfile
 */

void N_VPrintFile_SComplex(N_Vector x, FILE* outfile)
{
  sunindextype i, N;
  suncomplextype* xd;

  xd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);

  for (i = 0; i < N; i++)
  {
#if defined(SUNDIALS_EXTENDED_PRECISION)
    fprintf(outfile, "%35.32Le + i%35.32Le\n", creal(xd[i]), cimag(xd[i]));
#elif defined(SUNDIALS_DOUBLE_PRECISION)
    fprintf(outfile, "%19.16e + i%19.16e\n", creal(xd[i]), cimag(xd[i]));
#else
    fprintf(outfile, "%11.8e + i%11.8e\n", creal(xd[i]), cimag(xd[i]));
#endif
  }
  fprintf(outfile, "\n");

  return;
}

/*
 * -----------------------------------------------------------------
 * implementation of vector operations
 * -----------------------------------------------------------------
 */

N_Vector N_VCloneEmpty_SComplex(N_Vector w)
{
  N_Vector v;
  CS_NVectorContent content;

  /* Create vector */
  v = NULL;
  v = N_VNewEmpty(w->sunctx);
  if (v == NULL) { return NULL; }

  /* Attach operations */
  N_VCopyOps(w, v);

  /* Create content */
  content = NULL;
  content = (CS_NVectorContent)malloc(sizeof *content);
  if (content == NULL) { return NULL; }

  /* Attach content */
  v->content = content;

  /* Initialize content */
  content->length   = NV_LENGTH_CS(w);
  content->own_data = SUNFALSE;
  content->complex_data = NULL;

  return (v);
}

N_Vector N_VClone_SComplex(N_Vector w)
{
  N_Vector v;
  suncomplextype* complex_data;
  sunindextype length;

  v = NULL;
  v = N_VCloneEmpty_SComplex(w);
  if (v == NULL) { return NULL; }

  length = NV_LENGTH_CS(w);

  /* Create complex_data */
  complex_data = NV_COMPLEX_DATA_CS(v) = NULL;
  if (length > 0)
  {
    complex_data = (suncomplextype*)malloc(length * sizeof(suncomplextype));
    if (complex_data == NULL) { return NULL; }
    
    /* Attach complex_data */
    NV_OWN_DATA_CS(v) = SUNTRUE;
    NV_COMPLEX_DATA_CS(v)     = complex_data;
  }

  return (v);
}

void N_VDestroy_SComplex(N_Vector v)
{
  if (v == NULL) { return; }

  /* free content */
  if (v->content != NULL)
  {
    /* free data array if it's owned by the vector */
    if (NV_OWN_DATA_CS(v) && (NV_COMPLEX_DATA_CS(v) != NULL))
    {
      free(NV_COMPLEX_DATA_CS(v));
      NV_COMPLEX_DATA_CS(v) = NULL;
    }
    free(v->content);
    v->content = NULL;
  }

  /* free ops and vector */
  if (v->ops != NULL)
  {
    free(v->ops);
    v->ops = NULL;
  }
  free(v);
  v = NULL;

  return;
}

void N_VSpace_SComplex(N_Vector v, sunindextype* lrw, sunindextype* liw)
{
  if (lrw == NULL) { return; }
  if (liw == NULL) { return; }

  *lrw = NV_LENGTH_CS(v);
  *liw = 1;

  return;
}

suncomplextype* N_VGetArrayPointer_SComplex(N_Vector v)
{
    
  return ((suncomplextype*)NV_COMPLEX_DATA_CS(v));
}

void N_VSetArrayPointer_SComplex(suncomplextype* v_data, N_Vector v)
{
  if (NV_LENGTH_CS(v) > 0) { NV_COMPLEX_DATA_CS(v) = v_data; }

  return;
}

void N_VLinearSum_SComplex(suncomplextype a, N_Vector x, suncomplextype b, N_Vector y,
                       N_Vector z)
{
  sunindextype i, N;
  suncomplextype c, *xd, *yd, *zd;
  N_Vector v1, v2;
  sunbooleantype test;

  xd = yd = zd = NULL;

  if (((creal(b) == ONE) && (cimag(b) == ZERO)) && (z == y))
  { /* BLAS usage: axpy y <- ax+y */
    Vaxpy_SComplex(a, x, y);
    return;
  }

  if (((creal(a) == ONE) && (cimag(a) == ZERO)) && (z == x))
  { /* BLAS usage: axpy x <- by+x */
    Vaxpy_SComplex(b, y, x);
    return;
  }

  /* Case: a == b == 1.0 */

  if (((creal(a) == ONE) && (cimag(a) == ZERO)) && ((creal(b) == ONE) && (cimag(b) == ZERO)))
  {
    VSum_SComplex(x, y, z);
    return;
  }

  /* Cases: (1) a == 1.0, b = -1.0, (2) a == -1.0, b == 1.0 */

  if ((test = (((creal(a) == ONE) && (cimag(a) == ZERO)) && ((creal(b) == -ONE) && (cimag(b) == ZERO)))) || (((creal(a) == -ONE) && (cimag(a) == ZERO)) && ((creal(b) == ONE) && (cimag(b) == ZERO))))
  {
    v1 = test ? y : x;
    v2 = test ? x : y;
    VDiff_SComplex(v2, v1, z);
    return;
  }

  /* Cases: (1) a == 1.0, b == other or 0.0, (2) a == other or 0.0, b == 1.0 */
  /* if a or b is 0.0, then user should have called N_VScale */

  if ((test = ((creal(a) == ONE) && (cimag(a) == ZERO))) || ((creal(b) == ONE) && (cimag(b) == ZERO)))
  {
    c  = test ? b : a;
    v1 = test ? y : x;
    v2 = test ? x : y;
    VLin1_SComplex(c, v1, v2, z);
    return;
  }

  /* Cases: (1) a == -1.0, b != 1.0, (2) a != 1.0, b == -1.0 */

  if ((test = ((creal(a) == -ONE) && (cimag(a) == ZERO))) || ((creal(b) == -ONE) && (cimag(b) == ZERO)))
  {
    c  = test ? b : a;
    v1 = test ? y : x;
    v2 = test ? x : y;
    VLin2_SComplex(c, v1, v2, z);
    return;
  }

  /* Case: a == b */
  /* catches case both a and b are 0.0 - user should have called N_VConst */

  if (a == b)
  {
    VScaleSum_SComplex(a, x, y, z);
    return;
  }

  /* Case: a == -b */

  if (a == -b)
  {
    VScaleDiff_SComplex(a, x, y, z);
    return;
  }

  /* Do all cases not handled above:
     (1) a == other, b == 0.0 - user should have called N_VScale
     (2) a == 0.0, b == other - user should have called N_VScale
     (3) a,b == other, a !=b, a != -b */

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  yd = NV_COMPLEX_DATA_CS(y);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = (a * xd[i]) + (b * yd[i]); }

  return;
}

void N_VLinearSum_Real(sunrealtype a, N_Vector x, sunrealtype b, N_Vector y,
                       N_Vector z)
{  
  sunindextype i, N;
  sunrealtype c;
  suncomplextype *xd, *yd, *zd;
  N_Vector v1, v2;
  sunbooleantype test;

  xd = yd = zd = NULL;

  if ((b == ONE) && (z == y))
  { /* BLAS usage: axpy y <- ax+y */
    Vaxpy_SComplex(a, x, y);
    return;
  }

  if ((a == ONE) && (z == x))
  { /* BLAS usage: axpy x <- by+x */
    Vaxpy_SComplex(b, y, x);
    return;
  }

  /* Case: a == b == 1.0 */

  if ((a == ONE) && (b == ONE))
  {
    VSum_SComplex(x, y, z);
    return;
  }

  /* Cases: (1) a == 1.0, b = -1.0, (2) a == -1.0, b == 1.0 */

  if ((test = ((a == ONE) && (b == -ONE))) || ((a == -ONE) && (b == ONE)))
  {
    v1 = test ? y : x;
    v2 = test ? x : y;
    VDiff_SComplex(v2, v1, z);
    return;
  }

  /* Cases: (1) a == 1.0, b == other or 0.0, (2) a == other or 0.0, b == 1.0 */
  /* if a or b is 0.0, then user should have called N_VScale */

  if ((test = (a == ONE)) || (b == ONE))
  {
    c  = test ? b : a;
    v1 = test ? y : x;
    v2 = test ? x : y;
    VLin1_SComplex(c, v1, v2, z);
    return;
  }

  /* Cases: (1) a == -1.0, b != 1.0, (2) a != 1.0, b == -1.0 */

  if ((test = (a == -ONE)) || (b == -ONE))
  {
    c  = test ? b : a;
    v1 = test ? y : x;
    v2 = test ? x : y;
    VLin2_SComplex(c, v1, v2, z);
    return;
  }

  /* Case: a == b */
  /* catches case both a and b are 0.0 - user should have called N_VConst */

  if (a == b)
  {
    VScaleSum_SComplex(a, x, y, z);
    return;
  }

  /* Case: a == -b */

  if (a == -b)
  {
    VScaleDiff_SComplex(a, x, y, z);
    return;
  }

  /* Do all cases not handled above:
     (1) a == other, b == 0.0 - user should have called N_VScale
     (2) a == 0.0, b == other - user should have called N_VScale
     (3) a,b == other, a !=b, a != -b */

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  yd = NV_COMPLEX_DATA_CS(y);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = (a * xd[i]) + (b * yd[i]); }

  return;
}

void N_VConst_SComplex(suncomplextype c, N_Vector z)
{
  sunindextype i, N;
  suncomplextype* zd;

  zd = NULL;

  N  = NV_LENGTH_CS(z);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = c; }

  return;
}

void N_VConst_Real(sunrealtype c, N_Vector z)
{
  sunindextype i, N;
  suncomplextype* zd;

  zd = NULL;

  N  = NV_LENGTH_CS(z);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = (suncomplextype)c; }

  return;
}

void N_VProd_SComplex(N_Vector x, N_Vector y, N_Vector z)
{
  sunindextype i, N;
  suncomplextype *xd, *yd, *zd;

  xd = yd = zd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  yd = NV_COMPLEX_DATA_CS(y);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = xd[i] * yd[i]; }

  return;
}

void N_VDiv_SComplex(N_Vector x, N_Vector y, N_Vector z)
{
  sunindextype i, N;
  suncomplextype *xd, *yd, *zd;

  xd = yd = zd = NULL;
  
  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  yd = NV_COMPLEX_DATA_CS(y);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = xd[i] / yd[i]; }

  return;
}

void N_VScale_SComplex(suncomplextype c, N_Vector x, N_Vector z)
{
  
  sunindextype i, N;
  suncomplextype *xd, *zd;

  xd = zd = NULL;

  if (z == x)
  { /* BLAS usage: scale x <- cx */
    VScaleBy_SComplex(c, x);
    return;
  }

  if ((creal(c) == ONE) && (cimag(c) == ZERO)) { VCopy_SComplex(x, z); }
  else if ((creal(c) == -ONE) && (cimag(c) == ZERO)) { VNeg_SComplex(x, z); }
  else
  {
    N  = NV_LENGTH_CS(x);
    xd = NV_COMPLEX_DATA_CS(x);
    zd = NV_COMPLEX_DATA_CS(z);
    for (i = 0; i < N; i++) { zd[i] = c * xd[i]; }
  }

  return;
}

void N_VScale_Real(sunrealtype c, N_Vector x, N_Vector z)
{  
  sunindextype i, N;
  suncomplextype *xd, *zd;

  xd = zd = NULL;

  if (z == x)
  { /* BLAS usage: scale x <- cx */
    VScaleBy_SComplex(c, x);
    return;
  }

  if ((creal(c) == ONE) && (cimag(c) == ZERO)) { VCopy_SComplex(x, z); }
  else if ((creal(c) == -ONE) && (cimag(c) == ZERO)) { VNeg_SComplex(x, z); }
  else
  {
    N  = NV_LENGTH_CS(x);
    xd = NV_COMPLEX_DATA_CS(x);
    zd = NV_COMPLEX_DATA_CS(z);
    for (i = 0; i < N; i++) { zd[i] = c * xd[i]; }
  }

  return;
}


void N_VAbs_SComplex(N_Vector x, N_Vector z)
{

  sunindextype i, N;
  suncomplextype *xd, *zd;

  xd = zd = NULL;
  
  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = SUNCabs(xd[i]); }

  return;
}

void N_VInv_SComplex(N_Vector x, N_Vector z)
{
  sunindextype i, N;
  suncomplextype *xd, *zd;

  xd = zd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = (suncomplextype)ONE / xd[i]; }

  return;
}

void N_VAddConst_SComplex(N_Vector x, suncomplextype b, N_Vector z)
{
  sunindextype i, N;
  suncomplextype *xd, *zd;

  xd = zd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = xd[i] + b; }

  return;
}

void N_VAddConst_Real(N_Vector x, sunrealtype b, N_Vector z)
{
  sunindextype i, N;
  suncomplextype *xd, *zd;

  xd = zd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = xd[i] + b; }

  return;
}

suncomplextype N_VDotProd_SComplex(N_Vector x, N_Vector y)
{
  sunindextype i, N;
  suncomplextype sum, *xd, *yd;

  sum = (suncomplextype)ZERO;
  xd = yd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  yd = NV_COMPLEX_DATA_CS(y);

  for (i = 0; i < N; i++) { sum += conj(xd[i]) * yd[i]; } //Amihere

  return (sum);
}

sunrealtype N_VDotProd_Real(N_Vector x, N_Vector y)
{
  printf("\nCalling N_VDotProd_Real, that must cause an error!\n\n");

  sunindextype i, N;
  sunrealtype sum;
  suncomplextype *xd, *yd;

  sum = ZERO;
  xd = yd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  yd = NV_COMPLEX_DATA_CS(y);

  for (i = 0; i < N; i++) { sum += xd[i] * yd[i]; }

  return (sum);
}

sunrealtype N_VMaxNorm_SComplex(N_Vector x)
{
  sunindextype i, N;
  sunrealtype max;
  suncomplextype *xd;

  max = ZERO;
  xd  = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);

  for (i = 0; i < N; i++)
  {
    if (SUNCabs(xd[i]) > max) { max = SUNCabs(xd[i]); }
  }

  return (max);
}

sunrealtype N_VWrmsNorm_SComplex(N_Vector x, N_Vector w)
{
  sunrealtype norm = N_VWSqrSumLocal_SComplex(x, w);
  norm = SUNRsqrt(norm / NV_LENGTH_CS(x));
  return norm;
}

sunrealtype N_VWSqrSumLocal_SComplex(N_Vector x, N_Vector w)
{
  sunindextype i, N;
  sunrealtype sum;
  suncomplextype prodi, *xd, *wd;

  sum = ZERO;
  xd = wd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  wd = NV_COMPLEX_DATA_CS(w);

  for (i = 0; i < N; i++)
  {
    prodi = xd[i] * wd[i];
    sum += SUNSQR(SUNCabs(prodi));
  }

  return (sum);
}

sunrealtype N_VWL2Norm_SComplex(N_Vector x, N_Vector w)
{
  sunindextype i, N;
  sunrealtype sum;
  suncomplextype prodi, *xd, *wd;

  sum = ZERO;
  xd = wd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  wd = NV_COMPLEX_DATA_CS(w);

  for (i = 0; i < N; i++)
  {
    prodi = xd[i] * wd[i];
    sum += SUNSQR(SUNCabs(prodi));
  }

  return (SUNRsqrt(sum));
}

sunrealtype N_VL1Norm_SComplex(N_Vector x)
{
  sunindextype i, N;
  sunrealtype sum; 
  suncomplextype *xd;

  sum = ZERO;
  xd  = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);

  for (i = 0; i < N; i++) { sum += SUNCabs(xd[i]); }

  return (sum);
}

/*
 * -----------------------------------------------------------------
 * fused vector operations
 * -----------------------------------------------------------------
 */

SUNErrCode N_VLinearCombination_SComplex(int nvec, suncomplextype* c, N_Vector* X,
                                     N_Vector z)
{
  int i;
  sunindextype j, N;
  suncomplextype* zd = NULL;
  suncomplextype* xd = NULL;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* should have called N_VScale */
  if (nvec == 1)
  {
    N_VScale_SComplex(c[0], X[0], z);
    return SUN_SUCCESS;
  }

  /* should have called N_VLinearSum */
  if (nvec == 2)
  {
    N_VLinearSum_SComplex(c[0], X[0], c[1], X[1], z);
    return SUN_SUCCESS;
  }

  /* get vector length and data array */
  N  = NV_LENGTH_CS(z);
  zd = NV_COMPLEX_DATA_CS(z);

  /*
   * X[0] += c[i]*X[i], i = 1,...,nvec-1
   */
  if ((X[0] == z) && ((creal(c[0]) == ONE) && (cimag(c[0]) == ZERO)))
  {
    for (i = 1; i < nvec; i++)
    {
      xd = NV_COMPLEX_DATA_CS(X[i]);
      for (j = 0; j < N; j++) { zd[j] += c[i] * xd[j]; }
    }
    return SUN_SUCCESS;
  }

  /*
   * X[0] = c[0] * X[0] + sum{ c[i] * X[i] }, i = 1,...,nvec-1
   */
  if (X[0] == z)
  {
    for (j = 0; j < N; j++) { zd[j] *= c[0]; }
    for (i = 1; i < nvec; i++)
    {
      xd = NV_COMPLEX_DATA_CS(X[i]);
      for (j = 0; j < N; j++) { zd[j] += c[i] * xd[j]; }
    }
    return SUN_SUCCESS;
  }

  /*
   * z = sum{ c[i] * X[i] }, i = 0,...,nvec-1
   */
  xd = NV_COMPLEX_DATA_CS(X[0]);
  for (j = 0; j < N; j++) { zd[j] = c[0] * xd[j]; }
  for (i = 1; i < nvec; i++)
  {
    xd = NV_COMPLEX_DATA_CS(X[i]);
    for (j = 0; j < N; j++) { zd[j] += c[i] * xd[j]; }
  }

  return SUN_SUCCESS;
}

SUNErrCode N_VLinearCombination_Real(int nvec, sunrealtype* c, N_Vector* X,
                                     N_Vector z)
{
  int i;
  sunindextype j, N;
  suncomplextype* zd = NULL;
  suncomplextype* xd = NULL;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* should have called N_VScale */
  if (nvec == 1)
  {
    N_VScale_SComplex(c[0], X[0], z);
    return SUN_SUCCESS;
  }

  /* should have called N_VLinearSum */
  if (nvec == 2)
  {
    N_VLinearSum_SComplex(c[0], X[0], c[1], X[1], z);
    return SUN_SUCCESS;
  }

  /* get vector length and data array */
  N  = NV_LENGTH_CS(z);
  zd = NV_COMPLEX_DATA_CS(z);

  /*
   * X[0] += c[i]*X[i], i = 1,...,nvec-1
   */
  if ((X[0] == z) && (c[0] == ONE))
  {
    for (i = 1; i < nvec; i++)
    {
      xd = NV_COMPLEX_DATA_CS(X[i]);
      for (j = 0; j < N; j++) { zd[j] += c[i] * xd[j]; }
    }
    return SUN_SUCCESS;
  }

  /*
   * X[0] = c[0] * X[0] + sum{ c[i] * X[i] }, i = 1,...,nvec-1
   */
  if (X[0] == z)
  {
    for (j = 0; j < N; j++) { zd[j] *= c[0]; }
    for (i = 1; i < nvec; i++)
    {
      xd = NV_COMPLEX_DATA_CS(X[i]);
      for (j = 0; j < N; j++) { zd[j] += c[i] * xd[j]; }
    }
    return SUN_SUCCESS;
  }

  /*
   * z = sum{ c[i] * X[i] }, i = 0,...,nvec-1
   */
  xd = NV_COMPLEX_DATA_CS(X[0]);
  for (j = 0; j < N; j++) { zd[j] = c[0] * xd[j]; }
  for (i = 1; i < nvec; i++)
  {
    xd = NV_COMPLEX_DATA_CS(X[i]);
    for (j = 0; j < N; j++) { zd[j] += c[i] * xd[j]; }
  }

  return SUN_SUCCESS;
}

SUNErrCode N_VScaleAddMulti_SComplex(int nvec, suncomplextype* a, N_Vector x,
                                 N_Vector* Y, N_Vector* Z)
{  
  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;
  suncomplextype* zd = NULL;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* should have called N_VLinearSum */
  if (nvec == 1)
  {
    N_VLinearSum_SComplex(a[0], x, (suncomplextype)ONE, Y[0], Z[0]);
    return SUN_SUCCESS;
  }

  /* get vector length and data array */
  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);

  /*
   * Y[i][j] += a[i] * x[j]
   */
  if (Y == Z)
  {
    for (i = 0; i < nvec; i++)
    {
      yd = NV_COMPLEX_DATA_CS(Y[i]);
      for (j = 0; j < N; j++) { yd[j] += a[i] * xd[j]; }
    }
    return SUN_SUCCESS;
  }

  /*
   * Z[i][j] = Y[i][j] + a[i] * x[j]
   */
  for (i = 0; i < nvec; i++)
  {
    yd = NV_COMPLEX_DATA_CS(Y[i]);
    zd = NV_COMPLEX_DATA_CS(Z[i]);
    for (j = 0; j < N; j++) { zd[j] = a[i] * xd[j] + yd[j]; }
  }
  return SUN_SUCCESS;
}

SUNErrCode N_VScaleAddMulti_Real(int nvec, sunrealtype* a, N_Vector x,
                                 N_Vector* Y, N_Vector* Z)
{  
  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;
  suncomplextype* zd = NULL;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* should have called N_VLinearSum */
  if (nvec == 1)
  {
    N_VLinearSum_SComplex(a[0], x, ONE, Y[0], Z[0]);
    return SUN_SUCCESS;
  }

  /* get vector length and data array */
  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);

  /*
   * Y[i][j] += a[i] * x[j]
   */
  if (Y == Z)
  {
    for (i = 0; i < nvec; i++)
    {
      yd = NV_COMPLEX_DATA_CS(Y[i]);
      for (j = 0; j < N; j++) { yd[j] += a[i] * xd[j]; }
    }
    return SUN_SUCCESS;
  }

  /*
   * Z[i][j] = Y[i][j] + a[i] * x[j]
   */
  for (i = 0; i < nvec; i++)
  {
    yd = NV_COMPLEX_DATA_CS(Y[i]);
    zd = NV_COMPLEX_DATA_CS(Z[i]);
    for (j = 0; j < N; j++) { zd[j] = a[i] * xd[j] + yd[j]; }
  }
  return SUN_SUCCESS;
}

SUNErrCode N_VDotProdMulti_SComplex(int nvec, N_Vector x, N_Vector* Y,
                                suncomplextype* dotprods)
{
  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* should have called N_VDotProd */
  if (nvec == 1)
  {
    dotprods[0] = N_VDotProd_SComplex(Y[0], x);
    return SUN_SUCCESS;
  }

  /* get vector length and data array */
  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);

  /* compute multiple dot products */
  for (i = 0; i < nvec; i++)
  {
    yd          = NV_COMPLEX_DATA_CS(Y[i]);
    dotprods[i] = (suncomplextype)ZERO;
    for (j = 0; j < N; j++) { dotprods[i] += conj(yd[j]) * xd[j]; } //Amihere
  }

  return SUN_SUCCESS;
}

SUNErrCode N_VDotProdMulti_Real(int nvec, N_Vector x, N_Vector* Y,
                                sunrealtype* dotprods)
{

  printf("\nCalling N_VDotProdMulti_Real, that must cause an error!\n\n");

  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* should have called N_VDotProd */
  if (nvec == 1)
  {
    dotprods[0] = N_VDotProd_SComplex(Y[0], x);
    return SUN_SUCCESS;
  }

  /* get vector length and data array */
  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);

  /* compute multiple dot products */
  for (i = 0; i < nvec; i++)
  {
    yd          = NV_COMPLEX_DATA_CS(Y[i]);
    dotprods[i] = ZERO;
    for (j = 0; j < N; j++) { dotprods[i] +=  yd[j] * xd[j]; }
  }

  return SUN_SUCCESS;
}


/*
 * -----------------------------------------------------------------
 * vector array operations
 * -----------------------------------------------------------------
 */

SUNErrCode N_VLinearSumVectorArray_SComplex(int nvec, suncomplextype a, N_Vector* X,
                                        suncomplextype b, N_Vector* Y, N_Vector* Z)
{
  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;
  suncomplextype* zd = NULL;
  suncomplextype c;
  N_Vector* V1;
  N_Vector* V2;
  sunbooleantype test;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* should have called N_VLinearSum */
  if (nvec == 1)
  {
    N_VLinearSum_SComplex(a, X[0], b, Y[0], Z[0]);
    return SUN_SUCCESS;
  }

  /* BLAS usage: axpy y <- ax+y */
  if (((creal(b) == ONE) && (cimag(b)==ZERO)) && (Z == Y))
  {
    VaxpyVectorArray_SComplex(nvec, a, X, Y);
    return SUN_SUCCESS;
  }

  /* BLAS usage: axpy x <- by+x */
  if (((creal(a) == ONE) && (cimag(a)==ZERO)) && (Z == X))
  {
    VaxpyVectorArray_SComplex(nvec, b, Y, X);
    return SUN_SUCCESS;
  }

  /* Case: a == b == 1.0 */
  if (((creal(a) == ONE) && (cimag(a)==ZERO)) && ((creal(b) == ONE) && (cimag(b)==ZERO)))
  {
    VSumVectorArray_SComplex(nvec, X, Y, Z);
    return SUN_SUCCESS;
  }

  /* Cases:                    */
  /*   (1) a == 1.0, b = -1.0, */
  /*   (2) a == -1.0, b == 1.0 */
  if ((test = (((creal(a) == ONE) && (cimag(a)==ZERO)) && ((creal(b) == -ONE) && (cimag(b)==ZERO)))) || (((creal(a) == -ONE) && (cimag(a)==ZERO)) && ((creal(b) == ONE) && (cimag(b)==ZERO))))
  {
    V1 = test ? Y : X;
    V2 = test ? X : Y;
    VDiffVectorArray_SComplex(nvec, V2, V1, Z);
    return SUN_SUCCESS;
  }

  /* Cases:                                                  */
  /*   (1) a == 1.0, b == other or 0.0,                      */
  /*   (2) a == other or 0.0, b == 1.0                       */
  /* if a or b is 0.0, then user should have called N_VScale */
  if ((test = ((creal(a) == ONE) && (cimag(a)==ZERO))) || ((creal(b) == ONE) && (cimag(b)==ZERO)))
  {
    c  = test ? b : a;
    V1 = test ? Y : X;
    V2 = test ? X : Y;
    VLin1VectorArray_SComplex(nvec, c, V1, V2, Z);
    return SUN_SUCCESS;
  }

  /* Cases:                     */
  /*   (1) a == -1.0, b != 1.0, */
  /*   (2) a != 1.0, b == -1.0  */
  if ((test = ((creal(a) == -ONE) && (cimag(a)==ZERO))) || ((creal(b) == -ONE) && (cimag(b)==ZERO)))
  {
    c  = test ? b : a;
    V1 = test ? Y : X;
    V2 = test ? X : Y;
    VLin2VectorArray_SComplex(nvec, c, V1, V2, Z);
    return SUN_SUCCESS;
  }

  /* Case: a == b                                                         */
  /* catches case both a and b are 0.0 - user should have called N_VConst */
  if (a == b)
  {
    VScaleSumVectorArray_SComplex(nvec, a, X, Y, Z);
    return SUN_SUCCESS;
  }

  /* Case: a == -b */
  if (a == -b)
  {
    VScaleDiffVectorArray_SComplex(nvec, a, X, Y, Z);
    return SUN_SUCCESS;
  }

  /* Do all cases not handled above:                               */
  /*   (1) a == other, b == 0.0 - user should have called N_VScale */
  /*   (2) a == 0.0, b == other - user should have called N_VScale */
  /*   (3) a,b == other, a !=b, a != -b                            */

  /* get vector length */
  N = NV_LENGTH_CS(Z[0]);

  /* compute linear sum for each vector pair in vector arrays */
  for (i = 0; i < nvec; i++)
  {
    xd = NV_COMPLEX_DATA_CS(X[i]);
    yd = NV_COMPLEX_DATA_CS(Y[i]);
    zd = NV_COMPLEX_DATA_CS(Z[i]);
    for (j = 0; j < N; j++) { zd[j] = a * xd[j] + b * yd[j]; }
  }

  return SUN_SUCCESS;
}

SUNErrCode N_VLinearSumVectorArray_Real(int nvec, sunrealtype a, N_Vector* X,
                                        sunrealtype b, N_Vector* Y, N_Vector* Z)
{
  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;
  suncomplextype* zd = NULL;
  sunrealtype c;
  N_Vector* V1;
  N_Vector* V2;
  sunbooleantype test;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* should have called N_VLinearSum */
  if (nvec == 1)
  {
    N_VLinearSum_SComplex(a, X[0], b, Y[0], Z[0]);
    return SUN_SUCCESS;
  }

  /* BLAS usage: axpy y <- ax+y */
  if (((creal(b) == ONE) && (cimag(b)==ZERO)) && (Z == Y))
  {
    VaxpyVectorArray_SComplex(nvec, a, X, Y);
    return SUN_SUCCESS;
  }

  /* BLAS usage: axpy x <- by+x */
  if (((creal(a) == ONE) && (cimag(a)==ZERO)) && (Z == X))
  {
    VaxpyVectorArray_SComplex(nvec, b, Y, X);
    return SUN_SUCCESS;
  }

  /* Case: a == b == 1.0 */
  if (((creal(a) == ONE) && (cimag(a)==ZERO)) && ((creal(b) == ONE) && (cimag(b)==ZERO)))
  {
    VSumVectorArray_SComplex(nvec, X, Y, Z);
    return SUN_SUCCESS;
  }

  /* Cases:                    */
  /*   (1) a == 1.0, b = -1.0, */
  /*   (2) a == -1.0, b == 1.0 */
  if ((test = (((creal(a) == ONE) && (cimag(a)==ZERO)) && ((creal(b) == -ONE) && (cimag(b)==ZERO)))) || (((creal(a) == -ONE) && (cimag(a)==ZERO)) && ((creal(b) == ONE) && (cimag(b)==ZERO))))
  {
    V1 = test ? Y : X;
    V2 = test ? X : Y;
    VDiffVectorArray_SComplex(nvec, V2, V1, Z);
    return SUN_SUCCESS;
  }

  /* Cases:                                                  */
  /*   (1) a == 1.0, b == other or 0.0,                      */
  /*   (2) a == other or 0.0, b == 1.0                       */
  /* if a or b is 0.0, then user should have called N_VScale */
  if ((test = ((creal(a) == ONE) && (cimag(a)==ZERO))) || ((creal(b) == ONE) && (cimag(b)==ZERO)))
  {
    c  = test ? b : a;
    V1 = test ? Y : X;
    V2 = test ? X : Y;
    VLin1VectorArray_SComplex(nvec, c, V1, V2, Z);
    return SUN_SUCCESS;
  }

  /* Cases:                     */
  /*   (1) a == -1.0, b != 1.0, */
  /*   (2) a != 1.0, b == -1.0  */
  if ((test = ((creal(a) == -ONE) && (cimag(a)==ZERO))) || ((creal(b) == -ONE) && (cimag(b)==ZERO)))
  {
    c  = test ? b : a;
    V1 = test ? Y : X;
    V2 = test ? X : Y;
    VLin2VectorArray_SComplex(nvec, c, V1, V2, Z);
    return SUN_SUCCESS;
  }

  /* Case: a == b                                                         */
  /* catches case both a and b are 0.0 - user should have called N_VConst */
  if (a == b)
  {
    VScaleSumVectorArray_SComplex(nvec, a, X, Y, Z);
    return SUN_SUCCESS;
  }

  /* Case: a == -b */
  if (a == -b)
  {
    VScaleDiffVectorArray_SComplex(nvec, a, X, Y, Z);
    return SUN_SUCCESS;
  }

  /* Do all cases not handled above:                               */
  /*   (1) a == other, b == 0.0 - user should have called N_VScale */
  /*   (2) a == 0.0, b == other - user should have called N_VScale */
  /*   (3) a,b == other, a !=b, a != -b                            */

  /* get vector length */
  N = NV_LENGTH_CS(Z[0]);

  /* compute linear sum for each vector pair in vector arrays */
  for (i = 0; i < nvec; i++)
  {
    xd = NV_COMPLEX_DATA_CS(X[i]);
    yd = NV_COMPLEX_DATA_CS(Y[i]);
    zd = NV_COMPLEX_DATA_CS(Z[i]);
    for (j = 0; j < N; j++) { zd[j] = a * xd[j] + b * yd[j]; }
  }

  return SUN_SUCCESS;
}

SUNErrCode N_VScaleVectorArray_SComplex(int nvec, suncomplextype* c, N_Vector* X,
                                    N_Vector* Z)
{
  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* zd = NULL;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* should have called N_VScale */
  if (nvec == 1)
  {
    N_VScale_SComplex(c[0], X[0], Z[0]);
    return SUN_SUCCESS;
  }

  /* get vector length */
  N = NV_LENGTH_CS(Z[0]);

  /*
   * X[i] *= c[i]
   */
  if (X == Z)
  {
    for (i = 0; i < nvec; i++)
    {
      xd = NV_COMPLEX_DATA_CS(X[i]);
      for (j = 0; j < N; j++) { xd[j] *= c[i]; }
    }
    return SUN_SUCCESS;
  }

  /*
   * Z[i] = c[i] * X[i]
   */
  for (i = 0; i < nvec; i++)
  {
    xd = NV_COMPLEX_DATA_CS(X[i]);
    zd = NV_COMPLEX_DATA_CS(Z[i]);
    for (j = 0; j < N; j++) { zd[j] = c[i] * xd[j]; }
  }
  return SUN_SUCCESS;
}

SUNErrCode N_VScaleVectorArray_Real(int nvec, sunrealtype* c, N_Vector* X,
                                    N_Vector* Z)
{  
  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* zd = NULL;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* should have called N_VScale */
  if (nvec == 1)
  {
    N_VScale_SComplex(c[0], X[0], Z[0]);
    return SUN_SUCCESS;
  }

  /* get vector length */
  N = NV_LENGTH_CS(Z[0]);

  /*
   * X[i] *= c[i]
   */
  if (X == Z)
  {
    for (i = 0; i < nvec; i++)
    {
      xd = NV_COMPLEX_DATA_CS(X[i]);
      for (j = 0; j < N; j++) { xd[j] *= c[i]; }
    }
    return SUN_SUCCESS;
  }

  /*
   * Z[i] = c[i] * X[i]
   */
  for (i = 0; i < nvec; i++)
  {
    xd = NV_COMPLEX_DATA_CS(X[i]);
    zd = NV_COMPLEX_DATA_CS(Z[i]);
    for (j = 0; j < N; j++) { zd[j] = c[i] * xd[j]; }
  }
  return SUN_SUCCESS;
}

SUNErrCode N_VConstVectorArray_SComplex(int nvec, suncomplextype c, N_Vector* Z)
{
  int i;
  sunindextype j, N;
  suncomplextype* zd = NULL;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* should have called N_VConst */
  if (nvec == 1)
  {
    N_VConst_SComplex((suncomplextype)ONE, Z[0]);
    return SUN_SUCCESS;
  }

  /* get vector length */
  N = NV_LENGTH_CS(Z[0]);

  /* set each vector in the vector array to a constant */
  for (i = 0; i < nvec; i++)
  {
    zd = NV_COMPLEX_DATA_CS(Z[i]);
    for (j = 0; j < N; j++) { zd[j] = c; }
  }

  return SUN_SUCCESS;
}

SUNErrCode N_VConstVectorArray_Real(int nvec, sunrealtype c, N_Vector* Z)
{
  int i;
  sunindextype j, N;
  suncomplextype* zd = NULL;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* should have called N_VConst */
  if (nvec == 1)
  {
    N_VConst_SComplex(ONE, Z[0]);
    return SUN_SUCCESS;
  }

  /* get vector length */
  N = NV_LENGTH_CS(Z[0]);

  /* set each vector in the vector array to a constant */
  for (i = 0; i < nvec; i++)
  {
    zd = NV_COMPLEX_DATA_CS(Z[i]);
    for (j = 0; j < N; j++) { zd[j] = (suncomplextype)c; }
  }

  return SUN_SUCCESS;
}

SUNErrCode N_VWrmsNormVectorArray_SComplex(int nvec, N_Vector* X, N_Vector* W,
                                       sunrealtype* nrm)
{
  int i;
  sunindextype j, N;
  suncomplextype* wd = NULL;
  suncomplextype* xd = NULL;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* should have called N_VWrmsNorm */
  if (nvec == 1)
  {
    nrm[0] = N_VWrmsNorm_SComplex(X[0], W[0]);
    return SUN_SUCCESS;
  }

  /* get vector length */
  N = NV_LENGTH_CS(X[0]);

  /* compute the WRMS norm for each vector in the vector array */
  for (i = 0; i < nvec; i++)
  {
    xd     = NV_COMPLEX_DATA_CS(X[i]);
    wd     = NV_COMPLEX_DATA_CS(W[i]);
    nrm[i] = ZERO;
    for (j = 0; j < N; j++) { nrm[i] += SUNSQR(SUNCabs(xd[j] * wd[j])); }
    nrm[i] = SUNRsqrt(nrm[i] / N);
  }

  return SUN_SUCCESS;
}

SUNErrCode N_VScaleAddMultiVectorArray_SComplex(int nvec, int nsum,
                                            suncomplextype* a, N_Vector* X,
                                            N_Vector** Y, N_Vector** Z)
{
  int i, j;
  sunindextype k, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;
  suncomplextype* zd = NULL;
  N_Vector* YY;
  N_Vector* ZZ;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* ---------------------------
   * Special cases for nvec == 1
   * --------------------------- */

  if (nvec == 1)
  {
    /* should have called N_VLinearSum */
    if (nsum == 1)
    {
      N_VLinearSum_SComplex(a[0], X[0], (suncomplextype)ONE, Y[0][0], Z[0][0]);

      return SUN_SUCCESS;
    }

    /* should have called N_VScaleAddMulti */
    YY = (N_Vector*)malloc(nsum * sizeof(N_Vector));
    if (YY == NULL) { return SUN_ERR_MALLOC_FAIL; }
    ZZ = (N_Vector*)malloc(nsum * sizeof(N_Vector));
    if (ZZ == NULL) { return SUN_ERR_MALLOC_FAIL; }

    for (j = 0; j < nsum; j++)
    {
      YY[j] = Y[j][0];
      ZZ[j] = Z[j][0];
    }

    N_VScaleAddMulti_SComplex(nsum, a, X[0], YY, ZZ);

    free(YY);
    free(ZZ);

    return SUN_SUCCESS;
  }

  /* --------------------------
   * Special cases for nvec > 1
   * -------------------------- */

  /* should have called N_VLinearSumVectorArray */
  if (nsum == 1)
  {
    N_VLinearSumVectorArray_SComplex(nvec, a[0], X, (suncomplextype)ONE, Y[0], Z[0]);
    return SUN_SUCCESS;
  }

  /* ----------------------------
   * Compute multiple linear sums
   * ---------------------------- */

  /* get vector length */
  N = NV_LENGTH_CS(X[0]);

  /*
   * Y[i][j] += a[i] * x[j]
   */
  if (Y == Z)
  {
    for (i = 0; i < nvec; i++)
    {
      xd = NV_COMPLEX_DATA_CS(X[i]);
      for (j = 0; j < nsum; j++)
      {
        yd = NV_COMPLEX_DATA_CS(Y[j][i]);
        for (k = 0; k < N; k++) { yd[k] += a[j] * xd[k]; }
      }
    }
    return SUN_SUCCESS;
  }

  /*
   * Z[i][j] = Y[i][j] + a[i] * x[j]
   */
  for (i = 0; i < nvec; i++)
  {
    xd = NV_COMPLEX_DATA_CS(X[i]);
    for (j = 0; j < nsum; j++)
    {
      yd = NV_COMPLEX_DATA_CS(Y[j][i]);
      zd = NV_COMPLEX_DATA_CS(Z[j][i]);
      for (k = 0; k < N; k++) { zd[k] = a[j] * xd[k] + yd[k]; }
    }
  }
  return SUN_SUCCESS;
}

SUNErrCode N_VScaleAddMultiVectorArray_Real(int nvec, int nsum,
                                            sunrealtype* a, N_Vector* X,
                                            N_Vector** Y, N_Vector** Z)
{
  int i, j;
  sunindextype k, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;
  suncomplextype* zd = NULL;
  N_Vector* YY;
  N_Vector* ZZ;

  /* invalid number of vectors */
  if (nvec < 0) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* ---------------------------
   * Special cases for nvec == 1
   * --------------------------- */

  if (nvec == 1)
  {
    /* should have called N_VLinearSum */
    if (nsum == 1)
    {
      N_VLinearSum_SComplex(a[0], X[0], ONE, Y[0][0], Z[0][0]);

      return SUN_SUCCESS;
    }

    /* should have called N_VScaleAddMulti */
    YY = (N_Vector*)malloc(nsum * sizeof(N_Vector));
    if (YY == NULL) { return SUN_ERR_MALLOC_FAIL; }
    ZZ = (N_Vector*)malloc(nsum * sizeof(N_Vector));
    if (ZZ == NULL) { return SUN_ERR_MALLOC_FAIL; }

    for (j = 0; j < nsum; j++)
    {
      YY[j] = Y[j][0];
      ZZ[j] = Z[j][0];
    }

    N_VScaleAddMulti_SComplex(nsum, (suncomplextype*)a, X[0], YY, ZZ);

    free(YY);
    free(ZZ);

    return SUN_SUCCESS;
  }

  /* --------------------------
   * Special cases for nvec > 1
   * -------------------------- */

  /* should have called N_VLinearSumVectorArray */
  if (nsum == 1)
  {
    N_VLinearSumVectorArray_SComplex(nvec, a[0], X, ONE, Y[0], Z[0]);
    return SUN_SUCCESS;
  }

  /* ----------------------------
   * Compute multiple linear sums
   * ---------------------------- */

  /* get vector length */
  N = NV_LENGTH_CS(X[0]);

  /*
   * Y[i][j] += a[i] * x[j]
   */
  if (Y == Z)
  {
    for (i = 0; i < nvec; i++)
    {
      xd = NV_COMPLEX_DATA_CS(X[i]);
      for (j = 0; j < nsum; j++)
      {
        yd = NV_COMPLEX_DATA_CS(Y[j][i]);
        for (k = 0; k < N; k++) { yd[k] += a[j] * xd[k]; }
      }
    }
    return SUN_SUCCESS;
  }

  /*
   * Z[i][j] = Y[i][j] + a[i] * x[j]
   */
  for (i = 0; i < nvec; i++)
  {
    xd = NV_COMPLEX_DATA_CS(X[i]);
    for (j = 0; j < nsum; j++)
    {
      yd = NV_COMPLEX_DATA_CS(Y[j][i]);
      zd = NV_COMPLEX_DATA_CS(Z[j][i]);
      for (k = 0; k < N; k++) { zd[k] = a[j] * xd[k] + yd[k]; }
    }
  }
  return SUN_SUCCESS;
}

SUNErrCode N_VLinearCombinationVectorArray_SComplex(int nvec, int nsum,
                                                suncomplextype* c, N_Vector** X,
                                                N_Vector* Z)
{
  int i;          /* vector arrays index in summation [0,nsum) */
  int j;          /* vector index in vector array     [0,nvec) */
  sunindextype k; /* element index in vector          [0,N)    */
  sunindextype N;
  suncomplextype* zd = NULL;
  suncomplextype* xd = NULL;
  suncomplextype* ctmp;
  N_Vector* Y;

  /* invalid number of vectors */
  if ((nvec < 1) || (nsum < 1)) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* ---------------------------
   * Special cases for nvec == 1
   * --------------------------- */

  if (nvec == 1)
  {
    /* should have called N_VScale */
    if (nsum == 1)
    {
      N_VScale_SComplex(c[0], X[0][0], Z[0]);
      return SUN_SUCCESS;
    }

    /* should have called N_VLinearSum */
    if (nsum == 2)
    {
      N_VLinearSum_SComplex(c[0], X[0][0], c[1], X[1][0], Z[0]);
      return SUN_SUCCESS;
    }

    /* should have called N_VLinearCombination */
    Y = (N_Vector*)malloc(nsum * sizeof(N_Vector));
    if (Y == NULL) { return SUN_ERR_MALLOC_FAIL; }

    for (i = 0; i < nsum; i++) { Y[i] = X[i][0]; }

    N_VLinearCombination_SComplex(nsum, c, Y, Z[0]);

    free(Y);

    return SUN_SUCCESS;
  }

  /* --------------------------
   * Special cases for nvec > 1
   * -------------------------- */

  /* should have called N_VScaleVectorArray */
  if (nsum == 1)
  {
    ctmp = (suncomplextype*)malloc(nvec * sizeof(suncomplextype));
    if (ctmp == NULL) { return SUN_ERR_MALLOC_FAIL; }

    for (j = 0; j < nvec; j++) { ctmp[j] = c[0]; }

    N_VScaleVectorArray_SComplex(nvec, ctmp, X[0], Z);

    free(ctmp);
    return SUN_SUCCESS;
  }

  /* should have called N_VLinearSumVectorArray */
  if (nsum == 2)
  {
    N_VLinearSumVectorArray_SComplex(nvec, c[0], X[0], c[1], X[1], Z);
    return SUN_SUCCESS;
  }

  /* --------------------------
   * Compute linear combination
   * -------------------------- */

  /* get vector length */
  N = NV_LENGTH_CS(Z[0]);

  /*
   * X[0][j] += c[i]*X[i][j], i = 1,...,nvec-1
   */
  if ((X[0] == Z) && ((creal(c[0]) == ONE) && cimag(c[0]) == ZERO))
  {
    for (j = 0; j < nvec; j++)
    {
      zd = NV_COMPLEX_DATA_CS(Z[j]);
      for (i = 1; i < nsum; i++)
      {
        xd = NV_COMPLEX_DATA_CS(X[i][j]);
        for (k = 0; k < N; k++) { zd[k] += c[i] * xd[k]; }
      }
    }
    return SUN_SUCCESS;
  }

  /*
   * X[0][j] = c[0] * X[0][j] + sum{ c[i] * X[i][j] }, i = 1,...,nvec-1
   */
  if (X[0] == Z)
  {
    for (j = 0; j < nvec; j++)
    {
      zd = NV_COMPLEX_DATA_CS(Z[j]);
      for (k = 0; k < N; k++) { zd[k] *= c[0]; }
      for (i = 1; i < nsum; i++)
      {
        xd = NV_COMPLEX_DATA_CS(X[i][j]);
        for (k = 0; k < N; k++) { zd[k] += c[i] * xd[k]; }
      }
    }
    return SUN_SUCCESS;
  }

  /*
   * Z[j] = sum{ c[i] * X[i][j] }, i = 0,...,nvec-1
   */
  for (j = 0; j < nvec; j++)
  {
    xd = NV_COMPLEX_DATA_CS(X[0][j]);
    zd = NV_COMPLEX_DATA_CS(Z[j]);
    for (k = 0; k < N; k++) { zd[k] = c[0] * xd[k]; }
    for (i = 1; i < nsum; i++)
    {
      xd = NV_COMPLEX_DATA_CS(X[i][j]);
      for (k = 0; k < N; k++) { zd[k] += c[i] * xd[k]; }
    }
  }
  return SUN_SUCCESS;
}

SUNErrCode N_VLinearCombinationVectorArray_Real(int nvec, int nsum,
                                                sunrealtype* c, N_Vector** X,
                                                N_Vector* Z)
{
  int i;          /* vector arrays index in summation [0,nsum) */
  int j;          /* vector index in vector array     [0,nvec) */
  sunindextype k; /* element index in vector          [0,N)    */
  sunindextype N;
  suncomplextype* zd = NULL;
  suncomplextype* xd = NULL;
  suncomplextype* ctmp;
  N_Vector* Y;

  /* invalid number of vectors */
  if ((nvec < 1) || (nsum < 1)) { return SUN_ERR_ARG_OUTOFRANGE; }

  /* ---------------------------
   * Special cases for nvec == 1
   * --------------------------- */

  if (nvec == 1)
  {
    /* should have called N_VScale */
    if (nsum == 1)
    {
      N_VScale_SComplex(c[0], X[0][0], Z[0]);
      return SUN_SUCCESS;
    }

    /* should have called N_VLinearSum */
    if (nsum == 2)
    {
      N_VLinearSum_SComplex(c[0], X[0][0], c[1], X[1][0], Z[0]);
      return SUN_SUCCESS;
    }

    /* should have called N_VLinearCombination */
    Y = (N_Vector*)malloc(nsum * sizeof(N_Vector));
    if (Y == NULL) { return SUN_ERR_MALLOC_FAIL; }

    for (i = 0; i < nsum; i++) { Y[i] = X[i][0]; }

    N_VLinearCombination_SComplex(nsum, (suncomplextype*)c, Y, Z[0]);

    free(Y);

    return SUN_SUCCESS;
  }

  /* --------------------------
   * Special cases for nvec > 1
   * -------------------------- */

  /* should have called N_VScaleVectorArray */
  if (nsum == 1)
  {
    ctmp = (suncomplextype*)malloc(nvec * sizeof(suncomplextype));
    if (ctmp == NULL) { return SUN_ERR_MALLOC_FAIL; }

    for (j = 0; j < nvec; j++) { ctmp[j] = (suncomplextype)c[0]; }

    N_VScaleVectorArray_SComplex(nvec, ctmp, X[0], Z);

    free(ctmp);
    return SUN_SUCCESS;
  }

  /* should have called N_VLinearSumVectorArray */
  if (nsum == 2)
  {
    N_VLinearSumVectorArray_SComplex(nvec, c[0], X[0], c[1], X[1], Z);
    return SUN_SUCCESS;
  }

  /* --------------------------
   * Compute linear combination
   * -------------------------- */

  /* get vector length */
  N = NV_LENGTH_CS(Z[0]);

  /*
   * X[0][j] += c[i]*X[i][j], i = 1,...,nvec-1
   */
  if ((X[0] == Z) && ((creal(c[0]) == ONE) && cimag(c[0]) == ZERO))
  {
    for (j = 0; j < nvec; j++)
    {
      zd = NV_COMPLEX_DATA_CS(Z[j]);
      for (i = 1; i < nsum; i++)
      {
        xd = NV_COMPLEX_DATA_CS(X[i][j]);
        for (k = 0; k < N; k++) { zd[k] += c[i] * xd[k]; }
      }
    }
    return SUN_SUCCESS;
  }

  /*
   * X[0][j] = c[0] * X[0][j] + sum{ c[i] * X[i][j] }, i = 1,...,nvec-1
   */
  if (X[0] == Z)
  {
    for (j = 0; j < nvec; j++)
    {
      zd = NV_COMPLEX_DATA_CS(Z[j]);
      for (k = 0; k < N; k++) { zd[k] *= c[0]; }
      for (i = 1; i < nsum; i++)
      {
        xd = NV_COMPLEX_DATA_CS(X[i][j]);
        for (k = 0; k < N; k++) { zd[k] += c[i] * xd[k]; }
      }
    }
    return SUN_SUCCESS;
  }

  /*
   * Z[j] = sum{ c[i] * X[i][j] }, i = 0,...,nvec-1
   */
  for (j = 0; j < nvec; j++)
  {
    xd = NV_COMPLEX_DATA_CS(X[0][j]);
    zd = NV_COMPLEX_DATA_CS(Z[j]);
    for (k = 0; k < N; k++) { zd[k] = c[0] * xd[k]; }
    for (i = 1; i < nsum; i++)
    {
      xd = NV_COMPLEX_DATA_CS(X[i][j]);
      for (k = 0; k < N; k++) { zd[k] += c[i] * xd[k]; }
    }
  }
  return SUN_SUCCESS;
}

/*
 * -----------------------------------------------------------------
 * OPTIONAL XBraid interface operations
 * -----------------------------------------------------------------
 */

SUNErrCode N_VBufSize_SComplex(N_Vector x, sunindextype* size)
{
  *size = NV_LENGTH_CS(x) * ((sunindextype)sizeof(suncomplextype));
  return SUN_SUCCESS;
}

SUNErrCode N_VBufPack_SComplex(N_Vector x, void* buf)
{
  sunindextype i, N;
  suncomplextype* xd = NULL;
  suncomplextype* bd = NULL;

  if (buf == NULL) { return SUN_ERR_ARG_CORRUPT; }
  
  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  bd = (suncomplextype*)buf;

  for (i = 0; i < N; i++) { bd[i] = xd[i]; }

  return SUN_SUCCESS;
}

SUNErrCode N_VBufUnpack_SComplex(N_Vector x, void* buf)
{
  sunindextype i, N;
  suncomplextype* xd = NULL;
  suncomplextype* bd = NULL;

  if (buf == NULL) { return SUN_ERR_ARG_CORRUPT; }

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  bd = (suncomplextype*)buf;

  for (i = 0; i < N; i++) { xd[i] = bd[i]; }

  return SUN_SUCCESS;
}

/*
 * -----------------------------------------------------------------
 * private functions for special cases of vector operations
 * -----------------------------------------------------------------
 */

static void VCopy_SComplex(N_Vector x, N_Vector z)
{
  sunindextype i, N;
  suncomplextype *xd, *zd;

  xd = zd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = xd[i]; }

  return;
}

static void VSum_SComplex(N_Vector x, N_Vector y, N_Vector z)
{
  sunindextype i, N;
  suncomplextype *xd, *yd, *zd;

  xd = yd = zd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  yd = NV_COMPLEX_DATA_CS(y);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = xd[i] + yd[i]; }

  return;
}

static void VDiff_SComplex(N_Vector x, N_Vector y, N_Vector z)
{
  sunindextype i, N;
  suncomplextype *xd, *yd, *zd;

  xd = yd = zd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  yd = NV_COMPLEX_DATA_CS(y);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = xd[i] - yd[i]; }

  return;
}

static void VNeg_SComplex(N_Vector x, N_Vector z)
{
  sunindextype i, N;
  suncomplextype *xd, *zd;

  xd = zd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = -xd[i]; }

  return;
}

static void VScaleSum_SComplex(suncomplextype c, N_Vector x, N_Vector y, N_Vector z)
{
  sunindextype i, N;
  suncomplextype *xd, *yd, *zd;

  xd = yd = zd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  yd = NV_COMPLEX_DATA_CS(y);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = c * (xd[i] + yd[i]); }

  return;
}

static void VScaleDiff_SComplex(suncomplextype c, N_Vector x, N_Vector y, N_Vector z)
{
  sunindextype i, N;
  suncomplextype *xd, *yd, *zd;

  xd = yd = zd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  yd = NV_COMPLEX_DATA_CS(y);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = c * (xd[i] - yd[i]); }

  return;
}

static void VLin1_SComplex(suncomplextype a, N_Vector x, N_Vector y, N_Vector z)
{
  sunindextype i, N;
  suncomplextype *xd, *yd, *zd;

  xd = yd = zd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  yd = NV_COMPLEX_DATA_CS(y);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = (a * xd[i]) + yd[i]; }

  return;
}

static void VLin2_SComplex(suncomplextype a, N_Vector x, N_Vector y, N_Vector z)
{
  sunindextype i, N;
  suncomplextype *xd, *yd, *zd;

  xd = yd = zd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  yd = NV_COMPLEX_DATA_CS(y);
  zd = NV_COMPLEX_DATA_CS(z);

  for (i = 0; i < N; i++) { zd[i] = (a * xd[i]) - yd[i]; }

  return;
}

static void Vaxpy_SComplex(suncomplextype a, N_Vector x, N_Vector y)
{
  sunindextype i, N;
  suncomplextype *xd, *yd;

  xd = yd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);
  yd = NV_COMPLEX_DATA_CS(y);

  if ((creal(a) == ONE) && (cimag(a) == ZERO))
  {
    for (i = 0; i < N; i++) { yd[i] += xd[i]; }
    return;
  }

  if ((creal(a) == -ONE) && (cimag(a) == ZERO))
  {
    for (i = 0; i < N; i++) { yd[i] -= xd[i]; }
    return;
  }

  for (i = 0; i < N; i++) { yd[i] += a * xd[i]; }

  return;
}

static void VScaleBy_SComplex(suncomplextype a, N_Vector x)
{
  sunindextype i, N;
  suncomplextype* xd;

  xd = NULL;

  N  = NV_LENGTH_CS(x);
  xd = NV_COMPLEX_DATA_CS(x);

  for (i = 0; i < N; i++) { xd[i] *= a; }

  return;
}

/*
 * -----------------------------------------------------------------
 * private functions for special cases of vector array operations
 * -----------------------------------------------------------------
 */

static void VSumVectorArray_SComplex(int nvec, N_Vector* X, N_Vector* Y, N_Vector* Z)
{
  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;
  suncomplextype* zd = NULL;

  N = NV_LENGTH_CS(X[0]);

  for (i = 0; i < nvec; i++)
  {
    xd = NV_COMPLEX_DATA_CS(X[i]);
    yd = NV_COMPLEX_DATA_CS(Y[i]);
    zd = NV_COMPLEX_DATA_CS(Z[i]);
    for (j = 0; j < N; j++) { zd[j] = xd[j] + yd[j]; }
  }
}

static void VDiffVectorArray_SComplex(int nvec, N_Vector* X, N_Vector* Y,
                                  N_Vector* Z)
{
  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;
  suncomplextype* zd = NULL;

  N = NV_LENGTH_CS(X[0]);

  for (i = 0; i < nvec; i++)
  {
    xd = NV_COMPLEX_DATA_CS(X[i]);
    yd = NV_COMPLEX_DATA_CS(Y[i]);
    zd = NV_COMPLEX_DATA_CS(Z[i]);
    for (j = 0; j < N; j++) { zd[j] = xd[j] - yd[j]; }
  }
}

static void VScaleSumVectorArray_SComplex(int nvec, suncomplextype c, N_Vector* X,
                                      N_Vector* Y, N_Vector* Z)
{
  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;
  suncomplextype* zd = NULL;

  N = NV_LENGTH_CS(X[0]);

  for (i = 0; i < nvec; i++)
  {
    xd = NV_COMPLEX_DATA_CS(X[i]);
    yd = NV_COMPLEX_DATA_CS(Y[i]);
    zd = NV_COMPLEX_DATA_CS(Z[i]);
    for (j = 0; j < N; j++) { zd[j] = c * (xd[j] + yd[j]); }
  }
}

static void VScaleDiffVectorArray_SComplex(int nvec, suncomplextype c, N_Vector* X,
                                       N_Vector* Y, N_Vector* Z)
{
  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;
  suncomplextype* zd = NULL;

  N = NV_LENGTH_CS(X[0]);

  for (i = 0; i < nvec; i++)
  {
    xd = NV_COMPLEX_DATA_CS(X[i]);
    yd = NV_COMPLEX_DATA_CS(Y[i]);
    zd = NV_COMPLEX_DATA_CS(Z[i]);
    for (j = 0; j < N; j++) { zd[j] = c * (xd[j] - yd[j]); }
  }
}

static void VLin1VectorArray_SComplex(int nvec, suncomplextype a, N_Vector* X,
                                  N_Vector* Y, N_Vector* Z)
{
  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;
  suncomplextype* zd = NULL;

  N = NV_LENGTH_CS(X[0]);

  for (i = 0; i < nvec; i++)
  {
    xd = NV_COMPLEX_DATA_CS(X[i]);
    yd = NV_COMPLEX_DATA_CS(Y[i]);
    zd = NV_COMPLEX_DATA_CS(Z[i]);
    for (j = 0; j < N; j++) { zd[j] = (a * xd[j]) + yd[j]; }
  }
}

static void VLin2VectorArray_SComplex(int nvec, suncomplextype a, N_Vector* X,
                                  N_Vector* Y, N_Vector* Z)
{
  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;
  suncomplextype* zd = NULL;

  N = NV_LENGTH_CS(X[0]);

  for (i = 0; i < nvec; i++)
  {
    xd = NV_COMPLEX_DATA_CS(X[i]);
    yd = NV_COMPLEX_DATA_CS(Y[i]);
    zd = NV_COMPLEX_DATA_CS(Z[i]);
    for (j = 0; j < N; j++) { zd[j] = (a * xd[j]) - yd[j]; }
  }
}

static void VaxpyVectorArray_SComplex(int nvec, suncomplextype a, N_Vector* X,
                                  N_Vector* Y)
{
  int i;
  sunindextype j, N;
  suncomplextype* xd = NULL;
  suncomplextype* yd = NULL;

  N = NV_LENGTH_CS(X[0]);

  if ((creal(a) == ONE) && (cimag(a) == ZERO))
  {
    for (i = 0; i < nvec; i++)
    {
      xd = NV_COMPLEX_DATA_CS(X[i]);
      yd = NV_COMPLEX_DATA_CS(Y[i]);
      for (j = 0; j < N; j++) { yd[j] += xd[j]; }
    }
    return;
  }

  if ((creal(a) == -ONE) && (cimag(a) == ZERO))
  {
    for (i = 0; i < nvec; i++)
    {
      xd = NV_COMPLEX_DATA_CS(X[i]);
      yd = NV_COMPLEX_DATA_CS(Y[i]);
      for (j = 0; j < N; j++) { yd[j] -= xd[j]; }
    }
    return;
  }

  for (i = 0; i < nvec; i++)
  {
      xd = NV_COMPLEX_DATA_CS(X[i]);
      yd = NV_COMPLEX_DATA_CS(Y[i]);
    for (j = 0; j < N; j++) { yd[j] += a * xd[j]; }
  }
}

/*
 * -----------------------------------------------------------------
 * Enable / Disable fused and vector array operations
 * -----------------------------------------------------------------
 */

SUNErrCode N_VEnableFusedOps_SComplex(N_Vector v, sunbooleantype tf)
{
  if (tf)
  {
    /* enable all fused vector operations */
    // v->ops->nvlinearcombination = N_VLinearCombination_SComplex;
    v->ops->nvlinearcombination = N_VLinearCombination_Real;

    // v->ops->nvscaleaddmulti     = N_VScaleAddMulti_SComplex;
    v->ops->nvscaleaddmulti     = N_VScaleAddMulti_Real;
    
    // v->ops->nvdotprodmulti      = N_VDotProdMulti_SComplex;
    v->ops->nvdotprodmulti      = N_VDotProdMulti_Real;


    /* enable all vector array operations */
    // v->ops->nvlinearsumvectorarray     = N_VLinearSumVectorArray_SComplex;
    v->ops->nvlinearsumvectorarray     = N_VLinearSumVectorArray_Real;

    // v->ops->nvscalevectorarray         = N_VScaleVectorArray_SComplex;
    v->ops->nvscalevectorarray         = N_VScaleVectorArray_Real;


    // v->ops->nvconstvectorarray         = N_VConstVectorArray_SComplex;
    v->ops->nvconstvectorarray         = N_VConstVectorArray_Real;

    v->ops->nvwrmsnormvectorarray      = N_VWrmsNormVectorArray_SComplex;
    
    // v->ops->nvscaleaddmultivectorarray = N_VScaleAddMultiVectorArray_SComplex;
    v->ops->nvscaleaddmultivectorarray = N_VScaleAddMultiVectorArray_Real;
    
    // v->ops->nvlinearcombinationvectorarray = N_VLinearCombinationVectorArray_SComplex;
    v->ops->nvlinearcombinationvectorarray = N_VLinearCombinationVectorArray_Real;

    /* enable single buffer reduction operations */
    
    // v->ops->nvdotprodmultilocal = N_VDotProdMulti_SComplex;
    v->ops->nvdotprodmultilocal = N_VDotProdMulti_Real;

  }
  else
  {
    /* disable all fused vector operations */
    v->ops->nvlinearcombination = NULL;
    v->ops->nvscaleaddmulti     = NULL;
    v->ops->nvdotprodmulti      = NULL;
    /* disable all vector array operations */
    v->ops->nvlinearsumvectorarray         = NULL;
    v->ops->nvscalevectorarray             = NULL;
    v->ops->nvconstvectorarray             = NULL;
    v->ops->nvwrmsnormvectorarray          = NULL;
    v->ops->nvwrmsnormmaskvectorarray      = NULL;
    v->ops->nvscaleaddmultivectorarray     = NULL;
    v->ops->nvlinearcombinationvectorarray = NULL;
    /* disable single buffer reduction operations */
    v->ops->nvdotprodmultilocal = NULL;
  }

  /* return success */
  return SUN_SUCCESS;
}